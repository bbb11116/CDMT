import torch
import torch.nn as nn
import torchvision


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 1, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_circles),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            #LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)


    def _forward(self, pd_scores, pd_circles, anc_points, gt_labels, gt_circles, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_circles (torch.Tensor): Target circles with shape (bs, num_total_anchors, 3).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps, distence= self.get_pos_mask(
            pd_scores, pd_circles, gt_labels, gt_circles, anc_points, mask_gt
        )  # mask_pos 与真实框匹配的topk个先验框的mask，形状为(bs, max_num_obj, h*w)；
        # align_metric 为真实框与先验框的对齐度量，形状为(bs, max_num_obj, h*w)；
        # overlaps 为预测框与真实框的iou，形状为(bs, max_num_obj, h*w)；
        # distence 为预测框与真实框的距离，形状为(bs, max_num_obj, h*w)；

        # 选出topk个先验框中最高得分的的下标和掩码
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        # Assigned target
        target_labels, target_circles, target_scores = self.get_targets(gt_labels, gt_circles, target_gt_idx, fg_mask)
        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_circles, target_scores, fg_mask.bool(), target_gt_idx










    def get_pos_mask(self, pd_scores, pd_circles, gt_labels, gt_circles, anc_points, mask_gt):
        """Get positive mask for each ground truth circle.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_circles)  # 找出真实框内的锚点（掩码）（8, max_objects, 2550000）
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, distence = self.get_box_metrics(pd_scores, pd_circles, gt_labels, gt_circles, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps, distence




    def get_box_metrics(self, pd_scores, pd_circles, gt_labels, gt_circles, mask_gt):
        """Compute alignment metric given predicted and ground truth circles.根据预测和真实的圆计算对齐度量

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 2550000).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_circles.shape[-2] #2550000
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_circles.dtype, device=pd_circles.device)#GT与预测框的IOU[bs,max_objects,2550000]
        distence = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_circles.dtype, device=pd_circles.device)
        circles_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)#预测框得分[bs,max_objects,2550000]

        # ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        # ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj（表示批次索引）
        # ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj（表示真实框的类别）
        # # Get the scores of each grid for each gt cls
        # circles_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w
        # 因为 num_classes == 1，所有预测得分都属于同一类
        pd_scores_squeezed = pd_scores.squeeze(-1)  # (B, N)

        # circles_scores: (B, max_num_obj, N)
        # 我们要把 pd_scores_squeezed (B, N) 扩展到每个 GT（max_num_obj 次）
        # 方法：unsqueeze(1) -> (B, 1, N)，然后自动广播
        circles_scores = pd_scores_squeezed.unsqueeze(1).expand(-1, self.n_max_boxes, -1)

        # 注意：mask_gt 会自动筛选有效位置，所以上面可以直接赋值，无需复杂索引
        # 如果你后续只用 circles_scores[mask_gt]，那这样完全等价且安全
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_circles = pd_circles.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]#(bs,max_objects,2550000,3)*mask_gt->(bs*max_objects*2550000,3)
        gt_circles = gt_circles.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]#(bs,max_objects,2550000,3)*mask_gt->(bs*max_objects*2550000,3)
        overlaps[mask_gt] = circle_ious(gt_circles, pd_circles)#(bs,max_objects,2550000) 计算真实圆与预测圆的IOU
        distence[mask_gt] = center_distanceLoss(gt_circles, pd_circles)#(bs,max_objects,2550000) 计算真实圆与预测圆圆心的距离

        align_metric = circles_scores.pow(self.alpha) * overlaps.pow(self.beta) * distence.pow(self.beta)
        return align_metric, overlaps, distence #(bs,max_objects,2550000) 对齐度量  (bs,max_objects,2550000) IOU (bs,max_objects,2550000) 中心距离损失



    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)  从255000个锚点中选择topk个得分最高的锚点
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)#[b,max_objects,1]
        for k in range(self.topk):#对于每个真实目标，找到其第 k好的候选锚点在 h*w中的位置，然后在 count_tensor的对应位置上加 1。
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)#masked_fill_操作将所有计数大于 1 的位置重置为 0。
                                                            # 这一步确保了每个锚点对于同一个真实目标最多只被计算一次，防御性编程

        return count_tensor.to(metrics.dtype)#将 count_tensor的数据类型转换为与输入 metrics一致后返回。


    def get_targets(self, gt_labels, gt_circles, target_gt_idx, fg_mask):
        """Compute target labels, target circles, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_circles (torch.Tensor): Ground truth circles of shape (b, max_num_obj, 3).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_circles (torch.Tensor): Target circles for positive anchor points with shape (b, h*w, 3).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target circles, (b, max_num_obj, 3) -> (b, h*w, 3)
        target_circles = gt_circles.view(-1, gt_circles.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 1)
        # target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        # 因为 num_classes == 1，正样本类别得分 = 1

        target_scores[fg_mask.unsqueeze(-1).bool()] = 1.0


        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 1)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_circles, target_scores
        # target_labels：正样本的类别标签（形状(b, h * w)）
        # target_circles：正样本对应的真实框坐标（形状(b, h * w, 3)）
        # target_scores：正样本的one - hot类别分数（形状(b, h * w, num_classes)）


    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_circles, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_circles (torch.Tensor): Ground truth circles, shape (b, n_boxes, 3).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0] # 2550000
        bs, n_boxes, _ = gt_circles.shape # (8, max_objects, 3)
        xy, r = gt_circles.view(-1, 1, 3).split([2, 1], dim=2)  # center, radius
        diff = xy - xy_centers[None, :, :]  # (8*max_objects, 2550000, 2)
        r = r.view(-1, 1) # (8*max_objects, 1)
        distances = torch.norm(diff, dim=2)  # (8*max_objects, 2550000)
        return (distances <= r).view(bs, n_boxes, n_anchors).float()  # [8,max_objects,2550000] bool掩码

        # bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors,
        #                                                                                     -1)  # 计算锚点与真实框的偏移量
        #return bbox_deltas.amin(3).gt_(eps)  # [8,max_objects,2550000] bool掩码

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos
        # 目标索引 target_gt_idx(b, h*w) 、前景掩码 fg_mask(b, h*w)和更新后的正样本掩码 mask_pose(b, n_max_boxes, h*w)。
        #target_gt_idx: 每个锚点具体负责第几个物体。
        #fg_mask: 前景掩码，指示哪些锚点是正样本。
        #mask_pos: 更新后的正样本掩码，形状为 (b, n_max_boxes, h*w)，用于指示每个锚点是否负责某个真实目标。





def center_distanceLoss(gt_circles, pd_circles,):
    """计算真实圆与预测圆的中心距离损失

    Args:
        gt_circles (torch.Tensor): Ground truth circles with shape (bs*max_objects*2550000, 3).
        pd_circles (torch.Tensor): Predicted circles with shape (bs*max_objects*2550000, 3).

    Returns:
        center_distance_loss (torch.Tensor): Center distance loss between predicted and ground truth circles.
    """
    x1, y1, r1 = gt_circles.split(1, dim=1)  # (bs*max_objects*2550000, 1)
    x2, y2, r2 = pd_circles.split(1, dim=1)  # (bs*max_objects*2550000, 1)
    d = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # (bs*max_objects*2550000, 1)
    r = r1 + r2  # (bs*max_objects*2550000, 1)
    distance = torch.clamp((r - d) / r, min=0)  # (bs*max_objects*2550000, 1)
    return distance.squeeze(1)


def circle_ious(gt_circles: torch.Tensor, pd_circles: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU for M pairs of circles.
    Args:
        gt_circles: (M, 3) [x, y, r]
        pd_circles: (M, 3) [x, y, r]
    Returns:
        iou: (M,) float tensor in [0, 1]
    """
    c0 = gt_circles[:, :2]      # (M, 2)
    r0 = gt_circles[:, 2]       # (M,)
    c1 = pd_circles[:, :2]      # (M, 2)
    r1 = pd_circles[:, 2]       # (M,)

    inter_area = circle_intersection_area_tensor(c0, r0, c1, r1)
    area0 = torch.pi * r0.square()
    area1 = torch.pi * r1.square()
    union_area = area0 + area1 - inter_area

    iou = torch.where(union_area > 0, inter_area / union_area, torch.zeros_like(inter_area))
    return iou  # shape: (M,)

def circle_intersection_area_tensor(c0, r0, c1, r1):
    """（同前，已优化到效率天花板）"""
    d = torch.linalg.norm(c0 - c1, dim=-1)
    no_inter = d >= r0 + r1
    contained = d <= torch.abs(r0 - r1)
    area = torch.zeros_like(d)

    # 包含情况：小圆面积
    area = torch.where(
        contained,
        torch.pi * torch.min(r0, r1) ** 2,
        area
    )

    # 仅处理相交样本（95%+样本直接跳过！）
    mask = (~no_inter) & (~contained)
    if not mask.any():
        return area

    # 精准切片
    dm, r0m, r1m = d[mask], r0[mask], r1[mask]

    # 防浮点误差
    t0 = (dm ** 2 + r0m ** 2 - r1m ** 2) / (2 * dm * r0m)
    t1 = (dm ** 2 + r1m ** 2 - r0m ** 2) / (2 * dm * r1m)
    t0, t1 = t0.clamp(-1.0, 1.0), t1.clamp(-1.0, 1.0)

    # 海伦公式安全计算
    sq = torch.sqrt(torch.clamp(
        (-dm + r0m + r1m) * (dm + r0m - r1m) * (dm - r0m + r1m) * (dm + r0m + r1m),
        min=0.0
    ))

    # 计算相交面积
    area_m = r0m ** 2 * torch.acos(t0) + r1m ** 2 * torch.acos(t1) - 0.5 * sq
    area[mask] = area_m

    return area


import torch

import torch

import torch


def make_anchor(feats, stride, grid_cell_offset=0.5):
    """Generate anchors from features for a batch."""
    B, _,H, W = feats.shape  # Assuming feats is of shape [B, C, H, W]
    dtype, device = feats.dtype, feats.device

    sx = (torch.arange(0, W, device=device, dtype=dtype) + grid_cell_offset)
    sy = (torch.arange(0, H, device=device, dtype=dtype) + grid_cell_offset)

    sy, sx = torch.meshgrid(sy, sx, indexing="ij")
    anchor_points = torch.stack((sx, sy), -1)  # Shape [H, W, 2]

    # Reshape to [1, H*W, 2] and repeat along batch dimension
    anchor_points_batch = anchor_points.view(1, -1, 2).repeat(B, 1, 1)
    stride_tensor_batch = torch.full((B, H * W, 1), stride, dtype=dtype, device=device)

    return anchor_points_batch, stride_tensor_batch

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)



# def dist2circle(distance, anchor_points, dim=-1):
#     """Transform distance(ltrb) to box(xywh or xyxy)."""
#     l, r = distance.chunk(2, dim)
#     t = l
#     b = r
#     lt = torch.cat((l, t), dim)
#     rb = torch.cat((r, b), dim)
#     #print(rb.shape , anchor_points.shape)
#     x1y1 = anchor_points - lt # (8, 2550000, 2)
#     x2y2 = anchor_points + rb # (8, 2550000, 2)
#     xy = (x1y1 + x2y2) / 2
#     r = (x2y2 - x1y1) / 2 # (8, 2550000, 2)
#     r = torch.mean(r, dim=-1, keepdim=True) # (8, 2550000, 1)
#     #rec = torch.cat((x1y1, x2y2), dim)
#
#     return torch.cat((xy, r), dim)  # xyr circle

def dist2circle(distance, anchor_points,stride_tensor, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    ratio = 100.
    l, r = distance.chunk(2, dim)
    # 确保预测值为正数且有意义
    l = torch.abs(l) * ratio  # 缩放因子，根据图像尺寸调整
    r = torch.abs(r) * ratio
    t = l
    b = r
    lt = torch.cat((l, t), dim)
    rb = torch.cat((r, b), dim)
    #print(rb.shape , anchor_points.shape)
    anchor_points = anchor_points * stride_tensor
    x1y1 = anchor_points - lt # (8, 2550000, 2)
    x2y2 = anchor_points + rb # (8, 2550000, 2)
    xy = (x1y1 + x2y2) / 2
    r = (x2y2 - x1y1) / 2 # (8, 2550000, 2)
    r = torch.mean(r, dim=-1, keepdim=True) # (8, 2550000, 1)
    r = torch.clamp(r, min=5.0)  # 最小半径5个像素
    #rec = torch.cat((x1y1, x2y2), dim)

    return torch.cat((xy, r), dim)  # xyr circle
