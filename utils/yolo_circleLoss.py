import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tal import *




class YoloCircleLoss(nn.Module):
    def __init__ (self, tal_topk: int = 10):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = [2, 4, 8]
        #self.reg_max = 16
        self.nc = 1
        self.no = self.nc +2
        self.device = device
        #self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.circle_loss = CircleLoss().to(device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor):
        """通过转换为张量格式并缩放坐标来预处理目标。"""
        nl, ne = targets.shape  # nl: 标注数量, ne: 每个标注的维度数(3+1+1)
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            W, H = scale_tensor[0], scale_tensor[1]  # 因为 scale_tensor = [W, H, W, H]
            diag = torch.sqrt(W ** 2 + H ** 2)
            circle_scale = torch.stack([W, H, diag])  # shape (3,)

            # 缩放 x, y, r
            out[..., 1:4] = out[..., 1:4] * circle_scale  # 或 .mul(circle_scale)
            #out[..., 1:4] = out[..., 1:4].mul_(scale_tensor)
        return out




    def clrcle_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor,stride_tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        b, a, c = pred_dist.shape
        pred_dist =pred_dist.view(b,a,c).softmax(dim=2)
        #pred_dist = pred_dist.sigmoid()
        # if self.use_dfl:
        #     b, a, c = pred_dist.shape  # batch, anchors, channels
        #     pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2circle(pred_dist, anchor_points,stride_tensor)


    def __call__(self,  preds: list[torch.Tensor], batch: dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (2, self.nc), 1)  # (8, 80+2, 2550000)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (8, 2550000, 80)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (8, 2550000,  2)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # (8,2550000,2), (8,2550000,1)

        # Targets [这一个批次一共有多少个框，2+1+nc]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["circles"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) # (8, max_objects, 4)
        gt_labels, gt_circles = targets.split((1, 3), 2)  # cls, xyr # (8, max_objects, 1), (8, max_objects, 3)
        mask_gt = gt_circles.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_circles = self.clrcle_decode(anchor_points, pred_distri,stride_tensor)  # xyr, (8, 2550000, 3)

        _, target_circles, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_circles.detach()).type(gt_circles.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_circles,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Circle loss
        if fg_mask.sum():
            loss[0], loss[2] = self.circle_loss(
                pred_distri,
                pred_circles,
                anchor_points,
                target_circles / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= 0.7  # box gain
        loss[1] *= 0.3  # cls gain
        loss[2] *= 0.7  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class CircleLoss(nn.Module):
    """Criterion class for computing training losses for circles."""

    def __init__(self):
        """Initialize the CircleLoss module with regularization maximum."""
        super().__init__()

    def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = circle_ious(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        distence = center_distanceLoss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        loss_dist = ((1.0 - distence) * weight).sum() / target_scores_sum


        return loss_iou, loss_dist
