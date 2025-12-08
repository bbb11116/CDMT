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
        #self.nc = 80
        self.no = 2
        self.device = device
        #self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, alpha=0.5, beta=6.0)
        self.circle_loss = CircleLoss().to(device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor):
        """通过转换为张量格式并缩放坐标来预处理目标。"""
        nl, ne = targets.shape  # nl: 标注数量, ne: 每个标注的维度数(3+1)
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
            out = out * circle_scale  # 或 .mul(circle_scale)
            # out[..., 1:4] = out[..., 1:4].mul_(scale_tensor)
        return out




    def clrcle_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        # if self.use_dfl:
        #     b, a, c = pred_dist.shape  # batch, anchors, channels
        #     pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2circle(pred_dist, anchor_points)


    def __call__(self,  preds: list[torch.Tensor], batch: dict[str, torch.Tensor]):
        loss = torch.zeros(2, device=self.device)  # box, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (8, 2550000,  2)

        dtype = pred_distri.dtype
        batch_size = pred_distri.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # (8,2550000,2), (8,2550000,1)

        # Targets [这一个批次一共有多少个框，3]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["circles"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # (8, max_objects, 4)
        gt_circles = targets
        mask_gt = gt_circles.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_circles = self.clrcle_decode(anchor_points, pred_distri)  # xyr, (8, 2550000, 3)

        target_circles, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            (pred_circles.detach() * stride_tensor).type(gt_circles.dtype),
            anchor_points * stride_tensor,
            gt_circles,
            mask_gt,
        )

        #target_scores_sum = max(target_scores.sum(), 1)
        # Cls loss
        #loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Circle loss
        if fg_mask.sum():
            loss[0], loss[1] = self.circle_loss(
                pred_distri,
                pred_circles,
                anchor_points,
                target_circles / stride_tensor,
                fg_mask,
            )

        loss[0] *= 0.3  # circle gain
        loss[1] *= 0.7  # dist gain

        return loss * batch_size, loss.detach()  # loss(box, dfl)


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
            target_circles: torch.Tensor,
            #target_scores: torch.Tensor,
            #target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        #weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = circle_ious(pred_bboxes[fg_mask], target_circles[fg_mask])
        distence = center_distanceLoss(pred_bboxes[fg_mask], target_circles[fg_mask])
        loss_iou = ((1.0 - iou)).sum() / fg_mask.sum()
        loss_dist = ((1.0 - distence)).sum() / fg_mask.sum()


        return loss_iou, loss_dist
