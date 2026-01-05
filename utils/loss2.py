import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft
from fontTools.ttLib.tables.D__e_b_g import table_D__e_b_g
#from paddle.base.libpaddle.eager.ops.legacy import sigmoid

#input:[4,H,W] targets:[N,4]
def regression_loss(inputs, targets):
    inputs = inputs.float()
    targets = targets.float()
    inputs = F.sigmoid(inputs)
    targets = F.sigmoid(targets)
    for i in range(targets.shape[0]):
        criterion = nn.MSELoss()
        loss = criterion(inputs[i], targets[i])

    return loss


def bdcn_loss2(inputs, targets, l_weight=1.1):
    """使用 BCEWithLogitsLoss 的优化版本 (推荐)"""
    targets = targets.long()
    mask = targets.float()

    # 计算正负样本数量
    num_positive = torch.sum((mask > 0.0).float()).float()
    num_negative = torch.sum((mask <= 0.0).float()).float()
    total = num_positive + num_negative + 1e-6  # 防除零

    # 创建权重掩码
    weight_mask = torch.ones_like(mask)
    weight_mask[mask > 0.] = num_negative / total
    weight_mask[mask <= 0.] = 1.1 * num_positive / total

    # 使用 BCEWithLogitsLoss (自动处理 sigmoid + 数值稳定)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    cost = bce_loss(inputs, targets.float())

    # 应用权重掩码
    cost = cost * weight_mask
    cost = torch.sum(cost.float().mean((1, 2, 3)))

    return l_weight * cost

# ------------ cats losses ----------

def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)



    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return torch.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))


def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss
    label = torch.clamp(label, 0.0, 1.0)

    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # cost = torch.nn.functional.binary_cross_entropy(
    #     prediction.float(), label.float(), weight=mask, reduction='none')
    # cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return  bdr_factor * bdrcost + tex_factor * textcost

def Dice_loss(prediction, label, l_weight=[0], device='cpu'):
    smooth = 1e-5
    prediction = prediction.to(device).float()  # 强制转移到 CPU
    label = label.to(device).float()  # 强制转移到 CPU
    prediction = torch.sigmoid(prediction)
    prediction = prediction.view(-1)
    label = label.view(-1)
    intersection = (prediction * label).sum()
    union = prediction.sum() + label.sum()
    diec = ((2.0 * intersection + smooth) / (union + smooth)) * l_weight
    return 1.0 - diec


import torch
import torch.nn as nn


class RankEDLoss(nn.Module):
    def __init__(self, alpha=0.5, margin=0.1, num_samples=5000):
        super().__init__()
        self.alpha = alpha  # LSort损失的权重系数
        self.margin = margin  # 排序间隔阈值
        self.num_samples = num_samples  # 最大采样对数

    def forward(self, pred, target, certainty=None):
        """
        pred:     模型输出 [N, H, W]
        target:   真实标签 [N, H, W] (值应为0或1)
        certainty: 确定性分数 [N, H, W] (可选)
        """
        # 类型安全预处理
        pred = torch.sigmoid(pred).float()
        target = target.float()

        # 展平张量处理
        pred_flat = pred.view(-1)  # [B*H*W]
        target_flat = target.view(-1)  # [B*H*W]
        pos_mask = target_flat == 1  # 布尔类型掩码

        # 正样本数量检查
        num_pos = pos_mask.sum().int().item()
        if num_pos < 2:
            return torch.tensor(0.0, device=pred.device)

        # ================= 全局排名损失 LRank =================
        # 生成浮点类型的排名张量
        _, indices = torch.sort(pred_flat, descending=True)
        ranks = torch.arange(
            1, len(pred_flat) + 1,
            dtype=torch.float32,
            device=pred.device
        )[indices]

        # 提取正样本排名并计算均值
        l_rank = ranks[pos_mask].mean() / len(pred_flat)  # 归一化到[0,1]

        # ================= 优化后的排序损失 LSort =================
        # 动态调整采样数量
        max_pairs = num_pos * (num_pos - 1)
        m = min(self.num_samples, max_pairs)

        # 生成随机样本对索引
        indices = torch.randint(
            0, num_pos,
            size=(2, m),
            device=pred.device
        )

        # 过滤i==j的无效对
        valid_mask = indices[0] != indices[1]
        i = indices[0][valid_mask]
        j = indices[1][valid_mask]

        # 计算预测差异
        pos_scores = pred_flat[pos_mask]  # [num_pos]
        pred_diff = pos_scores[i] - pos_scores[j]
        valid_pairs = (pred_diff < self.margin).float()  # [valid_pairs]

        # 处理确定性分数
        if certainty is not None:
            pos_certainty = certainty.view(-1)[pos_mask]  # [num_pos]
            c_diff = (pos_certainty[i] - pos_certainty[j] + 1) / 2  # 映射到[0,1]
            loss_terms = valid_pairs * (1 - c_diff)
        else:
            loss_terms = valid_pairs

        # 计算均值损失（处理无有效采样情况）
        l_sort = loss_terms.mean() if len(loss_terms) > 0 else 0.0

        return l_rank + self.alpha * l_sort


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        """
        Focal Loss 二分类实现
        :param alpha: 正样本权重 (用于类别平衡，建议0.75-0.95)
        :param gamma: 困难样本调节因子 (γ↑ 更关注困难样本)
        :param reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入校验
        targets = targets.squeeze(1)  # 关键修改
        assert inputs.shape == targets.shape, \
            f"预测值与标签形状不匹配: inputs {inputs.shape}, targets {targets.shape}"

        # 计算二分类交叉熵 (无需sigmoid，使用logits更稳定)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # 计算概率值pt
        pt = torch.exp(-bce_loss)  # pt = p if y=1 else 1-p
        focal_term = (1 - pt) ** self.gamma

        # 应用类别权重alpha
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 组合得到Focal Loss
        loss = alpha_factor * focal_term * bce_loss

        # 聚合方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class HybridLoss(nn.Module):
    def __init__(self,
                 max_epochs=50,
                 scheduler_type='cosine',
                 hard_threshold=(0.3, 0.7),
                 hard_weight=2.0,
                 hard_gamma=2.0,):
        super().__init__()
        # 基础损失组件
        self.rank_loss = RankEDLoss(alpha=0,num_samples=10000)
        self.focal_loss = FocalLoss(gamma=hard_gamma)

        # 渐进式调度参数
        self.max_epochs = max_epochs
        self.scheduler_type = scheduler_type
        self.current_epoch = 0

        # 困难样本参数
        self.hard_threshold = hard_threshold
        self.hard_weight = hard_weight
        self.hard_gamma = hard_gamma

    def _get_weight_ratio(self):
        """根据调度类型计算当前权重比例"""
        if self.scheduler_type == 'cosine':
            ratio = 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.max_epochs))
        elif self.scheduler_type == 'linear':
            ratio = 1 - self.current_epoch / self.max_epochs
        else:
            ratio = 1.0  # 固定权重
        return ratio

    def _get_hard_weights(self, pred):
        # 混合使用阈值和连续权重
        with torch.no_grad():
            prob = torch.sigmoid(pred)

            # 离散困难区域检测
            hard_mask = (prob > self.hard_threshold[0]) & (prob < self.hard_threshold[1])

            # 连续困难度权重
            hardness = 1 - torch.abs(prob - 0.5) * 2
            cont_weights = hardness ** self.hard_gamma

            # 组合权重
            weights = torch.where(hard_mask, cont_weights * self.hard_weight, 1.0)

        return weights

    def forward(self, pred, target,lweight):
        pred = pred.squeeze(1)  # [B,1,H,W] → [B,H,W]
        target = target.float()
        # 获取渐进式权重
        with torch.no_grad():  # 🟢 关闭梯度计算
            ratio = self._get_weight_ratio()
            hard_weights = self._get_hard_weights(pred.detach())  # 🟢 分离计算图
            hard_weights = hard_weights / (hard_weights.mean() + 1e-6) * 0.5 + 0.5
        #ratio = self._get_weight_ratio()
        # 获取困难样本权重
        #hard_weights = self._get_hard_weights(pred)
        # 均值归一化 (关键步骤)
        # 添加平滑系数
        #hard_weights = hard_weights / (hard_weights.mean() + 1e-6) * 0.5 + 0.5

        # 加权损失计算
        rank_loss = (self.rank_loss(pred, target) * hard_weights).mean()
        focal_loss = (self.focal_loss(pred, target) * hard_weights).mean()

        return (ratio * rank_loss + (1 - ratio) * focal_loss) * lweight
