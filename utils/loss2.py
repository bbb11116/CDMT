import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft
from fontTools.ttLib.tables.D__e_b_g import table_D__e_b_g
#from paddle.base.libpaddle.eager.ops.legacy import sigmoid
import numpy as np

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
import torch.nn.functional as F


def Dice_loss_seg(inputs, target, beta=1, smooth=1e-5):
    """
    计算二分类的 Dice Loss。

    参数:
    - inputs: [B, C, H, W] 模型输出的logits或概率分布（C=2）
    - target: [B, H, W] 或 [B, C, H, W] one-hot编码的真实标签（C=2）
    - beta: Dice系数的beta参数，默认为1（此时等同于F1-score）
    - smooth: 平滑项，防止除零，默认为1e-5

    返回:
    - dice_loss: 标量损失值
    """
    n, c, h, w = inputs.size()

    # 如果target是[B, H, W]格式，则转换为one-hot格式[B, C, H, W]
    if len(target.shape) == 3:
        target = F.one_hot(target.long(), num_classes=c).permute(0, 3, 1, 2)

    nt, ct, ht, wt = target.size()

    # 如果输入尺寸不一致，调整inputs大小
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 将inputs从[B, C, H, W]转换为[B, H*W, C]，并应用softmax
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    # 将target从[B, C, H, W]转换为[B, H*W, C]
    temp_target = target.view(n, -1, ct)

    # 只考虑前景类（即第1通道）
    temp_inputs_fg = temp_inputs[:, :, 1]  # [B, H*W]
    temp_target_fg = temp_target[:, :, 1]  # [B, H*W]

    # 计算tp, fp, fn
    tp = torch.sum(temp_inputs_fg * temp_target_fg, dim=1)  # true positives
    fp = torch.sum(temp_inputs_fg, dim=1) - tp  # false positives
    fn = torch.sum(temp_target_fg, dim=1) - tp  # false negatives

    # 计算Dice系数
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)

    return dice_loss

def seg_diceloss(prediction, label, l_weight=[0], device='cpu'):
    # 在 dataset __getitem__ 中打印
    label_flat = label.view(-1)
    seg_labels = torch.eye(2, device=label.device)[label_flat]
    seg_labels = seg_labels.reshape((int(label.shape[0]), 2, int(label.shape[1]),int(label.shape[2])))

    loss = Dice_loss_seg(prediction, seg_labels)

    return loss


