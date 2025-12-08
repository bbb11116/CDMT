import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from attentions.conv_pdc import Conv2d
import numpy as np
import torchvision
#from scipy.linalg.cython_lapack import sgebd2

from model.lifting import LiftingScheme2D, WaveletHaar2D, WaveletHaar
from attentions.ChannelAtt import ChannelAttention
from utils.AF.Fsmish import smish as Fsmish
from utils.AF.Xsmish import Smish
from utils.yolo_circleLoss import *


class Regression(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Regression, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((30,4))
        self.classifier = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_ch//2, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.classifier(self.global_pool(x)).squeeze()

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # 按照通道数将attn分割，并保存为图像
        # for i in range(attn.shape[1]):
        #     attn_slice = attn[0, i, :, :]
        #     attn_slice = attn_slice.cpu().detach().numpy()
        #     attn_slice = np.array(attn_slice * 255, dtype=np.uint8)
        #     cv2.imwrite(f'./attn_{i}.png', attn_slice)


        return ((x * attn).sum(1)).unsqueeze(1)


class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch*8, kernel_size=3,
                               stride=1, padding=1, groups=in_ch) # before 64
        self.PSconv1 = nn.PixelShuffle(1)

        self.DWconv2 = nn.Conv2d(32, 32*1, kernel_size=3,
                               stride=1, padding=1,groups=32)# before 64  instead of 32

        self.AF= Smish()#XAF() #nn.Tanh()# XAF() #   # Smish()#


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.PSconv1(self.DWconv1(self.AF(x))) # #TEED best res TEDv14 [8, 32, 352, 352]

        attn2 = self.PSconv1(self.DWconv2(self.AF(attn))) # #TEED best res TEDv14[8, 3, 352, 352]

        return Fsmish(((attn2 +attn).sum(1)).unsqueeze(1)) #TED best res

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()

        return 0.5 * (new_features + x2), x2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac=use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish= Smish()#nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x

class SideHaarBlock(nn.Module):
    def __init__(self,in_features):
        super(SideHaarBlock, self).__init__()

        self.out_channels = in_features // 2
        self.out_features = in_features * 2

        self.CAM = ChannelAttention(in_features)
        self.down = nn.Conv2d(in_channels=in_features, out_channels=self.out_channels, kernel_size=1, stride=1)
        self.haar = WaveletHaar2D()
        self.bn = nn.BatchNorm2d(self.out_features)

    def forward(self, x):
        x = self.CAM(x)
        x = self.down(x)
        (LL, LH, HL, HH) = self.haar(x)
        x = torch.cat([HH, HL, LH, LL], dim=1)
        x = self.bn(x)
        return x






class LDC_side_lifting(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(LDC_side_lifting, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 64, 96)# 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.CA1 = ChannelAttention(32)
        self.down1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1)

        self.CA2 = ChannelAttention(64)
        self.down2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)

        self.wavelet1 = LiftingScheme2D(in_planes=8, share_weights=False)
        self.wavelet2 = LiftingScheme2D(in_planes=16, share_weights=False)

        # self.wavelet1 = LiftingScheme2D(in_planes=32, share_weights=False)
        # self.wavelet2 = LiftingScheme2D(in_planes=64, share_weights=False)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        self.side_haar1 = SideHaarBlock(16)
        self.side_haar2 = SideHaarBlock(32)
        self.side_haar3 = SideHaarBlock(64)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)# 128

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)# 128
        # self.block_cat = SingleConvBlock(4, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = DoubleFusion(4,4)# cats fusion method


        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape
         # supose the image size is 352x352
        # Block 1
        block_1 = self.block_1(x) # [8,16,176,176]
        # block_1_side = self.side_1(block_1) # 16 [8,32,88,88]

        block_1_side = self.side_haar1(block_1) # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        # block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2 = self.CA1(block_2)
        block_2_14 = self.down1(block_2)
        (c1, d1, LL1, LH1, HL1, HH1) = self.wavelet1(block_2_14)
        block_2_down = torch.cat([HH1, HL1, LH1, LL1], dim=1)
        block_2_down = self.CA1(block_2_down)

        # block_2_down = self.wavelet1(block_2)[-1]
        block_2_add = block_2_down + block_1_side # [8,32,88,88]
        # block_2_side = self.side_2(block_2_add) # [8,64,44,44] block 3 R connection

        block_2_side = self.side_haar2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down) # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense]) # [8,64,88,88]
        # block_3_down = self.maxpool(block_3) # [8,64,44,44]
        block_3 = self.CA2(block_3)
        block_3_14 = self.down2(block_3)
        (c2, d2, LL2, LH2, HL2, HH2) = self.wavelet2(block_3_14)
        block_3_down = torch.cat([HH2, HL2, LH2, LL2], dim=1)
        block_3_down = self.CA2(block_3_down)


        # block_3_down = self.wavelet2(block_3)[-1]
        block_3_add = block_3_down + block_2_side # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down) # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half) # [8,96,44,44]
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense]) # [8,96,44,44]


        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        # results = [out_1, out_2, out_3, out_4, out_5, out_6]
        results = [out_1, out_2, out_3, out_4]
        pre = [block_1, block_3, block_4]
        clrcle_model = CustomDetect(nc=80, ch=[16, 64, 96]).to(x.device)
        clrcle_results = clrcle_model(pre)

        # 将results中的每个结果保存为图像
        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx4xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results ,clrcle_results


class CustomDetect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.no = 2  # 只预测2个偏移量

        # 定义三个尺度的回归分支
        self.cv2 = nn.ModuleList()  # 回归分支 (Regression Branch)

        for x in ch:
            # 回归分支: 输入通道 -> 2个输出通道
            self.cv2.append(nn.Conv2d(x, self.no, 1))

    def forward(self, x):
        """
        x: list of tensors
        x[0]: (B, 16, 600, 800)
        x[1]: (B, 64, 300, 400)
        x[2]: (B, 96, 150, 200)
        """
        res = []
        for i in range(len(x)):
            # 只计算回归分支 (B, 2, H, W)
            reg_out = self.cv2[i](x[i])

            res.append(reg_out)

        # 返回三个张量，每个张量只包含2个偏移量：
        # res[0]: (B, 2, 600, 800)
        # res[1]: (B, 2, 300, 400)
        # res[2]: (B, 2, 150, 200)
        return res


class PostProcess:
    def __init__(self, strides=[2, 4, 8]):
        self.strides = strides  # 对应你的三层输出 stride

    def __call__(self, preds):
        """
        preds: 列表，包含三个 tensor
               Scale 0: (B, 2, 600, 800)
               Scale 1: (B, 2, 300, 400)
               Scale 2: (B, 2, 150, 200)
        """
        # 1. 初始化一个列表，长度为 Batch_Size，用来存放每张图的结果
        batch_size = preds[0].shape[0]
        output = [torch.zeros((0, 4), device=preds[0].device) for _ in range(batch_size)]  # 改为 (0, 4)

        # 遍历每一个尺度 (scale)
        for i, pred in enumerate(preds):
            stride = self.strides[i]
            B, C, H, W = pred.shape
            # 1. 维度变换: (B, 2, H, W) -> (B, H, W, 2)
            offsets = pred.permute(0, 2, 3, 1)

            # 2. 生成锚点
            anchor_points, stride_tensor = make_anchor(pred, stride, 0.5)

            # 3. 解码偏移量得到圆参数
            yoloCircleLoss = YoloCircleLoss()
            pred_circles = yoloCircleLoss.clrcle_decode(anchor_points, offsets.reshape(B, -1, 2)).reshape(B, H, W, 3)

            # 4. 整合当前尺度的结果（不再筛选，全部保留）
            for b in range(batch_size):
                # 获取当前 batch 的所有预测圆
                circles = pred_circles[b]  # (H, W, 3)

                # 展平为 (H*W, 3)
                circles_flat = circles.reshape(-1, 3)

                # 提取 x, y, r
                valid_x = circles_flat[:, 0]  # (H*W,)
                valid_y = circles_flat[:, 1]  # (H*W,)
                valid_r = circles_flat[:, 2]  # (H*W,)

                # 堆叠结果: [x, y, r, score=1.0] -> (H*W, 4)
                # 由于移除了分类，我们给每个预测一个默认的置信度分数1.0
                dets = torch.stack([valid_x, valid_y, valid_r, torch.ones_like(valid_x)], dim=1)

                # 将当前尺度的结果拼接到该图片的总结果中
                output[b] = torch.cat((output[b], dets), dim=0)

        # 返回 list of tensors，每个 tensor 对应一张图的检测结果
        # 每个 tensor 的形状是 (num_detections, 4) -> [x, y, r, score=1.0]
        return output


def standard_nms_with_fixed_size(detections, fixed_size=40, iou_thres=0.45):
    """
    detections: (N, 4) -> [x, y, r, score]
    fixed_size: 假定的目标大小
    """
    if len(detections) == 0:
        return []

    x = detections[:, 0]
    y = detections[:, 1]
    r = detections[:, 2]
    scores = detections[:, 3]

    # 伪造左上角和右下角坐标
    x1 = x - r
    y1 = y - r
    x2 = x + r
    y2 = y + r

    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # 确保 boxes 的坐标是非负的，并且 x2 > x1, y2 > y1
    boxes = torch.clamp(boxes, min=0)
    boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1e-6)  # x2 > x1
    boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1e-6)  # y2 > y1

    # 转换数据类型为 float32
    boxes = boxes.to(dtype=torch.float32, device=boxes.device)
    scores = scores.to(dtype=torch.float32, device=scores.device)

    # 限制处理的数量，避免内存问题
    if len(scores) > fixed_size:
        _, topk_indices = scores.topk(min(fixed_size, len(scores)))
        boxes = boxes[topk_indices]
        scores = scores[topk_indices]
        detections = detections[topk_indices]

    # 使用官方 NMS
    keep_indices = torchvision.ops.nms(boxes, scores, iou_thres)

    return detections[keep_indices]


def final_results(PostProcess, output):
    if not output:
        return []

    final_results = []
    postprocessor = PostProcess()
    detections = postprocessor(output)

    for img_dets in detections:
        # img_dets: (Total_N, 4)
        if img_dets.shape[0] == 0:
            final_results.append(img_dets)
            continue

        # 使用伪造 Box NMS 策略
        # 输入: (Total_N, 4) -> 输出: (Keep_N, 4)
        keep_dets = standard_nms_with_fixed_size(img_dets, fixed_size=40)
        final_results.append(keep_dets)

    return final_results




