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



def smish(x):
    """Smish activation: x * tanh(log(1 + sigmoid(x)))"""
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x):
        return smish(x)


class EnhancedDoubleFusion(nn.Module):
    """
    Enhanced version of DoubleFusion for edge detection.

    Features:
      - Supports arbitrary input channels
      - Optional normalization (GroupNorm recommended for small batch)
      - Optional upscale via PixelShuffle (e.g., r=2 for 2x upsample)
      - Clean, modular, and efficient
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int = 1,
            mid_factor: int = 8,  # expansion factor (was fixed to 8)
            use_norm: bool = True,  # add GroupNorm after conv
            norm_groups: int = 8,  # groups for GroupNorm
            upscale_factor: int = 1,  # set >1 to upsample (e.g., 2)
            activation: nn.Module = Smish()
    ):
        super().__init__()
        self.out_ch = out_ch
        self.upscale_factor = upscale_factor

        mid_ch = in_ch * mid_factor

        # First depthwise block
        self.block1 = self._make_dw_block(in_ch, mid_ch, use_norm, norm_groups, activation)

        # Second depthwise block
        self.block2 = self._make_dw_block(mid_ch, mid_ch, use_norm, norm_groups, activation)

        # Final projection to out_ch (usually 1 for edge map)
        final_in_ch = mid_ch
        if upscale_factor > 1:
            # PixelShuffle reduces channels by r^2, so we need to expand first
            final_in_ch = mid_ch * (upscale_factor ** 2)
            self.pre_shuffle = nn.Conv2d(mid_ch, final_in_ch, 1)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        else:
            self.pre_shuffle = None
            self.pixel_shuffle = None

        self.final_conv = nn.Conv2d(final_in_ch // (upscale_factor ** 2) if upscale_factor > 1 else mid_ch,
                                    out_ch, kernel_size=1)

        self.final_act = activation

    def _make_dw_block(self, in_c, out_c, use_norm, norm_groups, act):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, groups=in_c),
        ]
        if use_norm:
            # Use GroupNorm (more stable than BN for small batches or variable input sizes)
            groups = min(norm_groups, out_c)
            layers.append(nn.GroupNorm(groups, out_c))
        layers.append(act)
        return nn.Sequential(*layers)

    def forward(self, x):
        # First path
        feat1 = self.block1(x)  # [B, mid_ch, H, W]
        feat2 = self.block2(feat1)  # [B, mid_ch, H, W]

        # Fuse: element-wise sum
        fused = feat1 + feat2  # [B, mid_ch, H, W]

        # Optional upsample
        if self.upscale_factor > 1:
            fused = self.pre_shuffle(fused)  # [B, mid_ch * r^2, H, W]
            fused = self.pixel_shuffle(fused)  # [B, mid_ch, H*r, W*r]

        # Project to output channels
        out = self.final_conv(fused)  # [B, out_ch, H', W']
        out = self.final_act(out)
        return out



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




#******************************************************








class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=21):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        probs = F.interpolate(probs, size=feats.shape[-2:], mode='bilinear', align_corners=True)  # b,21,h/2,w/2

        b, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(b, c, -1)
        feats = feats.view(b, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # b*hw/4*64
        probs = F.softmax(probs, dim=2)  # b*21*hw/4
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # b*64*21*1
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(ObjectAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        # x 64*h/2*w/2
        # proxy 64*21*1
        b, h, w = x.size(0), x.size(2), x.size(3)
        query = self.f_pixel(x).view(b, self.key_channels, -1)  # b*32*hw/4
        query = query.permute(0, 2, 1)  # b*hw/4*32
        key = self.f_object(proxy).view(b, self.key_channels, -1)  # b*32*21
        value = self.f_down(proxy).view(b, self.key_channels, -1)
        value = value.permute(0, 2, 1)  # b*21*32

        sim_map = torch.matmul(query, key)  # b*hw/4*21
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)  # b*hw/4*32
        context = context.permute(0, 2, 1).contiguous()  # b*32*hw/4
        context = context.view(b, self.key_channels, *x.size()[2:])  # b*32*h/2*w/2
        context = self.f_up(context)  # b*64*h/2*w/2

        return context


class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels)

        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        # feats 64*h/2*w/2
        # proxy_feats 64*21*1
        context = self.object_context_block(feats, proxy_feats)  # b*64*h/2*w/2

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))  # b*64*h/2*w/2

        return output

#增强特征表示
class Ocr(nn.Module):
    def __init__(self, in_channels=64, num_class=21):
        super(Ocr, self).__init__()
        # self.conv3x3_ocr = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(1, in_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.ocr_gather_head = SpatialGather_Module(cls_num=num_class)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=in_channels,
                                                 key_channels=int(in_channels / 2),
                                                 out_channels=in_channels,
                                                 dropout=0.05
                                                 )

    def forward(self, feats, out_aux):
        # feats 64*h/2*w/2
        # out_aux 21*h*w
        # feats = self.conv3x3_ocr(feats)  # 64*h/2*w/2
        context = self.ocr_gather_head(feats, out_aux)  # 64*21*1
        feats = self.ocr_distri_head(feats, context)  # 512*h/2*w/2

        return feats


class FuseGFF(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(FuseGFF, self).__init__()
        self.FG = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.FG(input)
        return output




class LDC_side_lifting(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self,nclasses=1,inter_channel=16):
        super(LDC_side_lifting, self).__init__()
        self.nclasses = nclasses
        feat_chans = [16, 32, 64, 96]
        self.inter_channels = inter_channel
        self.down5 = nn.Conv2d(feat_chans[-1], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down4 = nn.Conv2d(feat_chans[-2], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down3 = nn.Conv2d(feat_chans[-3], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down22 = nn.Conv2d(feat_chans[-4], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.adjust = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        # 特征融合前增强各层级的特征表示
        self.FuseGFF2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF3 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF4 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF5 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)

        self.FuseGFF2_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF3_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF4_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF5_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)

        self.FuseGFFC1 = nn.Conv2d(in_channels=feat_chans[0], out_channels=feat_chans[0], kernel_size=1, stride=1, bias=False)
        self.FuseGFFC2 = nn.Conv2d(in_channels=feat_chans[1], out_channels=feat_chans[1], kernel_size=1, stride=1, bias=False)
        self.FuseGFFC3 = nn.Conv2d(in_channels=feat_chans[2], out_channels=feat_chans[2], kernel_size=1, stride=1, bias=False)
        self.FuseGFFC4 = nn.Conv2d(in_channels=feat_chans[3], out_channels=feat_chans[3], kernel_size=1, stride=1, bias=False)
        # 全局上下文特征提取
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_chans[-1], self.inter_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(1, self.inter_channels),
            nn.ReLU(inplace=True)
        )

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

        self.head = nn.Conv2d(self.inter_channels, self.nclasses, kernel_size=1, stride=1)
        self.seg_head = nn.Conv2d(self.inter_channels * 2, self.nclasses + 1, kernel_size=1, stride=1)
        self.ocr = Ocr(in_channels=self.inter_channels, num_class=self.nclasses + 1)  # 增强特征表示
        self.edge_head = nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1, stride=1)

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
        n, c, h, w = x.shape
         # supose the image size is 352x352
        # Block 1
        block_1 = self.block_1(x) # [8,16,176,176] 1/2
        # block_1_side = self.side_1(block_1) # 16 [8,32,88,88]

        block_1_side = self.side_haar1(block_1) # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        # block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2 = self.CA1(block_2)#1/2
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
        block_3 = self.CA2(block_3)#1/4
        block_3_14 = self.down2(block_3)
        (c2, d2, LL2, LH2, HL2, HH2) = self.wavelet2(block_3_14)
        block_3_down = torch.cat([HH2, HL2, LH2, LL2], dim=1)
        block_3_down = self.CA2(block_3_down)


        # block_3_down = self.wavelet2(block_3)[-1]
        block_3_add = block_3_down + block_2_side # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down) # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half) # [8,96,44,44]
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense]) # [8,96,44,44] 1/8

        side2 = self.down22(block_1)  # 64,1/2
        side3 = self.down3(block_2)  # 64,1/2
        side4 = self.down4(block_3)  # 64,1/4
        side5 = self.down5(block_4)  # 64,1/8

        g2 = torch.sigmoid(side2)
        g3 = torch.sigmoid(side3)
        g4 = torch.sigmoid(side4)
        g5 = torch.sigmoid(side5)

        gs2 = F.interpolate(g2 * side2, size=side3.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/2
        gs3 = F.interpolate(g3 * side3, size=side2.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/2
        gs4 = F.interpolate(g4 * side4, size=side5.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/8
        gs5 = F.interpolate(g5 * side5, size=side4.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/4

        side5gff = (1 + g5) * side5 + (1 - g5) * gs4  # 64,1/8
        side4gff = (1 + g4) * side4 + (1 - g4) * gs5  # 64,1/4
        side3gff = (1 + g3) * side3 + (1 - g3) * gs2  # 64,1/2
        side2gff = (1 + g2) * side2 + (1 - g2) * gs3  # 64,1/2

        side5gff = self.FuseGFF5(side5gff)  # 64,1/8
        side4gff = self.FuseGFF4(side4gff)  # 64,1/4
        side3gff = self.FuseGFF3(side3gff)  # 64,1/2
        side2gff = self.FuseGFF2(side2gff)  # 64,1/2

        side5gff = F.interpolate(side5gff, size=side4gff.shape[-2:], mode='bilinear', align_corners=True)
        seg = torch.cat([side5gff, side4gff], dim=1)
        seg = self.seg_head(seg)
        seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=True)


        # side5gff_2 = self.FuseGFF5_2(side5gff)  # 64,1/8
        # side4gff_2 = self.FuseGFF4_2(side4gff)  # 64,1/4
        # side3gff_2 = self.FuseGFF3_2(side3gff)  # 64,1/2
        # side2gff_2 = self.FuseGFF2_2(side2gff)  # 64,1/2
        # global_context = self.global_context(block_4)  # 64,1*1
        # global_context = F.interpolate(global_context, size=side5gff_2.size()[2:], mode='bilinear',
        #                                align_corners=True)  # 64,1/8






        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        # results = [out_1, out_2, out_3, out_4, out_5, out_6]
        results = [out_1, out_2, out_3, out_4]


        # 将results中的每个结果保存为图像
        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx4xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        block_cat = self.adjust(block_cat)


        # return results


        sum_23 = self.ocr(block_cat, seg)
        final_feature = self.head(sum_23)  # 1,1/2
        sedge = F.interpolate(final_feature, size=(h, w), mode='bilinear', align_corners=True)
        results.append(sedge)


        circle1 = self.FuseGFFC1(block_1)
        #circle2 = self.FuseGFFC2(block_2)
        circle3 = self.FuseGFFC3(block_3)
        circle4 = self.FuseGFFC4(block_4)
        pre = [circle1, circle3, circle4]
        clrcle_model = CustomDetect(nc=1, ch=[16, 64, 96]).to(x.device)
        clrcle_results = clrcle_model(pre)

        return results , clrcle_results , seg



class CustomDetect(nn.Module):
    def __init__(self, nc=1, ch=(16, 64, 96)):
        super().__init__()
        self.nc = nc  # 类别数: 80
        self.no = 2  # 你的特有偏移量: 2 (例如 dx, dy 或者 r, offset)

        # 你的输入通道列表: [16, 64, 96]
        # 对应 strides: [2, 4, 8] (基于你的输入1200x1600和输出600x800推算)
        self.sigmoid = nn.Sigmoid()
        # 定义三个尺度的处理模块
        self.cv2 = nn.ModuleList()  # 回归分支 (Regression Branch)
        self.cv3 = nn.ModuleList()  # 分类分支 (Classification Branch)


        for x in ch:
            # 1. 回归分支: 输入通道 -> 2个输出通道
            # 这里使用 1x1 卷积直接映射，也可以先加 3x3 卷积增加非线性
            self.cv2.append(nn.Conv2d(x, self.no, kernel_size=1, stride=1))

            # 2. 分类分支: 输入通道 -> 80个输出通道
            self.cv3.append(nn.Conv2d(x, self.nc, kernel_size=1, stride=1))

    def forward(self, x):
        """
        x: list of tensors
        x[0]: (B, 16, 600, 800)
        x[1]: (B, 64, 300, 400)
        x[2]: (B, 96, 150, 200)
        """
        res = []
        for i in range(len(x)):
            # 1. 计算分类分支 (B, 1, H, W)
            cls_out = self.cv3[i](x[i])
            #cls_out = self.sigmoid(cls_out)

            # 2. 计算回归分支 (B, 2, H, W)
            reg_out = self.cv2[i](x[i])
            #reg_out = self.sigmoid(reg_out)

            # 3. 拼接 (Concatenate) -> (B, 1+2, H, W)
            # dim=1 代表在通道维度拼接
            out = torch.cat((cls_out, reg_out), 1)

            res.append(out)

        # 此时 res 包含了三个张量，形状完全符合你的要求：
        # res[0]: (8, 3, 600, 800)
        # res[1]: (8, 3, 300, 400)
        # res[2]: (8, 3, 150, 200)
        return res


import torch
import torch.nn as nn
import torch.nn.functional as F


class PostProcess:
    def __init__(self, conf_thres=0.6, strides=[2, 4, 8]):
        self.conf_thres = conf_thres
        self.strides = strides  # 对应你的三层输出 stride
        self.num_classes = 1

    def __call__(self, preds):
        """
        preds: 列表，包含三个 tensor
               Scale 0: (B, 3, 600, 800)
               Scale 1: (B, 3, 300, 400)
               Scale 2: (B, 3, 150, 200)
        """
        # 1. 初始化一个列表，长度为 Batch_Size，用来存放每张图的结果
        batch_size = preds[0].shape[0]
        output = [torch.zeros((0, 5), device=preds[0].device) for _ in range(batch_size)]  # 修复：改为 (0, 5)

        # 遍历每一个尺度 (scale)
        for i, pred in enumerate(preds):
            stride = self.strides[i]
            B, C, H, W = pred.shape
            # 1. 维度变换: (B, 3, H, W) -> (B, H, W, 3)
            pred1 = pred.permute(0, 2, 3, 1)
            # 2. 分割通道: 前1是类别，后2是偏移量
            scores = pred1[..., :self.num_classes].sigmoid()
            offsets = pred1[..., self.num_classes:]
            anchor_points, stride_tensor = make_anchor(pred, stride, 0.5)
            yoloCircleLoss = YoloCircleLoss()
            pred_circles = yoloCircleLoss.clrcle_decode(anchor_points, offsets.reshape(B, -1, 2), stride_tensor).reshape(B, H, W, 3)

            # 6. 整合当前尺度的结果
            # 找到每个网格中分数最大的类别
            max_scores, class_ids = scores.max(dim=-1)  # (B, H, W)
            # 筛选大于阈值的点
            mask = max_scores > self.conf_thres

            for b in range(batch_size):
                # 获取当前 batch 中满足阈值的索引
                b_mask = mask[b]
                if b_mask.sum() == 0:
                    continue

                # 修复：不要直接解包 pred_circles[b][b_mask]
                # 因为 pred_circles[b][b_mask] 的形状是 (N, 3)，不能直接解包成三个变量
                filtered_circles = pred_circles[b][b_mask]  # (N, 3)

                # 从 (N, 3) 中分别提取 x, y, r
                valid_x = filtered_circles[:, 0]  # (N,)
                valid_y = filtered_circles[:, 1]  # (N,)
                valid_r = filtered_circles[:, 2]  # (N,)

                valid_scores = max_scores[b][b_mask]  # (N,)
                valid_classes = class_ids[b][b_mask]  # (N,)

                # 堆叠结果: [x, y, r, score, class_id] -> (Num_Valid, 5)
                dets = torch.stack([valid_x, valid_y, valid_r, valid_scores, valid_classes.float()], dim=1)

                # 关键步骤: 将当前尺度的结果拼接到该图片的总结果中
                output[b] = torch.cat((output[b], dets), dim=0)

        # 返回 list of tensors，每个 tensor 对应一张图的检测结果
        # 每个 tensor 的形状是 (num_detections, 5) -> [x, y, r, score, class_id]
        return output


def standard_nms_with_fixed_size(detections, fixed_size=10, iou_thres=0.9):
    """
    detections: (N, 5) -> [x, y, r, score, class]
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







def final_results(PostProcess,output):
    if not output:
        return []
    #output = [torch.sigmoid(o) for o in output]
    final_results = []
    postprocessor = PostProcess()
    detections = postprocessor(output)
    for img_dets in detections:
        # img_dets: (Total_N, 5)
        if img_dets.shape[0] == 0:
            final_results.append(img_dets)
            continue

        # 使用之前提到的伪造 Box NMS 策略
        # 输入: (Total_N, 5) -> 输出: (Keep_N, 5)
        keep_dets = standard_nms_with_fixed_size(img_dets, fixed_size=20,iou_thres=0.95)
        final_results.append(keep_dets)
    return final_results




