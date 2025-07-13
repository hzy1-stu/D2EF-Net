import torch.nn as nn
import torch
from torch.nn.init import normal_
from my_math import real2complex, complex2real, torch_fft2c, torch_ifft2c, sos
import math
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import h5py
from network.CTO_net import CTO
from network.Res2Net import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s




class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1, dropout=False):
        super(double_conv_up, self).__init__()
        self.drop = 0.
        self.ConvV2 = ConvV2_concat_Block(in_ch)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.ConvV2(x)
        x = self.conv_3(x)
        x3 = x + x1
        return x3


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(inconv, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding),
            nn.ReLU(inplace=True),  # 设置成TRUE这样做的好处就是能够节省运算内存，不用多存储额外的变量
            nn.Conv2d(out_ch, out_ch, 3, padding=padding),
            nn.ReLU(inplace=True),  # 设置成TRUE这样做的好处就是能够节省运算内存，不用多存储额外的变量
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Conv2d(1, 64, 3, padding=padding)

    def forward(self, x):
        x1 = self.conv_2(self.conv_1(x))
        x2 = x1 + self.conv_3(x)
        return x2

class up(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                # 用双线插值对特征图放大两倍，使其跟拼接过来的浅层特征图大小保持一致
                # scale_factor指定输出大小为输入的多少倍数;mode:可使用的上采样算法;align_corners为True，输入的角像素将与输出张量对齐
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 使用卷积层将其通道数减半（加入padding来保证特征图不变），利于拼接
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x1, x2):
        # 上采样的过程中需要对两个特征图进行融合，通道数一样并且尺寸也应该一样，x1是上采样获得的特征，而x2是下采样获得的特征，
        # 首先对x1进行反卷积使其大小变为输入时的2倍，首先需要计算两张图长宽的差值，作为填补padding的依据，由于此时图片的表示为（C,H,W）
        # 因此diffY对应的图片的高，diffX对应图片的宽度， F.pad指的是（左填充，右填充，上填充，下填充），其数值代表填充次数，因此需要/2，最后进行融合剪裁

        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        # Math.ceil()  “向上取整”， 即小数部分直接舍去，并向正数部分进1
        x1 = F.pad(x1, (math.ceil(diffY / 2), int(diffY / 2),
                        math.ceil(diffX / 2), int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
                # 最大池化只改变图片尺寸的大小，不改变通道数。参数stride默认值为kernel大小
                nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)


    def forward(self, x):
        x = self.conv(x)
        return x


# UNet with channel-wise attention, input arguments of CSE_block should change according to image size
class UNetCSE(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetCSE, self).__init__()
        self.conv_layer_in = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        self.conv_layer = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.inc = inconv(n_channels, 64, 1)
        self.inc_res1 = nn.Conv2d(256, 64, 1)
        self.inc_res2 = nn.Conv2d(512, 64, 1)
        self.inc_res3 = nn.Conv2d(1024, 64, 1)
        self.inc_res4 = nn.Conv2d(2048, 64, 1)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.trans1 = trans(64, 64, 1)
        self.trans2 = trans(64, 64, 1)
        self.trans3 = trans(64, 64, 1)
        self.trans4 = trans(64, 64, 1)

        self.up3 = up(64, 64, 1)

        self.up2 = up(64, 64, 1)

        self.up1 = up(64, 64, 1)

        self.outc = outconv(64, n_classes)

        self.down = down(64, 64, 1)

        self.invert1 = AdaptiveMultiScaleConvBlock(64, 64)
        self.invert2 = AdaptiveMultiScaleConvBlock(64, 64)
        self.invert3 = AdaptiveMultiScaleConvBlock(64, 64)
        self.invert4 = AdaptiveMultiScaleConvBlock(64, 64)

        self.mlp = Mlp(192, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.down(x1)

        "SW-MSA编码器"
        x2 = self.trans1(x1)
        x3 = self.down(x2)
        x4 = self.trans2(x3)
        x5 = self.down(x4)
        x6 = self.trans3(x5)
        x7 = self.down(x6)
        x8 = self.trans4(x7)

        "Res2Net编码器"
        res1, res2, res3, res4 = self.resnet(self.conv_layer_in(x))
        res1 = self.inc_res1(res1)  # 1,64,128,128
        res2 = self.inc_res2(res2)  # 1,64,64,64
        res3 = self.inc_res3(res3)  # 1,64,32,32
        res4 = self.inc_res4(res4)  # 1,64,16,16

        x2 = F.interpolate(x2, res1.size()[2:], mode='bilinear', align_corners=False)
        x1 = F.interpolate(x1, res1.size()[2:], mode='bilinear', align_corners=False)
        dual_attention1 = self.mlp(torch.cat([x2, res1, x1], dim=1))  # 1,64,16,16

        x4 = F.interpolate(x4, res2.size()[2:], mode='bilinear', align_corners=False)
        dual_attention1 = F.interpolate(dual_attention1, res2.size()[2:], mode='bilinear', align_corners=False)
        dual_attention2 = self.mlp(torch.cat([x4, res2, dual_attention1], dim=1))  # 1,64,16,16

        x6 = F.interpolate(x6, res3.size()[2:], mode='bilinear', align_corners=False)
        dual_attention2 = F.interpolate(dual_attention2, res3.size()[2:], mode='bilinear', align_corners=False)
        dual_attention3 = self.mlp(torch.cat([x6, res3, dual_attention2], dim=1))  # 1,64,16,16

        x8 = F.interpolate(x8, res4.size()[2:], mode='bilinear', align_corners=False)
        dual_attention3 = F.interpolate(dual_attention3, res4.size()[2:], mode='bilinear', align_corners=False)
        dual_attention = self.mlp(torch.cat([x8, res4, dual_attention3], dim=1))   # 1,64,16,16

        "解码器"
        decode_1 = torch.cat([dual_attention, res4], dim=1)
        decode_2 = self.invert1(self.conv_layer(decode_1))
        decode_3 = self.up1(decode_2, res3)
        decode_4 = self.invert2(self.conv_layer(decode_3))
        decode_5 = self.up2(decode_4, res2)
        decode_6 = self.invert3(self.conv_layer(decode_5))
        decode_7 = self.up3(decode_6, res1)
        decode_8 = self.invert4(self.conv_layer(decode_7))
        # decode_8 = self.scope(decode_8)
        decode_8 = F.interpolate(decode_8, x.size()[2:], mode='bilinear', align_corners=False)
        x = self.outc(decode_8)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveMultiScaleConvBlock, self).__init__()

        # Multi-scale convolutions with different kernel sizes
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=1)

        # Channel attention mechanism
        self.attention_fc1 = nn.Linear(out_channels, out_channels // 8)
        self.attention_fc2 = nn.Linear(out_channels // 8, out_channels)

        # Batch Normalization and Activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply convolutions with different scales
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)
        conv7x7_out = self.conv7x7(x)

        # Sum the outputs from different scales
        multi_scale_out = conv3x3_out + conv5x5_out + conv7x7_out

        # Global Average Pooling to generate channel-wise attention weights
        batch_size, channels, _, _ = multi_scale_out.size()
        attention = F.adaptive_avg_pool2d(multi_scale_out, 1).view(batch_size, channels)
        attention = self.attention_fc1(attention)
        attention = F.relu(attention)
        attention = self.attention_fc2(attention)
        attention = torch.sigmoid(attention).view(batch_size, channels, 1, 1)

        # Apply attention weights to multi-scale output
        out = multi_scale_out * attention

        # Batch normalization and ReLU activation
        out = self.bn(out)
        out = self.relu(out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.GELU()
        )
        self.proj = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch)
        self.proj_act = nn.GELU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=True),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW'):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils import LayerNorm, GRN


class ConvV2_concat_Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.conv = nn.Conv2d(dim, 64, 1)
        self.dwconv = nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64)  # depthwise conv
        self.norm = LayerNorm(64, eps=1e-6)
        self.pwconv1 = nn.Linear(64, 256)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(256)
        self.pwconv2 = nn.Linear(256, 64)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class CSE_Block(nn.Module):
    def __init__(self, in_channel, r, w, h):  # r是压缩比
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            # 在这里相当于把宽为80，高为80，通道为128的特征图直接池化成C维的向量（全局池化，并不是只池化一部分，所以要（w,h））
            nn.AvgPool2d((w, h)),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s*x