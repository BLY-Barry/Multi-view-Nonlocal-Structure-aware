from turtle import forward
from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import torchvision.models as models
from lib.modules import *
from lib.transformer import TransformerBlock
from pytorch_wavelets import DWTForward, DWTInverse
import torch.fft
from scipy.ndimage import convolve
import numpy as np
import cv2


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.1)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class DWT(nn.Module):
    """ 兼容多通道输入的二维离散小波分解（支持自动微分） """

    def __init__(self, wavelet='Haar'):
        super().__init__()
        self.dwt = DWTForward(wave=wavelet, mode='symmetric', J=1)  # J=1 表示单层分解

    def forward(self, x):
        # 输入 x: [B, C, H, W]
        LL, high_coeffs = self.dwt(x)  # LL: [B, C, H/2, W/2]
        LH, HL, HH = torch.unbind(high_coeffs[0], dim=2)  # 分解三个高频分量
        return LL, (LH, HL, HH)


class IDWT(nn.Module):
    """ 兼容多通道输入的二维离散小波重构（支持自动微分） """

    def __init__(self, wavelet='Haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')

    def forward(self, LL, LH, HL, HH):
        # 输入 LL/LH/HL/HH: [B, C, H/2, W/2]
        high_coeffs = torch.stack([LH, HL, HH], dim=2)  # 合并为 [B, C, 3, H/2, W/2]
        return self.idwt((LL, [high_coeffs]))  # 输出重构图像 [B, C, H, W]


class WaveletEnhanceDenoise(nn.Module):
    """ 小波域增强与重构模块 """

    def __init__(self, wavelet='Haar', channels=3):
        super().__init__()
        self.dwt = DWT(wavelet=wavelet)
        self.idwt = IDWT(wavelet=wavelet)

        # 边缘增强模块（适用于LH,HL高频）
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.ReLU()
        )

        # 噪声抑制模块（适用于HH高频）
        self.denoise = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 初始化卷积核参数
        self._init_weights()

    def _init_weights(self):
        """ 初始化边缘增强和降噪的卷积核参数 """
        # 边缘增强核（类Sobel算子）
        edge_kernel = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]]).expand(3, 1, 3, 3).clone()
        self.edge_enhance[0].weight.data = edge_kernel

        # 降噪核（类高斯模糊）
        denoise_kernel = torch.tensor([[1., 2., 1.],
                                       [2., 4., 2.],
                                       [1., 2., 1.]]).expand(3, 1, 3, 3).clone() / 16.0
        self.denoise[0].weight.data = denoise_kernel

    def forward(self, x):
        # 小波分解
        LL, (LH, HL, HH) = self.dwt(x)

        # 高频分量处理
        LH = self.edge_enhance(LH)  # 水平边缘增强
        HL = self.edge_enhance(HL)  # 垂直边缘增强
        HH = self.denoise(HH)  # 对角噪声抑制

        # 小波重构
        return self.idwt(LL, LH, HL, HH)


class FourierProcessor(nn.Module):
    def __init__(self, D0=30, high_gain=1, threshold=0.1):
        super().__init__()
        self.D0 = D0  # 截止频率
        self.high_gain = high_gain  # 高频增益系数
        self.threshold = threshold  # 降噪阈值

    def forward(self, x):
        """
        输入:
            x - 输入图像张量 (batch, channels, H, W)
        输出:
            处理后的图像张量 (保持相同维度)
        """
        # 傅里叶变换
        f = torch.fft.fft2(x, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))

        # 创建距离矩阵
        _, _, H, W = x.shape
        y = torch.arange(H, device=x.device).view(-1, 1)
        x_coord = torch.arange(W, device=x.device)
        y_centered = y - H // 2
        x_centered = x_coord - W // 2
        d = torch.sqrt(y_centered.float().pow(2) + x_centered.float().pow(2))  # (H, W)

        # 创建滤波器掩码
        lowpass_mask = (d <= self.D0).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        highpass_mask = (d > self.D0).float().unsqueeze(0).unsqueeze(0)

        # 频率分解
        f_low = fshift * lowpass_mask
        f_high = fshift * highpass_mask

        # 高频处理（幅度增强 + 降噪）
        magnitude = torch.abs(f_high)
        phase = torch.angle(f_high)

        # 幅度增强
        magnitude_enhanced = magnitude * self.high_gain

        # 软阈值降噪
        magnitude_denoised = torch.sign(magnitude_enhanced) * F.relu(magnitude_enhanced - self.threshold)

        # 重建复数信号
        f_high_processed = magnitude_denoised * torch.exp(1j * phase)

        # 合并频率分量
        f_processed = f_low + f_high_processed

        # 逆傅里叶变换
        ishift = torch.fft.ifftshift(f_processed, dim=(-2, -1))
        img_back = torch.fft.ifft2(ishift, dim=(-2, -1))
        img_back = torch.abs(img_back)

        # 动态范围归一化
        min_val = img_back.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = img_back.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        img_normalized = (img_back - min_val) / (max_val - min_val + 1e-6)

        return img_normalized


class BaseNet(nn.Module):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        super(BaseNet, self).__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine, momentum=0.1)

    def MakeBlk(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, ):
        d = self.dilation * dilation

        conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
        t = nn.ModuleList([])
        t.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn: t.append(self._make_bn(outd))
        if relu: t.append(nn.ReLU(inplace=True))
        blk = nn.Sequential(*t)
        self.curchan = outd
        self.dilation *= stride

        return blk


class Adapter(BaseNet):
    def __init__(self, mchan=4, **kw):
        super(Adapter, self).__init__()
        t = BaseNet()
        tt = nn.ModuleList([])
        ops = nn.ModuleList([])
        ops.append(t.MakeBlk(8 * mchan))
        ops.append(t.MakeBlk(8 * mchan))
        ops.append(t.MakeBlk(16 * mchan, stride=2))
        ops.append(t.MakeBlk(16 * mchan))
        ops.append(t.MakeBlk(32 * mchan, stride=2))
        ops.append(t.MakeBlk(32 * mchan))
        self.ops = ops
        self.RLNs = tt

    def forward(self, x):
        for i, layer in enumerate(self.ops):
            if i % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x

class BaseNet2(nn.Module):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=12, dilated=True, dilation=1, bn=True, bn_affine=False):
        super(BaseNet2, self).__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine, momentum=0.1)

    def MakeBlk(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, ):
        d = self.dilation * dilation

        conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
        t = nn.ModuleList([])
        t.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn: t.append(self._make_bn(outd))
        if relu: t.append(nn.ReLU(inplace=True))
        blk = nn.Sequential(*t)
        self.curchan = outd
        self.dilation *= stride

        return blk


class Adapter2(BaseNet):
    def __init__(self, mchan=4, **kw):
        super(Adapter2, self).__init__()
        t = BaseNet2()
        tt = nn.ModuleList([])
        ops = nn.ModuleList([])
        ops.append(t.MakeBlk(8 * mchan))
        ops.append(t.MakeBlk(8 * mchan))
        ops.append(t.MakeBlk(16 * mchan, stride=2))
        ops.append(t.MakeBlk(16 * mchan))
        ops.append(t.MakeBlk(32 * mchan, stride=2))
        ops.append(t.MakeBlk(32 * mchan))
        self.ops = ops
        self.RLNs = tt

    def forward(self, x):
        for i, layer in enumerate(self.ops):
            if i % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim=128, mchan=4, relu22=False, dilation=4, **kw):
        super(Encoder, self).__init__()
        t = BaseNet(inchan=32 * mchan, dilation=dilation)
        ops = nn.ModuleList([])
        ops.append(t.MakeBlk(32 * mchan, k=3, stride=2, relu=False))
        ops.append(t.MakeBlk(32 * mchan, k=3, stride=2, relu=False))
        ops.append(t.MakeBlk(dim, k=3, stride=2, bn=False, relu=False))
        self.out_dim = dim
        self.ops = ops

    def forward(self, x):
        for i in range(len(self.ops)):
            if i % 2 == 1:
                x = self.ops[i](x) + x
            else:
                x = self.ops[i](x)
        return x


class ConditionalEstimator(nn.Module):
    def __init__(self) -> None:
        super(ConditionalEstimator, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.preconv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64, affine=False),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 32, kernel_size=3, dilation=2, padding=2, bias=False),
                                     nn.BatchNorm2d(32, affine=False),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 16, kernel_size=3, dilation=4, padding=4, bias=False),
                                     nn.BatchNorm2d(16, affine=False),
                                     nn.ReLU()
                                     )
        self.bn1 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
                                 nn.ReLU(),
                                 nn.InstanceNorm2d(16, affine=False),
                                 nn.ReLU())
        self.bn2 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
                                 nn.ReLU(),
                                 nn.InstanceNorm2d(16, affine=False),
                                 nn.ReLU())
        self.bn3 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
                                 nn.ReLU(),
                                 nn.InstanceNorm2d(16, affine=False),
                                 nn.ReLU())
        self.pool1 = nn.AvgPool2d(3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(16, affine=False),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(16, affine=False),
                                    nn.ReLU(),
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(16, affine=False),
                                    nn.ReLU())
        self.postconv = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1, bias=False))

    def LN(self, x, conv, pool, bn):
        x = conv(x)
        x = x.exp().clamp(max=1e4)  # 指数变换（将负值特征映射到接近零，正值特征被非线性放大，增强特征的对比度。突出显著特征，抑制噪声。） + 数值截断（防止指数爆炸）
        x = x / (pool(x) + 1e-5)  # 局部归一化（将每个特征值除以其局部邻域的统计量，实现局部对比度归一化）。抑制局部高响应区域的过度激活（防止单一特征主导）
        x = bn(x)  # 全局（bn+in）归一化
        return x

    def forward(self, x):
        x = self.dropout(x)
        x = self.preconv(x)
        x = self.LN(x, self.layer1, self.pool1, self.bn1)
        x = self.LN(x, self.layer2, self.pool2, self.bn2)
        x = self.LN(x, self.layer3, self.pool3, self.bn3)
        x = self.postconv(x)
        x = F.softmax(x, dim=1)[:, 0].unsqueeze(1)  # 对通道维度进行Softmax归一化，并提取第一个通道的概率值
        return x


class Superhead(nn.Module):
    def __init__(self) -> None:
        super(Superhead, self).__init__()
        self.PriorEstimator = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1))  # 先验概率

        self.ConditionalEstimator = ConditionalEstimator()  # 条件概率

    def p_x(self, x):
        x = self.PriorEstimator(x)
        p_x = F.softplus(x)
        p_x = p_x / (1 + p_x)
        # p_x = x/(1+x)
        return p_x

    def forward(self, x):
        p_c = self.ConditionalEstimator(x)  # 条件概率
        p_x = self.p_x(x)  # 处理后的先验概率
        # p_y = F.softplus(x1+x2)
        # p_y = p_y/(1+p_y)
        # p_x = self.p_x(x)
        p_y = (p_x * p_c)  # 最终的概率，即分数图
        return p_y


class ChannelAttention(nn.Module):
    """
    通道注意力模块。

    参数:
        in_channels (int): 输入通道数。
        reduction (int): 通道压缩比例, 默认为16。
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return out * x


class SpatialAttention(nn.Module):
    """
    空间注意力模块。

    参数:
        kernel_size (int): 卷积核大小, 默认为7。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out * x


class Hybrid(BaseNet):
    def __init__(self, mchan=4, **kw):
        super(Hybrid, self).__init__()
        t = BaseNet(inchan=128)
        # 混合注意力编码
        self.ca = ChannelAttention(32 * mchan)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力
        self.cn = t.MakeBlk(32 * mchan, k=1, bn=True, relu=True)

    def forward(self, x):
        x = self.sa(x) * self.ca(x)
        x = self.cn(x)

        return x


class SPEMDescriptor(nn.Module):
    def __init__(self, sigma1=[0.6, 0.8, 1.0], sigma2=[0.6, 0.8, 1.0], dilation_rates=[1, 2, 3],
                 orientations=6, orientations2=6):
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dilation_rates = dilation_rates
        self.angles = nn.Parameter(torch.linspace(0, math.pi, orientations), requires_grad=False)
        self.angles2 = nn.Parameter(torch.linspace(0, math.pi, orientations2), requires_grad=False)

    def _gaussian_kernel(self, size, sigma):  # 高斯核
        ax = torch.linspace(-(size // 2), size // 2, steps=size, device=self.angles.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (1 / (2 * math.pi * sigma ** 2)) * torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))  # 高斯函数
        return kernel / kernel.sum()

    def _first_order_filter(self, sigma, theta):  # 一阶高斯导数滤波器
        size = int(6 * sigma) | 1
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.angles.device)
        y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.angles.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        Gx = -xx * torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 4)  # 对x求导（0度）
        Gy = -yy * torch.exp(-(xx ** 2 + yy ** 2) / (2 * math.pi * sigma ** 4))  # 对y求导（90度）

        # 任意方向一阶响应的线性组合（由0度、90度组合）
        return torch.cos(theta) * Gx + torch.sin(theta) * Gy

    def _second_order_filter(self, sigma, theta):  # 二阶高斯导数滤波器
        size = int(6 * sigma) | 1
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.angles.device)
        y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=self.angles.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r2 = xx ** 2 + yy ** 2
        sigma_sq = sigma ** 2
        G_xx = (-1 / (2 * math.pi * sigma ** 4)) * (1 - (xx ** 2) / (sigma_sq ** 2)) * torch.exp(
            -r2 / (2 * sigma_sq))  # 0度
        G_yy = (-1 / (2 * math.pi * sigma ** 4)) * (1 - (yy ** 2) / (sigma_sq ** 2)) * torch.exp(
            -r2 / (2 * sigma_sq))  # 90度
        G_xy = (xx * yy) / (2 * math.pi * sigma ** 6) * torch.exp(-r2 / (2 * sigma_sq))

        # 任意方向二阶响应的线性组合（由0度、60度、120度组合）
        return (torch.cos(theta)**2) * G_xx + (torch.sin(theta)**2) * (G_yy - G_xy) - (2 * torch.sin(theta) * torch.cos(theta)) * (G_yy + G_xy)

    def _dilated_convolution(self, feat, kernel, dilation):  # 膨胀卷积
        k_size = kernel.shape[0]
        dilated_kernel = torch.zeros((k_size + (k_size - 1) * (dilation - 1),
                                      k_size + (k_size - 1) * (dilation - 1)), device=feat.device)
        dilated_kernel[::dilation, ::dilation] = kernel
        input_4d = feat.unsqueeze(1)
        weight = dilated_kernel.unsqueeze(0).unsqueeze(0)
        padding = (dilated_kernel.shape[0] // 2, dilated_kernel.shape[1] // 2)
        output = F.conv2d(input_4d, weight, padding=padding, stride=1)
        return output.squeeze(1)

    def build_sfoc(self, image):
        if image.size(1) == 3:
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = image[:, :1]
        gray = gray.float() / 255.0

        features = []

        # 一阶响应
        for theta in self.angles:  # 遍历每个方向
            combined_response = None
            for sigma in self.sigma1:  # 遍历每个标准差
                kernel = self._first_order_filter(sigma, theta)  # 一阶滤波器
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                pad = kernel.shape[-1] // 2
                # 计算单个一阶滤波器的响应
                response = F.conv2d(F.pad(gray, (pad, pad, pad, pad)), kernel).squeeze(1)
                # 累加同一方向下 不同标准差的响应
                if combined_response is None:
                    combined_response = response
                else:
                    combined_response += response
            features.append(combined_response)  # 添加方向的特征

        # 二阶响应
        for theta in self.angles2:  # 遍历每个方向
            combined_response = None
            for sigma in self.sigma2:  # 遍历每个标准差
                kernel = self._second_order_filter(sigma, theta)  # 一阶滤波器
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                pad = kernel.shape[-1] // 2
                # 计算单个一阶滤波器的响应
                response = F.conv2d(F.pad(gray, (pad, pad, pad, pad)), kernel).squeeze(1)
                # 累加同一方向下 不同标准差的响应
                if combined_response is None:
                    combined_response = response
                else:
                    combined_response += response
            features.append(combined_response)  # 添加各方向的特征

        dilated_features = []

        for feat in features:
            channel = []
            for rate in self.dilation_rates:  # 3个膨胀率
                g_kernel = self._gaussian_kernel(3, 1.0)
                conv = self._dilated_convolution(feat, g_kernel, rate)  # 膨胀卷积
                channel.append(conv)
            merged = torch.stack(channel, dim=1).sum(dim=1)  # 叠加3个膨胀率下的特征
            dilated_features.append(merged)  # 添加膨胀卷积处理后的特征

        features_tensor = torch.stack(dilated_features, dim=1)  # 拼接所有特征
        mean = features_tensor.mean(dim=(2, 3), keepdim=True)
        std = features_tensor.std(dim=(2, 3), keepdim=True)
        normalized_features = (features_tensor - mean) / (std + 1e-6)
        return normalized_features


class MMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sfoc = SPEMDescriptor()

        self.ada1 = Adapter()
        self.ada2 = Adapter2()

        self.hy = Hybrid()

        self.enc = Encoder()
        self.det = Superhead()

    def forward1(self, imgs):
        feat_in = self.ada1(imgs)

        feat_in = self.hy(feat_in)

        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat, dim=1), score

    def forward2(self, imgs):
        sfoc_feat = self.sfoc.build_sfoc(imgs)

        feat_in = self.ada2(sfoc_feat)

        feat_in = self.hy(feat_in)

        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat, dim=1), score

    def forward(self, img1, img2):
        feat1, score1 = self.forward1(img1)
        feat2, score2 = self.forward2(img2)
        return {'feat': [feat1, feat2], 'score': [score1, score2]}
