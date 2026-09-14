import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.model.components.attention_utils import LayerNorm
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.model.components.common_function import get_conv, get_norm, get_traspose_conv

# 所有Norm采用GN

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1, groups=1, dim = 3):
        super(DoubleConv, self).__init__()
        hidden_ch = out_ch // reduction

        self.conv = nn.Sequential(
            get_conv(dim)(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            get_norm("IN", dim)(hidden_ch),
            nn.GELU(),
            get_conv(dim)(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            get_norm("IN", dim)(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=1, reduction=1, dropout=0.0, dim = 3):
        super().__init__()
        hidden_ch = out_ch // reduction

        self.conv = nn.Sequential(
            get_conv(dim)(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            get_norm("IN", dim)(hidden_ch),
            nn.GELU(),
            get_conv(dim)(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            get_norm("IN", dim)(out_ch),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            get_conv(dim)(in_ch, out_ch, 1, 1, groups=groups),
            get_norm("IN", dim)(out_ch),
            nn.Dropout(dropout),
        )
        self.relu = nn.GELU()

    def forward(self, x):
        net = self.conv(x)
        net = net + self.residual(x)
        net = self.relu(net)
        return net



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size=2, groups=1, use_norm=True, dim=3):

        super().__init__()

        self.down = get_conv(dim)(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = tuple(2*s-1 for s in ensure_tuple_rep(patch_size, dim)),
            stride = patch_size,
            padding = tuple(s-1 for s in ensure_tuple_rep(patch_size, dim)),
            groups=groups
        )
        self.norm = get_norm("IN", dim)(out_channels) if use_norm else nn.Identity()
    def forward(self, x):

        return self.norm(self.down(x))

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, up_rate=2, groups=1, dim=3):

        super().__init__()

        self.up = get_traspose_conv(dim)(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = up_rate,
            stride = up_rate,
            groups=groups,
        )
        self.norm = get_norm("IN", dim)(out_channels)
    def forward(self, x):

        return self.norm(self.up(x))


class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, conv_op=DoubleConv,
                 groups=1, reduction=1, dim = 3):
        super(Upsample, self).__init__()
        self.conv = conv_op(in_ch1+in_ch2, out_ch, reduction=reduction, dim=dim)
        if dim == 3:
            mode = "trilinear"
        else:
            mode = "bilinear"

        self.mode = mode
        self.up_conv = nn.Sequential(
            get_conv(dim)(in_ch2, in_ch2, 1, 1, groups=groups),
            nn.GELU(),
        )

    def forward(self, x1, x2):
        up = F.interpolate(x2, x1.size()[2:], mode=self.mode)
        up = self.up_conv(up)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, groups=1, dropout=0.0, dim = 3):
        super(UpsampleConv, self).__init__()

        kernel_size_ = ensure_tuple_rep(kernel_size, dim)
        scale_factor_ = ensure_tuple_rep(2, dim)
        padding = tuple((k - 1) // 2 for k in kernel_size_)  # type: ignore
        output_padding = tuple(s - 1 - (k - 1) % 2 for k, s in zip(kernel_size_, scale_factor_))  # type: ignore

        # out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding + 1
        self.up_conv = nn.Sequential(
            get_conv(dim)(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size_,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                ),
            get_norm("IN", dim)(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.up_conv(x)

class TransposeUpsample(Upsample):
    def __init__(self, in_ch1, in_ch2, out_ch, kernel_size=4, stride=2,
                 groups=1, conv_op=DoubleConv, reduction=1, dim = 3):
        super().__init__(in_ch1, in_ch2, out_ch, conv_op, reduction, dim)
        self.up_conv = nn.Sequential(
            get_traspose_conv(dim)(in_ch2, in_ch2, kernel_size=kernel_size, groups=groups,
                                   stride=stride, padding=1, output_padding=0),
            nn.GELU()
        )

    def forward(self, x1, x2):
        up = self.up_conv(x2)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net
