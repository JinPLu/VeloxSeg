import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from .utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from .kan import KANLinear, KAN
from torch.nn import init


class KANLayer3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]
        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
        self.dwconv_1 = DW_bn_relu3D(hidden_features)
        self.dwconv_2 = DW_bn_relu3D(hidden_features)
        self.dwconv_3 = DW_bn_relu3D(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, D, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, D, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, D, H, W)
        return x

class KANBlock3D(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = KANLayer3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), D, H, W))
        return x


class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DW_bn_relu3D(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding """
    def __init__(self, img_size=64, patch_size=3, stride=2, in_chans=1, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.D, self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        self.num_patches = self.D * self.H * self.W
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C), N=D*H*W
        x = self.norm(x)
        return x, D, H, W


class ConvLayer3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class UKAN(nn.Module):
    def __init__(self, 
                 num_classes, 
                 input_channels=3, 
                 deep_supervision=False, 
                 img_size=[96, 96, 96], 
                 patch_size=16, 
                 embed_dims=[256, 320, 512], 
                 no_kan=False,
                 drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 depths=[1, 1, 1], 
                 **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer3D(input_channels, kan_input_dim//8)  # output: [B, C1, D, H, W]
        self.encoder2 = ConvLayer3D(kan_input_dim//8, kan_input_dim//4)  # output: [B, C2, D/2, H/2, W/2]
        self.encoder3 = ConvLayer3D(kan_input_dim//4, kan_input_dim)  # output: [B, C3, D/4, H/4, W/4]

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed3D(img_size=[s // 4 for s in img_size], patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed3D(img_size=[s // 8 for s in img_size], patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer3D(embed_dims[2], embed_dims[1])  # [B, C2, D, H, W]
        self.decoder2 = D_ConvLayer3D(embed_dims[1], embed_dims[0])  # [B, C0, D, H, W]
        self.decoder3 = D_ConvLayer3D(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer3D(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer3D(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv3d(embed_dims[0]//8, num_classes, kernel_size=1)  # output: [B, num_classes, D, H, W]
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        # x: [B, C_in, D, H, W]
        B = x.shape[0]
        ### Encoder
        # Stage 1
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # [B, C1, D/2, H/2, W/2]
        t1 = out
        # Stage 2
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))  # [B, C2, D/4, H/4, W/4]
        t2 = out
        # Stage 3
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))  # [B, C3, D/8, H/8, W/8]
        t3 = out

        ### Tokenized KAN Stage
        # Patch embedding 1
        out, D, H, W = self.patch_embed3(out)  # out: [B, N, C], N=D*H*W, C=embed_dims[1]
        for i, blk in enumerate(self.block1):
            out = blk(out, D, H, W)  # [B, N, C]
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        t4 = out

        ### Bottleneck
        out, D, H, W = self.patch_embed4(out)  # out: [B, N, C], N=D*H*W, C=embed_dims[2]
        for i, blk in enumerate(self.block2):
            out = blk(out, D, H, W)  # [B, N, C]
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]

        ### Decoder
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2,2), mode='trilinear', align_corners=True))  # [B, C2, D*2, H*2, W*2]
        out = torch.add(out, t4)  # [B, C2, D*2, H*2, W*2]
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1,2)  # [B, N, C], N=D*H*W
        for i, blk in enumerate(self.dblock1):
            out = blk(out, D, H, W)  # [B, N, C]
        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2,2), mode='trilinear', align_corners=True))  # [B, C0, D*2, H*2, W*2]
        out = torch.add(out, t3)  # [B, C0, D*2, H*2, W*2]
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1,2)  # [B, N, C]
        for i, blk in enumerate(self.dblock2):
            out = blk(out, D, H, W)  # [B, N, C]
        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2,2), mode='trilinear', align_corners=True))

        # 输出: [B, num_classes, D, H, W]
        return self.final(out)
