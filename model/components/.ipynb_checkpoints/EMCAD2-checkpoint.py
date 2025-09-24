import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from einops import rearrange
from .common_function import get_conv, get_norm, get_pool
from .unet_blocks import UpsampleConv

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='gelu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups, spatial_dim=3):
    
    if spatial_dim == 2:
        x = rearrange(x, 'b (g c) h w -> b (c g) h w', g=groups)
    elif spatial_dim == 3:
        x = rearrange(x, 'b (g c) d h w -> b (c g) d h w', g=groups)

    return x


class MSUNeXtBlock(nn.Module):

    def __init__(self, in_channels, kernel_sizes=[1,3,5], expansion_factor=2, activation='gelu', dropout=0.0, spatial_dim=3):
        super(MSUNeXtBlock, self).__init__()
        
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                get_conv(spatial_dim)(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels),
                get_norm("IN", spatial_dim)(in_channels),
                act_layer(activation, inplace=True),
                nn.Dropout(dropout)
            )
            for kernel_size in kernel_sizes
        ])

        self.pwconv = nn.Sequential(
            get_conv(spatial_dim)(in_channels, in_channels * expansion_factor, 1, 1, 0, bias=False),
            act_layer(activation, inplace=True),
            get_conv(spatial_dim)(in_channels * expansion_factor, in_channels, 1, 1, 0, bias=False), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        for dwconv in self.dwconvs:
            x = x + dwconv(x)
        return x + self.pwconv(x)


def MSUNeXtLayer(in_channels, depth=1, kernel_sizes=[1,3,5], expansion_factor=2, activation='gelu', dropout=0.0, spatial_dim=3):
        """
        create a series of multi-scale convolution blocks.
        """
        msconvs = nn.Sequential(*[
                        MSUNeXtBlock(in_channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                     activation=activation, dropout=dropout, spatial_dim=spatial_dim)
                        for _ in range(depth)
                    ])
        return msconvs     
    
# Contour Attention Gating (CAG)
class CAG(nn.Module):
    def __init__(self, in_channels, activation='gelu', spatial_dim=3):
        super(CAG, self).__init__()

        self.cag = nn.Sequential(
            get_conv(spatial_dim)(in_channels * 2, in_channels, 3, 1, 1, groups=in_channels),
            get_norm("IN", spatial_dim)(in_channels),
            act_layer(activation, inplace=True),
            get_conv(spatial_dim)(in_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )
  
    def forward(self, x1, x2):
        
        x = torch.cat([x1, x2], dim=1)
        x = channel_shuffle(x, 2)
        
        mask = self.cag(x)

        return x1 * mask + x2 * (1 - mask)
    


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='gelu', spatial_dim=3):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = get_pool('avg', spatial_dim)(1)
        self.max_pool = get_pool('max', spatial_dim)(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = get_conv(spatial_dim)(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = get_conv(spatial_dim)(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    

class SAB(nn.Module):
    def __init__(self, kernel_size=7, spatial_dim=3):
        super(SAB, self).__init__()

        self.spatial_attn = nn.Sequential(
            get_conv(spatial_dim)(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.spatial_attn(x)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=64, kernel_size=7, activation='gelu', spatial_dim=3):
        super(CBAM, self).__init__()

        self.spatial_attention = SAB(kernel_size=kernel_size, spatial_dim=spatial_dim)
        self.channel_attention = CAB(in_channels, ratio=ratio, activation=activation, spatial_dim=spatial_dim)
        
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class EMCAD(nn.Module):
    def __init__(self, base_ch, num_modalities=2, num_classes=2, 
                 kernel_sizes=[1,3,5], depths = [2,2,2,2], expansion_factor=6, 
                 dropout=0.1, activation='gelu', spatial_dim=3):
        super(EMCAD,self).__init__()
        
        
        self.up3 = UpsampleConv(in_ch=base_ch*8, out_ch=base_ch*4, kernel_size=2, groups=num_modalities, dropout=dropout, dim=spatial_dim)
        self.up2 = UpsampleConv(in_ch=base_ch*4, out_ch=base_ch*2, kernel_size=2, groups=num_modalities, dropout=dropout, dim=spatial_dim)
        self.up1 = UpsampleConv(in_ch=base_ch*2, out_ch=base_ch, kernel_size=2, groups=num_modalities, dropout=dropout, dim=spatial_dim)
        
        self.cag3 = CAG(in_channels=base_ch*4, activation=activation, spatial_dim=spatial_dim)
        self.cag2 = CAG(in_channels=base_ch*2, activation=activation, spatial_dim=spatial_dim)
        self.cag1 = CAG(in_channels=base_ch, activation=activation, spatial_dim=spatial_dim)
        
        self.cbam4 = CBAM(in_channels=base_ch*8, activation=activation, spatial_dim=spatial_dim)
        self.cbam3 = CBAM(in_channels=base_ch*4, activation=activation, spatial_dim=spatial_dim)
        self.cbam2 = CBAM(in_channels=base_ch*2, activation=activation, spatial_dim=spatial_dim)
        self.cbam1 = CBAM(in_channels=base_ch, activation=activation, spatial_dim=spatial_dim)
        
        self.mscb4 = MSUNeXtLayer(base_ch*8, depth=depths[0], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.mscb3 = MSUNeXtLayer(base_ch*4, depth=depths[1], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.mscb2 = MSUNeXtLayer(base_ch*2, depth=depths[2], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.mscb1 = MSUNeXtLayer(base_ch, depth=depths[3], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation, dropout=dropout, spatial_dim=spatial_dim)

        self.out_head4 = get_conv(spatial_dim)(base_ch*8, num_classes, 1)
        self.out_head3 = get_conv(spatial_dim)(base_ch*4, num_classes, 1)
        self.out_head2 = get_conv(spatial_dim)(base_ch*2, num_classes, 1)
        self.out_head1 = get_conv(spatial_dim)(base_ch, num_classes, 1)
       
      
    def forward(self, x, skips):
            
        d4 = self.cbam4(x) + x
        d4 = self.mscb4(d4)
        
        d3 = self.up3(d4)
        d3 = self.cag3(skips[0], d3)
        d3 = self.cbam3(d3) + d3
        d3 = self.mscb3(d3)

        d2 = self.up2(d3)
        d2 = self.cag2(skips[1], d2)
        d2 = self.cbam2(d2) + d2
        d2 = self.mscb2(d2)
        
        d1 = self.up1(d2)
        d1 = self.cag1(skips[2], d1)
        d1 = self.cbam1(d1) + d1
        d1 = self.mscb1(d1)
    
        p4 = self.out_head4(d4)
        p3 = self.out_head3(d3)
        p2 = self.out_head2(d2) 
        p1 = self.out_head1(d1)
        
        return p1, p2, p3, p4