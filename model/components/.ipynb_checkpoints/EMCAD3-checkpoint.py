import torch
import torch.nn as nn
from einops import rearrange
from .common_function import get_conv, get_norm, get_pool, get_act
from monai.networks.layers import DropPath
from .unet_blocks import UpsampleConv


def channel_shuffle(x, groups, spatial_dim=3):
    
    if spatial_dim == 2:
        x = rearrange(x, 'b (g c) h w -> b (c g) h w', g=groups)
    elif spatial_dim == 3:
        x = rearrange(x, 'b (g c) d h w -> b (c g) d h w', g=groups)

    return x


class MSUNeXtBlock(nn.Module):

    def __init__(self, in_channels, kernel_sizes=[1,3,5], expansion_factor=2, groups=1, 
                 norm_type = "IN", activation='gelu', dropout=0.0, spatial_dim=3):
        super(MSUNeXtBlock, self).__init__()

        self.use_skip = use_skip
        
        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                get_conv(spatial_dim)(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            )
            for kernel_size in kernel_sizes
        ])
        mid_channels = in_channels * expansion_factor
        self.channel_conv = nn.Sequential(
            get_norm(norm_type, spatial_dim)(in_channels),
            get_conv(spatial_dim)(in_channels, mid_channels, 1, 1, 0),
            get_act(activation, inplace=True),
            get_conv(spatial_dim)(mid_channels, in_channels, 1, 1, 0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        for dwconv in self.spatial_convs:
            x = x + dwconv(x)
        x = x + self.channel_conv(x)
        return x

def MSUNeXtLayer(in_channels, depth=1, kernel_sizes=[1,3,5], expansion_factor=2, 
                 groups_size=1, activation='gelu', dropout=0.0, spatial_dim=3):
    
        msconvs = nn.Sequential(*[
                        MSUNeXtBlock(in_channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                     groups_size=groups_size, activation=activation, dropout=dropout, 
                                     spatial_dim=spatial_dim)
                        for _ in range(depth)
                    ])
        return msconvs     
    
# Contour Attention Gating (CAG)
class CAG(nn.Module):
    
    '''
    需要将感受野相差较大的特征进行融合时
    为了避免直接相加造成的小感受野特征的细节丢失，首先用大小核的卷积调整感受野，
    采用这种方式进行加权融合
    '''
    
    def __init__(self, channels, k_x, k_y, groups=1, activation="gelu", spatial_dim=3):
        super(CAG, self).__init__()
        
        self.conv_x = get_conv(spatial_dim)(channels, channels, k_x, 1, k_x // 2, groups=groups)
        self.conv_y = get_conv(spatial_dim)(channels, channels, k_y, 1, k_y // 2, groups=groups)
        
        self.norm = get_norm("IN", spatial_dim)(channels * 2)
        self.act_layer = get_act(activation)
        self.gate = nn.Sequential(
            get_conv(spatial_dim)(channels * 2, 1, 1, 1, 0),
            nn.Tanh()
        )
    def forward(self, x, y):
        xy = torch.cat([self.conv_x(x), self.conv_y(y)], dim=1)
        xy = self.act_layer(self.norm(xy))
        mask = self.gate(xy)

        return x + (1 + mask) * y
    

class CAB(nn.Module):
    def __init__(self, in_channels, ratio=4, activation='gelu', spatial_dim=3):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = self.in_channels // ratio


        self.avg_pool = get_pool('avg', spatial_dim)(1)
        self.max_pool = get_pool('max', spatial_dim)(1)
        self.ca = nn.Sequential(
            get_conv(spatial_dim)(self.in_channels * 2, self.mid_channels * 2, 1, bias=False),
            get_act(activation, inplace=True),
            get_conv(spatial_dim)(self.mid_channels * 2, self.in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.ca(x)
    

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
    def __init__(self, in_channels, ratio=4, kernel_size=7, activation='gelu', spatial_dim=3):
        super(CBAM, self).__init__()
        
        self.channel_attention = CAB(in_channels, ratio=ratio, activation=activation, spatial_dim=spatial_dim)
        self.spatial_attention = SAB(kernel_size=kernel_size, spatial_dim=spatial_dim)
        
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=1, activation='gelu', dropout=0.0, spatial_dim=3):
        super(UpsampleConv,self).__init__()

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            get_conv(spatial_dim)(in_ch, in_ch, 3, 1, 1, groups=groups),
	        get_norm("IN", spatial_dim)(in_ch),
            get_act(activation, inplace=True),
            get_conv(spatial_dim)(in_ch, out_ch, 1, 1, 0),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.up_conv(x)
        return x
        

class EMCAD(nn.Module):
    def __init__(self, base_ch, num_classes=2, kernel_sizes=[1,3,5], 
                 depths = [2,2,2,2], expansion_factor=6, groups_sizes=[8,8,16,16],
                 dropout=0.1, activation='gelu', spatial_dim=3):
        super(EMCAD,self).__init__()
        
        
        self.up3 = UpsampleConv(in_ch=base_ch*8, out_ch=base_ch*4, groups_size=groups_sizes[3], dropout=dropout, spatial_dim=spatial_dim)
        self.up2 = UpsampleConv(in_ch=base_ch*4, out_ch=base_ch*2, groups_size=groups_sizes[2], dropout=dropout, spatial_dim=spatial_dim)
        self.up1 = UpsampleConv(in_ch=base_ch*2, out_ch=base_ch  , groups_size=groups_sizes[1], dropout=dropout, spatial_dim=spatial_dim)
        
        self.cag3 = CAG(in_channels=base_ch*4, groups_size=groups_sizes[2], activation=activation, spatial_dim=spatial_dim)
        self.cag2 = CAG(in_channels=base_ch*2, groups_size=groups_sizes[1], activation=activation, spatial_dim=spatial_dim)
        self.cag1 = CAG(in_channels=base_ch  , groups_size=groups_sizes[0], activation=activation, spatial_dim=spatial_dim)
        
        # self.cbam4 = CBAM(in_channels=base_ch*8, activation=activation, spatial_dim=spatial_dim)
        # self.cbam3 = CBAM(in_channels=base_ch*4, activation=activation, spatial_dim=spatial_dim)
        # self.cbam2 = CBAM(in_channels=base_ch*2, activation=activation, spatial_dim=spatial_dim)
        # self.cbam1 = CBAM(in_channels=base_ch  , activation=activation, spatial_dim=spatial_dim)
        
        self.msnx4 = MSUNeXtLayer(base_ch*8, depth=depths[0], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                  groups_size=groups_sizes[3], activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.msnx3 = MSUNeXtLayer(base_ch*4, depth=depths[1], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                  groups_size=groups_sizes[2], activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.msnx2 = MSUNeXtLayer(base_ch*2, depth=depths[2], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                  groups_size=groups_sizes[1], activation=activation, dropout=dropout, spatial_dim=spatial_dim)
        self.msnx1 = MSUNeXtLayer(base_ch  , depth=depths[3], kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, 
                                  groups_size=groups_sizes[0], activation=activation, dropout=dropout, spatial_dim=spatial_dim)

        self.out_head4 = get_conv(spatial_dim)(base_ch*8, num_classes, 1)
        self.out_head3 = get_conv(spatial_dim)(base_ch*4, num_classes, 1)
        self.out_head2 = get_conv(spatial_dim)(base_ch*2, num_classes, 1)
        self.out_head1 = get_conv(spatial_dim)(base_ch  , num_classes, 1)
       
      
    def forward(self, x, skips):
            
        # d4 = self.cbam4(x)
        d4 = self.msnx4(d4)
        
        d3 = self.up3(d4)
        d3 = self.cag3(skips[0], d3)
        # d3 = self.cbam3(d3)
        d3 = self.msnx3(d3)

        d2 = self.up2(d3)
        d2 = self.cag2(skips[1], d2)
        # d2 = self.cbam2(d2)
        d2 = self.msnx2(d2)
        
        d1 = self.up1(d2)
        d1 = self.cag1(skips[2], d1)
        # d1 = self.cbam1(d1)
        d1 = self.msnx1(d1)
    
        p4 = self.out_head4(d4)
        p3 = self.out_head3(d3)
        p2 = self.out_head2(d2) 
        p1 = self.out_head1(d1)
        
        return p1, p2, p3, p4