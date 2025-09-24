import torch.nn as nn
from .common_function import get_conv, get_norm, get_traspose_conv, get_act

class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size=2, groups=1, use_norm=True, dim=3):

        super().__init__()

        self.down = get_conv(dim)(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 2*patch_size-1,
            stride = patch_size,
            padding = patch_size-1,
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

class JLC(nn.Module):

    def __init__(self, in_channels, kernel_sizes=[1,3,5], groups=1, epansion_factor=4,
                 norm_type = "IN", activation='gelu', dropout=0.0, spatial_dim=3):
        super(JLC, self).__init__()
        
        conv = get_conv(spatial_dim)
        norm = get_norm(norm_type, spatial_dim)
        
        if len(kernel_sizes) > 1:
            self.spatial_convs = nn.ModuleList([
                nn.Sequential(
                    conv(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=groups),
                    norm(in_channels),
                    get_act(activation, inplace=True),
                )
                for kernel_size in kernel_sizes
            ])
        else:
            self.spatial_convs = nn.ModuleList([
                conv(in_channels, in_channels, kernel_sizes[0], padding=kernel_sizes[0] // 2, groups=groups)
            ])

        self.channel_conv = nn.Sequential(
            norm(in_channels),
            conv(in_channels, in_channels*epansion_factor, 1, 1, 0),
            get_act(activation, inplace=True),
            conv(in_channels*epansion_factor, in_channels, 1, 1, 0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = x + sum([conv(x) for conv in self.spatial_convs])
        out = out + self.channel_conv(out)
        return out

def JLCLayer(in_channels, depth=1, kernel_sizes=[1,3,5], groups=1, epansion_factor=4,
                 activation='gelu', dropout=0.0, spatial_dim=3):
    
        jlcs = nn.Sequential(*[
                        JLC(in_channels, kernel_sizes=kernel_sizes, groups=groups, epansion_factor=epansion_factor,
                                     activation=activation, dropout=dropout, spatial_dim=spatial_dim)
                        for _ in range(depth)
                    ])
        return jlcs     