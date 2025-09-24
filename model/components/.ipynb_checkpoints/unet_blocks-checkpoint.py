import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from .attention_utils import LayerNorm

class Normalization(nn.Module):

    def __init__(self, in_channels, norm_type='instance', dim = 3):
        super().__init__()
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )
        elif norm_type=='instance':
            if dim == 3:
                self.norm = nn.InstanceNorm3d(
                    num_features=in_channels, 
                    affine=True,)
            elif dim == 2:
                self.norm = nn.InstanceNorm2d(
                    num_features=in_channels, 
                    affine=True,)
            else:
                raise NotImplementedError
    def forward(self, x):
        return self.norm(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1, dim = 3):
        super(DoubleConv, self).__init__()
        hidden_ch = out_ch // reduction
        if dim == 3:
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d
        else:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
            
        self.conv = nn.Sequential(
            conv(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            norm(hidden_ch),
            nn.GELU(),
            conv(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1, dropout=0.0, dim = 3):
        super().__init__()
        hidden_ch = out_ch // reduction
        if dim == 3:
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d
        else:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
            
        self.conv = nn.Sequential(
            conv(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            norm(hidden_ch),
            nn.GELU(),
            conv(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm(out_ch),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            conv(in_ch, out_ch, 1, 1),
            norm(out_ch),
            nn.Dropout(dropout),
        )
        self.relu = nn.GELU()
    
    def forward(self, x):
        net = self.conv(x)
        net = net + self.residual(x)
        net = self.relu(net)
        return net
    

class ConvNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                expasion_ratio:int=4, 
                kernel_size:int=7, 
                do_skip:int=True,
                norm_type:str = 'instance',
                n_groups = None,
                dim = 3,
                dropout_rate: float = 0.0,
                ):

        super().__init__()

        self.do_skip = do_skip

        assert dim in [2, 3], "dim should be 2 or 3, now is {}".format(dim)
        self.dim = dim
        if self.dim == 2:
            conv = nn.Conv2d
        elif self.dim == 3:
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions

        
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        self.norm = Normalization(in_channels, norm_type)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = expasion_ratio*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = expasion_ratio*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

 
    def forward(self, x):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.dropout(self.conv3(x1))
        if self.do_skip:
            x1 = x + x1  
        return x1

class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, 
                 stride=2, norm_type = 'instance', dim=3):

        super().__init__()

        if dim == 2:
            conv = nn.Conv2d
        elif dim == 3:
            conv = nn.Conv3d
        self.down = conv(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size//2,
        )
        self.norm = Normalization(out_channels, norm_type)
    def forward(self, x):
        
        return self.norm(self.down(x))
    
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, up_rate=2, norm_type = 'instance', dim=3):

        super().__init__()

        if dim == 2:
            conv = nn.ConvTranspose2d
        elif dim == 3:
            conv = nn.ConvTranspose3d
        self.up = conv(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = up_rate,
            stride = up_rate
        )
        self.norm = Normalization(out_channels, norm_type)
    def forward(self, x):
        
        return self.norm(self.up(x))

class ConvNeXtUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, 
                 norm_type='instance', dim=3):
        super().__init__()
        
        if dim == 2:
            conv = nn.ConvTranspose2d
        elif dim == 3:
            conv = nn.ConvTranspose3d

        kernel_size_ = ensure_tuple_rep(kernel_size, dim)
        scale_factor_ = ensure_tuple_rep(2, dim)
        padding = tuple((k - 1) // 2 for k in kernel_size_)  # type: ignore
        output_padding = tuple(s - 1 - (k - 1) % 2 for k, s in zip(kernel_size_, scale_factor_))  # type: ignore
        
        # out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding + 1
        self.norm = Normalization(in_channels, norm_type)

        self.up = conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size_,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                )
        
        self.dim = dim

    def forward(self, x, dummy_tensor=None):
        
        x = self.up(self.norm(x))

        return x

class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, conv_op=DoubleConv, reduction=1, dim = 3):
        super(Upsample, self).__init__()
        self.conv = conv_op(in_ch1+in_ch2, out_ch, reduction=reduction, dim=dim)
        if dim == 3:
            mode = "trilinear"
            conv = nn.Conv3d
        else:
            mode = "bilinear"
            conv = nn.Conv2d
            
        self.mode = mode
        self.up_conv = nn.Sequential(
            conv(in_ch2, in_ch2, 1, 1),
            nn.GELU()
        )

    def forward(self, x1, x2):
        up = F.interpolate(x2, x1.size()[2:], mode=self.mode)
        up = self.up_conv(up)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, dim = 3):
        super(UpsampleConv, self).__init__()
        if dim == 2:
            conv = nn.ConvTranspose2d
            norm = nn.InstanceNorm2d
        elif dim == 3:
            conv = nn.ConvTranspose3d
            norm = nn.InstanceNorm3d
        
        kernel_size_ = ensure_tuple_rep(kernel_size, dim)
        scale_factor_ = ensure_tuple_rep(2, dim)
        padding = tuple((k - 1) // 2 for k in kernel_size_)  # type: ignore
        output_padding = tuple(s - 1 - (k - 1) % 2 for k, s in zip(kernel_size_, scale_factor_))  # type: ignore
        
        # out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding + 1
        self.up_conv = nn.Sequential(
            conv(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size_,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                ),
            norm(out_ch),
            nn.GELU() 
        )
    
    def forward(self, x):
        return self.up_conv(x)

class TransposeUpsample(Upsample):
    def __init__(self, in_ch1, in_ch2, out_ch, kernel_size=4, stride=2, conv_op=DoubleConv, reduction=1, dim = 3):
        super().__init__(in_ch1, in_ch2, out_ch, conv_op, reduction, dim)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch2, in_ch2, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0),
            nn.GELU() 
        )
    
    def forward(self, x1, x2):
        up = self.up_conv(x2)
        net = torch.cat([x1, up], dim=1)
        net = self.conv(net)
        return net

