from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .components.attention_utils import LayerNorm
from .components.Attention2 import Transformer_BasicLayer
from monai.networks.blocks import PatchEmbed
from einops import rearrange
from .components.common_function import check_input
from .components.unet_blocks import ResidualDoubleConv, TransposeUpsample
from .components.initialization import InitWeights_He


class Conv_Encoder(nn.Module):
    def __init__(self, 
                patch_size: int = 4, 
                in_ch: int = 1, 
                base_ch: int = 32, 
                kernel_size: int = 7, 
                spatial_dim: int = 3):
        super(Conv_Encoder, self).__init__()
        if spatial_dim == 3:
            conv = nn.Conv3d
            norm_layer = nn.InstanceNorm3d
        elif spatial_dim == 2:
            conv = nn.Conv2d
            norm_layer = nn.InstanceNorm2d
        
        assert kernel_size >= patch_size, "kernel_size should be greater than patch_size"
        
        self.downs = nn.ModuleList([conv(in_ch, base_ch, kernel_size=kernel_size, stride=patch_size, padding=kernel_size//2)])
        self.norms = nn.ModuleList([norm_layer(base_ch)])
        # self.layers = nn.ModuleList([ConvNeXtBlock(base_ch, base_ch, kernel_size=kernel_size, dim=spatial_dim)])
        self.layers = nn.ModuleList([ResidualDoubleConv(base_ch, base_ch, dim=spatial_dim)])
        for i in range(3):
            self.downs.append(conv(base_ch * 2**i, base_ch * 2**(i+1), kernel_size=3, stride=2, padding=1))
            self.norms.append(norm_layer(base_ch * 2**(i+1)))
            # self.layers.append(ConvNeXtBlock(base_ch * 2**(i+1), base_ch * 2**(i+1), 
            #                                  kernel_size=kernel_size, dim=spatial_dim))
            self.layers.append(ResidualDoubleConv(base_ch * 2**(i+1), base_ch * 2**(i+1), dim=spatial_dim))

    
    def forward(self, x) -> Sequence[torch.Tensor]:
        
        x = self.norms[0](self.downs[0](x))
        enc1 = self.layers[0](x)
        
        x = self.norms[1](self.downs[1](enc1))
        enc2 = self.layers[1](x)
        
        x = self.norms[2](self.downs[2](enc2))
        enc3 = self.layers[2](x)
        
        x = self.norms[3](self.downs[3](enc3))
        enc4 = self.layers[3](x)
        
        return enc1, enc2, enc3, enc4
        
         

class SlimUM_Decoder(nn.Module):
    def __init__(self, patch_size: int, base_ch: int = 32, out_ch: int = 2, 
                 deep_supervision: bool = False, spatial_dim: int = 3):
        super(SlimUM_Decoder, self).__init__()
        self.deep_supervision = deep_supervision
        
        self.layer_up3 = TransposeUpsample(base_ch*4, base_ch*8, base_ch*4, conv_op=ResidualDoubleConv, dim=spatial_dim)
        self.layer_up2 = TransposeUpsample(base_ch*2, base_ch*4, base_ch*2, conv_op=ResidualDoubleConv, dim=spatial_dim)
        self.layer_up1 = TransposeUpsample(base_ch  , base_ch*2, base_ch  , conv_op=ResidualDoubleConv, dim=spatial_dim)
        
        if spatial_dim == 3:
            conv = nn.Conv3d
            transpose_conv = nn.ConvTranspose3d
        elif spatial_dim == 2:
            conv = nn.Conv2d
            transpose_conv = nn.ConvTranspose2d
            
        self.out_conv1 = transpose_conv(base_ch, out_ch, kernel_size=patch_size, 
                                        stride=patch_size, padding=0, output_padding=0)
        if deep_supervision:
            self.out_conv2 = conv(base_ch * 2, out_ch, 1, 1)
            self.out_conv3 = conv(base_ch * 4, out_ch, 1, 1)
            self.out_conv4 = conv(base_ch * 8, out_ch, 1, 1)
    
    def forward(self, enc1, enc2, enc3, enc4) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        up3 = self.layer_up3(enc3, enc4)
        up2 = self.layer_up2(enc2, up3)
        up1 = self.layer_up1(enc1, up2)
        out = self.out_conv1(up1)
        
        if self.deep_supervision:
            out4 = self.out_conv4(enc4)
            out3 = self.out_conv3(up3)
            out2 = self.out_conv2(up2)

            return out, out2, out3, out4
        return out

class SlimUM(nn.Module):
    
    def __init__(self,
                patch_size: int,
                in_ch: Sequence[int],
                n_classes: int = 2,
                base_ch: int = 32,
                deep_supervision: bool = True,
                spatial_dim: str = 3,):
        super(SlimUM, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_classes = n_classes
        
        self.encoder = Conv_Encoder(
            patch_size              = patch_size,
            in_ch                   = sum(in_ch),    
            base_ch                 = base_ch,
            spatial_dim             = spatial_dim
        )
        
        self.decoder = SlimUM_Decoder(
            patch_size              = patch_size,
            base_ch                 = base_ch,
            out_ch                  = n_classes,
            deep_supervision        = deep_supervision,
            spatial_dim             = spatial_dim
        )
        
        self.init_weights()
        
    def init_weights(self):
        self.apply(InitWeights_He(neg_slope=1e-2))
        
    def scale_prediction(self, pred, i_depth, is_prediction=True):
        if self.spatial_dim == 3:
            mode = "trilinear"
        elif self.spatial_dim == 2:
            mode = "bilinear"
        
        if (i_depth == 0) and is_prediction:
            return pred
        else:
            pred = F.interpolate(pred, scale_factor=self.patch_size * 2 ** i_depth, mode=mode, align_corners=True)
            return pred
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        enc1, enc2, enc3, enc4 = self.encoder(x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            
            return pred
        else:
            return pred[0]