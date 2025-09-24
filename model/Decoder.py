from typing import Sequence, Union
import torch
import torch.nn as nn

# Core components for the VeloxSeg decoder architecture
from .components.conv_blocks import JLCLayer, UpConv  # Johnson-Lindenstrauss lemma-guided convolution & upsampling
from .components.common_function import get_conv, get_norm, get_pram_matrix  # Utility functions
from .components.superpixel import PixelShuffle        # Pixel shuffle for upsampling


class RC_Decoder(nn.Module):
    """
    Reconstruction Decoder (Self-Supervised Textual Teacher)

    Architecture Components:
    - Takes input from Modal-Cooperative Transformer Layer (bottom branch)
    - Uses Concatenation Operation (Ⓒ) to combine features from both encoder branches
    - Performs reconstruction of multimodal images (R_m, m=1,...,M)
    - Provides supervision signal for reconstruction loss (Lrc)
    - Acts as teacher for knowledge distillation (SDKT) to Segmentation Decoder
    
    Args:
        in_channel: Number of input channels for reconstruction
        enc_channel: Number of encoder channels (combined from both branches)
        dec_channel: Number of decoder channels
        patch_size: Size of patches for upsampling
        depths: Number of JLC blocks at each decoder level
        kernel_sizes: Kernel sizes for group convolutions (parallel: [1,3,5])
        min_dim_group: Lower bound of group size for convolutions (JL-guided: [4,8,8,16])
        expansion_factor: Expansion factors for FFN
        spatial_dim: Spatial dimensions (2D or 3D)
        dropout: Dropout rate
    """
    def __init__(self, 
                in_channel: int, 
                enc_channel: int, 
                dec_channel: int, 
                patch_size: int, 
                
                depths: Sequence[int] = [1, 1, 1, 1],
                kernel_sizes: Sequence[int] = [1, 3, 5],
                min_dim_group: Sequence[int] = [4, 8, 8, 16],
                expansion_factor: Sequence[int] = [3, 3, 2, 2],

                spatial_dim: int = 3, 
                dropout: float = 0.0):
        super(RC_Decoder, self).__init__()
        
        conv = get_conv(spatial_dim)
        norm = get_norm("IN", spatial_dim)

        # Feature adaptation layers for Concatenation Operation (Ⓒ)
        # These adapt encoder features from both branches for reconstruction
        self.enc2rc_4 = nn.Sequential(conv(enc_channel*8, dec_channel*8, 1, 1, 0), norm(dec_channel*8))
        self.enc2rc_3 = nn.Sequential(conv(enc_channel*4, dec_channel*4, 1, 1, 0), norm(dec_channel*4))
        self.enc2rc_2 = nn.Sequential(conv(enc_channel*2, dec_channel*2, 1, 1, 0), norm(dec_channel*2))
        self.enc2rc_1 = nn.Sequential(conv(enc_channel  , dec_channel  , 1, 1, 0), norm(dec_channel  ))
            
        # Upsampling layers (RC. Decoder Layer 3, 2, 1 in diagram)
        self.layer_up3 = UpConv(dec_channel*8, dec_channel*4, up_rate=2, dim=spatial_dim)
        self.layer_up2 = UpConv(dec_channel*4, dec_channel*2, up_rate=2, dim=spatial_dim)
        self.layer_up1 = UpConv(dec_channel*2, dec_channel, up_rate=2, dim=spatial_dim)
        
        # JLC blocks for reconstruction processing
        groups = [dec_channel * 2 ** i // min_dim_group[i] for i in range(4)]
        self.layer1 = JLCLayer(dec_channel  , depths[0], kernel_sizes, groups[0], expansion_factor[0], dropout=dropout, spatial_dim=spatial_dim)
        self.layer2 = JLCLayer(dec_channel*2, depths[1], kernel_sizes, groups[1], expansion_factor[1], dropout=dropout, spatial_dim=spatial_dim)
        self.layer3 = JLCLayer(dec_channel*4, depths[2], kernel_sizes, groups[2], expansion_factor[2], dropout=dropout, spatial_dim=spatial_dim)

        # Output layers: 3x3 Conv + Pixel Shuffle for reconstruction
        # This implements the "Conv+PixelShuffle" operation that unfolds channel relationships
        # into spatial details, providing the foundation for SDKT knowledge transfer
        self.out_conv = nn.Sequential(
            conv(dec_channel, (patch_size ** 3) * in_channel, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=patch_size, spatial_dim=spatial_dim)
        )
    
    def forward(self, enc1, enc2, enc3, enc4) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        # Adapt encoder features for reconstruction
        enc4 = self.enc2rc_4(enc4)
        enc3 = self.enc2rc_3(enc3)
        enc2 = self.enc2rc_2(enc2)
        enc1 = self.enc2rc_1(enc1)
        
        # Upsampling through RC. Decoder Layers (Layer 3, 2, 1 in diagram)
        up3 = self.layer3(enc3 + self.layer_up3(enc4))
        up2 = self.layer2(enc2 + self.layer_up2(up3))
        up1 = self.layer1(enc1 + self.layer_up1(up2))
        
        # Generate reconstructed image and return parameter matrix for SDKT
        if self.training:
            return self.out_conv(up1), get_pram_matrix(up1)
        else:
            return self.out_conv(up1)


class Seg_Decoder(nn.Module):
    """
    Segmentation Decoder (Student)
    
    Architecture Components:
    - Takes input from Modal-Fusion Convolution Layer (top branch)
    - Performs segmentation through Seg. Decoder Layers (Layer 3, 2, 1)
    - Generates predicted segmentation mask (Pred)
    - Receives supervision signal for segmentation loss (Lseg)
    - Acts as student receiving knowledge distillation (SDKT) from RC_Decoder
    
    Args:
        patch_size: Size of patches for upsampling
        base_ch: Base number of channels
        out_ch: Number of output classes
        depths: Number of JLC blocks at each decoder level
        kernel_sizes: Kernel sizes for group convolutions (parallel: [1,3,5])
        min_dim_group: Lower bound of group size for convolutions (JL-guided: [4,8,8,16])
        expansion_factor: Expansion factors for FFN
        dropout: Dropout rate
        deep_supervision: Whether to use deep supervision
        spatial_dim: Spatial dimensions (2D or 3D)
    """
    def __init__(self, 
                patch_size: int, 
                base_ch: int = 32, 
                out_ch: int = 2, 
                
                depths: Sequence[int] = [1, 1, 1, 1],
                kernel_sizes: Sequence[int] = [1, 3, 5],
                min_dim_group: Sequence[int] = [4, 8, 8, 16],
                expansion_factor: Sequence[int] = [3, 3, 2, 2],
                
                dropout: float = 0.0, 
                deep_supervision: bool = False, 
                spatial_dim: int = 3):
        super(Seg_Decoder, self).__init__()
        self.deep_supervision = deep_supervision
        
        conv = get_conv(spatial_dim)
        
        # Upsampling layers (Seg. Decoder Layer 3, 2, 1 in diagram)
        self.layer_up3 = UpConv(base_ch*8, base_ch*4, up_rate=2, dim=spatial_dim)
        self.layer_up2 = UpConv(base_ch*4, base_ch*2, up_rate=2, dim=spatial_dim)
        self.layer_up1 = UpConv(base_ch*2, base_ch, up_rate=2, dim=spatial_dim)
        
        # JLC blocks for segmentation processing
        groups = [base_ch * 2 ** i // min_dim_group[i] for i in range(4)]
        self.layer1 = JLCLayer(base_ch  , depths[0], kernel_sizes, groups[0], expansion_factor[0], dropout=dropout, spatial_dim=spatial_dim)
        self.layer2 = JLCLayer(base_ch*2, depths[1], kernel_sizes, groups[1], expansion_factor[1], dropout=dropout, spatial_dim=spatial_dim)
        self.layer3 = JLCLayer(base_ch*4, depths[2], kernel_sizes, groups[2], expansion_factor[2], dropout=dropout, spatial_dim=spatial_dim)

        # Main output layer: 3x3 Conv + Pixel Shuffle for segmentation
        self.out_conv1 = nn.Sequential(
            conv(base_ch, (patch_size ** 3) * out_ch, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=patch_size, spatial_dim=spatial_dim)
        )
        # Deep supervision outputs for multi-scale training
        if deep_supervision:
            self.out_conv2 = conv(base_ch * 2, out_ch, 1, 1)
            self.out_conv3 = conv(base_ch * 4, out_ch, 1, 1)
            self.out_conv4 = conv(base_ch * 8, out_ch, 1, 1)
    
    def forward(self, enc1, enc2, enc3, enc4) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        # Upsampling through Seg. Decoder Layers (Layer 3, 2, 1 in diagram)
        up3 = self.layer3(enc3 + self.layer_up3(enc4))
        up2 = self.layer2(enc2 + self.layer_up2(up3))
        up1 = self.layer1(enc1 + self.layer_up1(up2))
        
        # Generate predicted segmentation mask (Pred in diagram)
        out = self.out_conv1(up1)
        
        if self.training:
            if self.deep_supervision:
                # Multi-scale outputs for deep supervision
                out4 = self.out_conv4(enc4)
                out3 = self.out_conv3(up3)
                out2 = self.out_conv2(up2)
                
                return [out, out2, out3, out4], get_pram_matrix(up1)
            else:
                return out, up1
        return out
