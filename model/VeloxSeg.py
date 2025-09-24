from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F

# Core components for the VeloxSeg architecture
from .components.initialization import InitWeights_He  # Weight initialization
from .components.attention_utils import LayerNorm     # Layer normalization
from .components.common_function import concat  # Utility functions

# Import encoder and decoder components
from .Encoder import Encoder
from .Decoder import RC_Decoder, Seg_Decoder


class VeloxSeg(nn.Module):
    """
    VeloxSeg: Multimodal Medical Image Segmentation Framework
    
    Core Architecture:
    1. Dual Encoder Branches:
       - Modal-Fusion Convolution Layer with JLC blocks (robust local features)
       - Modal-Cooperative Transformer Layer with PWA blocks (multimodal and global context)
    
    2. Dual Decoder Branches:
       - Segmentation Decoder (Student): Primary segmentation task with SDKT enhancement
       - Reconstruction Decoder (Teacher): Self-supervised texture teacher for knowledge transfer
    
    3. Spatially Decoupled Knowledge Transfer (SDKT):
       - Uses Gram matrices to characterize feature channel relationships
       - Establishes positive knowledge transfer path avoiding ROI discrepancies
       - Zero inference overhead enhancement: L_sdkt = Σ_m w_T^m * ||GM(D_T^m) - GM(D_seg)||^2
    
    Args:
        input_size: Input spatial dimensions
        patch_size: Size of patches for embedding
        in_ch: Number of input channels per modality (e.g., [1, 1] for PET+CT)
        n_classes: Number of segmentation classes
        base_ch: Base number of channels for convolution branch
        conv_depths: Number of JLC blocks at each level in conv branch
        kernel_sizes: Kernel sizes for group convolutions (parallel: [1,3,5])
        min_dim_group: Lower bound of group size for convolutions (JL-guided: [4,8,8,16])
        conv_expansion_factor: Expansion factors for FFN in conv branch
        attn_base_ch: Base number of channels for attention branch
        depths: Number of PWA blocks at each level in attention branch
        min_big_window_sizes: Window sizes for attention at each level
        min_small_window_sizes: Small window sizes for attention
        min_dim_head: Lower bound of head size for PWA (JL-guided)
        scale_factors: Downsampling factors
        num_heads: Number of attention heads
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        drop_path: Drop path rate
        ffn_expansion_ratio: FFN expansion ratios
        act_layer: Activation function
        norm_layer: Normalization layer type
        patch_norm: Whether to apply normalization after patch embedding
        qkv_bias: Whether to use bias in QKV projection
        conv_drop: Dropout rate for convolution branch
        deep_supervision: Whether to use deep supervision
        spatial_dim: Spatial dimensions (2D or 3D)
    """
    
    def __init__(self,
                input_size: Sequence[int],
                patch_size: int,
                in_ch: Sequence[int],
                n_classes: int = 2,
                base_ch: int = 16,
                
                conv_depths: Sequence[int] = [1, 1, 1, 1],
                kernel_sizes: Sequence[int] = [1, 3, 5],
                min_dim_group: Sequence[int] = [4, 8, 8, 16],
                conv_expansion_factor: Sequence[int] = [3, 3, 2, 2],
                
                attn_base_ch: int = 16,
                depths: Sequence[int] = [2, 2, 2, 2],
                min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [6, 6, 6], [3, 3, 3], [3, 3, 3]],
                min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                min_dim_head: Sequence[int] = [4, 8, 8, 16],
                scale_factors: Sequence[int] = [2, 2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 2, 4],
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: Sequence[int] = [3, 3, 2, 2],
                act_layer: str = "GELU",
                norm_layer: type[LayerNorm] = LayerNorm,
                patch_norm: bool = False,
                qkv_bias: bool = True,
                
                conv_drop: float = 0.0,
                deep_supervision: bool = True,
                spatial_dim: str = 3,):
        super(VeloxSeg, self).__init__()
        # Store model configuration
        self.size = input_size
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.num_modalities = len(in_ch)
        
        # Main encoder with dual branches (Modal-Fusion + Modal-Cooperative)
        self.encoder = Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_ch                   = in_ch,    
            base_ch                 = base_ch,
            
            conv_depths             = conv_depths,
            kernel_sizes            = kernel_sizes,
            min_dim_group           = min_dim_group,
            conv_expansion_factor   = conv_expansion_factor,
            
            attn_base_ch            = attn_base_ch,
            depths                  = depths,
            min_big_window_sizes    = min_big_window_sizes,
            min_small_window_sizes  = min_small_window_sizes,
            min_dim_head            = min_dim_head,
            scale_factors           = scale_factors,
            num_heads               = num_heads,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            drop_path               = drop_path,
            ffn_expansion_ratio     = ffn_expansion_ratio,
            act_layer               = act_layer,
            norm_layer              = norm_layer,
            patch_norm              = patch_norm,
            qkv_bias                = qkv_bias,
            conv_drop               = conv_drop,
            spatial_dim             = spatial_dim,
        )
        
        # Segmentation Decoder (Student branch)
        self.decoder = Seg_Decoder(
            patch_size              = patch_size,
            base_ch                 = base_ch,
            out_ch                  = n_classes,
            
            depths                  = conv_depths,
            kernel_sizes            = kernel_sizes,
            min_dim_group           = min_dim_group,
            expansion_factor        = conv_expansion_factor,
            
            dropout                 = conv_drop,
            deep_supervision        = deep_supervision,
            spatial_dim             = spatial_dim,
        )
        
        # Reconstruction Decoders (Teacher branch) - one for each modality
        # These implement the Self-Supervised Textual Teacher for SDKT knowledge distillation
        # Each teacher T_m learns rich textural details optimized by reconstruction tasks
        self.rc_decoders = nn.ModuleList([
            RC_Decoder(
                in_channel              = in_ch[i],
                enc_channel             = attn_base_ch + base_ch,  # Combined features from both branches
                dec_channel             = base_ch,
                patch_size              = patch_size,
                
                depths                  = conv_depths,
                kernel_sizes            = kernel_sizes,
                min_dim_group           = min_dim_group,
                expansion_factor        = conv_expansion_factor,

                spatial_dim             = spatial_dim,
                dropout                 = conv_drop,
            ) for i in range(len(in_ch))
        ])
        
        # Initialize model weights
        self.init_weights()
    
    def init_weights(self):
        self.apply(InitWeights_He(neg_slope=1e-2))
        
    def scale_prediction(self, pred):
        if self.spatial_dim == 3:
            mode = "trilinear"
        elif self.spatial_dim == 2:
            mode = "bilinear"

        pred = F.interpolate(pred, size=self.size, mode=mode, align_corners=True)
        return pred
    
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Returns:
            Training: [seg_preds, recon_preds, seg_params, recon_params]
            Inference: seg_prediction
        """
        # Process through dual encoder branches
        # Returns attention features (for knowledge distillation) and fused features
        
        if self.training:
            [attn1, attn2, attn3, attn4], [enc1, enc2, enc3, enc4] = self.encoder(x)
            # Segmentation Decoder (Student branch)
            # Generates predicted segmentation mask (Pred) with supervision signal (Lseg)
            pred, dec_pram = self.decoder(enc1, enc2, enc3, enc4)
            pred = [self.scale_prediction(p) for p in pred]
            
            # Reconstruction Decoders (Teacher branch)
            # Generate reconstructed PET and CT images (RPET, RCT) with supervision signal (Lrc)
            # Each teacher T_m extracts rich textural details for SDKT knowledge transfer
            rcs = []
            rc_prams = []
            for m in range(self.num_modalities):
                # Enabling teachers to access both local (JLC) and global (PWA) features
                rc, rc_pram = self.rc_decoders[m](concat(attn1[m], enc1), concat(attn2[m], enc2), 
                                                    concat(attn3[m], enc3), concat(attn4[m], enc4))
                rcs.append(rc)
                rc_prams.append(rc_pram)
            rcs = torch.cat(rcs, dim=1)
            
            # Return outputs for training:
            # - Segmentation predictions (for Lseg)
            # - Reconstruction predictions (for Lrc) 
            # - Gram matrices (for SDKT knowledge distillation via Gram matrices)
            return pred + [rcs] + [dec_pram] + rc_prams
        else:
            enc1, enc2, enc3, enc4 = self.encoder(x)
            # Inference mode: only segmentation prediction
            pred = self.decoder(enc1, enc2, enc3, enc4)
            return pred