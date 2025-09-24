from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .components.attention_utils import LayerNorm
from .SlimUAM import SlimUAM
from .SlimUM_SR import SR_Decoder 
from einops import rearrange



class SlimUAM_SR(SlimUAM):
    
    def __init__(self,
                input_size: Sequence[int],
                patch_size: int,
                in_ch: Sequence[int],
                n_classes: int = 2,
                base_ch: int = 16,
                depths: Sequence[int] = [2, 2, 2, 2],
                min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                scale_factors: Sequence[int] = [2, 2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 4, 8],
                min_dim_head: int = 4,
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: int = 4,
                act_layer: str = "GELU",
                norm_layer: type[LayerNorm] = LayerNorm,
                patch_norm: bool = False,
                qkv_bias: bool = True,
                deep_supervision: bool = True,
                spatial_dim: str = 3,):
        super(SlimUAM_SR, self).__init__(
            input_size              = input_size,
            patch_size              = patch_size,
            in_ch                   = in_ch,    
            n_classes               = n_classes,    
            base_ch                 = base_ch,
            depths                  = depths,
            min_big_window_sizes    = min_big_window_sizes,
            min_small_window_sizes  = min_small_window_sizes,
            scale_factors           = scale_factors,
            num_heads               = num_heads,    
            min_dim_head            = min_dim_head,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            drop_path               = drop_path,
            ffn_expansion_ratio     = ffn_expansion_ratio,
            act_layer               = act_layer,
            norm_layer              = norm_layer,
            patch_norm              = patch_norm,
            qkv_bias                = qkv_bias,
            deep_supervision        = deep_supervision,
            spatial_dim             = spatial_dim
        )
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_classes = n_classes
        
        self.sr_decoder = SR_Decoder(in_channels=sum(in_ch), patch_size=patch_size, base_ch=base_ch, spatial_dim=spatial_dim)
        
        self.init_weights()
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        _, [enc1, enc2, enc3, enc4] = self.encoder(x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)

        if self.training:
            sr = self.sr_decoder(enc1, enc2, enc3, enc4)
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            return pred + [sr]
        else:
            return pred[0]