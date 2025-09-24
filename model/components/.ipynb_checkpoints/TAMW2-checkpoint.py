import torch
import torch.nn as nn
from .Attention2 import Transformer_BasicLayer
from .attention_utils import LayerNorm
from typing import Sequence


class TAMW_Block(nn.Module):    
    def __init__(self, 
                input_size: Sequence[int],
                in_channels: int,
                out_channels: int,
                depth: int = 2,
                min_big_window_size: Sequence[int] = [3, 3, 3],
                min_small_window_size: Sequence[int] = [1, 1, 1],
                scale_factor: int = 2,
                num_heads: int = 1,
                min_dim_head: int = 4,
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: int = 4,
                act_layer: str = "GELU",
                norm_layer: type[LayerNorm] = LayerNorm,
                qkv_bias: bool = True,
                spatial_dim: int = 3,
                ):
        super(TAMW_Block, self).__init__()
        
        if spatial_dim == 2:
            norm = nn.InstanceNorm2d
            conv = nn.Conv2d
        elif spatial_dim == 3:
            norm = nn.InstanceNorm3d
            conv = nn.Conv3d

        self.attn_fore = Transformer_BasicLayer(
                    input_size              = input_size,
                    in_channels             = [in_channels],
                    depth                   = depth,
                    min_big_window_size     = min_big_window_size,
                    min_small_window_size   = min_small_window_size,
                    scale_factor            = scale_factor,
                    num_heads               = num_heads,
                    min_dim_head            = min_dim_head,
                    attn_drop               = attn_drop,
                    proj_drop               = proj_drop,
                    drop_path               = drop_path,
                    ffn_expansion_ratio     = ffn_expansion_ratio,
                    act_layer               = act_layer,
                    norm_layer              = norm_layer,
                    qkv_bias                = qkv_bias,
                    do_downsample           = False,
                    dim                     = spatial_dim,
                )
        
        self.attn_back = Transformer_BasicLayer(
                    input_size              = input_size,
                    in_channels             = [in_channels],
                    depth                   = depth,
                    min_big_window_size     = min_big_window_size,
                    min_small_window_size   = min_small_window_size,
                    scale_factor            = scale_factor,
                    num_heads               = num_heads,
                    min_dim_head            = min_dim_head,
                    attn_drop               = attn_drop,
                    proj_drop               = proj_drop,
                    drop_path               = drop_path,
                    ffn_expansion_ratio     = ffn_expansion_ratio,
                    act_layer               = act_layer,
                    norm_layer              = norm_layer,
                    qkv_bias                = qkv_bias,
                    do_downsample           = False,
                    dim                     = spatial_dim,
                )
        
        self.point_wise_conv = nn.Sequential(
            norm(in_channels),
            conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )
        
        self.output_fp = conv(in_channels, 2, kernel_size=1, stride=1, padding=0)
        self.output_fn = conv(in_channels, 2, kernel_size=1, stride=1, padding=0)
        
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.softmax = nn.Softmax(1)
        
    def forward(self, features, p_hi=None):
        '''
        feature: the feature after concatenation of encoder and decoder features
        p_hi: prediction map from (k+1)th decoder (after upsampling)
        '''
        p_hi = self.softmax(p_hi)
        
        foreground_slice = p_hi.argmax(dim=1, keepdim=True)
        background_slice = 1 - foreground_slice

        uncertain_mask = 1 - 2 * (p_hi[:, 0:1] - 0.5).abs()
        
        # attn return: [attns, down]
        # attns: Sequence[torch.Tensor], len(attns) = num_modalities
        fp_feature = self.attn_fore([features * foreground_slice * uncertain_mask])[0][0]
        fn_feature = self.attn_back([features * background_slice * uncertain_mask])[0][0]
        features = self.point_wise_conv(features - self.alpha * fp_feature + self.beta * fn_feature)
        
        fp = self.output_fp(fp_feature)
        fn = self.output_fn(fn_feature)
        
        return features, fp, fn