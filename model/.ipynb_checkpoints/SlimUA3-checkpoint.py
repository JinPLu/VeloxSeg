from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .SlimUM3 import SlimUM_Decoder
from .components.attention_utils import OverlapPatchEmbed
from .components.Attention3 import Transformer_BasicLayer
from .components.common_function import get_conv
from .components.initialization import InitWeights_He

# 改版：
# transformer encoder 的模态特异特征，直接采用简单的卷积与 conv encoder 特征进行通道对齐

class Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_channels: Sequence[int],
        embed_dim: int = 24,
        depths: Sequence[int] = [3, 3, 9, 3],
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
        qkv_bias: bool = True,
        
        use_mix_modal: bool = True,
        spatial_dim: str = 3
    ) -> None:

        super(Encoder, self).__init__()

        
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)
        self.use_mix_modal = use_mix_modal

        self.patch_size = patch_size
        self.patch_embeds = OverlapPatchEmbed(patch_size=patch_size, in_channels=sum(in_channels),
                                              embed_dim=embed_dim*self.num_modalities,
                                              groups=self.num_modalities, spatial_dims=spatial_dim)
        
        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.layers = nn.ModuleList()
        if use_mix_modal:
            self.mix_modal = nn.ModuleList()

        input_size = torch.tensor(input_size) // self.patch_size
        for i_layer in range(self.num_layers):
            self.layers.append(Transformer_BasicLayer(
                input_size              = input_size.tolist(),
                in_channels             = [embed_dim * 2**i_layer] * self.num_modalities,
                depth                   = depths[i_layer],
                min_big_window_size     = min_big_window_sizes[i_layer],
                min_small_window_size   = min_small_window_sizes[i_layer],
                scale_factor            = scale_factors[i_layer],
                num_heads               = num_heads[i_layer],
                min_dim_head            = min_dim_head,
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio,
                act_layer               = act_layer,
                qkv_bias                = qkv_bias,
                do_downsample           = i_layer < self.num_layers - 1,
                dim                     = spatial_dim,
            ))
            if use_mix_modal:
                self.mix_modal.append(nn.Sequential(
                    nn.GroupNorm(num_groups=self.num_modalities, num_channels=embed_dim * 2**i_layer*self.num_modalities),
                    get_conv(spatial_dim)(embed_dim * 2**i_layer * self.num_modalities, embed_dim * 2**i_layer, 1, 1, 0, bias=False),
                    nn.GELU()
                ))
            input_size = input_size // 2


    def forward(self, xs) -> Sequence[torch.Tensor]:
        
        # xs: torch.Tensor
        xs = self.patch_embeds(xs)
        xs = self.pos_drop(xs)
        
        attn1, down = self.layers[0](xs)
        attn2, down = self.layers[1](down)
        attn3, down = self.layers[2](down)
        attn4, _ = self.layers[3](down)
        
        if self.use_mix_modal:
            return self.mix_modal[0](attn1), self.mix_modal[1](attn2), self.mix_modal[2](attn3), self.mix_modal[3](attn4)
        else:
            return attn1, attn2, attn3, attn4


class SlimUA(nn.Module):
    
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
                qkv_bias: bool = True,
                
                use_mix_modal: bool = True,
                conv_drop: float = 0.0,
                deep_supervision: bool = True,
                output_feature: bool = False,
                spatial_dim: str = 3,):
        super(SlimUA, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.num_modality = len(in_ch)
        self.n_classes = n_classes
        
        self.encoder = Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_channels             = in_ch,    
            embed_dim               = base_ch,
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
            qkv_bias                = qkv_bias,
            use_mix_modal           = use_mix_modal,
            spatial_dim             = spatial_dim,
        )
        
        self.decoder = SlimUM_Decoder(
            patch_size              = patch_size,
            base_ch                 = base_ch if use_mix_modal else base_ch*self.num_modality,
            out_ch                  = n_classes,
            dropout                 = conv_drop,
            deep_supervision        = deep_supervision,
            spatial_dim             = spatial_dim,
            output_feature          = output_feature,
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