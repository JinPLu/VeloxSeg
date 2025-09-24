from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .SlimUM3 import SlimUM_Decoder
from .components.attention_utils import OverlapPatchEmbed
from .components.Attention7 import Transformer_BasicLayer
from .components.initialization import InitWeights_He
from .components.EMCAD2 import Fusion_Conv
from monai.utils import ensure_tuple_rep

# 改版：
# transformer encoder 的模态特异特征，直接采用简单的卷积与 conv encoder 特征进行通道对齐


class Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_channels: Sequence[int],
        embed_dim: int = 16,
        depths: Sequence[int] = [1, 1, 1, 1],
        kernel_size: Sequence[int] = [3, 5, 7],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        min_dim_head: int = 4,
        groups_size: Sequence[int] = [4, 4, 8, 8],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        conv_drop: float = 0.0,
        drop_path: float = 0,
        ffn_expansion_ratio: Sequence[int] = [2, 2, 2, 2],
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        conv_norm_type: str = "IN",
        use_mix_modal: bool = True,
        output_feature: bool = False,
        spatial_dim: str = 3
    ) -> None:

        super(Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)
        self.use_mix_modal = use_mix_modal

        self.patch_size = patch_size
        self.patch_embeds = OverlapPatchEmbed(patch_size=patch_size, in_channels=sum(in_channels),
                                              embed_dim=embed_dim*self.num_modalities, norm_type=conv_norm_type,
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
                kernel_size             = kernel_size,
                min_big_window_size     = min_big_window_sizes[i_layer],
                min_small_window_size   = min_small_window_sizes[i_layer],
                scale_factor            = scale_factors[i_layer],
                num_heads               = num_heads[i_layer],
                min_dim_head            = min_dim_head,
                groups_size             = groups_size[i_layer],
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio[i_layer],
                act_layer               = act_layer,
                qkv_bias                = qkv_bias,
                conv_norm_type          = conv_norm_type,
                do_downsample           = i_layer < self.num_layers - 1,
                dim                     = spatial_dim,
            ))
            if use_mix_modal:
                self.mix_modal.append(Fusion_Conv(in_ch=embed_dim * 2**i_layer * self.num_modalities,
                                                  out_ch=embed_dim * 2**i_layer, activation=act_layer,
                                                  dropout=conv_drop, spatial_dim=spatial_dim))
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
                kernel_size: Sequence[int] = [3, 5, 7],
                min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                scale_factors: Sequence[int] = [2, 2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 4, 8],
                min_dim_head: int = 4,
                groups_size: Sequence[int] = [4, 4, 4, 4],
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: Sequence[int] = [2, 2, 2, 2],
                act_layer: str = "GELU",
                qkv_bias: bool = True,
                
                conv_norm_type: str = "IN",
                conv_drop: float = 0.0,
                use_mix_modal: bool = True,
                deep_supervision: bool = True,
                output_feature: bool = False,
                spatial_dim: str = 3,):
        super(SlimUA, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.num_modality = len(in_ch)
        self.n_classes = n_classes
        self.groups_size = ensure_tuple_rep(groups_size, len(depths))
        
        self.encoder = Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_channels             = in_ch,    
            embed_dim               = base_ch,
            depths                  = depths,
            kernel_size             = kernel_size,
            min_big_window_sizes    = min_big_window_sizes,
            min_small_window_sizes  = min_small_window_sizes,
            scale_factors           = scale_factors,
            num_heads               = num_heads,
            min_dim_head            = min_dim_head,
            groups_size             = self.groups_size,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            conv_drop               = conv_drop,
            drop_path               = drop_path,
            ffn_expansion_ratio     = ffn_expansion_ratio,
            act_layer               = act_layer,
            qkv_bias                = qkv_bias,
            conv_norm_type          = conv_norm_type,
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
        
    def scale_prediction(self, pred):
        if self.spatial_dim == 3:
            mode = "trilinear"
        elif self.spatial_dim == 2:
            mode = "bilinear"

        pred = F.interpolate(pred, size=self.size, mode=mode, align_corners=True)
        return pred
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        self.size = x.shape[-self.spatial_dim:]
        
        enc1, enc2, enc3, enc4 = self.encoder(x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p) for  p in pred]
            
            return pred
        else:
            return pred[0]