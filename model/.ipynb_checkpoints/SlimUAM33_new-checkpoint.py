from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .SlimUM3 import SlimUM_Decoder, Conv_Encoder
from .components.attention_utils import LayerNorm
from .components.Attention2_new import Transformer_BasicLayer
from monai.networks.blocks import PatchEmbed
from einops import rearrange
from .components.common_function import check_input
from .components.unet_blocks import ResidualDoubleConv
from .components.initialization import InitWeights_He

# 改版：
# transformer encoder 的模态特异特征，直接采用简单的卷积与 conv encoder 特征进行通道对齐

class Transformer_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_channels: Sequence[int],
        embed_dim: int = 24,
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        min_dim_head: Sequence[int] = [4, 8, 8, 16],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        spatial_dim: str = 3
    ) -> None:

        super(Transformer_Encoder, self).__init__()
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)

        self.patch_size = patch_size
        self.patch_embeds = nn.ModuleList([PatchEmbed(
                                    patch_size      = self.patch_size,
                                    in_chans        = self.in_channels[m],
                                    embed_dim       = embed_dim,
                                    spatial_dims    = spatial_dim,
                                    ) for m in range(self.num_modalities)])
        
        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.layers = nn.ModuleList()

        input_size = torch.tensor(input_size) // self.patch_size
        for i_layer in range(self.num_layers):
            self.layers.append(Transformer_BasicLayer(
                input_size              = input_size.tolist(),
                in_channels             = [int(embed_dim * 2**i_layer)] * self.num_modalities,
                depth                   = depths[i_layer],
                min_big_window_size     = min_big_window_sizes[i_layer],
                min_small_window_size   = min_small_window_sizes[i_layer],
                scale_factor            = scale_factors[i_layer],
                num_heads               = num_heads[i_layer],
                min_dim_head            = min_dim_head[i_layer],
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio,
                act_layer               = act_layer,
                qkv_bias                = qkv_bias,
                do_downsample           = i_layer < self.num_layers - 1,
                dim                     = spatial_dim,
            ))
            input_size = input_size // 2

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c h w d -> n h w d c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w d c -> n c h w d")
                # x = F.gelu(x)
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
                # x = F.gelu(x)
        return x

    def forward(self, xs, normalize=False) -> Sequence[torch.Tensor]:
        
        # xs: Sequence[torch.Tensor], length = num_modalities
        xs = check_input(xs, self.in_channels)
        xs = [self.patch_embeds[m](xs[m]) for m in range(self.num_modalities)]
        xs = [self.pos_drop(xs[m]) for m in range(self.num_modalities)]
        
        # attn: (b, base_ch, h // 2, w // 2, d // 2)
        # down: (b, base_ch * 2, h // 4, w // 4, d // 4)
        attn, down = self.layers[0]([x.contiguous() for x in xs])
        attn1 = [self.proj_out(x, normalize) for x in attn]
        
        # attn: (b, base_ch * 2, h // 4, w // 4, d // 4)
        # down: (b, base_ch * 4, h // 8, w // 8, d // 8)
        attn, down = self.layers[1]([d.contiguous() for d in down])
        attn2 = [self.proj_out(x, normalize) for x in attn]
        
        # attn: (b, base_ch * 4, h // 8, w // 8, d // 8)
        # down: (b, base_ch * 8, h // 16, w // 16, d // 16)
        attn, down = self.layers[2]([d.contiguous() for d in down])
        attn3 = [self.proj_out(x, normalize) for x in attn]
        
        # attn: (b, base_ch * 8, h // 16, w // 16, d // 16)
        attn, _ = self.layers[3]([d.contiguous() for d in down])
        attn4 = [self.proj_out(x, normalize) for x in attn]
        
        return attn1, attn2, attn3, attn4
        
class SlimUAM_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_ch: Sequence[int],
        base_ch: int = 32,
        attn_base_ch: int = 16,
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        min_dim_head: Sequence[int] = [4, 8, 8, 16],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        conv_drop: float = 0.0,
        spatial_dim: str = 3,
        output_attn: bool = True,
    ):

        super(SlimUAM_Encoder, self).__init__()
        
        if spatial_dim == 3:
            conv = nn.Conv3d
        elif spatial_dim == 2:
            conv = nn.Conv2d
        
        self.in_channels = in_ch
        self.num_modalities = len(in_ch)
        self.output_attn = output_attn
        
        self.encoder_attn = Transformer_Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_channels             = in_ch,
            embed_dim               = attn_base_ch,
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
            spatial_dim             = spatial_dim
        )
        self.encoder_conv = Conv_Encoder(
            patch_size  = patch_size, 
            in_ch       = sum(in_ch), 
            base_ch     = base_ch, 
            dropout     = conv_drop,
            spatial_dim = spatial_dim
        )
        self.attn2conv_1 = conv(attn_base_ch     * self.num_modalities, base_ch    , 1, 1)
        self.attn2conv_2 = conv(attn_base_ch * 2 * self.num_modalities, base_ch * 2, 1, 1)
        self.attn2conv_3 = conv(attn_base_ch * 4 * self.num_modalities, base_ch * 4, 1, 1)
        self.attn2conv_4 = conv(attn_base_ch * 8 * self.num_modalities, base_ch * 8, 1, 1)
        
    def forward(self, x) -> Sequence[torch.Tensor]:

        attn1_, attn2_, attn3_, attn4_ = self.encoder_attn(x, normalize=False)
        attn1 = self.attn2conv_1(torch.cat(attn1_, dim=1))
        attn2 = self.attn2conv_2(torch.cat(attn2_, dim=1))
        attn3 = self.attn2conv_3(torch.cat(attn3_, dim=1))
        attn4 = self.attn2conv_4(torch.cat(attn4_, dim=1))
        
        x = self.encoder_conv.downs[0](x) + attn1
        x = self.encoder_conv.norms[0](x)
        enc1 = self.encoder_conv.layers[0](x)
        
        x = self.encoder_conv.downs[1](enc1) + attn2
        x = self.encoder_conv.norms[1](x)
        enc2 = self.encoder_conv.layers[1](x)
        
        x = self.encoder_conv.downs[2](enc2) + attn3
        x = self.encoder_conv.norms[2](x)
        enc3 = self.encoder_conv.layers[2](x)
        
        x = self.encoder_conv.downs[3](enc3) + attn4
        x = self.encoder_conv.norms[3](x)
        enc4 = self.encoder_conv.layers[3](x)
        
        if self.output_attn:
            return [attn1_, attn2_, attn3_, attn4_], [enc1, enc2, enc3, enc4]
        else:
            return enc1, enc2, enc3, enc4

class SlimUAM(nn.Module):
    
    def __init__(self,
                input_size: Sequence[int],
                patch_size: int,
                in_ch: Sequence[int],
                n_classes: int = 2,
                base_ch: int = 16,
                attn_base_ch: int = 16,
                depths: Sequence[int] = [2, 2, 2, 2],
                min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                scale_factors: Sequence[int] = [2, 2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 4, 8],
                min_dim_head: Sequence[int] = [4, 8, 8, 16],
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: int = 4,
                act_layer: str = "GELU",
                qkv_bias: bool = True,
                
                conv_drop: float = 0.0,
                deep_supervision: bool = True,
                output_feature: bool = False,
                output_attn: bool = True,
                spatial_dim: str = 3,):
        super(SlimUAM, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_classes = n_classes
        
        self.encoder = SlimUAM_Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_ch                   = in_ch,    
            base_ch                 = base_ch,
            attn_base_ch            = attn_base_ch,
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
            conv_drop               = conv_drop,
            spatial_dim             = spatial_dim,
            output_attn             = output_attn,
        )
        
        self.decoder = SlimUM_Decoder(
            patch_size              = patch_size,
            base_ch                 = base_ch,
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
        
        _, [enc1, enc2, enc3, enc4] = self.encoder(x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            
            return pred
        else:
            return pred[0]