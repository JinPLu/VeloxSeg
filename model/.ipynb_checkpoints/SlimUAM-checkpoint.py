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
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        patch_norm: bool = False,
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
                                    norm_layer      = norm_layer if patch_norm else None,
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
                min_dim_head            = min_dim_head,
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio,
                act_layer               = act_layer,
                norm_layer              = norm_layer,
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
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
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
        
        
class SlimUAM_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_ch: Sequence[int],
        base_ch: int = 32,
        kernel_size: int = 7,
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
        norm_layer = LayerNorm,
        patch_norm: bool = False,
        qkv_bias: bool = True,
        spatial_dim: str = 3,
    ):

        super(SlimUAM_Encoder, self).__init__()
        
        self.in_channels = in_ch
        self.num_modalities = len(in_ch)
        
        assert base_ch % self.num_modalities == 0, "base_ch should be divisible by num_modalities"
        
        self.encoder_attn = Transformer_Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_channels             = in_ch,
            embed_dim               = base_ch // self.num_modalities,
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
            spatial_dim             = spatial_dim
        )
        self.encoder_conv = Conv_Encoder(
            patch_size  = patch_size, 
            in_ch       = sum(in_ch), 
            base_ch     = base_ch, 
            kernel_size = kernel_size,
            spatial_dim = spatial_dim
        )
        
    def forward(self, xs, x) -> Sequence[torch.Tensor]:

        attn1, attn2, attn3, attn4 = self.encoder_attn(xs, normalize=False)
        attn1 = torch.cat(attn1, dim=1)
        attn2 = torch.cat(attn2, dim=1)
        attn3 = torch.cat(attn3, dim=1)
        attn4 = torch.cat(attn4, dim=1)
        
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
        
        return enc1, enc2, enc3, enc4
    

class SlimUAM_Decoder(nn.Module):
    def __init__(self, patch_size: int, base_ch: int = 32, out_ch: int = 2, 
                 deep_supervision: bool = False, spatial_dim: int = 3):
        super(SlimUAM_Decoder, self).__init__()
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

class SlimUAM(nn.Module):
    
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
            spatial_dim             = spatial_dim
        )
        
        self.decoder = SlimUAM_Decoder(
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
        
        xs = check_input(x, self.in_ch)
        enc1, enc2, enc3, enc4 = self.encoder(xs, x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            
            return pred
        else:
            return pred[0]