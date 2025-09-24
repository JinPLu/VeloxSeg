from typing import Sequence, Union
import torch
import torch.nn as nn
from .components.attention_utils import LayerNorm
from .components.Attention2 import Cross_Channel_Attention
from .SlimUAM import SlimUAM
from .components.common_function import check_input
from .components.TAMW2 import TAMW_Block

    
class SlimUAM_TAMW_Decoder(nn.Module):
    def __init__(self, 
                input_size: Sequence[int],
                patch_size: int,
                base_ch: int = 32, 
                out_ch: int = 2, 
                 
                depths: Sequence[int] = [2, 2, 2],
                min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                scale_factors: Sequence[int] = [2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 4],
                min_dim_head: int = 4,
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                ffn_expansion_ratio: int = 4,
                act_layer: str = "GELU",
                norm_layer = LayerNorm,
                qkv_bias: bool = True,
                 
                deep_supervision: bool = True, 
                spatial_dim: int = 3):

        super(SlimUAM_TAMW_Decoder, self).__init__()
        
        assert deep_supervision == True, "TAMW only supports deep supervision"
        
        if spatial_dim == 3:
            conv = nn.Conv3d
            self.up_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            transpose_conv = nn.ConvTranspose3d
            
        elif spatial_dim == 2:
            conv = nn.Conv2d
            self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            transpose_conv = nn.ConvTranspose2d
        
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        input_size = torch.tensor(input_size) // patch_size
        num_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.upsample = nn.ModuleList([])
        self.tamws = nn.ModuleList([])
        self.enc_dec_attns = nn.ModuleList([])
        for i_layer in range(self.num_layers):
            if i_layer > 0:
                input_size = input_size // 2
            self.upsample.append(transpose_conv(num_channels[i_layer+1], num_channels[i_layer], kernel_size=2, stride=2))
            self.tamws.append(
                TAMW_Block(
                    input_size              = input_size.tolist(),
                    in_channels             = num_channels[i_layer],
                    out_channels            = num_channels[i_layer],
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
                    spatial_dim             = spatial_dim,)
                )
            self.enc_dec_attns.append(
                Cross_Channel_Attention(
                    ch1         = num_channels[i_layer], 
                    ch2         = num_channels[i_layer],
                    spatial_dim = spatial_dim,
                    output_both = False,
                )
            )

        # pred out
        self.out_conv1 = transpose_conv(base_ch, out_ch, kernel_size=patch_size, stride=patch_size, padding=0, output_padding=0)
        self.out_conv2 = conv(base_ch * 2, out_ch, 1, 1)
        self.out_conv3 = conv(base_ch * 4, out_ch, 1, 1)
        self.out_conv4 = conv(base_ch * 8, out_ch, 1, 1)


    def forward(self, enc1, enc2, enc3, enc4):
        pred4 = self.out_conv4(enc4)
        
        dec = self.upsample[2](enc4)
        dec = self.enc_dec_attns[2](enc3, dec)
        dec, fp3, fn3 = self.tamws[2](dec, self.up_2(pred4))
        pred3 = self.out_conv3(dec)
        
        dec = self.upsample[1](dec)
        dec = self.enc_dec_attns[1](enc2, dec)
        dec, fp2, fn2 = self.tamws[1](dec, self.up_2(pred3))
        pred2 = self.out_conv2(dec)
        
        dec = self.upsample[0](dec)
        dec = self.enc_dec_attns[0](enc1, dec)
        dec, fp1, fn1 = self.tamws[0](dec, self.up_2(pred2))
        pred1 = self.out_conv1(dec)

        # print(f"pred1.shape: {pred1.shape}, pred2.shape: {pred2.shape}, pred3.shape: {pred3.shape}, pred4.shape: {pred4.shape}")
        
        return [pred1, pred2, pred3, pred4], [fp1, fp2, fp3], [fn1, fn2, fn3]
        
class SlimUAM_TAMW(SlimUAM):
    
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
                spatial_dim: str = 3,):
        super(SlimUAM_TAMW, self).__init__(input_size, patch_size, in_ch, n_classes, 
                                           base_ch, depths, min_big_window_sizes, min_small_window_sizes, 
                                           scale_factors, num_heads, min_dim_head, attn_drop, proj_drop, 
                                           drop_path, ffn_expansion_ratio, act_layer, norm_layer, 
                                           patch_norm, qkv_bias)
        
        self.decoder = SlimUAM_TAMW_Decoder(
            input_size              = input_size,
            patch_size              = patch_size,
            base_ch                 = base_ch,
            out_ch                  = n_classes,
            depths                  = depths[:-1],
            min_big_window_sizes    = min_big_window_sizes[:-1],
            min_small_window_sizes  = min_small_window_sizes[:-1],
            scale_factors           = scale_factors[:-1],
            num_heads               = num_heads[:-1],
            min_dim_head            = min_dim_head,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            drop_path               = drop_path,
            ffn_expansion_ratio     = ffn_expansion_ratio,
            act_layer               = act_layer,
            norm_layer              = norm_layer,
            qkv_bias                = qkv_bias,
            spatial_dim             = spatial_dim
        )
        
        self.init_weights()
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        xs = check_input(x, self.in_ch)
        enc1, enc2, enc3, enc4 = self.encoder(xs, x)
        
        pred, fp, fn = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            fp = [self.scale_prediction(p, i_depth, is_prediction=False) for i_depth, p in enumerate(fp)]
            fn = [self.scale_prediction(p, i_depth, is_prediction=False) for i_depth, p in enumerate(fn)]
            
            return pred + fp + fn
        else:
            return pred[0]