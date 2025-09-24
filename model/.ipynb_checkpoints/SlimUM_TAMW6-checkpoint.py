from typing import Sequence, Union
import torch
from .SlimUM import SlimUM, SlimUM_Decoder
from .components.TAMW6 import TAMW_Block
from .components.attention_utils import LayerNorm
from torch import nn

# 改进：
# 1）将TAMW5的FP，FN预测的模块改为注意力


class SlimUM_TAMW_Decoder(SlimUM_Decoder):

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
                
                tamw_temperature: float = 2.0,
                deep_supervision: bool = True, 
                spatial_dim: int = 3):

        super(SlimUM_TAMW_Decoder, self).__init__(patch_size, base_ch, out_ch, deep_supervision, spatial_dim)
        
        assert deep_supervision == True, "TAMW only supports deep supervision"

        
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        input_size = torch.tensor(input_size) // patch_size
        num_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.tamws = nn.ModuleList([])
        for i_layer in range(self.num_layers):
            if i_layer > 0:
                input_size = input_size // 2
            self.tamws.append(
                TAMW_Block(
                    input_size              = input_size.tolist(),
                    enc_in_channels         = num_channels[i_layer],
                    dec_in_channels         = num_channels[i_layer+1],
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
                    temperature             = tamw_temperature,
                    spatial_dim             = spatial_dim,)
                )


    def forward(self, enc1, enc2, enc3, enc4):
        
        pred4 = self.out_conv4(enc4)
        
        dec, fp3, fn3 = self.tamws[2](enc3, enc4, pred4)
        pred3 = self.out_conv3(dec)
        
        dec, fp2, fn2 = self.tamws[1](enc2, enc3, pred3)
        pred2 = self.out_conv2(dec)
        
        dec, fp1, fn1 = self.tamws[0](enc1, enc2, pred2)
        pred1 = self.out_conv1(dec)
        
        return [pred1, pred2, pred3, pred4], [fp1, fp2, fp3], [fn1, fn2, fn3]
    
   
class SlimUM_TAMW(SlimUM):
    
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
                qkv_bias: bool = True,
                tamw_temperature: float = 2.0,
                
                deep_supervision: bool = True, 
                spatial_dim: str = 3,):
        super(SlimUM_TAMW, self).__init__(patch_size, in_ch, n_classes, base_ch, deep_supervision=True, spatial_dim=spatial_dim)
        
        self.decoder = SlimUM_TAMW_Decoder(
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
            tamw_temperature        = tamw_temperature,
            deep_supervision        = deep_supervision,
            spatial_dim             = spatial_dim
        )
        
        self.init_weights()
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        enc1, enc2, enc3, enc4 = self.encoder(x)
        
        pred, fp, fn = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p, i_depth, is_prediction=True) for i_depth, p in enumerate(pred)]
            fp = [self.scale_prediction(p, i_depth, is_prediction=False) for i_depth, p in enumerate(fp)]
            fn = [self.scale_prediction(p, i_depth, is_prediction=False) for i_depth, p in enumerate(fn)]
            
            return pred + fp + fn
        else:
            return pred[0]