from typing import Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from .components.unet_blocks3 import ResidualDoubleConv, DownConv, UpConv
from .components.Attention10 import Transformer_BasicLayer
from .components.attention_utils import OverlapPatchEmbed
from .components.initialization import InitWeights_He
from .components.common_function import get_conv, get_traspose_conv, get_norm

# 改版：
# 主要是改进了trans分支和conv分支相加部分，加了norm统一两者量纲

class Conv_Encoder(nn.Module):
    def __init__(self, 
                patch_size: int = 4, 
                in_ch: int = 1, 
                base_ch: int = 32,
                dropout: float = 0.0,
                spatial_dim: int = 3):
        super(Conv_Encoder, self).__init__()
        
        self.downs = nn.ModuleList([DownConv(in_ch, base_ch, patch_size=patch_size, dim=spatial_dim)])
        self.layers = nn.ModuleList([ResidualDoubleConv(base_ch, base_ch, dropout=dropout, dim=spatial_dim)])
        
        for i in range(3):
            self.downs.append(DownConv(base_ch * 2**i, base_ch * 2**(i+1), patch_size=2, dim=spatial_dim))
            self.layers.append(ResidualDoubleConv(base_ch * 2**(i+1), base_ch * 2**(i+1), dropout=dropout, dim=spatial_dim))

    
    def forward(self, x) -> Sequence[torch.Tensor]:
        
        x = self.downs[0](x)
        enc1 = self.layers[0](x)
        
        x = self.downs[1](enc1)
        enc2 = self.layers[1](x)
        
        x = self.downs[2](enc2)
        enc3 = self.layers[2](x)
        
        x = self.downs[3](enc3)
        enc4 = self.layers[3](x)
        
        return enc1, enc2, enc3, enc4
        
         

class SlimUM_Decoder(nn.Module):
    def __init__(self, patch_size: int, base_ch: int = 32, out_ch: int = 2, 
                 dropout: float = 0.0, deep_supervision: bool = False, 
                 spatial_dim: int = 3, output_feature: bool = False):
        super(SlimUM_Decoder, self).__init__()
        self.deep_supervision = deep_supervision
        self.output_feature = output_feature
        
        self.layer_up3 = UpConv(base_ch*8, base_ch*4, up_rate=2, dim=spatial_dim)
        self.layer_up2 = UpConv(base_ch*4, base_ch*2, up_rate=2, dim=spatial_dim)
        self.layer_up1 = UpConv(base_ch*2, base_ch  , up_rate=2, dim=spatial_dim)
        
        self.layer3 = ResidualDoubleConv(base_ch*4, base_ch*4, dropout=dropout, dim=spatial_dim)
        self.layer2 = ResidualDoubleConv(base_ch*2, base_ch*2, dropout=dropout, dim=spatial_dim)
        self.layer1 = ResidualDoubleConv(base_ch, base_ch, dropout=dropout, dim=spatial_dim)
        
        self.out_conv1 = get_traspose_conv(spatial_dim)(base_ch, out_ch, kernel_size=patch_size, 
                                                        stride=patch_size, padding=0, output_padding=0)
        if deep_supervision:
            self.out_conv2 = get_conv(spatial_dim)(base_ch * 2, out_ch, 1, 1)
            self.out_conv3 = get_conv(spatial_dim)(base_ch * 4, out_ch, 1, 1)
            self.out_conv4 = get_conv(spatial_dim)(base_ch * 8, out_ch, 1, 1)
    
    def forward(self, enc1, enc2, enc3, enc4) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        up3 = self.layer3(enc3 + self.layer_up3(enc4))
        up2 = self.layer2(enc2 + self.layer_up2(up3))
        up1 = self.layer1(enc1 + self.layer_up1(up2))
        out = self.out_conv1(up1)
        
        if self.deep_supervision:
            out4 = self.out_conv4(enc4)
            out3 = self.out_conv3(up3)
            out2 = self.out_conv2(up2)

            if self.output_feature:
                return [out, out2, out3, out4], [up1, up2, up3]
            else:
                return out, out2, out3, out4
        return out

class Transformer_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_channels: Sequence[int],
        embed_dim: int = 16,
        
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ffn_expansion_ratio: Sequence[int] = [4, 4, 4, 4],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        min_dim_head: Sequence[int] = [4, 4, 4, 4],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        spatial_dim: str = 3
    ) -> None:

        super(Transformer_Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)

        self.patch_size = patch_size
        self.patch_embeds = OverlapPatchEmbed(in_channels=sum(in_channels), out_channels=embed_dim*self.num_modalities,
                                            patch_size=patch_size, num_modalities=self.num_modalities, dim=spatial_dim)
        
        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.layers = nn.ModuleList()

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
                min_dim_head            = min_dim_head[i_layer],
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio[i_layer],
                act_layer               = act_layer,
                qkv_bias                = qkv_bias,
                
                do_downsample           = i_layer < self.num_layers - 1,
                dim                     = spatial_dim,
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
        
        return attn1, attn2, attn3, attn4
        
class SlimUAM_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_ch: Sequence[int],
        base_ch: int = 16,
        attn_base_ch: int = 16,
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ffn_expansion_ratio: Sequence[int] = [4, 4, 4, 4],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        min_dim_head: Sequence[int] = [8, 8, 16, 16],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        conv_drop: float = 0.0,
        spatial_dim: str = 3,
        output_attn: bool = True,
    ):

        super(SlimUAM_Encoder, self).__init__()
        
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
            ffn_expansion_ratio     = ffn_expansion_ratio,
            scale_factors           = scale_factors,
            num_heads               = num_heads,
            min_dim_head            = min_dim_head,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            drop_path               = drop_path,
            
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
        self.attn2conv_1 = nn.Sequential(get_conv(spatial_dim)(attn_base_ch     * self.num_modalities, base_ch    , 1, 1), 
                                         get_norm("IN", spatial_dim)(base_ch))
        self.attn2conv_2 = nn.Sequential(get_conv(spatial_dim)(attn_base_ch * 2 * self.num_modalities, base_ch * 2, 1, 1), 
                                         get_norm("IN", spatial_dim)(base_ch * 2))
        self.attn2conv_3 = nn.Sequential(get_conv(spatial_dim)(attn_base_ch * 4 * self.num_modalities, base_ch * 4, 1, 1), 
                                         get_norm("IN", spatial_dim)(base_ch * 4))
        self.attn2conv_4 = nn.Sequential(get_conv(spatial_dim)(attn_base_ch * 8 * self.num_modalities, base_ch * 8, 1, 1), 
                                         get_norm("IN", spatial_dim)(base_ch * 8))
        
    def forward(self, x) -> Sequence[torch.Tensor]:
        
        attn1_, attn2_, attn3_, attn4_ = self.encoder_attn(x)
        attn1 = self.attn2conv_1(attn1_)
        attn2 = self.attn2conv_2(attn2_)
        attn3 = self.attn2conv_3(attn3_)
        attn4 = self.attn2conv_4(attn4_)
        
        
        x = self.encoder_conv.downs[0](x) + attn1
        enc1 = self.encoder_conv.layers[0](x)
        
        x = self.encoder_conv.downs[1](enc1) + attn2
        enc2 = self.encoder_conv.layers[1](x)
        
        x = self.encoder_conv.downs[2](enc2) + attn3
        enc3 = self.encoder_conv.layers[2](x)
        
        x = self.encoder_conv.downs[3](enc3) + attn4
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
                ffn_expansion_ratio: Sequence[int] = [4, 4, 4, 4],
                scale_factors: Sequence[int] = [2, 2, 2, 2],
                num_heads: Sequence[int] = [1, 2, 4, 8],
                min_dim_head: Sequence[int] = [4, 4, 4, 4],
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                drop_path: float = 0,
                act_layer: str = "GELU",
                qkv_bias: bool = True,
                
                conv_drop: float = 0.1,
                deep_supervision: bool = True,
                output_feature: bool = False,
                output_attn: bool = True,
                spatial_dim: str = 3,):
        super(SlimUAM, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.size = None
        
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
        
    def scale_prediction(self, pred):
        if self.spatial_dim == 3:
            mode = "trilinear"
        elif self.spatial_dim == 2:
            mode = "bilinear"

        pred = F.interpolate(pred, size=self.size, mode=mode, align_corners=True)
        return pred
        
    def forward(self, x) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        
        self.size = x.shape[-self.spatial_dim:]
        
        _, [enc1, enc2, enc3, enc4] = self.encoder(x)
        
        pred = self.decoder(enc1, enc2, enc3, enc4)
        
        if self.training:
            pred = [self.scale_prediction(p) for  p in pred]
            
            return pred
        else:
            return pred[0]