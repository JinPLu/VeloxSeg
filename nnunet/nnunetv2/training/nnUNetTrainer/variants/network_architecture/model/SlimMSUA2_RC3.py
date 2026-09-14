"""VeloxSeg RC3: a single stage specification drives both encoders and decoders."""
from math import prod
import torch
from torch import nn
from monai.networks.blocks import PatchEmbed
from nnunetv2.utilities.network_initialization import InitWeights_He
from .components.attention_utils import LayerNorm
from .components.Attention2_new2 import Transformer_BasicLayer
from .components.EMCAD4 import MSUNeXtLayer
from .components.common_function import get_conv, get_norm, get_gram_matrix
from .components.superpixel import PixelShuffle
from .components.unet_blocks3 import DownConv, UpConv


class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, strides, in_ch, channels, depths, big, small, scales,
                 heads, head_dims, attn_drop, proj_drop, drop_path, expansion,
                 act_layer, norm_layer, patch_norm, qkv_bias, spatial_dim):
        super().__init__()
        self.in_channels = in_ch
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=strides[0], in_chans=c, embed_dim=channels[0],
                       norm_layer=norm_layer if patch_norm else None, spatial_dims=spatial_dim)
            for c in in_ch])
        self.pos_drop = nn.Dropout(proj_drop)
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        shape = list(input_size)
        self.layers = nn.ModuleList()
        for i, c in enumerate(channels):
            shape = [n // s for n, s in zip(shape, strides[i])]
            self.layers.append(Transformer_BasicLayer(
                input_size=shape, in_channels=[c] * len(in_ch),
                down_channels=[channels[i+1]] * len(in_ch) if i+1 < len(channels) else [],
                down_stride=strides[i+1] if i+1 < len(channels) else [1]*spatial_dim,
                depth=depths[i], min_big_window_size=big[i], min_small_window_size=small[i],
                scale_factor=scales[i], num_heads=heads[i], min_dim_head=head_dims[i],
                attn_drop=attn_drop, proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                ffn_expansion_ratio=expansion[i], act_layer=act_layer,
                norm_layer=norm_layer, qkv_bias=qkv_bias, dim=spatial_dim))

    def forward(self, x):
        xs = [self.pos_drop(embed(part)) for embed, part in
              zip(self.patch_embeds, torch.split(x, self.in_channels, dim=1))]
        outputs = []
        for layer in self.layers:
            output, xs = layer([part.contiguous() for part in xs])
            outputs.append(output)
        return outputs


class Conv_Encoder(nn.Module):
    def __init__(self, in_ch, strides, channels, depths, kernels, groups, expansion, dropout, dim):
        super().__init__()
        self.num_stages = len(channels)
        for i, (cin, cout) in enumerate(zip([in_ch]+channels[:-1], channels)):
            self.add_module(f'down{i+1}', DownConv(cin, cout, patch_size=strides[i], dim=dim))
        for i, c in enumerate(channels):
            self.add_module(f'layer{i+1}', MSUNeXtLayer(c, depths[i], kernels[i], c//groups[i],
                                                      expansion[i], dropout=dropout, spatial_dim=dim))


class Encoder(nn.Module):
    def __init__(self, input_size, strides, in_ch, conv_channels, attn_channels,
                 conv_depths, kernel_sizes, min_dim_group, conv_expansion_factor,
                 depths, min_big_window_sizes, min_small_window_sizes, scale_factors,
                 num_heads, min_dim_head, attn_drop, proj_drop, drop_path,
                 ffn_expansion_ratio, act_layer, norm_layer, patch_norm, qkv_bias, conv_drop, spatial_dim):
        super().__init__()
        self.num_stages = len(conv_channels)
        self.encoder_attn = Transformer_Encoder(
            input_size, strides, in_ch, attn_channels, depths, min_big_window_sizes,
            min_small_window_sizes, scale_factors, num_heads, min_dim_head, attn_drop,
            proj_drop, drop_path, ffn_expansion_ratio, act_layer, norm_layer, patch_norm,
            qkv_bias, spatial_dim)
        self.encoder_conv = Conv_Encoder(sum(in_ch), strides, conv_channels, conv_depths,
                                        kernel_sizes, min_dim_group, conv_expansion_factor, conv_drop, spatial_dim)
        conv, norm = get_conv(spatial_dim), get_norm('IN', spatial_dim)
        for i, (a, c) in enumerate(zip(attn_channels, conv_channels)):
            self.add_module(f'attn2conv_{i+1}', nn.Sequential(conv(a*len(in_ch), c, 1, 1), norm(c)))

    def forward(self, x):
        attention = self.encoder_attn(x)
        fused = [getattr(self, f'attn2conv_{i+1}')(torch.cat(parts, dim=1))
                 for i, parts in enumerate(attention)]
        encoded = []
        for i, attn in enumerate(fused):
            x = getattr(self.encoder_conv, f'down{i+1}')(x) + attn
            x = getattr(self.encoder_conv, f'layer{i+1}')(x)
            encoded.append(x)
        return attention, encoded


def add_decoder_stages(module, channels, strides, depths, kernels, groups, expansion, dropout, dim):
    for i in reversed(range(len(channels)-1)):
        module.add_module(f'layer_up{i+1}', UpConv(channels[i+1], channels[i], up_rate=strides[i+1], dim=dim))
    for i, c in enumerate(channels[:-1]):
        module.add_module(f'layer{i+1}', MSUNeXtLayer(c, depths[i], kernels[i], c//groups[i],
                                                   expansion[i], dropout=dropout, spatial_dim=dim))


def decode(module, encoded):
    outputs = list(encoded)
    for i in reversed(range(len(outputs)-1)):
        outputs[i] = getattr(module, f'layer{i+1}')(
            encoded[i] + getattr(module, f'layer_up{i+1}')(outputs[i+1]))
    return outputs


class RC_Decoder(nn.Module):
    def __init__(self, in_channel, enc_channels, channels, strides, depths, kernels,
                 groups, expansion, dropout, dim):
        super().__init__()
        conv, norm = get_conv(dim), get_norm('IN', dim)
        for i in reversed(range(len(channels))):
            self.add_module(f'enc2rc_{i+1}', nn.Sequential(conv(enc_channels[i], channels[i], 1, 1, 0), norm(channels[i])))
        add_decoder_stages(self, channels, strides, depths, kernels, groups, expansion, dropout, dim)
        self.out_conv = nn.Sequential(conv(channels[0], prod(strides[0])*in_channel, 3, 1, 1),
                                      PixelShuffle(strides[0], dim))
        self.norm_before_out = norm(channels[0])

    def forward(self, *encoded):
        encoded = [getattr(self, f'enc2rc_{i+1}')(x) for i, x in enumerate(encoded)]
        x = self.norm_before_out(decode(self, encoded)[0])
        return self.out_conv(x), get_gram_matrix(x)


class Seg_Decoder(nn.Module):
    def __init__(self, out_ch, channels, strides, depths, kernels, groups, expansion,
                 dropout, deep_supervision, dim):
        super().__init__()
        self.deep_supervision = deep_supervision
        conv, norm = get_conv(dim), get_norm('IN', dim)
        add_decoder_stages(self, channels, strides, depths, kernels, groups, expansion, dropout, dim)
        self.out_conv1 = nn.Sequential(conv(channels[0], prod(strides[0])*out_ch, 3, 1, 1), PixelShuffle(strides[0], dim))
        for i, c in enumerate(channels):
            self.add_module(f'norm_before_out{i+1}', norm(c))
        for i, c in enumerate(channels[1:], 1):
            self.add_module(f'out_conv{i+1}', conv(c, out_ch, 1, 1))

    def forward(self, *encoded):
        decoded = decode(self, encoded)
        x = self.norm_before_out1(decoded[0])
        out = self.out_conv1(x)
        if not self.training:
            return out
        if self.deep_supervision:
            aux = [getattr(self, f'out_conv{i+1}')(getattr(self, f'norm_before_out{i+1}')(decoded[i]))
                   for i in reversed(range(1, len(decoded)))]
            out = [out, *reversed(aux)]
        return out, get_gram_matrix(x)


class SlimMSUA_RC(nn.Module):
    def __init__(self, input_size, strides, in_ch, conv_channels, attn_channels,
                 conv_depths, kernel_sizes, min_dim_group, conv_expansion_factor,
                 depths, min_big_window_sizes, min_small_window_sizes, min_dim_head,
                 scale_factors, num_heads, ffn_expansion_ratio,
                 n_classes=2, attn_drop=0.1, proj_drop=0.1, drop_path=0,
                 act_layer='GELU', norm_layer=LayerNorm, patch_norm=False,
                 qkv_bias=True, conv_drop=0.0, deep_supervision=True, spatial_dim=3):
        super().__init__()
        n = len(strides)
        stage_values = [conv_channels, attn_channels, conv_depths, kernel_sizes, min_dim_group,
                        conv_expansion_factor, depths, min_big_window_sizes, min_small_window_sizes,
                        min_dim_head, scale_factors, num_heads, ffn_expansion_ratio]
        if n < 2 or any(len(value) != n for value in stage_values):
            raise ValueError('Every stage field must match strides; at least two stages are required')
        shape = list(input_size)
        for i, stride in enumerate(strides):
            if any(s < 1 or p % s for p, s in zip(shape, stride)):
                raise ValueError('Input shape must be divisible by every cumulative stride')
            shape = [p//s for p, s in zip(shape, stride)]
            if prod(shape) < 2 or conv_channels[i] % min_dim_group[i]:
                raise ValueError('Invalid normalization shape or JLC channel grouping')
        self.in_ch, self.num_modalities = in_ch, len(in_ch)
        self.encoder = Encoder(input_size, strides, in_ch, conv_channels, attn_channels,
                               conv_depths, kernel_sizes, min_dim_group, conv_expansion_factor,
                               depths, min_big_window_sizes, min_small_window_sizes, scale_factors,
                               num_heads, min_dim_head, attn_drop, proj_drop, drop_path,
                               ffn_expansion_ratio, act_layer, norm_layer, patch_norm, qkv_bias, conv_drop, spatial_dim)
        decoder_args = (n_classes, conv_channels, strides, conv_depths, kernel_sizes,
                        min_dim_group, conv_expansion_factor, conv_drop, deep_supervision, spatial_dim)
        # RC3 historically initializes a segmentation model before replacing its decoder.
        # Preserve that RNG sequence, including the discarded decoder, for baseline parity.
        self.decoder = Seg_Decoder(*decoder_args)
        self.apply(InitWeights_He(neg_slope=1e-2))
        self.decoder = Seg_Decoder(*decoder_args)
        self.rc_decoders = nn.ModuleList([
            RC_Decoder(c, [a+b for a,b in zip(attn_channels, conv_channels)], conv_channels,
                       strides, conv_depths, kernel_sizes, min_dim_group, conv_expansion_factor, conv_drop, spatial_dim)
            for c in in_ch])
        self.apply(InitWeights_He(neg_slope=1e-2))

    def forward(self, x):
        attention, encoded = self.encoder(x)
        output = self.decoder(*encoded)
        if not self.training:
            return output
        seg, gram_seg = output
        rcs, grams = [], []
        for m, decoder in enumerate(self.rc_decoders):
            rc, gram = decoder(*[torch.cat((attn[m], enc), dim=1) for attn, enc in zip(attention, encoded)])
            rcs.append(rc)
            grams.append(gram)
        return {'seg': seg, 'rc': torch.cat(rcs, dim=1), 'gram_seg': gram_seg, 'gram_rc': grams}
