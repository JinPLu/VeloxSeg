"""VeloxSeg's two encoders, driven by one shared stage specification."""
from torch import nn
import torch
from monai.networks.blocks import PatchEmbed

from .components.attention_utils import PatchMerging
from .components.PWA import Transformer_BasicLayer
from .components.conv_blocks import DownConv, JLCLayer
from .components.common_function import get_conv, get_norm


class Conv_Encoder(nn.Module):
    def __init__(self, in_channels, stages, dropout, spatial_dim):
        super().__init__()
        self.downs = nn.ModuleList()
        self.layers = nn.ModuleList()
        previous = in_channels
        for stage in stages:
            channels = stage['channels']
            self.downs.append(DownConv(previous, channels, stage['stride'], dim=spatial_dim))
            self.layers.append(JLCLayer(
                channels, stage['conv_depth'], stage['kernels'],
                channels // stage['group_width'], stage['expansion'],
                dropout=dropout, spatial_dim=spatial_dim))
            previous = channels

    def forward(self, x, fusion_features):
        features = []
        for down, layer, fusion in zip(self.downs, self.layers, fusion_features):
            x = layer(down(x) + fusion)
            features.append(x)
        return features


class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, in_channels, stages, dropout, spatial_dim):
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=tuple(stages[0]['stride']), in_chans=channels,
                       embed_dim=stages[0]['channels'], spatial_dims=spatial_dim)
            for channels in in_channels])
        self.downs = nn.ModuleList()
        self.layers = nn.ModuleList()
        shape = list(input_size)
        for index, stage in enumerate(stages):
            shape = [n // stride for n, stride in zip(shape, stage['stride'])]
            if index:
                self.downs.append(nn.ModuleList([
                    PatchMerging(stages[index - 1]['channels'], stage['channels'],
                                 stage['stride'], dim=spatial_dim)
                    for _ in in_channels]))
            self.layers.append(Transformer_BasicLayer(
                input_size=shape, in_channels=[stage['channels']] * len(in_channels),
                depth=stage['attn_depth'], min_big_window_size=stage['big_window'],
                min_small_window_size=stage['small_window'], scale_factor=2,
                num_heads=stage['heads'], min_dim_head=stage['head_dim'],
                ffn_expansion_ratio=stage['expansion'], attn_drop=dropout,
                proj_drop=dropout, dim=spatial_dim))

    def forward(self, x):
        modalities = x.split(self.in_channels, dim=1)
        # Real padded MRI patches produce a large embedding-bias gradient.
        # Its FP16 convolution reduction overflows even after repeated scaler
        # backoff. Keep this compact input projection/normalization in FP32;
        # the subsequent attention and convolution stages still use autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xs = [embed(modality.float()) for embed, modality in zip(self.patch_embeds, modalities)]
        features = []
        for index, layer in enumerate(self.layers):
            if index:
                xs = [down(value) for down, value in zip(self.downs[index - 1], xs)]
            xs = layer(xs)
            features.append(xs)
        return features


class Encoder(nn.Module):
    def __init__(self, input_size, in_ch, stages, dropout, spatial_dim):
        super().__init__()
        conv, norm = get_conv(spatial_dim), get_norm('IN', spatial_dim)
        self.encoder_attn = Transformer_Encoder(input_size, in_ch, stages, dropout, spatial_dim)
        self.encoder_conv = Conv_Encoder(sum(in_ch), stages, dropout, spatial_dim)
        self.attn2conv = nn.ModuleList([
            nn.Sequential(conv(stage['channels'] * len(in_ch), stage['channels'], 1),
                          norm(stage['channels'])) for stage in stages])

    def forward(self, x):
        attention = self.encoder_attn(x)
        fusion = [mixer(torch.cat(features, dim=1))
                  for mixer, features in zip(self.attn2conv, attention)]
        convolution = self.encoder_conv(x, fusion)
        return attention, convolution
