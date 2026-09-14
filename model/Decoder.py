"""Native VeloxSeg segmentation and reconstruction decoders."""
from math import prod
from torch import nn

from .components.conv_blocks import JLCLayer, UpConv
from .components.common_function import get_conv, get_norm, get_pram_matrix
from .components.superpixel import PixelShuffle


class DecoderStages(nn.Module):
    def __init__(self, stages, dropout, spatial_dim):
        super().__init__()
        self.ups = nn.ModuleList()
        self.layers = nn.ModuleList()
        for index, stage in enumerate(stages[:-1]):
            channels = stage['channels']
            self.ups.append(UpConv(stages[index + 1]['channels'], channels,
                                   stages[index + 1]['stride'], dim=spatial_dim))
            self.layers.append(JLCLayer(channels, stage['conv_depth'], stage['kernels'],
                                       channels // stage['group_width'], stage['conv_expansion'],
                                       dropout=dropout, spatial_dim=spatial_dim))

    def forward(self, features):
        decoded = [features[-1]]
        x = features[-1]
        for index in reversed(range(len(self.ups))):
            x = self.layers[index](features[index] + self.ups[index](x))
            decoded.append(x)
        return list(reversed(decoded))


def output_layer(channels, output_channels, stride, spatial_dim):
    return nn.Sequential(
        get_conv(spatial_dim)(channels, prod(stride) * output_channels, 3, padding=1),
        PixelShuffle(stride, spatial_dim))


class RC_Decoder(nn.Module):
    def __init__(self, in_channel, stages, dropout, spatial_dim):
        super().__init__()
        conv, norm = get_conv(spatial_dim), get_norm('IN', spatial_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(conv(2 * stage['channels'], stage['channels'], 1), norm(stage['channels']))
            for stage in stages])
        self.decoder = DecoderStages(stages, dropout, spatial_dim)
        self.output = output_layer(stages[0]['channels'], in_channel, stages[0]['stride'], spatial_dim)

    def forward(self, features):
        decoded = self.decoder([adapter(value) for adapter, value in zip(self.adapters, features)])
        return self.output(decoded[0]), get_pram_matrix(decoded[0])


class Seg_Decoder(nn.Module):
    def __init__(self, n_classes, stages, dropout, deep_supervision, spatial_dim):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.decoder = DecoderStages(stages, dropout, spatial_dim)
        self.output = output_layer(stages[0]['channels'], n_classes, stages[0]['stride'], spatial_dim)
        # Supervise decoded features, not the undecoded bottleneck.
        self.auxiliary_heads = nn.ModuleList([
            get_conv(spatial_dim)(stage['channels'], n_classes, 1) for stage in stages[1:-1]])

    def forward(self, features):
        decoded = self.decoder(features)
        prediction = self.output(decoded[0])
        if not self.training:
            return prediction
        predictions = [prediction]
        if self.deep_supervision:
            predictions.extend(head(value) for head, value in zip(self.auxiliary_heads, decoded[1:-1]))
        return predictions, get_pram_matrix(decoded[0])
