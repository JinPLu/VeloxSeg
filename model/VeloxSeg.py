"""VeloxSeg: JLC/PWA encoders and segmentation/texture-transfer decoders."""
import torch
from torch import nn

from .Encoder import Encoder
from .Decoder import RC_Decoder, Seg_Decoder
from .components.initialization import InitWeights_He


class VeloxSeg(nn.Module):
    def __init__(self, input_size, in_ch, stages, n_classes=2, dropout=0.0,
                 deep_supervision=True, spatial_dim=3):
        super().__init__()
        if len(stages) < 2:
            raise ValueError('VeloxSeg requires an encoder transition and a decoder')
        if len(input_size) != spatial_dim or spatial_dim not in (2, 3):
            raise ValueError('Input dimensions and spatial_dim must agree')
        if not in_ch or any(channels < 1 for channels in in_ch):
            raise ValueError('Specify a positive channel count for every modality group')
        shape = list(input_size)
        for stage in stages:
            if len(stage['stride']) != spatial_dim or any(stride < 1 for stride in stage['stride']):
                raise ValueError('Each stage requires a positive stride per spatial axis')
            if any(n % stride for n, stride in zip(shape, stage['stride'])):
                raise ValueError('Input must be divisible by the cumulative stage strides')
            shape = [n // stride for n, stride in zip(shape, stage['stride'])]
            if stage['channels'] % stage['group_width']:
                raise ValueError('JLC group width must divide stage channels')
        self.size = tuple(input_size)
        self.in_ch = tuple(in_ch)
        self.num_modalities = len(in_ch)
        self.spatial_dim = spatial_dim
        self.encoder = Encoder(input_size, in_ch, stages, dropout, spatial_dim)
        self.decoder = Seg_Decoder(n_classes, stages, dropout, deep_supervision, spatial_dim)
        self.rc_decoders = nn.ModuleList([
            RC_Decoder(channels, stages, dropout, spatial_dim) for channels in in_ch])
        self.apply(InitWeights_He(neg_slope=1e-2))

    def forward(self, x):
        # Under any caller's autocast (nnU-Net's predictor requests FP16) the
        # network runs in BF16, so training, validation and inference share one
        # numeric range; the FP32 islands inside stay FP32.
        with torch.autocast(x.device.type, dtype=torch.bfloat16,
                            enabled=torch.is_autocast_enabled(x.device.type)):
            attention, encoded = self.encoder(x)
            if not self.training:
                return self.decoder(encoded)
            predictions, student = self.decoder(encoded)
            reconstructions, teachers = [], []
            for modality, decoder in enumerate(self.rc_decoders):
                features = [torch.cat((local, cooperative[modality]), dim=1)
                            for local, cooperative in zip(encoded, attention)]
                reconstruction, teacher = decoder(features)
                reconstructions.append(reconstruction)
                teachers.append(teacher)
            return [*predictions, torch.cat(reconstructions, dim=1), student, *teachers]
