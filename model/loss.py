"""Training objective for the native VeloxSeg output contract."""

import torch
from torch import nn
from torch.nn import functional as F

from utils.runtime import veloxseg_output_layout


class VeloxSegLoss(nn.Module):
    """Resolution-weighted segmentation, modality-mean reconstruction, and summed SDKT.

    The model returns Gram matrices normalized by C*N. SDKT uses a squared
    Frobenius norm per sample, sums teachers, and averages the batch. Taking
    another elementwise mean here would introduce an extra factor of 1/C**2.
    """

    def __init__(self, segmentation_loss, modality_channels,
                 reconstruction_weight=0.5, sdkt_weight=2.0):
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.modality_channels = tuple(modality_channels)
        self.reconstruction_weight = reconstruction_weight
        self.sdkt_weight = sdkt_weight

    def forward(self, output, target, reconstruction_target):
        layout = veloxseg_output_layout(len(output), len(self.modality_channels))
        start, end = layout['seg']
        weights = [2.0 ** -index for index in range(end - start)]
        # All objective reductions stay FP32; autocast is reserved for the network.
        with torch.autocast(device_type=target.device.type, enabled=False):
            segmentation = sum(
                weight * self.segmentation_loss(
                    pred.float(), target if pred.shape[2:] == target.shape[2:] else
                    F.interpolate(target.float(), size=pred.shape[2:], mode='nearest').to(target.dtype))
                for weight, pred in zip(weights, output[start:end])) / sum(weights)

        reconstructions = output[layout['reconstruction']].split(self.modality_channels, dim=1)
        targets = reconstruction_target.split(self.modality_channels, dim=1)
        reconstruction = sum(F.mse_loss(pred.float(), truth.float()) for pred, truth in zip(reconstructions, targets))
        reconstruction = reconstruction / len(self.modality_channels)

        student = output[layout['decoder_gram']].float()
        sdkt = sum((student - output[index].float()).square().sum(dim=(-2, -1)).mean()
                   for index in layout['teacher_grams'])
        return (segmentation + self.reconstruction_weight * reconstruction
                + self.sdkt_weight * sdkt)
