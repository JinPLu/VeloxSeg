import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from einops import rearrange
from monai.networks.layers import trunc_normal_
from monai.networks.layers import get_act_layer
from .common_function import get_conv


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, 
                 data_format: str = "channels_last", dim: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
        self.dim = dim
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.dim == 2:
                weight = self.weight[:, None, None]
                bias = self.bias[:, None, None]
            elif self.dim == 3:
                weight = self.weight[:, None, None, None]
                bias = self.bias[:, None, None, None]
            x = weight * x + bias
            return x

class FFN(nn.Module):
    
    def __init__(
        self, in_channels: int, groups: int = 1, expansion_ratio: int = 4, 
        dropout_rate: float = 0.0, act: str = "GELU", dim: int = 3,):

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        

        self.linear1 = get_conv(dim)(in_channels, in_channels * expansion_ratio, 1, 1, 0, groups=groups)
        self.linear2 = get_conv(dim)(in_channels * expansion_ratio, in_channels, 1, 1, 0, groups=groups)
        self.fn = get_act_layer(act)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = self.drop1


    def forward(self, x):

        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: Sequence[int]):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Positional Embedding
        if self.dim == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing = 'ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif self.dim == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing = 'ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def get_relative_position_bias(self, l):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:l, :l].reshape(-1)
        ]
        relative_position_bias = rearrange(relative_position_bias, '(l1 l2) head -> head l1 l2', l1=l, l2=l)
        return relative_position_bias

class PatchMerging(nn.Module):
    """Gather adjacent voxels using the planned per-axis stride, then project."""
    def __init__(self, in_ch, out_ch, stride, norm_layer=LayerNorm, dim=3):
        super().__init__()
        from math import prod
        self.stride = tuple(stride)
        channels = in_ch * prod(stride)
        self.norm = norm_layer(channels, data_format='channels_first', dim=dim)
        self.reduction = get_conv(dim)(channels, out_ch, 1, bias=False)

    def forward(self, x):
        from itertools import product
        sampled = [x[(slice(None), slice(None), *[slice(offset, None, step)
                    for offset, step in zip(offsets, self.stride)])]
                   for offsets in product(*(range(step) for step in self.stride))]
        return self.reduction(self.norm(torch.cat(sampled, dim=1)))
