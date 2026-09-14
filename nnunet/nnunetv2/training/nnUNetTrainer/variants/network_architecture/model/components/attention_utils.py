import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union
from einops import rearrange
from monai.networks.layers import trunc_normal_
from monai.networks.layers import get_act_layer
from monai.utils import ensure_tuple_rep
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.model.components.common_function import get_conv, get_norm

def channel_shuffle(x, groups, spatial_dim=3):

    if spatial_dim == 2:
        x = rearrange(x, 'b (g c) h w -> b (c g) h w', g=groups)
    elif spatial_dim == 3:
        x = rearrange(x, 'b (g c) d h w -> b (c g) d h w', g=groups)

    return x


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

class OverlapPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: Union[Sequence[int], int] = 2,
        in_channels: int = 1,
        embed_dim: int = 16,
        groups: int = 1,
        use_norm: bool = True,
        norm_type: str = "GN",
        spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_norm = use_norm

        self.proj = get_conv(spatial_dims)(
            in_channels=in_channels, out_channels=embed_dim, kernel_size=[2*p-1 for p in patch_size],
            stride=patch_size, padding=[p-1 for p in patch_size], groups=groups)
        if use_norm:
            if norm_type in ["BN", 'LN', 'IN']:
                self.norm = get_norm(norm_type, spatial_dims)(embed_dim)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_groups=groups, num_channels=embed_dim)


    def forward(self, x):
        x = self.proj(x)
        if self.use_norm:
            x = self.norm(x)
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


class FFN2(nn.Module):

    def __init__(self, in_channels: int, groups_size: int = 1, expansion_ratio: int = 4,
                dropout_rate: float = 0.0, act: str = "GELU", dim: int = 3,):

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")


        groups = in_channels // groups_size
        mid_channels = in_channels * expansion_ratio

        self.linear1 = get_conv(dim)(in_channels, mid_channels, 1, 1, 0, groups=groups)
        self.dwconv = get_conv(dim)(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels)
        self.fn = get_act_layer(act)
        self.linear2 = get_conv(dim)(mid_channels, in_channels, 1, 1, 0, groups=groups)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = self.drop1


    def forward(self, x):

        x = self.linear1(x)
        x = self.fn(self.dwconv(x))
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
    def __init__(self, in_ch, out_ch, stride, norm_layer=LayerNorm, dim=3):
        super().__init__()
        from math import prod
        self.stride = tuple(stride)
        self.mid_ch = in_ch * prod(stride)
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.reduction = conv(self.mid_ch, out_ch, 1, 1, 0, bias=False)
        self.norm = norm_layer(self.mid_ch, data_format="channels_first", dim=dim)

    def forward(self, x):
        from itertools import product
        xs = [x[(slice(None), slice(None), *[slice(o, None, s) for o, s in zip(offset, self.stride)])]
              for offset in product(*(range(s) for s in self.stride))]
        return self.reduction(self.norm(torch.cat(xs, dim=1)))


class PatchMerging_v2(nn.Module):

    def __init__(self, in_ch: int, num_modalities:int=1, dim: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.num_modalities = num_modalities
        self.dim = dim

        if dim == 2:
            self.mid_ch = self.in_ch * 4
        elif dim == 3:
            self.mid_ch = self.in_ch * 8

        self.reduction = get_conv(dim)(self.mid_ch, 2 * self.in_ch, 1, 1, 0, bias=False, groups=num_modalities)
        self.norm = get_norm("GN", dim)(num_groups=num_modalities, num_channels=self.mid_ch)

    def faeture_sample(self, x):
        if self.dim == 2:
            x = rearrange(x, 'b (m c) (h r1) (w r2) -> b (m r1 r2 c) h w', m=self.num_modalities, r1=2, r2=2)
        elif self.dim == 3:
            x = rearrange(x, 'b (m c) (h r1) (w r2) (d r3) -> b (m r1 r2 r3 c) h w d', m=self.num_modalities, r1=2, r2=2, r3=2)
        return x

    def forward(self, x):
        x = self.faeture_sample(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x
