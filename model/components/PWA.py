import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.layers import DropPath
from typing import Sequence
from .attention_utils import FFN, LayerNorm, PositionalEmbedding
from math import ceil, prod

# Memory-efficient SDPA raises "CUDA error: invalid configuration argument" above
# 65535 batch entries (RTX 3090, PyTorch 2.6: 65535 passes, 65536 fails for 1/2/4 heads).
SDPA_MAX_BATCH = 65535

class Paired_Windows_Attention(nn.Module):

    def __init__(self,
                 input_size: Sequence[int],
                 in_channels: int,
                 min_big_window_size: Sequence[int] = [3, 3, 3], 
                 min_small_window_size: Sequence[int] = [1, 1, 1],
                 scale_factor: int = 2,
                 num_heads: int = 1,
                 min_dim_head: int = 4,
                 dropout: float = 0.1,
                 dim: str = 3,
                 use_pos_embed: bool = True,
                ):
        super(Paired_Windows_Attention, self).__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        
        self.channels_qk = in_channels
        self.channels_v = in_channels
        
        self.min_big_window_size = min_big_window_size
        self.min_small_window_size = min_small_window_size
        self.scale_factor = scale_factor
        self.num_heads = num_heads
        self.min_dim_head = min_dim_head
        self.dim = dim
        self.use_pos_embed = use_pos_embed
        
        if self.num_heads > 0:
            self.big_window_size, self.small_window_size = self.get_window_sizes()
            self.n_hwd = [self.min_big_window_size[i] // self.min_small_window_size[i] for i in range(self.dim)]

            self.num_bswin = len(self.big_window_size)
            self.num_heads_bswin = num_heads * self.num_bswin
            self.mid_channels_perhead = self.in_channels // self.num_heads_bswin
            
            if self.use_pos_embed:
                self.position_embedding = PositionalEmbedding(dim=self.dim, num_heads=self.num_heads, window_size=self.n_hwd)

            self.dropout_weight = nn.Dropout(dropout)
    
    def get_window_sizes(self):
        """Expand the paired windows per axis until every axis covers the input.

        Axis a tiles the input with ratio 2**q_a. Scale k doubles each axis at
        most q_a times, so every scale keeps the token grid big/small of the
        smallest window and the last scale is the whole input.
        """
        if self.scale_factor != 2:
            raise ValueError('PWA requires a power-of-two ratio to global coverage')
        exponents = []
        for n, big, small in zip(self.input_size, self.min_big_window_size, self.min_small_window_size):
            ratio, remainder = divmod(int(n), int(big))
            if remainder or ratio < 1 or ratio & (ratio - 1):
                raise ValueError('PWA windows must tile every axis with a power-of-two ratio')
            if big % small:
                raise ValueError('PWA pooling windows must divide the attention windows')
            exponents.append(ratio.bit_length() - 1)

        bw_sizes = []
        sw_sizes = []
        for scale in range(max(exponents) + 1):
            growth = [2 ** min(scale, q) for q in exponents]
            bw_sizes.append([big * g for big, g in zip(self.min_big_window_size, growth)])
            sw_sizes.append([small * g for small, g in zip(self.min_small_window_size, growth)])

        channels_need = len(bw_sizes) * self.num_heads * self.min_dim_head
        channels_qk = channels_need
        channels_v = ceil(self.channels_v / channels_need) * channels_need
        
        self.channels_qk = channels_qk
        self.channels_v = channels_v
        
        return bw_sizes, sw_sizes

    def attention_operation(self, query, key, value):
        # query/key/value: (windows, head, tokens, c). Windows of every scale are
        # independent SDPA batch entries.
        bias = None
        if self.use_pos_embed:
            spatial_tokens = self.position_embedding.relative_position_index.shape[0]
            modalities = query.shape[-2] // spatial_tokens
            bias = self.position_embedding.get_relative_position_bias(l=spatial_tokens)
            bias = bias.repeat(1, modalities, modalities).to(query.dtype)
        # PyTorch's memory-efficient kernel needs head dims divisible by 8. Zero
        # channels add nothing to q.k and are sliced off v; scale keeps 1/sqrt(c).
        channels, scale = value.shape[-1], query.shape[-1] ** -0.5
        query, key, value = (F.pad(t, (0, -t.shape[-1] % 8)) if t.shape[-1] % 8 else t
                             for t in (query, key, value))
        # The kernel's CUDA launch fails beyond SDPA_MAX_BATCH batch entries
        # (windows x samples), whatever the head count, so large batches attend
        # in chunks.
        attention = [F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout_weight.p if self.training else 0.0, scale=scale)
            for q, k, v in zip(*(t.split(SDPA_MAX_BATCH) for t in (query, key, value)))]
        attention = attention[0] if len(attention) == 1 else torch.cat(attention)
        return attention[..., :channels]

    def window_gathering(self, x):
        """Pool every scale on the full grid, then partition it into windows.

        x: (b, m, bswin*head*c, *spatial) -> (sum_i b*N_i, head, m*l, c), scale-major.
        Pooling before partitioning equals pooling inside each big window because
        every big window is a multiple of its pooling window and windows tile x.
        """
        b, m, _, *spatial = x.shape
        pool = F.max_pool3d if self.dim == 3 else F.max_pool2d
        axes = range(self.dim)
        windows = []
        for xi, big, small in zip(x.chunk(self.num_bswin, dim=2), self.big_window_size, self.small_window_size):
            if prod(small) > 1:
                xi = pool(xi.flatten(0, 1), kernel_size=small, stride=small).unflatten(0, (b, m))
            counts = [n // w for n, w in zip(spatial, big)]
            xi = xi.reshape(b, m, self.num_heads, -1, *[v for pair in zip(counts, self.n_hwd) for v in pair])
            # (b, m, head, c, N_1, n_1, ...) -> (b, N_1.., head, m, n_1.., c)
            xi = xi.permute(0, *[4 + 2 * a for a in axes], 2, 1, *[5 + 2 * a for a in axes], 3)
            # A single-window scale reshapes to a view with channels ahead of tokens in
            # memory; compiled cat kept that layout and memory-efficient SDPA rejected a
            # last dimension that is not contiguous.
            windows.append(xi.reshape(b * prod(counts), self.num_heads, m * prod(self.n_hwd), -1).contiguous())
        return torch.cat(windows)
        
    def window_scattering(self, out, b, m, spatial):
        """Inverse of window_gathering; every window is upsampled on its own.

        out: (sum_i b*N_i, head, m*l, c) -> (b, m, bswin*head*c, *spatial)
        """
        mode = 'trilinear' if self.dim == 3 else 'bilinear'
        d = self.dim
        axes = range(d)
        counts = [[n // w for n, w in zip(spatial, big)] for big in self.big_window_size]
        scales = []
        for oi, count, small in zip(out.split([b * prod(c) for c in counts]), counts, self.small_window_size):
            # (b, N.., head, m, n.., c) -> (b*m*N, head*c, n..)
            oi = oi.reshape(b, *count, self.num_heads, m, *self.n_hwd, -1)
            oi = oi.permute(0, d + 2, *[1 + a for a in axes], d + 1, 2 * d + 3, *[d + 3 + a for a in axes])
            oi = oi.reshape(b * m * prod(count), -1, *self.n_hwd)
            if prod(small) > 1:
                oi = F.interpolate(oi, scale_factor=small, mode=mode, align_corners=True)
            # (b, m, N.., head*c, w..) -> (b, m, head*c, *spatial)
            oi = oi.reshape(b, m, *count, *oi.shape[1:])
            oi = oi.permute(0, 1, d + 2, *[v for a in axes for v in (2 + a, d + 3 + a)])
            scales.append(oi.reshape(b, m, -1, *spatial))
        return torch.cat(scales, dim=2)

    def forward(self, query, key, value):
        if self.num_heads == 0:
            return query
        # q,k,v: (b, bswin*head*c, *spatial)
        q, k, v = (self.window_gathering(t.unsqueeze(1)) for t in (query, key, value))
        attn = self.attention_operation(q, k, v)
        return self.window_scattering(attn, query.shape[0], 1, query.shape[2:]).squeeze(1)
    
class MultiModal_Paired_Windows_Attention(Paired_Windows_Attention):

    def __init__(self,
                input_size: Sequence[int],
                in_channels: Sequence[int],
                min_big_window_size: Sequence[int] = [3, 3, 3],
                min_small_window_size: Sequence[int] = [1, 1, 1],
                scale_factor: int = 2,
                num_heads: int = 1,
                min_dim_head: int = 4,
                qkv_bias: bool = True,
                attn_drop: float = 0.1,
                proj_drop: float = 0.1,
                norm_layer = LayerNorm,
                dim: str = 3,
                use_pos_embed: bool = True
                ):
        self.mid_channels = max(in_channels)
        super(MultiModal_Paired_Windows_Attention, self).__init__(
                                                    input_size=input_size,
                                                    in_channels=self.mid_channels,
                                                    min_big_window_size=min_big_window_size,
                                                    min_small_window_size=min_small_window_size,
                                                    scale_factor=scale_factor,
                                                    num_heads=num_heads,
                                                    min_dim_head=min_dim_head,
                                                    dropout=attn_drop, dim=dim, 
                                                    use_pos_embed=use_pos_embed)
        if self.num_heads > 0:
            
            self.in_channels = in_channels
            self.num_modalities = len(in_channels)

            if dim == 3:
                conv = nn.Conv3d
            else:
                conv = nn.Conv2d
            
            input_norms = []
            qkv_proj = []
            mix_channels = []
            dropout_attns = []
            for m in range(self.num_modalities):
                    
                input_norms.append(norm_layer(self.in_channels[m], data_format='channels_first', dim=self.dim))
                qkv_proj.append(
                    nn.ModuleList(
                        [conv(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                        conv(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                        conv(self.in_channels[m], self.channels_v, kernel_size=1, bias=qkv_bias)]
                    )
                )
                mix_channels.append(conv(self.channels_v, self.in_channels[m], kernel_size=1))
                dropout_attns.append(nn.Dropout(proj_drop))
            
            self.input_norms = nn.ModuleList(input_norms)
            self.qkv_proj = nn.ModuleList(qkv_proj)
            self.mix_channels = nn.ModuleList(mix_channels)
            self.dropout_attns = nn.ModuleList(dropout_attns)

    def forward(self, inputs):
        
        # Residual branch only: Paired_Windows_TransformerBlock adds the skip once.
        if self.num_heads == 0:
            return [torch.zeros_like(x) for x in inputs]

        # inputs: List[Tensor], (b, c, *spatial)
        assert len(inputs) == self.num_modalities, f"The number of modalities should be {self.num_modalities}, but got {len(inputs)}"
        normed = [norm(x) for norm, x in zip(self.input_norms, inputs)]
        # q, k, v: (windows, head, m*l, c); modalities share every window
        q, k, v = (self.window_gathering(torch.stack([proj[j](x) for proj, x in zip(self.qkv_proj, normed)], dim=1))
                   for j in range(3))
        attn = self.attention_operation(q, k, v)
        # attn: (b, m, bswin*head*c, *spatial)
        attn = self.window_scattering(attn, inputs[0].shape[0], self.num_modalities, inputs[0].shape[2:])
        return [drop(mix(attn[:, m])) for m, (mix, drop) in enumerate(zip(self.mix_channels, self.dropout_attns))]
    

class Paired_Windows_TransformerBlock(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        min_big_window_size: Sequence[int] = [3, 3, 3],
        min_small_window_size: Sequence[int] = [1, 1, 1],
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0.0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        qkv_bias: bool = True,
        dim: str = 3,
    ) -> None:

        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)

        self.attn = MultiModal_Paired_Windows_Attention(
                    input_size              = input_size,
                    in_channels             = in_channels,
                    min_big_window_size     = min_big_window_size,
                    min_small_window_size   = min_small_window_size,
                    scale_factor            = scale_factor,
                    num_heads               = num_heads,
                    min_dim_head            = min_dim_head,
                    qkv_bias                = qkv_bias,
                    attn_drop               = attn_drop,
                    proj_drop               = proj_drop,
                    norm_layer              = norm_layer,
                    dim                     = dim,
                    use_pos_embed           = True
                )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.ffns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for m in range(self.num_modalities):
            self.ffns.append(FFN(in_channels[m], expansion_ratio=ffn_expansion_ratio, 
                                 dropout_rate=proj_drop, act=act_layer, dim = dim))
            self.norms.append(norm_layer(in_channels[m], data_format='channels_first', dim = dim))
    
    def forward(self, xs: Sequence[torch.Tensor]):
        
        attns = self.attn(xs)
        attns = [xs[m] + self.drop_path(attns[m]) for m in range(self.num_modalities)]
        attns = [attns[m] + self.drop_path(self.ffns[m](self.norms[m](attns[m]))) for m in range(self.num_modalities)]
        
        return attns
    
    


class Transformer_BasicLayer(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        depth: int = 2,
        min_big_window_size: Sequence[int] = [3, 3, 3],
        min_small_window_size: Sequence[int] = [1, 1, 1],
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: int = 4,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        qkv_bias: bool = True,
        dim: str = 3
    ):

        super().__init__()
        
        self.num_modalities = len(in_channels)
        self.blocks = nn.ModuleList(
            [   
                Paired_Windows_TransformerBlock(
                    input_size              = input_size,
                    in_channels             = in_channels,
                    min_big_window_size     = min_big_window_size,
                    min_small_window_size   = min_small_window_size,
                    scale_factor            = scale_factor,
                    num_heads               = num_heads,
                    min_dim_head            = min_dim_head,
                    attn_drop               = attn_drop,
                    proj_drop               = proj_drop,
                    drop_path               = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    ffn_expansion_ratio     = ffn_expansion_ratio,
                    act_layer               = act_layer,
                    norm_layer              = norm_layer,
                    qkv_bias                = qkv_bias,
                    dim                     = dim,
                )
                for i in range(depth)
            ]
        )
    def forward(self, xs):
        for block in self.blocks:
            xs = block(xs)
        return xs


class Cross_Channel_Attention(nn.Module):

    def __init__(self, ch1: Sequence[int], ch2: int, channel_reduction: int = 4, spatial_dim: int = 3,
                 output_both: bool = False):
    
        super(Cross_Channel_Attention, self).__init__()
        
        self.chs1 = ch1
        self.ch2 = ch2
        self.spatial_dim = spatial_dim
        self.output_both = output_both
        
        if spatial_dim == 3:
            avp = nn.AdaptiveAvgPool3d
            conv = nn.Conv3d
        elif spatial_dim == 2:
            avp = nn.AdaptiveAvgPool2d
            conv = nn.Conv2d
        
        self.ch1 = sum(ch1)
        self.squeeze_extract_1 = nn.Sequential(
            avp(1),
            conv(self.ch1, self.ch1 // channel_reduction, kernel_size=1),
            nn.GELU(),
            conv(self.ch1 // channel_reduction, self.ch1, kernel_size=1),
            nn.Flatten(2),
        )
        self.squeeze_extract_2 = nn.Sequential(
            avp(1),
            conv(ch2, ch2 // channel_reduction, kernel_size=1),
            nn.GELU(),
            conv(ch2 // channel_reduction, ch2, kernel_size=1),
            nn.Flatten(2),
        )
        
    
    def forward(self, x1, x2) -> Sequence[torch.Tensor]:
        
        # qkv: (b, c, 1)
        x1 = torch.cat(x1, dim=1)
        qkv_attn = self.squeeze_extract_1(x1)
        qkv_conv = self.squeeze_extract_2(x2)
        
        scores = torch.einsum("b m d, b n d -> b m n", qkv_attn, qkv_conv)
        weight_1_to_2 = F.softmax(scores, dim=1) / self.ch1 ** 0.5
        if self.output_both:
            weight_2_to_1 = F.softmax(scores, dim=2) / self.ch2 ** 0.5
        
        if self.spatial_dim == 3:            
            x2_ = torch.einsum("b m n, b m h w d -> b n h w d", weight_1_to_2, x1) + x2
            
            if self.output_both:
                x1_ = torch.einsum("b m n, b n h w d -> b m h w d", weight_2_to_1, x2) + x1
        
                xs = []
                c = 0
                for c1 in self.chs1:
                    xs.append(x1_[:, c:c+c1])
                    c += c1
                return xs, x2_
            return x2_
                
        elif self.spatial_dim == 2:
            
            x2_ = torch.einsum("b m n, b m h w -> b n h w", weight_1_to_2, x1) + x2
            
            if self.output_both:
                x1_ = torch.einsum("b m n, b n h w -> b m h w", weight_2_to_1, x2) + x1
        
                xs = []
                c = 0
                for c1 in self.chs1:
                    xs.append(x1_[:, c:c+c1])
                    c += c1
                return xs, x2_
            return x2_

            

  