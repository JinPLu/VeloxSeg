import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from monai.networks.layers import DropPath
from monai.networks.blocks import PatchEmbed
from typing import Sequence
from .attention_utils import FFN, LayerNorm, PositionalEmbedding, PatchMerging


# Parameter Explanation:
# bn = b * Nh * Nw * Nd
# l = winh * winw * wind
# ln = l // dr
# hd = head // dr
# 动机
# 1）将全局注意力改为多头->多窗的窗内注意力的叠加，避免全图attention的超高计算量
# 2）利用大小配对窗口，将窗内注意力计算复杂度降低为固定大小，降低计算复杂度的同时实现了高效的并行
# 3）固定大小窗口的比例，建模完整的局部信息和全局信息

class Paired_Windows_Attention(nn.Module):

    def __init__(self,
                 input_size: list,
                 in_channels: int,
                 min_big_window_size: list = [3, 3, 3], 
                 min_small_window_size: list = [1, 1, 1],
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
            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d

            self.softmax = nn.Softmax(dim=-1)
            self.dropout_weight = nn.Dropout(dropout)
    
    def get_window_sizes(self):
        input_size = torch.tensor(self.input_size)
        min_big_window_size = torch.tensor(self.min_big_window_size)
        min_small_window_size = torch.tensor(self.min_small_window_size)
        
        bw_sizes = []
        sw_sizes = []
        
        bw = min_big_window_size
        sw = min_small_window_size
        num_window_sizes = 0
        max_num_windows = self.in_channels // (self.num_heads * self.min_dim_head)
        
        for _ in range(max_num_windows):
            bw_sizes.append(bw.tolist())
            sw_sizes.append(sw.tolist())
            
            if max_num_windows % len(bw_sizes) == 0:
                num_window_sizes = len(bw_sizes)
            
            bw = bw * self.scale_factor
            sw = sw * self.scale_factor
            
            if (bw > input_size).any():
                break
        
        bw_sizes = bw_sizes[:num_window_sizes]
        sw_sizes = sw_sizes[:num_window_sizes]
        
        # print(f"max_num_windows: {max_num_windows}")
        # print(f"big_window_sizes: {bw_sizes}")
        # print(f"small_window_sizes: {sw_sizes}")
        
        return bw_sizes, sw_sizes

    def attention_operation(self, query, key, value):
        # query, key, value: (b, head, Ns, l, c)
        l, c = query.shape[-2:]

        scores = torch.einsum('bhNmc, bhNnc -> bhNmn', [query, key])
        weights = self.softmax(scores / (c ** 0.5))

        if self.use_pos_embed:
            relative_position_bias = self.position_embedding.get_relative_position_bias(l = l)
            relative_position_bias = rearrange(relative_position_bias, 'head l1 l2 -> 1 head 1 l1 l2')
            weights = weights + relative_position_bias
            
        weights = self.dropout_weight(weights)
        
        attention = torch.einsum('bhNmn, bhNnc -> bhNmc', [weights, value])
        # attention: (b, head, Ns, l, c)
        return attention


    def window_gathering_3d(self, x):
        # x: (b, c, h, w, d)
        b, _, h, w, d = x.size()
        # x: (b, bswin, head, c, h, w, d)
        
        x = rearrange(x, 'b (bswin head c) h w d -> b bswin head c h w d', bswin=self.num_bswin, head=self.num_heads)

        n = 0
        Ns = []
        xs = []
        for i in range(self.num_bswin):
            # x: (b, head, c, h, w, d)
            b_win_h, b_win_w, b_win_d = self.big_window_size[i]
            s_win_h, s_win_w, s_win_d = self.small_window_size[i]

            Nh, Nw, Nd = h // b_win_h, w // b_win_h, d // b_win_d
            nh, nw, nd = b_win_h // s_win_h, b_win_w // s_win_w, b_win_d // s_win_d
            
            # xi: (bhN, c, nh, nw, nd)            
            xi = rearrange(x[:, i], 'b head c (Nh winh) (Nw winw) (Nd wind) -> b (head Nh Nw Nd c) winh winw wind', 
                                                                    winh=b_win_h, winw=b_win_w, wind=b_win_d)

            xi = F.max_pool3d(xi, kernel_size=self.small_window_size[i], stride=self.small_window_size[i])
            
            xi = rearrange(xi, 'b (head Nh Nw Nd c) nh nw nd -> b head (Nh Nw Nd) (nh nw nd) c', 
                                                        head=self.num_heads, Nh=Nh, Nw=Nw, Nd=Nd)

            xs.append(xi)
            Ns.append([Nh, Nw, Nd])

            assert n == 0 or (n[0] == nh and n[1] == nw and n[2] == nd), "Please check that the number of small windows in all big windows is equal to ensure parallel calculation of attention."
            n = [nh, nw, nd]
        
        # x: (b, head, Ns, l, c)
        x = torch.cat(xs, dim=2)
        return x, Ns, n
    
    def window_gathering_2d(self, x):
        # x: (b, c, h, w)
        b, _, h, w = x.size()
        # x: (b, bswin, head, c, h, w)
        x = rearrange(x, 'b (bswin head c) h w -> b bswin head c h w', bswin=self.num_bswin, head=self.num_heads)
        
        n = 0
        Ns = []
        xs = []
        for i in range(self.num_bswin):
            # x: (b, head, c, h, w)
            b_win_h, b_win_w = self.big_window_size[i]
            s_win_h, s_win_w = self.small_window_size[i]

            Nh, Nw = h // b_win_h, w // b_win_h
            nh, nw = b_win_h // s_win_h, b_win_w // s_win_w
            
            # xi: (bhN, c, nh, nw)
            xi = rearrange(x[:, i], 'b head c (Nh winh) (Nw winw) -> b (head Nh Nw c) winh winw', 
                                                                    winh=b_win_h, winw=b_win_w)
            xi = F.max_pool2d(xi, kernel_size=self.small_window_size[i], stride=self.small_window_size[i])
            
            xi = rearrange(xi, 'b (head Nh Nw c) nh nw -> b head (Nh Nw) (nh nw) c', 
                                                        head=self.num_heads, Nh=Nh, Nw=Nw)

            xs.append(xi)
            Ns.append([Nh, Nw])

            assert n == 0 or (n[0] == nh and n[1] == nw), "Please check that the number of small windows in all big windows is equal to ensure parallel calculation of attention."
            n = [nh, nw]
        
        # x: (b, head, Ns, l, c)
        x = torch.cat(xs, dim=2)
        return x, Ns, n
    
    def window_scattering_3d(self, outs, Ns, n):
        nh, nw, nd = n
        outs = rearrange(outs, 'b head Ns (nh nw nd) c -> b head Ns c nh nw nd', nh=nh, nw=nw, nd=nd)

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, head, Ns, c, nh, nw, nd)
            Nh, Nw, Nd = Ns[i]
            N = Nh * Nw * Nd

            # out: (b, head, N, c, s_win_h, s_win_w, s_win_d)
            out = rearrange(outs[:, :, idx:idx+N], 'b head N c nh nw nd -> b (head N c) nh nw nd', nh=nh, nw=nw, nd=nd)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='trilinear', align_corners=True)
            
            # out: (b, 1, head, c, h, w, d)
            out = rearrange(out, 'b (head Nh Nw Nd c) winh winw wind -> b 1 head c (Nh winh) (Nw winw) (Nd wind)', 
                                                                            head=self.num_heads, Nh=Nh, Nw=Nw, Nd=Nd)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=1)
        out = rearrange(out, 'b bswin head c h w d -> b (bswin head c) h w d')
        # out: (b, bswin*head*c, h, w, d)
        return out
    
    def window_scattering_2d(self, outs, Ns, n):
        nh, nw = n
        outs = rearrange(outs, 'b hNs (nh nw) c -> b hNs c nh nw', nh=nh, nw=nw)

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, head, Ns, c, nh, nw)
            Nh, Nw = Ns[i]
            N = Nh * Nw

            # out: (b, head, N, c, s_win_h, s_win_w)
            out = rearrange(outs[:, :, idx:idx+N], 'b head N c nh nw -> b (head N c) nh nw', nh=nh, nw=nw)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='bilinear', align_corners=True)
            
            # out: (b, 1, head, c, h, w, d)
            out = rearrange(out, 'b (head Nh Nw c) winh winw -> b 1 head c (Nh winh) (Nw winw)', 
                                                            head=self.num_heads, Nh=Nh, Nw=Nw)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=1)
        out = rearrange(out, 'b bswin head c h w -> b (bswin head c) h w')
        # out: (b, bswin*head*c, h, w)
        return out

    def forward(self, query, key, value):
        if self.num_heads == 0:
            return query
        
        # q,k,v: (b, bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
        input_size = query.size()

        # q, k, v: (b, head, Ns, l, c)
        q, Ns, n = self.window_gathering(query)
        k, _ , _ = self.window_gathering(key)
        v, _ , _ = self.window_gathering(value)

        # attn: (b, head, Ns, l, c)
        attn = self.attention_operation(q, k, v)

        # attn: (b, bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
        attn = self.window_scattering(attn, Ns, n)
        return attn
    
class MultiModal_Paired_Windows_Attention(Paired_Windows_Attention):

    def __init__(self,
                input_size: list,
                in_channels: list,
                min_big_window_size: list = [3, 3, 3],
                min_small_window_size: list = [1, 1, 1],
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
                        [conv(self.in_channels[m], self.mid_channels, kernel_size=1, bias=qkv_bias),
                        conv(self.in_channels[m], self.mid_channels, kernel_size=1, bias=qkv_bias),
                        conv(self.in_channels[m], self.mid_channels, kernel_size=1, bias=qkv_bias)]
                    )
                )
                mix_channels.append(conv(self.mid_channels, self.in_channels[m], kernel_size=1))
                dropout_attns.append(nn.Dropout(proj_drop))
            
            self.input_norms = nn.ModuleList(input_norms)
            self.qkv_proj = nn.ModuleList(qkv_proj)
            self.mix_channels = nn.ModuleList(mix_channels)
            self.dropout_attns = nn.ModuleList(dropout_attns)
            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d

    def attention_operation(self, query, key, value):
        # query, key, value: (b, head, Ns, ml, c)
        ml, c = query.shape[-2:]
        l = ml // self.num_modalities

        scores = torch.einsum('bhNmc, bhNnc -> bhNmn', [query, key])
        weights = self.softmax(scores / (c ** 0.5))

        if self.use_pos_embed:
            weights = rearrange(weights, 'b head Ns (m1 l1) (m2 l2) -> b head Ns (m1 m2) l1 l2', 
                                m1=self.num_modalities, m2=self.num_modalities)
            relative_position_bias = self.position_embedding.get_relative_position_bias(l = l)
            relative_position_bias = rearrange(relative_position_bias, 'head l1 l2 -> 1 head 1 1 l1 l2')
            weights = weights + relative_position_bias
            weights = rearrange(weights, 'b head Ns (m1 m2) l1 l2 -> b head Ns (m1 l1) (m2 l2)', 
                                m1=self.num_modalities, m2=self.num_modalities)
            
        weights = self.dropout_weight(weights)
        
        attention = torch.einsum('bhNmn, bhNnc -> bhNmc', [weights, value])
        # attention: (b, head, Ns, l, c)
        return attention
    
    def forward(self, inputs):
        
        if self.num_heads == 0:
            return inputs

        # inputs: List[Tensor], (b, c, h, w, d) or (b, c, h, w)
        # num_modalities: len(inputs)
        assert len(inputs) == self.num_modalities, f"The number of modalities should be {self.num_modalities}, but got {len(inputs)}"
        
        querys, keys, values = [], [], []
        for m in range(self.num_modalities):
            # q, k, v: (b, bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
            querys.append(self.qkv_proj[m][0](self.input_norms[m](inputs[m])))
            keys.append(self.qkv_proj[m][1](self.input_norms[m](inputs[m])))
            values.append(self.qkv_proj[m][2](self.input_norms[m](inputs[m])))
        
        q, k, v = None, None, None
        n, Ns = None, None
        l = None
        
        for m in range(self.num_modalities):
            # q, k, v: (b, head, Ns, l, c)
            q, Ns, n = self.window_gathering(querys[m])
            k, _ , _ = self.window_gathering(keys[m])
            v, _ , _ = self.window_gathering(values[m])
            # print(f'v.size(): {v.size()}')
            querys[m] = q
            keys[m]   = k
            values[m] = v
            if l is None:
                l = q.shape[-2]
            else:
                assert l == q.shape[-2], f"The seq length in all modalities should be equal, but got {l} and {q.shape[-2]}"

        # attn: (b, head, Ns, ml, c)
        # print([v.size() for v in values])
        querys = torch.cat(querys, dim=-2)
        keys   = torch.cat(keys,   dim=-2)
        values = torch.cat(values, dim=-2)

        # attn: (b, head, Ns, ml, c)
        attn = self.attention_operation(querys, keys, values)
        attn = rearrange(attn, 'b head Ns (m l) c -> b head Ns m l c', l=l)

        attns = []
        for m in range(self.num_modalities):
            # attn_m: (b, bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
            attn_m = self.window_scattering(attn[:, :, :, m], Ns, n)
            attn_m = inputs[m] + self.dropout_attns[m](self.mix_channels[m](attn_m))
            attns.append(attn_m)
        return attns
    

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
            self.ffns.append(FFN(in_channels[m], expansion_ratio=ffn_expansion_ratio, dropout_rate=proj_drop, act=act_layer, 
                        norm_layer = norm_layer, dim = dim))
            self.norms.append(norm_layer(in_channels[m], data_format='channels_first', dim = dim))
    
    def forward(self, xs: Sequence[torch.Tensor]):
        
        attns = self.attn(xs)
        attns = [xs[m] + self.drop_path(attns[m]) for m in range(self.num_modalities)]
        attns = [attns[m] + self.drop_path(self.ffns[m](self.norms[m](attns[m]))) for m in range(self.num_modalities)]
        
        return attns
    
    


class BasicLayer(nn.Module):

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
        do_downsample: bool = True,
        dim: str = 3
    ) -> None:

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
        self.downsample = None
        if do_downsample:
            self.downs = nn.ModuleList([PatchMerging(in_ch=in_channels[m], norm_layer=norm_layer, dim=dim)\
                                for m in range(self.num_modalities)])

    def forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        for blk in self.blocks:
            xs = blk(xs)
        if self.downs is not None:
            xs = [self.downs[m](xs[m]) for m in range(self.num_modalities)]
        return xs

class Transformer_Encoder(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        embed_dim: int = 24,
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
        patch_norm: bool = False,
        qkv_bias: bool = True,
        spatial_dims: str = 3
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)

        self.patch_size = 2
        self.patch_embeds = nn.ModuleList([PatchEmbed(
                                    patch_size      = self.patch_size,
                                    in_chans        = self.in_channels[m],
                                    embed_dim       = embed_dim,
                                    norm_layer      = norm_layer if patch_norm else None,
                                    spatial_dims    = spatial_dims,
                                    ) for m in range(self.num_modalities)])
        
        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.layers = nn.ModuleList()

        input_size = torch.tensor(input_size) // self.patch_size
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                input_size              = input_size.tolist(),
                in_channels             = [int(embed_dim * 2**i_layer)] * self.num_modalities,
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
                do_downsample           = True,
                dim                     = spatial_dims
            ))
            input_size = input_size // 2

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x
    
    def check_input(self, x):
        if not isinstance(x, list):
            channels = x.shape[1]
            assert channels == sum(self.in_channels), f"Input channels should be equal to the sum of in_channels, but got {channels} and {sum(self.in_channels)}"
            xs = []
            c = 0
            for i in range(len(self.in_channels)):
                xs.append(x[:, c:c+self.in_channels[i]])
                c += self.in_channels[i]
            x = xs
        return x

    def forward(self, xs, normalize=True):
        
        xs = self.check_input(xs)
        xs = [self.patch_embeds[m](xs[m]) for m in range(self.num_modalities)]
        
        xs = [self.pos_drop(x) for x in xs]
        outs = [[self.proj_out(x, normalize) for x in xs]]
        
        for layer in self.layers:
            xs = layer([x.contiguous() for x in xs])
            outs.append([[self.proj_out(x, normalize) for x in xs]])
        
        return outs
        
        
            
  