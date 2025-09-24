import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from monai.networks.layers import DropPath
from typing import Sequence
from .attention_utils import FFN, LayerNorm, PositionalEmbedding, PatchMerging
from .common_function import get_conv, get_norm
from math import ceil

# 动机
# 1）将全局注意力改为多头->多窗的窗内注意力的叠加，避免全图attention的超高计算量
# 2）利用大小配对窗口，将窗内注意力计算复杂度降低为固定大小，降低计算复杂度的同时实现了高效的并行
# 3）固定大小窗口的比例，建模完整的局部信息和全局信息

class Paired_Windows_Attention(nn.Module):

    def __init__(self,
                 input_size: Sequence[int],
                 in_channels: int,
                 group: int,
                 min_big_window_size: Sequence[int] = [3, 3, 3], 
                 min_small_window_size: Sequence[int] = [1, 1, 1],
                 scale_factor: int = 2,
                 num_heads: int = 1,
                 min_dim_head: int = 4,
                 dropout: float = 0.1,
                 dim: int= 3,
                 use_pos_embed: bool=True,
                ):
        super(Paired_Windows_Attention, self).__init__()

        self.input_size = input_size
        self.channels_qk = in_channels
        self.channels_v = in_channels
        
        self.group = group
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
        max_num_windows = self.channels_qk // (self.num_heads * self.min_dim_head)
        
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
        
        print(f"in_channels: {self.channels_qk}")
        print(f"max_num_windows: {max_num_windows}")
        print(f"big_window_sizes: {bw_sizes}")
        print(f"small_window_sizes: {sw_sizes}")
        
        return bw_sizes, sw_sizes

    # def get_window_sizes(self):
    #     input_size = torch.tensor(self.input_size)
    #     min_big_window_size = torch.tensor(self.min_big_window_size)
    #     min_small_window_size = torch.tensor(self.min_small_window_size)
        
    #     bw_sizes = []
    #     sw_sizes = []
        
    #     bw = min_big_window_size
    #     sw = min_small_window_size
        
    #     while (bw <= input_size).any():
    #         bw_sizes.append(bw.tolist())
    #         sw_sizes.append(sw.tolist())

    #         bw = bw * self.scale_factor
    #         sw = sw * self.scale_factor
            
    #     channels_need = len(bw_sizes) * self.num_heads * self.min_dim_head
    #     channels_qk = channels_need
    #     channels_v = ceil(self.channels_v / channels_need) * channels_need
        
    #     print(f"in_channels: {self.channels_qk}, channels_qk: {channels_qk}, channels_v: {channels_v}")
    #     print(f"big_window_sizes: {bw_sizes}")
    #     print(f"small_window_sizes: {sw_sizes}")
        
    #     self.channels_qk = channels_qk
    #     self.channels_v = channels_v
        
    #     return bw_sizes, sw_sizes

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
        
        x = rearrange(x, 'b (group bswin head c) h w d -> b group bswin head c h w d', bswin=self.num_bswin, head=self.num_heads,
                                                                                        group=self.group)   

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
            xi = rearrange(x[:, :, i], 'b group head c (Nh winh) (Nw winw) (Nd wind) -> b (group head Nh Nw Nd c) winh winw wind', 
                                                                    winh=b_win_h, winw=b_win_w, wind=b_win_d)

            xi = F.max_pool3d(xi, kernel_size=self.small_window_size[i], stride=self.small_window_size[i])
            
            xi = rearrange(xi, 'b (group head Nh Nw Nd c) nh nw nd -> b head (Nh Nw Nd) (group nh nw nd) c', 
                                                        group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw, Nd=Nd)
            
            
            

            xs.append(xi)
            Ns.append([Nh, Nw, Nd])

            assert n == 0 or (n[0] == nh and n[1] == nw and n[2] == nd), "Please check that the number of small windows in all big windows is equal to ensure parallel calculation of attention."
            n = [nh, nw, nd]
        
        # x: (b, head, Ns, g*l, c)
        x = torch.cat(xs, dim=2)
        return x, Ns, n
    
    def window_gathering_2d(self, x):
        # x: (b, c, h, w)
        b, _, h, w = x.size()
        # x: (b, bswin, head, c, h, w)
        
        x = rearrange(x, 'b (group bswin head c) h w -> b group bswin head c h w', bswin=self.num_bswin, head=self.num_heads,
                                                                                        group=self.group)   

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
            xi = rearrange(x[:, :, i], 'b group head c (Nh winh) (Nw winw) -> b (group head Nh Nw c) winh winw', 
                                                                    winh=b_win_h, winw=b_win_w)

            xi = F.max_pool2d(xi, kernel_size=self.small_window_size[i], stride=self.small_window_size[i])
            
            xi = rearrange(xi, 'b (group head Nh Nw c) nh nw -> b head (Nh Nw) (group nh nw) c', 
                                                        group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw)

            xs.append(xi)
            Ns.append([Nh, Nw])

            assert n == 0 or (n[0] == nh and n[1] == nw), "Please check that the number of small windows in all big windows is equal to ensure parallel calculation of attention."
            n = [nh, nw]
        
        # x: (b, head, Ns, g*l, c)
        x = torch.cat(xs, dim=2)
        return x, Ns, n
    
    def window_scattering_3d(self, outs, Ns, n):
        nh, nw, nd = n

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, head, Ns, (group nh nw nd), c)
            Nh, Nw, Nd = Ns[i]
            N = Nh * Nw * Nd

            # out: (b, group*head*N*c, s_win_h, s_win_w, s_win_d)
            out = rearrange(outs[:, :, idx:idx+N], 'b head N (group nh nw nd) c -> b (group head N c) nh nw nd', nh=nh, nw=nw, nd=nd)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='trilinear', align_corners=True)
            
            # out: (b, group, 1, head, c, h, w, d)
            out = rearrange(out, 'b (group head Nh Nw Nd c) winh winw wind -> b group 1 head c (Nh winh) (Nw winw) (Nd wind)', 
                                                                            group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw, Nd=Nd)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=2)
        out = rearrange(out, 'b group bswin head c h w d -> b group (bswin head c) h w d')
        # out: (b, group, bswin*head*c, h, w, d)
        return out
    
    def window_scattering_2d(self, outs, Ns, n):
        nh, nw = n

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, head, Ns, (group nh nw), c)
            Nh, Nw = Ns[i]
            N = Nh * Nw

            # out: (b, group*head*N*c, s_win_h, s_win_w)
            out = rearrange(outs[:, :, idx:idx+N], 'b head N (group nh nw) c -> b (group head N c) nh nw', nh=nh, nw=nw)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='bilinear', align_corners=True)
            
            # out: (b, group, 1, head, c, h, w)
            out = rearrange(out, 'b (group head Nh Nw c) winh winw -> b group 1 head c (Nh winh) (Nw winw)', 
                                                        group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=2)
        out = rearrange(out, 'b group bswin head c h w -> b group (bswin head c) h w')
        # out: (b, group, bswin*head*c, h, w)
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
                dim: str = 3,
                use_pos_embed: bool = True
                ):
        self.mid_channels = max(in_channels)
        self.num_modalities = len(in_channels)
        super(MultiModal_Paired_Windows_Attention, self).__init__(
                                                    input_size=input_size,
                                                    in_channels=self.mid_channels,
                                                    group=self.num_modalities,
                                                    min_big_window_size=min_big_window_size,
                                                    min_small_window_size=min_small_window_size,
                                                    scale_factor=scale_factor,
                                                    num_heads=num_heads,
                                                    min_dim_head=min_dim_head,
                                                    dropout=attn_drop, dim=dim, 
                                                    use_pos_embed=use_pos_embed)
        if self.num_heads > 0:
            self.in_channels = in_channels
            self.input_norms = nn.ModuleList([])
            self.qkv_proj = nn.ModuleList([])
            self.mix_channels = nn.ModuleList([])
            self.dropout_attns = nn.ModuleList([])
            for m in range(self.num_modalities):
                    
                self.input_norms.append(LayerNorm(self.in_channels[m], data_format='channels_first', dim=self.dim))
                self.qkv_proj.append(
                    nn.ModuleList(
                        [get_conv(dim)(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                        get_conv(dim)(self.in_channels[m], self.channels_qk, kernel_size=1, bias=qkv_bias),
                        get_conv(dim)(self.in_channels[m], self.channels_v, kernel_size=1, bias=qkv_bias)]
                    )
                )
                self.mix_channels.append(get_conv(dim)(self.channels_v, self.in_channels[m], kernel_size=1))
                self.dropout_attns.append(nn.Dropout(proj_drop))

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
        
        querys = torch.cat(querys, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        
        # querys, keys, values: (b, head, Ns, ml, c)
        querys, Ns, n = self.window_gathering(querys)
        keys, _ , _ = self.window_gathering(keys)
        values, _ , _ = self.window_gathering(values)

        # attn: (b, head, Ns, ml, c)
        attn = self.attention_operation(querys, keys, values)
        
        # attn: (b, m, bswin*head*c, h, w, d) or (b, m, bswin*head*c, h, w)
        attn = self.window_scattering(attn, Ns, n)

        attns = [self.dropout_attns[m](self.mix_channels[m](attn[:, m])) for m in range(self.num_modalities)]
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
                    dim                     = dim,
                    use_pos_embed           = True
                )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.ffns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for m in range(self.num_modalities):
            self.ffns.append(FFN(in_channels[m], expansion_ratio=ffn_expansion_ratio, 
                                 dropout_rate=proj_drop, act=act_layer, dim = dim))
            self.norms.append(LayerNorm(in_channels[m], data_format='channels_first', dim = dim))
    
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
        qkv_bias: bool = True,
        do_downsample: bool = True,
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
                    qkv_bias                = qkv_bias,
                    dim                     = dim,
                )
                for i in range(depth)
            ]
        )
        self.downs = None
        if do_downsample:
            self.downs = nn.ModuleList([PatchMerging(in_ch=in_channels[m], dim=dim) for m in range(self.num_modalities)])

    def attn_forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        for blk in self.blocks:
            xs = blk(xs)
        return xs
    
    def down_forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        down = None
        if self.downs is not None:
            down = [self.downs[m](xs[m]) for m in range(self.num_modalities)]
        return down
    
    def forward(self, xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        xs = self.attn_forward(xs)
        down = self.down_forward(xs)
        return xs, down


  