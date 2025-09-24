import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from monai.networks.layers import DropPath
from typing import Sequence
from .attention_utils import OverlapPatchEmbed, FFN
from .common_function import get_conv, get_norm
from math import ceil

# 改动：
# 1）为了让window覆盖全图，对channels进行了调整，在Linear Proj.时拓展通道数量，使得channels能够被window整除
# 2）发现位置编码的错误，进行了修正

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
                ):
        super(Paired_Windows_Attention, self).__init__()

        self.input_size = input_size
        self.channels = in_channels
        self.group = group
        self.min_big_window_size = min_big_window_size
        self.min_small_window_size = min_small_window_size
        self.scale_factor = scale_factor
        self.num_heads = num_heads
        self.min_dim_head = min_dim_head
        self.dim = dim
        
        if self.num_heads > 0:
            self.big_window_size, self.small_window_size = self.get_window_sizes()
            self.n_hwd = [self.min_big_window_size[i] // self.min_small_window_size[i] for i in range(self.dim)]

            self.num_bswin = len(self.big_window_size)
            self.num_heads_bswin = num_heads * self.num_bswin
            self.mid_channels_perhead = self.channels // self.num_heads_bswin
            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d

            self.softmax = nn.Softmax(dim=-1)
            self.dropout_weight = nn.Dropout(dropout)
    
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
    #     channels = ceil(self.channels / channels_need) * channels_need                              # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!修改
    #     # channels = channels_need
        
    #     print(f"in_channels: {self.channels}, need_channels: {channels_need}, channels: {channels}")
    #     print(f"big_window_sizes: {bw_sizes}")
    #     print(f"small_window_sizes: {sw_sizes}")
        
    #     self.channels = channels
    #     return bw_sizes, sw_sizes
    
    def get_window_sizes(self):
        input_size = torch.tensor(self.input_size)
        min_big_window_size = torch.tensor(self.min_big_window_size)
        min_small_window_size = torch.tensor(self.min_small_window_size)
        
        bw_sizes = []
        sw_sizes = []
        
        bw = min_big_window_size
        sw = min_small_window_size
        num_window_sizes = 0
        max_num_windows = self.channels // (self.num_heads * self.min_dim_head)
        
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
        
        print(f"in_channels: {self.channels}")
        print(f"big_window_sizes: {bw_sizes}")
        print(f"small_window_sizes: {sw_sizes}")
        
        return bw_sizes, sw_sizes

    def attention_operation(self, query, key, value):
        # query, key, value: (b, head, Ns, ml, c)
        ml, c = query.shape[-2:]
        l = ml // self.num_modalities

        scores = torch.einsum('bhNmc, bhNnc -> bhNmn', [query, key]) / (c ** 0.5)
        weights = self.dropout_weight(self.softmax(scores))
        
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
        outs = rearrange(outs, 'b head Ns (group nh nw nd) c -> b group head Ns c nh nw nd', nh=nh, nw=nw, nd=nd, group=self.group)

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, group, head, Ns, c, nh, nw, nd)
            Nh, Nw, Nd = Ns[i]
            N = Nh * Nw * Nd

            # out: (b, group, head, N, c, s_win_h, s_win_w, s_win_d)
            out = rearrange(outs[:, :, :, idx:idx+N], 'b group head N c nh nw nd -> b (group head N c) nh nw nd', nh=nh, nw=nw, nd=nd)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='trilinear', align_corners=True)
            
            # out: (b, 1, head, c, h, w, d)
            out = rearrange(out, 'b (group head Nh Nw Nd c) winh winw wind -> b group 1 head c (Nh winh) (Nw winw) (Nd wind)', 
                                                                            group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw, Nd=Nd)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=2)
        out = rearrange(out, 'b group bswin head c h w d -> b (group bswin head c) h w d')
        # out: (b, group*bswin*head*c, h, w, d)
        return out
    
    def window_scattering_2d(self, outs, Ns, n):
        nh, nw = n
        outs = rearrange(outs, 'b head Ns (group nh nw) c -> b group head Ns c nh nw', nh=nh, nw=nw, group=self.group)

        idx = 0
        outs_ = []
        for i in range(self.num_bswin):
            # outs: (b, group, head, Ns, c, nh, nw)
            Nh, Nw = Ns[i]
            N = Nh * Nw

            # out: (b, group, head, N, c, s_win_h, s_win_w)
            out = rearrange(outs[:, :, :, idx:idx+N], 'b group head N c nh nw -> b (group head N c) nh nw', nh=nh, nw=nw)
            out = F.interpolate(out, scale_factor=self.small_window_size[i], mode='bilinear', align_corners=True)
            
            # out: (b, group, 1, head, c, h, w, d)
            out = rearrange(out, 'b (group head Nh Nw c) winh winw -> b group 1 head c (Nh winh) (Nw winw)', 
                                                                            group=self.group, head=self.num_heads, Nh=Nh, Nw=Nw)
            outs_.append(out)
            idx += N
        out = torch.cat(outs_, dim=2)
        out = rearrange(out, 'b group bswin head c h w -> b (group bswin head c) h w')
        # out: (b, group*bswin*head*c, h, w)
        return out

    def forward(self, query, key, value):
        if self.num_heads == 0:
            return query

        # q, k, v: (b, head, Ns, group*l, c)
        q, Ns, n = self.window_gathering(query)
        k, _ , _ = self.window_gathering(key)
        v, _ , _ = self.window_gathering(value)

        # attn: (b, head, Ns, group*l, c)
        attn = self.attention_operation(q, k, v)

        # attn: (b, group*bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
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
                
                dim: int= 3,
                ):
        mid_channels = max(in_channels)
        num_modalities = len(in_channels)
        super(MultiModal_Paired_Windows_Attention, self).__init__(
                                                    input_size              = input_size,
                                                    in_channels             = mid_channels,
                                                    group                   = num_modalities,
                                                    min_big_window_size     = min_big_window_size,
                                                    min_small_window_size   = min_small_window_size,
                                                    scale_factor            = scale_factor,
                                                    num_heads               = num_heads,
                                                    min_dim_head            = min_dim_head,
                                                    dropout                 = attn_drop, 
                                                    dim                     = dim, )
        if self.num_heads > 0:
            in_channels = sum(in_channels)
            mid_channels = self.channels * num_modalities
            
            self.input_norms = nn.GroupNorm(num_groups=num_modalities, num_channels=in_channels)
            self.qkv_proj = nn.ModuleList(
                        [get_conv(dim)(in_channels, mid_channels, kernel_size=1, bias=qkv_bias, groups=num_modalities),
                        get_conv(dim)(in_channels, mid_channels, kernel_size=1, bias=qkv_bias, groups=num_modalities),
                        get_conv(dim)(in_channels, mid_channels, kernel_size=1, bias=qkv_bias, groups=num_modalities)]
                    )
            self.mix_channels = get_conv(dim)(mid_channels, in_channels, kernel_size=1, groups=num_modalities, bias=False)
            
            self.dropout_attns = nn.Dropout(proj_drop)
            
            self.window_gathering = self.window_gathering_3d if self.dim == 3 else self.window_gathering_2d
            self.window_scattering = self.window_scattering_3d if self.dim == 3 else self.window_scattering_2d
            self.num_modalities = num_modalities
    
    def forward(self, inputs):
        
        if self.num_heads == 0:
            return inputs

        # inputs: Tensor, (b, m*c, h, w, d) or (b, m*c, h, w)
        x = self.input_norms(inputs) 
        querys, keys, values = [proj(x) for proj in self.qkv_proj]

        # querys, keys, values: (b, head, Ns, ml, c)
        querys, Ns, n = self.window_gathering(querys)
        keys, _ , _ = self.window_gathering(keys)
        values, _ , _ = self.window_gathering(values)
        
        # attn: (b, head, Ns, ml, c)
        attn = self.attention_operation(querys, keys, values)
        
        # attn: (b, m*bswin*head*c, h, w, d) or (b, bswin*head*c, h, w)
        attn = self.window_scattering(attn, Ns, n)
        attn = self.dropout_attns(self.mix_channels(attn))
        return attn
    

class Paired_Windows_TransformerBlock(nn.Module):

    def __init__(
        self,
        input_size: Sequence[int],
        in_channels: Sequence[int],
        
        min_big_window_size: Sequence[int] = [3, 3, 3],
        min_small_window_size: Sequence[int] = [1, 1, 1],
        ffn_expansion_ratio: int = 4,
        scale_factor: int = 2,
        num_heads: int = 1,
        min_dim_head: int = 4,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        qkv_bias: bool = True,
        dim: int= 3,
    ) -> None:

        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.dim = dim

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
                )
        
        sum_channels = sum(in_channels)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm = get_norm("GN", dim)(num_groups=self.num_modalities, num_channels=sum_channels)
        
        self.ffn = FFN(sum_channels, groups=self.num_modalities, expansion_ratio=ffn_expansion_ratio, 
                       dropout_rate=proj_drop, act=act_layer, dim=dim)

    def forward(self, xs: torch.Tensor):
        
        attns = self.attn(xs)
        attns = xs + self.drop_path(attns)

        attns = attns + self.drop_path(self.ffn(self.norm(attns)))
        
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
        dim: int = 3
    ):

        super().__init__()
        
        self.num_modalities = len(in_channels)
        self.do_downsample = do_downsample
        
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
        
        self.norm = get_norm("GN")(self.num_modalities, sum(in_channels))

        if self.do_downsample:
            # self.downs = PatchMerging_v2(in_ch=sum(in_channels), num_modalities=self.num_modalities, dim=dim)
            self.downs = OverlapPatchEmbed(in_channels=sum(in_channels), out_channels=sum(in_channels)*2, 
                                            patch_size=2, num_modalities=self.num_modalities, dim=dim)
    
    def attn_forward(self, xs: torch.Tensor) -> torch.Tensor:

        for blk in self.blocks:
            xs = blk(xs)
        xs = self.norm(xs)
        return xs
    
    def down_forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.do_downsample:
            return self.downs(xs)
        else:
            return None
    
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.attn_forward(xs)
        down = self.down_forward(xs)
        return xs, down

  