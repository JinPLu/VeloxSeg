import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.distributions as td
from torch.amp import autocast

# Third-party libraries
from mamba_ssm import Mamba
from einops import rearrange, repeat
import timm
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

# Python standard libraries
import time
import math
import copy
from functools import partial
from typing import Optional, Callable
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops



    
class TDFringeConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
    

        self.height_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(kernel_size//2, 0), 
                    groups=in_channels
                    )
        )
        self.width_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                    kernel_size=(1,kernel_size),
                    stride=1,
                    padding=(0,kernel_size//2),
                    groups=in_channels
                    )
        )
    def forward(self, input_features):

        height_features = self.height_conv(input_features) 
        width_features = self.width_conv(input_features)
        
        fused_features = height_features + width_features
        
        return fused_features
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", 
        d_conv=3,
        expand=0.5,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.conv2d = TDFringeConvolution(in_channels=self.d_inner,kernel_size=d_conv)

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_core_windows
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core_windows(self, x: torch.Tensor, layer=1):
        return self.forward_corev0(x)



    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W

        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        B, H, W, C = x.shape


        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        z = z.permute(0, 3, 1, 2)
        
        z = z.permute(0, 2, 3, 1).contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        y = self.forward_core(x, layer)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        layer: int = 1,
        **kwargs,
    ):
        super().__init__()
        factor = 2.0 
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state,d_conv=3, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer
        
    def forward(self, input: torch.Tensor):
        input_x = self.down(input)
        input_x = input_x + self.drop_path(self.self_attention(self.ln_1(input_x), self.layer))
        x = self.up(input_x) + input
        return x
class LayerNormBatchFirst(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        ## x.shape = (B, C, D, H, W) ##
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False, device_type="cuda")
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out
class Spatial3DAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size,
                                    padding=kernel_size//2,
                                    bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, features):
        # Calculate channel-wise statistics
        avg_pooled = torch.mean(features, dim=1, keepdim=True) 
        max_pooled, _ = torch.max(features, dim=1, keepdim=True)
        
        # Concatenate statistics and compute attention weights
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        attention_weights = self.spatial_conv(pooled_features)
        
        return self.activation(attention_weights)


class ConvNeXtConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_rate=4,stride=1):
        super(ConvNeXtConv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=in_channels,
        )
        self.ln = nn.LayerNorm(in_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0)
        self.conv2 = nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
class MultiHeadAxialAttention3D(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, num_heads=8, axis='D'):
        """
        Parameters
        ----------
        in_dim : int
            channel of input tensor
        q_k_dim : int 
            channel of Q, K vector
        num_heads : int
            number of attention heads
        axis : str
            attention axis, can be 'D', 'H', or 'W'
        """
        super(MultiHeadAxialAttention3D, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.num_heads = num_heads
        self.head_dim = q_k_dim // num_heads
        # print("q_k_dim:",q_k_dim)
        # print("head_dim:",self.head_dim)
        self.axis = axis
        D,H,W = patch_ini[0],patch_ini[1],patch_ini[2]
        
        assert q_k_dim % num_heads == 0, "q_k_dim must be divisible by num_heads"

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")
        
        nn.init.xavier_uniform_(self.pos_embed)
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, processed):
        B, C, D, H, W = x.size()
        
        Q = self.query_conv(x) + self.pos_embed
        K = self.key_conv(processed) + self.pos_embed
        V = self.value_conv(processed)

        Q = Q.view(B, self.num_heads, self.head_dim, D, H, W)
        K = K.view(B, self.num_heads, self.head_dim, D, H, W)
        V = V.view(B, self.num_heads, self.in_dim // self.num_heads, D, H, W)

        scale = math.sqrt(self.head_dim)

        if self.axis == 'D':
            Q = Q.permute(0, 1, 4, 5, 3, 2).contiguous()  # (B, num_heads, H, W, D, head_dim)
            Q = Q.view(B*self.num_heads*H*W, D, self.head_dim)
            
            K = K.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, num_heads, H, W, head_dim, D)
            K = K.view(B*self.num_heads*H*W, self.head_dim, D)
            
            V = V.permute(0, 1, 4, 5, 3, 2).contiguous()  # (B, num_heads, H, W, D, head_dim)
            V = V.view(B*self.num_heads*H*W, D, self.in_dim//self.num_heads)
            
            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, H, W, D, self.in_dim//self.num_heads)
            out = out.permute(0, 1, 5, 4, 2, 3).contiguous()
            
        elif self.axis == 'H':
            Q = Q.permute(0, 1, 3, 5, 4, 2).contiguous()  # (B, num_heads, D, W, H, head_dim)
            Q = Q.view(B*self.num_heads*D*W, H, self.head_dim)
            
            K = K.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, num_heads, D, W, head_dim, H)
            K = K.view(B*self.num_heads*D*W, self.head_dim, H)
            
            V = V.permute(0, 1, 3, 5, 4, 2).contiguous()  # (B, num_heads, D, W, H, head_dim)
            V = V.view(B*self.num_heads*D*W, H, self.in_dim//self.num_heads)
            
            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, D, W, H, self.in_dim//self.num_heads)
            out = out.permute(0, 1, 5, 2, 4, 3).contiguous()
            
        else:  # self.axis == 'W'
            Q = Q.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, num_heads, D, H, W, head_dim)
            Q = Q.view(B*self.num_heads*D*H, W, self.head_dim)
            
            K = K.permute(0, 1, 3, 4, 2, 5).contiguous()  # (B, num_heads, D, H, head_dim, W)
            K = K.view(B*self.num_heads*D*H, self.head_dim, W)
            
            V = V.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, num_heads, D, H, W, head_dim)
            V = V.view(B*self.num_heads*D*H, W, self.in_dim//self.num_heads)
            
            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, D, H, W, self.in_dim//self.num_heads)
            out = out.permute(0, 1, 5, 2, 3, 4).contiguous()
        
        # 合并多头
        out = out.view(B, C, D, H, W)
        
        # 残差连接
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1-gamma) * x
        return out

class AxialAttention3D(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini,axis='D'):
        """
        Parameters
        ----------
        in_dim : int
            channel of input tensor
        q_k_dim : int
            channel of Q, K vector
        axis : str
            attention axis, can be 'D', 'H', or 'W'
        """
        super(AxialAttention3D, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        D,H,W=patch_ini[0],patch_ini[1],patch_ini[2]


        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")
        
        nn.init.xavier_uniform_(self.pos_embed)
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x,processed):
        """
        Parameterse
        ----------
        x : Tensor
            5-D tensor, (batch, channels, depth, height, width)
        """
        B, C, D, H, W = x.size()
        
        Q = self.query_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        K = self.key_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        V = self.value_conv(processed)  # (B, in_dim, D, H, W)
        # Q = self.query_conv(x)  # (B, q_k_dim, D, H, W)
        # K = self.key_conv(x)   # (B, q_k_dim, D, H, W)
        # V = self.value_conv(x)  # (B, in_dim, D, H, W)
        scale = math.sqrt(self.q_k_dim)
        if self.axis == 'D':
            Q = Q.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, q_k_dim)
            Q = Q.view(B*H*W, D, self.q_k_dim)  # (B*H*W, D, q_k_dim)
            
            K = K.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, q_k_dim, D)
            K = K.view(B*H*W, self.q_k_dim, D)  # (B*H*W, q_k_dim, D)
            
            V = V.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, in_dim)
            V = V.view(B*H*W, D, self.in_dim)  # (B*H*W, D, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*H*W, D, D)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*H*W, D, in_dim)

            out = out.view(B, H, W, D, self.in_dim)
            out = out.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D, H, W)
            
        elif self.axis == 'H':
            Q = Q.permute(0, 2, 4, 3, 1).contiguous()  # (B, D, W, H, q_k_dim)
            Q = Q.view(B*D*W, H, self.q_k_dim)  # (B*D*W, H, q_k_dim)
            
            K = K.permute(0, 2, 4, 1, 3).contiguous()  # (B, D, W, q_k_dim, H)
            K = K.view(B*D*W, self.q_k_dim, H)  # (B*D*W, q_k_dim, H)
            
            V = V.permute(0, 2, 4, 3, 1).contiguous()  # (B, D, W, H, in_dim)
            V = V.view(B*D*W, H, self.in_dim)  # (B*D*W, H, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*D*W, H, H)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*D*W, H, in_dim)
            out = out.view(B, D, W, H, self.in_dim)
            out = out.permute(0, 4, 1, 3, 2).contiguous()  # (B, C, D, H, W)
            
        else:  # self.axis == 'W'
            Q = Q.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, q_k_dim)
            Q = Q.view(B*D*H, W, self.q_k_dim)  # (B*D*H, W, q_k_dim)
            
            K = K.permute(0, 2, 3, 1, 4).contiguous()  # (B, D, H, q_k_dim, W)
            K = K.view(B*D*H, self.q_k_dim, W)  # (B*D*H, q_k_dim, W)
            
            V = V.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, in_dim)
            V = V.view(B*D*H, W, self.in_dim)  # (B*D*H, W, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*D*H, W, W)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*D*H, W, in_dim)
            out = out.view(B, D, H, W, self.in_dim)
            out = out.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1-gamma) * x
        return out


class DirectionalMamba(nn.Module):
    def __init__(self, d_model,patch_ini ,d_state=32, axis='D', is_vssb=True, is_slice_attention=True):
        super().__init__()
        self.is_vssb = is_vssb
        self.axis = axis
        
        self.permute_dict = {
            'D': [(0, 1, 2, 3, 4), (0, 2, 3, 4, 1), (0, 4, 1, 2, 3)],  # 原始->mamba输入->输出
            'H': [(0, 1, 2, 3, 4), (0, 3, 2, 4, 1), (0, 4, 2, 1, 3)],
            'W': [(0, 1, 2, 3, 4), (0, 4, 2, 3, 1), (0, 4, 2, 3, 1)]
        }
        
        if self.is_vssb:
            self.mamba = VSSBlock(hidden_dim=d_model, d_state=d_state)
        else:
            self.mamba = MambaLayer(dim=d_model, d_state=d_state)

        if is_slice_attention:
            # self.slice_attention = MultiHeadAxialAttention3D(in_dim=d_model, q_k_dim=d_model, axis=axis,patch_ini=patch_ini)
            self.slice_attention = AxialAttention3D(in_dim=d_model, q_k_dim=d_model, axis=axis,patch_ini=patch_ini)
        else:
            self.slice_attention = nn.Identity()
        
    def forward(self, x):
        # x: [B, C, D, H, W]
        
        original_order, to_mamba, from_mamba = self.permute_dict[self.axis]
        
        if self.is_vssb:

            x_mamba = x.permute(*to_mamba)  # [B, L, *, *, C]，L是处理维度(D/H/W)
            B, L = x_mamba.shape[:2]
            x_mamba = x_mamba.reshape(B * L, *x_mamba.shape[2:])
            
            processed = self.mamba(x_mamba)
            
            processed = processed.reshape(B, L, *processed.shape[1:])
            processed = processed.permute(*from_mamba)  # [B, C, D, H, W]
        else :
            x_mamba = x.permute(*to_mamba)  # [B, L, *, *, C]，L是处理维度(D/H/W)
            B, L = x_mamba.shape[:2]
            x_mamba = x_mamba.reshape(B * L, x_mamba.shape[-1],*x_mamba.shape[2:-1])
            processed = self.mamba(x_mamba)
             

        if isinstance(self.slice_attention, nn.Identity):
            return processed
        else:
            attn_result = self.slice_attention(x,processed)
            return attn_result
def shuffle_channels_3d(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    3D Channel Shuffle operation for ShuffleNet
    
    Args:
        x: Input tensor of shape [batch_size, channels, depth, height, width]
        groups: Number of groups for channel shuffle
        
    Returns:
        Tensor after channel shuffle operation
    """
    batch_size, channels, depth, height, width = x.size()
    
    assert channels % groups == 0
    
    channels_per_group = channels // groups
    
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)
    
    x = x.transpose(1, 2).contiguous()
    
    x = x.view(batch_size, channels, depth, height, width)
    
    return x

class ShuffleChannel(nn.Module):
    
    def __init__(self, groups: int):
        super(ShuffleChannel, self).__init__()
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shuffle_channels_3d(x, self.groups)
class TriplaneMamba3DConcat(nn.Module):
    def __init__(self, input_channels,patch_ini,reduce_factor=1,is_slice_attention=True,is_shuffle=False,is_split=True,is_res=True,is_proj=False):
        super().__init__()
        
        hidden_channels = input_channels
        self.is_proj=is_proj
        if self.is_proj:
            self.proj_in = nn.Sequential(
            nn.Conv3d(hidden_channels, input_channels, 1,stride=1, padding=0),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(input_channels)
        )
            hidden_channels = input_channels//reduce_factor
        self.hidden_channels = hidden_channels
        self.is_res=is_res
        self.is_shuffle=is_shuffle
        self.is_split=is_split
        if self.is_split:
            self.mamba_x = DirectionalMamba(d_model=hidden_channels//2,axis="D",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_y = DirectionalMamba(d_model=hidden_channels//4,axis="H",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_z = DirectionalMamba(d_model=hidden_channels//4,axis="W",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
        else:
            self.mamba_x = DirectionalMamba(d_model=hidden_channels,axis="D",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_y = DirectionalMamba(d_model=hidden_channels,axis="H",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_z = DirectionalMamba(d_model=hidden_channels,axis="W",patch_ini=patch_ini,is_slice_attention=is_slice_attention)

        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_channels, input_channels, 1,stride=1, padding=0),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(input_channels)
        )
    def forward(self, x):
        # x shape: [B, C, D, H, W]
        if self.is_proj:
            x = self.proj_in(x)
        C=self.hidden_channels
        if self.is_split:
            channel_quarter = C // 4
            x_feat = x[:, :channel_quarter*2]  # 1/2
            y_feat = x[:, channel_quarter*2:channel_quarter*3]  # 1/4
            z_feat = x[:, channel_quarter*3:]  # 1/4

            feat_list = []
            
            feat_list.append(self.mamba_x(x_feat))

            feat_list.append(self.mamba_y(y_feat))

            feat_list.append(self.mamba_z(z_feat))
            
            # feat_list.append(res_feat)
            out = self.fusion(torch.cat(feat_list, dim=1))
            if self.is_shuffle:
                ShuffleChannelBlock = ShuffleChannel(groups=4)
                out = ShuffleChannelBlock(out)
        else:
            feat_list = []
            
            feat_list.append(self.mamba_x(x))

            feat_list.append(self.mamba_y(x))

            feat_list.append(self.mamba_z(x))
            
            out = self.fusion(feat_list[0]+feat_list[1]+feat_list[2])
        if self.is_res:
            return out + x
        else:
            return out


class ResNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=2,
        kernel_size=3
    ):
        super().__init__()
        self.stride = stride
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels * expand_rate,
                    in_channels * expand_rate,
                    kernel_size,
                    stride,
                    kernel_size//2,
                    groups=in_channels,
                ),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.residual = in_channels == out_channels
        self.act = nn.LeakyReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x = self.conv_list[0](x)
        x = self.conv_list[1](x)
        x = self.conv_list[2](x)
        x = x + res if self.residual and self.stride == 1 else x
        return x



class DenseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=4,
        dropout_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                # LayerNormBatchFirst(in_channels),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels, in_channels * expand_rate, 1, 1, 0
                ),
                # nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                # nn.LeakyReLU(inplace=True),
                nn.GELU(),
            )
        )
        temp_in_channels = in_channels + in_channels + in_channels * expand_rate
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                # nn.InstanceNorm3d(out_channels, affine=True),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.residual = in_channels == out_channels
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.residual:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = self.conv_list[1](torch.cat([x, x1], dim=1))
        x2 = self.dp_2(x2)
        x = (
            self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
            if self.residual
            else self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        )
        return x
class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        stride=2,
        patch_size=[128,128,128],
        is_split=True,
        is_slice_attention=True,
        **kwargs,
    ):
        super().__init__()
        assert num_conv >= 1, "num_conv must be greater than or equal to 1"
        self.downsample_avg = nn.AvgPool3d(stride)
        self.downsample_resnext = ResNeXtConv(in_channels, in_channels, stride=stride)
        self.tmamba=TriplaneMamba3DConcat(input_channels=in_channels,is_split=is_split,patch_ini=patch_size,is_slice_attention=is_slice_attention)
        # self.tmamba = MambaLayer(dim=in_channels)

        # self.tmamba = TriplaneMamba3DCasCade(input_channels=in_channels,patch_ini=patch_size)
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels, out_channels, **kwargs)
                    if _ == 0
                    else conv(out_channels, out_channels, **kwargs)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x):
        x = self.downsample_avg(x) + self.downsample_resnext(x)
        x = self.tmamba(x)
        x_down = x
        for extractor in self.extractor:
            x = extractor(x)
        # x = self.tmamba(x)

        return x_down,x


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        patch_size=64,
        fusion_mode="add",
        stride=2,
        is_split=True,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up_transpose = nn.ConvTranspose3d(
            high_channels, high_channels, stride, stride
        )
        in_channels = 2 * high_channels if fusion_mode == "cat" else high_channels
        self.extractor = nn.ModuleList(
            [
                (
                    # TriplaneMamba3D(input_channels=high_channels,num_slices=patch_size,is_split=is_split)
                    # conv(low_channels, high_channels)
                    
                    nn.Sequential(
                        nn.Conv3d(low_channels, high_channels, 1, 1, 0),
                        nn.InstanceNorm3d(high_channels, affine=True),
                        nn.LeakyReLU(inplace=True),
                    )
                    
                    if _ == 0
                    else conv(high_channels, high_channels)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x_low, x_high):
        for extractor in self.extractor:
            x_low = extractor(x_low)
        x = (
            torch.cat([x_high, x_low], dim=1)
            if self.fusion_mode == "cat"
            else x_low + x_high
        )
        x = self.up_transpose(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.1):
        super().__init__()
        self.dp = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv3d(in_channels, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.dp(x)
        p = self.conv1(x)
        return p



class DenseConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=4,
        dropout_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.res = in_channels == out_channels
        self.dense = stride == 1

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                # LayerNormBatchFirst(in_channels),
                # nn.LeakyReLU(inplace=True),
            )
            
        )
        temp_in_channels = in_channels + in_channels if self.dense else in_channels
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, in_channels * expand_rate, 1, 1, 0),
                # nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                # nn.LeakyReLU(inplace=True),
                nn.GELU(),
            )
        )
        temp_in_channels = (
            in_channels + in_channels + in_channels * expand_rate
            if self.dense
            else in_channels + in_channels * expand_rate
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                # nn.InstanceNorm3d(out_channels, affine=True),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.dense:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = (
            self.conv_list[1](torch.cat([x, x1], dim=1))
            if self.dense
            else self.conv_list[1](x1)
        )
        x2 = self.dp_2(x2)
        if self.dense and self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
        elif self.dense and not self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        else:
            x = self.conv_list[2](torch.cat([x1, x2], dim=1))
        return x

class HCMA(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        patch_ini = [128, 128, 128],
        depth=4,
        conv=DenseConv,
        channels=[2**i for i in range(5, 10)],
        encoder_num_conv=[1, 1, 1, 1],
        decoder_num_conv=[1, 1, 1, 1],
        encoder_expand_rate=[4] * 4,
        decoder_expand_rate=[4] * 4,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)],
        dropout_rate_list=[0.025, 0.05, 0.1, 0.1],
        drop_path_rate_list=[0.025, 0.05, 0.1, 0.1],
        deep_supervision=False,
        predict_mode=False,
        is_skip=False,
        is_split=True,
        is_slice_attention=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        self.is_skip = is_skip
        assert len(channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 
        self.decoders = nn.ModuleList()  # 
        self.skips = nn.ModuleList()  # 
        self.encoders.append(DenseConv(in_channels, channels[0]))
        # patch_ini= [128, 128, 128]# [64,192,128]
        patch_ini = list(patch_ini)
        for i in range(self.depth):
            for j in range(3):
                patch_ini[j] = int(patch_ini[j]/strides[i][0])
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    patch_size=patch_ini,
                    is_split=is_split,
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                    is_slice_attention=is_slice_attention
                ),
            )

        for i in range(self.depth):
            patch_ini*=strides[self.depth - i - 1][0]
            for j in range(3):
                patch_ini[j]*=strides[self.depth - i - 1][0]
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    # conv=conv,
                    patch_size=patch_ini,
                    is_split=is_split,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    dropout_rate=dropout_rate_list[self.depth - i - 1],
                    drop_path_rate=drop_path_rate_list[self.depth - i - 1],
                )
            )
        for i in range(self.depth):
            if self.is_skip:
                self.skips.append(
                    TripleLine3DFusion(
                        in_channels = channels[self.depth - i],
                        kernel_size=7
                    )
                )
            else:
                self.skips.append(
                    nn.Identity()
                )
        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )
        

    def forward(self, x):
        encoder_features = []  # 
        decoder_features = []  # 

        for i,encoder in enumerate(self.encoders):
            if i==0:
                x = encoder(x)
                encoder_features.append([x])
            else:
                x_down,x = encoder(x)
                encoder_features.append([x_down,x])

        for i in range(self.depth+1):
            if i == 0:
                x_down,x_dec =  encoder_features[self.depth-i][0],encoder_features[self.depth-i][1]
                x_dec = self.skips[i](x_dec)
                # print(x_dec.shape)
            elif i == self.depth:
                x_dec = self.decoders[i-1](x_dec, x_down)
                decoder_features.append(x_dec)
                # print(x_dec.shape)
            else:
                x_dec = self.decoders[i-1](x_dec, self.skips[i](x_down))
                x_down = encoder_features[self.depth-i][0]
                decoder_features.append(x_dec)
                # print(x_dec.shape)

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        elif self.predict_mode:
            return self.out[-1](decoder_features[-1])
        else:
            return x_dec, self.out[-1](decoder_features[-1])

