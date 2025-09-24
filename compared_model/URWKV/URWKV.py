import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath
from einops import rearrange
# from .scan_scan_inv import vertical_forward_scan, vertical_forward_scan_inv, vertical_backward_scan,vertical_backward_scan_inv
# from .scan_scan_inv import horizontal_forward_scan, horizontal_forward_scan_inv, horizontal_backward_scan,horizontal_backward_scan_inv
from .scan_scan_inv_3d import scan_left_to_right, scan_left_to_right_inv
from .scan_scan_inv_3d import scan_right_to_left, scan_right_to_left_inv

from .scan_scan_inv_3d import scan_up_to_down, scan_up_to_down_inv
from .scan_scan_inv_3d import scan_down_to_up, scan_down_to_up_inv

from .scan_scan_inv_3d import scan_front_to_back, scan_front_to_back_inv
from .scan_scan_inv_3d import scan_back_to_front, scan_back_to_front_inv


T_MAX = 4096 #128*128 2048 均不可以 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


wkv_cuda = load(name="wkv", sources=['/data115_3/chenjy/RC_MM/compared_model/URWKV/cuda/wkv_op.cpp', '/data115_3/chenjy/RC_MM/compared_model/URWKV/cuda/wkv_cuda.cu'],
                verbose=True, extra_cuda_cflags=['-res-usage','--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']) #'--maxrregcount 60', 

def q_shift(x, shift_pixel=1, gamma=1 / 4, resolution=(3, 3, 3), scan_type='left_to_right'):
    """
    对输入张量进行 q_shift 操作，并根据 scan_type 选择不同的扫描方式
    """
    assert gamma <= 1 / 4
    if len(x.shape) == 3:
        B, N, C = x.shape
        x = x.reshape(B, C, resolution[0], resolution[1], resolution[2])
    output = scan_left_to_right(x)
    return output

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        
        device = v.device
        y = torch.empty((B, T, C), device=device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        device = v.device
        gw = torch.zeros((B, C), device=device).contiguous()
        gu = torch.zeros((B, C), device=device).contiguous()
        gk = torch.zeros((B, T, C), device=device).contiguous()
        gv = torch.zeros((B, T, C), device=device).contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    device = v.device
    return WKV.apply(B, T, C, w.to(device), u.to(device), k.to(device), v.to(device))

class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        # 定义线性层
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        
        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        
        # 输出线性层
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        # 初始化线性层的 scale_init
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv

class SpectralMixer(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        # 定义线性层
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        # 初始化线性层的 scale_init
        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.key.scale_init = 1

    def forward(self, x, resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        # 计算 key、value 和 receptance
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        # 计算最终输出
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv



def v_enc_256_fffse_dec_fusion_rwkv_with2x4_3d(input_channel=2, num_classes=2, rwkv_n_layer=8, dims=[8, 16, 64, 80, 128]): #_withinpool dims=[16, 32, 128, 160, 256]
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm3d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm3d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, ...], x[:, 1::2, ...]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm3d(ch_in),
                nn.Conv3d(ch_in, ch_out * 4, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(ch_out * 4),
                nn.Conv3d(ch_out * 4, ch_out, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x
            
            x1 = scan_left_to_right(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = scan_left_to_right_inv(x1, x4d_shape)
            
            x2 = scan_right_to_left(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = scan_right_to_left_inv(x2, x4d_shape)
            
            x3 = scan_up_to_down(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = scan_up_to_down_inv(x3, x4d_shape)
            
            x4 = scan_down_to_up(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = scan_down_to_up_inv(x4, x4d_shape)
            
            x5 = scan_front_to_back(x_4d)
            x5 = x5 + self.gamma1 * super().forward(self.ln1(x5), resolution)
            x5_4d = scan_front_to_back_inv(x5, x4d_shape)
            
            x6 = scan_back_to_front(x_4d)
            x6 = x6 + self.gamma1 * super().forward(self.ln1(x6), resolution)
            x6_4d = scan_back_to_front_inv(x6, x4d_shape)
            
            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d, x5_4d, x6_4d]), dim=0)
            # x4dout = torch.mean(torch.stack([x1_4d, x3_4d, x5_4d]), dim=0)
            
            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout      

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            
        def forward(self, x):
            b,c,d,h,w = x.shape
            x = self.allinone_spa(x, resolution=(d,h,w))   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=(d,h,w))            
            x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c,d,h,w = x.shape
            x_r = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)    
            x_r4d = x_r.transpose(1, 2).view(b, c, d, h, w)
            return super().forward(x)  + super().forward(x_r4d)
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=input_channel, num_classes=num_classes, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv3d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm3d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv3d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()
