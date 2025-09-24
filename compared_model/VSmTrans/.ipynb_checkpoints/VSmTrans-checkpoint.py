from typing import Sequence, Tuple, Union
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

def window_partition(x, D_sp, H_sp, W_sp, num_heads=None, is_Mask=False):
    B, D, H, W, C = x.shape
    if is_Mask:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C).contiguous()
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, D_sp * H_sp * W_sp, C)
    else:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C // num_heads, num_heads).contiguous()
        x = x.permute(0, 1, 3, 5, 8, 2, 4, 6, 7).contiguous().view(-1, num_heads, D_sp * H_sp * W_sp, C // num_heads)
    return x

def window_reverse(x, D_sp, H_sp, W_sp, D, H, W):
    _, _, C = x.shape
    x = x.view(-1, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(-1, D, H, W, C).contiguous()
    return x

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size[0], window_size[1], window_size[2], is_Mask=True)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = out[:,:, :skip.size()[2], :skip.size()[3], :skip.size()[4]]
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out

class PatchEmbed(nn.Module):
    """
        utilize a non-overlapping convolution to Down Sampling and Add Channels
    """
    def __init__(self, in_channels, feature_size, patch_size, norm_layer=None):
        super().__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels=in_channels, out_channels=feature_size, kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(feature_size)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)

        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.feature_size, D, Wh, Ww)
        return x

class PatchMerging(nn.Module):
    def __init__(self, feature_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.feature_size = feature_size
        self.norm = norm_layer(feature_size * 8)
        self.reduction = nn.Linear(8 * feature_size, 2 * feature_size, bias=False)

    def forward(self, x):
        B, D, H, W, C = x.shape
        # padding
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, D % 2))
            x = rearrange(x, 'b c d h w -> b d h w c')
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class VariableShapeAttention(nn.Module):
    def __init__(self, feature_size, idx, split_size, window_size, num_head, img_size, shift=False, attn_drop_rate=0.):
        super(VariableShapeAttention, self).__init__()
        self.num_head = num_head
        self.init_window_size(idx, img_size, split_size, window_size)
        head_dim = 4 * feature_size // num_head
        self.scale = head_dim ** -0.5
        self.shift = shift
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        mesh_args = torch.meshgrid.__kwdefaults__
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.D_sp - 1) * (2 * self.H_sp - 1) * (2 * self.W_sp - 1),
                num_head,
            )
        )
        coords_d = torch.arange(self.D_sp)
        coords_h = torch.arange(self.H_sp)
        coords_w = torch.arange(self.W_sp)
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.D_sp - 1
        relative_coords[:, :, 1] += self.H_sp - 1
        relative_coords[:, :, 2] += self.W_sp - 1
        relative_coords[:, :, 0] *= (2 * self.H_sp - 1) * (2 * self.W_sp - 1)
        relative_coords[:, :, 1] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)


    def init_window_size(self, idx, img_size, split_size, window_size):
        if idx == 0:
            self.D_sp, self.H_sp, self.W_sp = window_size if img_size[0] > window_size else img_size[0], \
                                              window_size if img_size[1] > window_size else img_size[1], \
                                              window_size if img_size[2] > window_size else img_size[2],
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                                              self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                                              self.W_sp // 2 if img_size[2] > self.W_sp else 0
        elif idx == 1:
            self.D_sp, self.H_sp, self.W_sp = split_size if img_size[0] > split_size else img_size[0], \
                                              img_size[1], \
                                              split_size if img_size[2] > split_size else img_size[2]
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                                              0, \
                                              self.W_sp // 2 if img_size[2] > self.W_sp else 0
        elif idx == 2:
            self.D_sp, self.H_sp, self.W_sp = split_size if img_size[0] > split_size else img_size[0], \
                                              split_size if img_size[1] > split_size else img_size[1], \
                                              img_size[2]
            self.D_sf, self.H_sf, self.W_sf = self.D_sp // 2 if img_size[0] > self.D_sp else 0, \
                                              self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                                              0
        elif idx == 3:
            self.D_sp, self.H_sp, self.W_sp = img_size[0], \
                                              split_size if img_size[1] > split_size else img_size[1], \
                                              split_size if img_size[2] > split_size else img_size[2]
            self.D_sf, self.H_sf, self.W_sf = 0, \
                                              self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                                              self.W_sp // 2 if img_size[2] > self.W_sp else 0

    def forward(self, qkv):
        B, D, H, W, C = qkv.shape
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.D_sp - D % self.D_sp) % self.D_sp
        pad_b = (self.H_sp - H % self.H_sp) % self.H_sp
        pad_r = (self.W_sp - W % self.W_sp) % self.W_sp
        qkv = F.pad(qkv, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = qkv.shape

        if self.shift:
            qkv = torch.roll(qkv, shifts=(-self.D_sf, -self.H_sf, -self.W_sf), dims=(1, 2, 3))

        qkv = qkv.reshape(B, Dp, Hp, Wp, 3, C // 3).permute(4, 0, 1, 2, 3, 5)
        q = window_partition(qkv[0], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        k = window_partition(qkv[1], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        v = window_partition(qkv[2], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        n = self.D_sp * self.H_sp * self.W_sp
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.shift:
            mask = compute_mask(dims=[Dp, Hp, Wp], window_size=(self.D_sp, self.H_sp, self.W_sp),
                                shift_size=(self.D_sf, self.H_sf, self.W_sf), device=qkv.device)
            nw = mask.shape[0]
            attn = attn.view(attn.shape[0] // nw, nw, self.num_head, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.D_sp * self.H_sp * self.W_sp, C // 3).contiguous()
        x = window_reverse(x, self.D_sp, self.H_sp, self.W_sp, Dp, Hp, Wp)
        if self.shift:
            x = torch.roll(x, shifts=(self.D_sf, self.H_sf, self.W_sf), dims=(1, 2, 3))

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

class VSmixWindow_MSA(nn.Module):
    def __init__(self,
                 feature_size,
                 split_size,
                 window_size,
                 num_head,
                 img_size,
                 shift=False,
                 qkv_bias=False,
                 attn_drop_rate=0.0,
                 drop_rate=0.0):
        super(VSmixWindow_MSA, self).__init__()
        self.num_head = num_head
        self.qkv = nn.Linear(feature_size, feature_size * 3, bias=qkv_bias)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(feature_size * 3, feature_size)
        self.norm1 = nn.LayerNorm(feature_size, eps=1e-6)
        self.dep_conv = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(num_features=feature_size)
        self.act2 = nn.LeakyReLU()

        self.attns = nn.ModuleList([
            VariableShapeAttention(
                feature_size=feature_size // 4,
                idx=i % 4,
                split_size=split_size,
                window_size=window_size,
                num_head=num_head,
                img_size=img_size,
                shift=shift,
                attn_drop_rate=attn_drop_rate
            )
            for i in range(4)])

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.drop = nn.Dropout(drop_rate)
        self.reset_parameters()

        self.proj = nn.Linear(feature_size, feature_size)
        self.proj_drop = nn.Dropout(drop_rate)

    def reset_parameters(self):
        if self.rate1 is not None:
            self.rate1.data.fill_(0.5)
        if self.rate2 is not None:
            self.rate2.data.fill_(0.5)
        self.dep_conv.bias.data.fill_(0.0)

    def forward(self, x):
        qkv = self.qkv(x)
        B, D, H, W, C = qkv.shape
        # Conv
        # B, D, H, W, C
        conv_x = self.conv1(self.act1(qkv))
        conv_x = self.norm1(conv_x).permute(0, 4, 1, 2, 3)
        conv_x = self.dep_conv(conv_x)
        conv_x = self.act2(self.norm2(conv_x)).permute(0, 2, 3, 4, 1)
        # Transformer
        x1 = self.attns[0](qkv[:, :, :, :, :C // 4])
        x2 = self.attns[1](qkv[:, :, :, :, C // 4:C // 4 * 2])
        x3 = self.attns[2](qkv[:, :, :, :, C // 4 * 2:C // 4 * 3])
        x4 = self.attns[3](qkv[:, :, :, :, C // 4 * 3:])
        attn_x = torch.cat([x1, x2, x3, x4], dim=-1)
        # 考虑注意力通道问题
        attn_x = self.proj_drop(self.proj(attn_x))
        x = self.rate1*attn_x+self.rate2*conv_x
        x = self.drop(x)
        return x

class VSmixedBlock(nn.Module):
    def __init__(self,
                 feature_size,
                 split_size,
                 window_size,
                 num_head,
                 img_size,
                 shift=False,
                 mlp_ratio=4,
                 qkv_bias=False,
                 drop_rate=0.0,
                 drop_path=0.0,
                 attn_drop_rate=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super(VSmixedBlock, self).__init__()
        self.shift = shift
        self.split_size = split_size

        self.norm1 = norm_layer(feature_size)
        self.attn = VSmixWindow_MSA(
            feature_size=feature_size, split_size=split_size, window_size=window_size,
            num_head=num_head, img_size=img_size, shift=shift, qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate, drop_rate=drop_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(feature_size)
        mlp_hidden_dim = int(feature_size * mlp_ratio)
        self.mlp = Mlp(in_features=feature_size, hidden_features=mlp_hidden_dim, drop=drop_rate, act_layer=act_layer)

    def forward_part1(self, x):
        x = self.norm1(x)
        return self.attn(x)

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 feature_size: int,
                 split_size: int,
                 window_size: int,
                 num_head: int,
                 depth: int,
                 img_size,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate=0.0,
                 drop_path=0.0,
                 attn_drop_rate=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super(BasicLayer, self).__init__()

        self.blocks = nn.ModuleList([
            VSmixedBlock(
                feature_size=feature_size,
                split_size=split_size,
                window_size=window_size,
                num_head=num_head,
                img_size=img_size,
                shift=False if i % 2 == 0 else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attn_drop_rate=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for i in range(depth)])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class VariableShapeMixedTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            feature_size: int,
            img_size: Sequence[int],
            split_size: Sequence[int],
            window_size: int,
            patch_size: Sequence[int],
            depths: Sequence[int],
            num_heads: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer=nn.LayerNorm,
    ) -> None:
        super(VariableShapeMixedTransformer, self).__init__()
        self.patch_norm = norm_layer
        self.img_size = img_size
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(in_channels=in_channels, feature_size=feature_size, patch_size=patch_size,
                                    norm_layer=norm_layer if self.patch_norm else None)
        self.embed_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        img_h, img_w, img_d = self.img_size[0], self.img_size[1], self.img_size[2]
        img_size_list = []
        for i in range(self.num_layers):
            img_h = img_h // 2 + (1 if img_h % 2 != 0 else 0)
            img_w = img_w // 2 + (1 if img_w % 2 != 0 else 0)
            img_d = img_d // 2 + (1 if img_d % 2 != 0 else 0)
            img_size_list.append([img_h, img_w, img_d])
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                feature_size=int(feature_size * 2**i_layer),
                depth=depths[i_layer],
                num_head=num_heads[i_layer],
                img_size=img_size_list[i_layer],
                split_size=split_size[i_layer],
                window_size=window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
            )
            down = PatchMerging(feature_size=feature_size * 2**i_layer, norm_layer=norm_layer)
            if i_layer == 0:
                self.layers1.append(layer)
                self.layers1.append(down)
            elif i_layer == 1:
                self.layers2.append(layer)
                self.layers2.append(down)
            elif i_layer == 2:
                self.layers3.append(layer)
                self.layers3.append(down)
            elif i_layer == 3:
                self.layers4.append(layer)
                self.layers4.append(down)


    def normalize_out(self, x):
        x = rearrange(x, "n d h w c -> n c d h w")
        x = F.group_norm(x, num_groups=12)
        return x

    def forward(self, x):
        x0 = self.patch_embed(x)
        x0 = self.embed_drop(x0)
        x0 = rearrange(x0, 'b c d h w -> b d h w c')

        x0 = self.layers1[0](x0.contiguous())
        x0_out = self.normalize_out(x0)

        x1 = self.layers1[1](x0.contiguous())
        x1 = self.layers2[0](x1.contiguous())
        x1_out = self.normalize_out(x1)

        x2 = self.layers2[1](x1.contiguous())
        x2 = self.layers3[0](x2.contiguous())
        x2_out = self.normalize_out(x2)

        x3 = self.layers3[1](x2.contiguous())
        x3 = self.layers4[0](x3.contiguous())
        x3_out = self.normalize_out(x3)

        x4 = self.layers4[1](x3.contiguous())
        x4_out = self.normalize_out(x4)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class Decoder(nn.Module):
    def __init__(self, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision

    def forward(self):
        None

class VSmixTUnet(nn.Module):
    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            feature_size: int = 48,
            split_size: Sequence[int] = [1, 3, 5, 7],
            window_size: int = 7,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            patch_size: Sequence[int] = (2, 2, 2),
            norm_name: Union[Tuple, str] = "instance",
            qkv_bias: bool = True,
            drop_rate: float = 0.1,
            attn_drop_rate: float = 0.1,
            drop_path_rate: float = 0.1,
            spatial_dims: int = 3,
            do_ds=False,
            ):
        super(VSmixTUnet, self).__init__()

        self.decoder = Decoder(do_ds)
        # self.decoder.deep_supervision = True

        self.VSmViT = VariableShapeMixedTransformer(
            img_size=img_size,
            in_channels=in_channels,
            feature_size=feature_size,
            split_size=split_size,
            window_size=window_size,
            num_heads=num_heads,
            depths=depths,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )

        self.do_ds = do_ds

        if self.do_ds:
            self.out_4 = nn.Conv3d(16 * feature_size, out_channels, 1)
            self.out_3 = nn.Conv3d(8 * feature_size, out_channels, 1)
            self.out_2 = nn.Conv3d(4 * feature_size, out_channels, 1)
            self.out_1 = nn.Conv3d(2 * feature_size, out_channels, 1)
            self.out_0 = nn.Conv3d(feature_size, out_channels, 1)

    def forward(self, x):
        hidden_states_out = self.VSmViT(x)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])

        dec3 = self.decoder5(hidden_states_out[4], enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        if self.do_ds:
            out1 = F.interpolate(self.out_0(dec0), scale_factor=2, mode='trilinear', align_corners=True)
            out2 = F.interpolate(self.out_1(dec1), scale_factor=4, mode='trilinear', align_corners=True)
            out3 = F.interpolate(self.out_2(dec2), scale_factor=8, mode='trilinear', align_corners=True)
            out4 = F.interpolate(self.out_3(dec3), scale_factor=16, mode='trilinear', align_corners=True)
            return [self.out(out), out1, out2, out3, out4]
        else:
            return self.out(out)



if __name__ == '__main__':
    data = torch.randn((1, 1, 96, 96, 96), dtype=torch.float32)
    model = VSmixTUnet(
        in_channels=1,
        out_channels=16,
        feature_size=48,
        split_size=[1, 3, 5, 7],
        window_size=7,
        num_heads=[3, 6, 12, 24],
        img_size=[96, 96, 96],
        depths=[2, 2, 2, 2],
        patch_size=(2, 2, 2),
    )
    result = model(data)
    import thop
    flops, params = thop.profile(model, inputs=(data,))
    print("flops = ", flops / 1024.0 / 1024.0 / 1024.0, 'G')
    print("params = ", params / 1024.0 / 1024.0, 'M')