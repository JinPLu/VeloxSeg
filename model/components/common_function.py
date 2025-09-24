import torch
from torch import nn
from einops import einsum

def concat(*args):
    return torch.cat(args, dim=1)

def get_pram_matrix(x):
    if len(x.shape) == 4:
        c, h, w = x.size()[-3:]
        return einsum(x, x, 'b m h w, b n h w -> b m n') / (c * h * w)
    elif len(x.shape) == 5:
        c, h, w, d = x.size()[-4:]
        return einsum(x, x, 'b m h w d, b n h w d -> b m n') / (c * h * w * d)

def split_output_channel(output, channels, mappings = lambda x: x):
    
    mapping = mappings
    res = []
    c = 0
    for i in range(len(channels)):
        if isinstance(mappings, list):
            mapping = mappings[i]
        res.append(mapping(output[:, c:c+channels[i]]))
        c += channels[i]
    return res

def check_input(x, in_channels):
    if not isinstance(x, list):
        channels = x.shape[1]
        assert channels == sum(in_channels), f"Input channels should be equal to the sum of in_channels, but got {channels} and {sum(in_channels)}"
        xs = []
        c = 0
        for i in range(len(in_channels)):
            xs.append(x[:, c:c+in_channels[i]])
            c += in_channels[i]
        x = xs
    return x

def get_conv(spatial_dim):
    if spatial_dim == 3:
        return nn.Conv3d
    elif spatial_dim == 2:
        return nn.Conv2d
    
def get_traspose_conv(spatial_dim):
    if spatial_dim == 3:
        return nn.ConvTranspose3d
    elif spatial_dim == 2:
        return nn.ConvTranspose2d

def get_norm(norm_type="BN", spatial_dim=3):
    assert norm_type in ["BN", 'GN', 'IN', "LN"]
    assert spatial_dim in [2, 3]
    if norm_type == "BN":
        if spatial_dim == 2:
            return nn.BatchNorm2d
        elif spatial_dim == 3:
            return nn.BatchNorm3d
    elif norm_type == "GN":
        return nn.GroupNorm
    elif norm_type == "IN":
        if spatial_dim == 2:
            return nn.InstanceNorm2d
        elif spatial_dim == 3:
            return nn.InstanceNorm3d
    elif norm_type == "LN":
        return nn.LayerNorm
    
def get_pool(pool_type, spatial_dim=3):
    if pool_type == "max":
        if spatial_dim == 2:
            return nn.AdaptiveMaxPool2d
        elif spatial_dim == 3:
            return nn.AdaptiveMaxPool3d
    elif pool_type == "avg":
        if spatial_dim == 2:
            return nn.AdaptiveAvgPool2d
        elif spatial_dim == 3:
            return nn.AdaptiveAvgPool3d
        
def get_act(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)


