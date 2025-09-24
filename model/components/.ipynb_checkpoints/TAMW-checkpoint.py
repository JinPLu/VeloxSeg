import torch
import torch.nn as nn
from .common_function import concat
from torch.nn import functional as F


class Adaptive_Weighting(nn.Module):
    """
    MutiModle-Channel Emphasize Attention 
    return: Enphasize(features)
    """
    def __init__(self, input_channels = [1, 1, 8*2], scale = 4, dim = 3):
        super(Adaptive_Weighting, self).__init__()
        
        self.input_channels = input_channels
        self.weight_size = [1, 1, 1]
        if dim == 2:
            self.weight_size = [1, 1]
            self.globalAvgPool = nn.AdaptiveAvgPool2d(self.weight_size)
        elif dim == 3:
            self.globalAvgPool = nn.AdaptiveAvgPool3d(self.weight_size)
        self.channels = sum(input_channels)
        self.fc1 = nn.Linear(in_features = self.channels         , out_features = self.channels // scale)
        self.fc2 = nn.Linear(in_features = self.channels // scale, out_features = self.channels         )
        self.tanh = nn.Tanh()

    def forward(self, features):
        b = features[0].size(0)
        num_modalities = len(features)
        
        original_features = features
        features = [self.globalAvgPool(f).view(b, -1) for f in features]
        features = concat(*features)

        # channel attention
        weight = self.fc1(features)
        weight = self.tanh(weight)
        weight = self.fc2(weight)
        weight = self.tanh(weight)
        weight = weight.view(b, self.channels, *self.weight_size)
        
        c = 0
        features = []
        for m in range(num_modalities):
            features.append(weight[:, c:c+self.input_channels[m]] * original_features[m])
            c = c + self.input_channels[m]
        return features

class Spatial_Conv(nn.Module):
    def __init__(
                self,
                in_channels:int, 
                num_dilated_ratio:int = 4,
                dropout:float = 0.25,
                dim = 3,):
        super().__init__()

        self.dim = dim
        if self.dim == 2:
            conv = nn.Conv2d
        elif self.dim == 3:
            conv = nn.Conv3d

        self.num_dilated_ratio = num_dilated_ratio
        spatial_conv = []
        for dr in range(1, self.num_dilated_ratio + 1):
            spatial_conv.append(
                conv(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = ((3 - 1) * dr) // 2,
                    dilation = dr,
                    groups = in_channels,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.spatial_conv = nn.ModuleList(spatial_conv)
        self.fusion = conv(self.num_dilated_ratio*in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        
        attn = []
        for i in range(self.num_dilated_ratio):
            attn.append(self.spatial_conv[i](x))
        attn = torch.cat(attn, dim=1)
        attn = self.dropout(self.fusion(attn))
        return attn

class Spatial_Channel_Attention(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_dilated_ratio: int = 4, 
                 channel_expansion:int = 4,
                 dropout: float = 0.25,
                 dim = 3):
        super().__init__()
        self.dim = dim
        if self.dim == 2:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
        elif self.dim == 3:
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d

        self.spatial_attn = Spatial_Conv(in_channels, num_dilated_ratio, dim=dim, dropout=dropout)
        self.channel_attn = nn.Sequential(
            norm(in_channels),
            conv(in_channels, in_channels * channel_expansion, 1, 1, 0),
            nn.LeakyReLU(),
            conv(in_channels * channel_expansion, in_channels, 1, 1, 0),
            nn.Dropout(dropout),
        )
        self.proj_weight = nn.Sequential(
            conv(in_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )
            
    def forward(self, features):
        
        features = self.spatial_attn(features) + features
        features = self.channel_attn(features) + features
        
        return self.proj_weight(features)


class TAMW_Block(nn.Module):    
    def __init__(self, 
                enc_in_channels: int,
                dec_in_channels: int,
                num_dilated_ratio: int = 4, 
                channel_expansion: int = 4,
                dropout: float = 0.1,
                enhance_enc: bool = True,
                fp_fn_weight: bool = False,
                dim: int = 3,
                ):
        super(TAMW_Block, self).__init__()
        
        if dim == 2:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif dim == 3:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.attn_fore = Spatial_Channel_Attention(
                        in_channels         = dec_in_channels,
                        num_dilated_ratio   = num_dilated_ratio,
                        channel_expansion   = channel_expansion,
                        dropout             = dropout,
                        dim                 = dim,)
        self.attn_back = Spatial_Channel_Attention(
                        in_channels         = dec_in_channels,
                        num_dilated_ratio   = num_dilated_ratio,
                        channel_expansion   = channel_expansion,
                        dropout             = dropout,
                        dim                 = dim,)
        
        self.enhance_enc = enhance_enc
        if self.enhance_enc:
            self.modality_weight = nn.Parameter(torch.zeros(enc_in_channels, 1, 1, 1), requires_grad=True)
        
        self.fp_fn_weight = fp_fn_weight
        if self.fp_fn_weight:
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        
    def forward(self, enc_features, dec_features, p_hi):
        '''
        enc_features: the feature from kth encoder
        dec_feature: the feature from (k+1)th decoder (before upsampling)
        p_hi: prediction map from (k+1)th decoder
        '''
        p_hi = self.softmax(p_hi)
        
        foreground_slice = p_hi.argmax(dim=1, keepdim=True)
        background_slice = 1 - foreground_slice
        
        uncertain_mask = 1 - 2 * (p_hi[:, 0:1] - 0.5).abs()

        f_fore = dec_features * foreground_slice * uncertain_mask
        f_back = dec_features * background_slice * uncertain_mask
        
        fp = self.attn_fore(f_fore)
        fn = self.attn_back(f_back)
        
        if self.fp_fn_weight:
            alpha = self.sigmoid(self.alpha)
            beta = self.sigmoid(self.beta)
            dec_features = dec_features * (1 - alpha * fp + beta * fn)
        else:
            dec_features = dec_features * (1 - fp + fn)
        
        if self.enhance_enc:
            modality_weights = self.sigmoid(self.modality_weight)
            enc_features = enc_features * (1 + modality_weights * self.upsample(uncertain_mask)) / 2
        
        return enc_features, dec_features, fp, fn
    

class TAMW(nn.Module):
    def __init__(self, in_channels: list[int], dim = 3):
        super(TAMW, self).__init__()
        
        self.in_channels = in_channels
        self.softmax = nn.Softmax(1)
        self.CA_fore = Adaptive_Weighting(self.in_channels)
        self.CA_back = Adaptive_Weighting(self.in_channels)
        if dim == 2:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif dim == 3:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, features, p_hi):
        '''
        feature: feature of m modalities
        '''
        p_hi = self.upsample(self.softmax(p_hi))
        f_fore = [feature * p_hi[:, 1:2] for feature in features]
        f_back = [feature * p_hi[:, 0:1] for feature in features]
        
        f_fore = self.CA_fore(f_fore)
        f_back = self.CA_back(f_back)

        f_enhance = [features[i] + f_fore[i] - f_back[i] for i in range(len(f_fore))]
        return f_enhance
    
    
# class ImagePool(nn.Module):
#     def __init__(self, in_channel, dim = 3):
#         super(ImagePool, self).__init__()
#         if dim == 2:
#             self.gpool = nn.AdaptiveAvgPool2d(1)
#             self.conv = nn.Conv2d(in_channel, in_channel, 1, 1)
#         elif dim == 3:
#             self.gpool = nn.AdaptiveAvgPool3d(1)
#             self.conv = nn.Conv3d(in_channel, in_channel, 1, 1)
#         self.dim = dim

#     def forward(self, x):
#         net = self.gpool(x)
#         net = self.conv(net)
        
#         mode = 'bilinear' if self.dim == 2 else 'trilinear'
#         net = F.interpolate(net, size=x.size()[2:], mode=mode, align_corners=True)
#         return net

# class SplitSpatialConv(nn.Module):
#     def __init__(self, in_channels, cards, dim = 3):
#         super(SplitSpatialConv, self).__init__()
        
#         if dim == 2:
#             conv = nn.Conv2d
#         elif dim == 3:
#             conv = nn.Conv3d
        
#         self.convs = nn.ModuleList()
#         for i in range(cards):
#             self.convs.append(
#                 conv(in_channels, in_channels, 
#                      kernel_size=3, stride=1, 
#                      padding=i+1, dilation=i+1, 
#                      groups=in_channels)
#             )

#         self.fusion = conv(in_channels*cards, in_channels, 1, 1, 0)
        
#     def forward(self, x):
#         nets = []
#         for conv in self.convs:
#             nets.append(conv(x))
#         return self.fusion(torch.cat(nets, dim=1))

# class CrossAttentionConv(nn.Module):
#     def __init__(self, x_ch, y_ch, dim=64):
#         super(CrossAttentionConv, self).__init__()
#         self.x_map_conv = nn.Sequential(
#             nn.Conv2d(x_ch, dim, 1, 1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#         self.y_map_conv = nn.Sequential(
#             nn.Conv2d(y_ch, dim, 1, 1, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         #spatial
#         self.x_spatial = nn.Sequential(
#             SplitSpatialConv(2, cards=4),
#             nn.Conv2d(2, 1, 1, 1, bias=False),
#             # nn.Conv2d(2, 1, 7, 1, 3),
#             nn.Sigmoid()
#         )

#         self.y_spatial = nn.Sequential(
#             SplitSpatialConv(2, cards=4),
#             nn.Conv2d(2, 1, 1, 1, bias=False),
#             # nn.Conv2d(2, 1, 7, 1, 3),
#             nn.Sigmoid()
#         )
#         #channel
#         self.x_channel = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // 4, 1, 1),
#             nn.Conv2d(dim//4, dim, 1, 1),
#             nn.Sigmoid()
#         )

#         self.y_channel = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // 4, 1, 1),
#             nn.Conv2d(dim // 4, dim, 1, 1),
#             nn.Sigmoid()
#         )

#         #
#         self.x_out = nn.Conv2d(dim, x_ch, 1, 1)
#         self.y_out = nn.Conv2d(dim, y_ch, 1, 1)

#     def forward(self, fes):
#         x, y = fes
#         x_hidden = self.x_map_conv(x)
#         y_hidden = self.y_map_conv(y)

#         #channel
#         x_channel = self.x_channel(x_hidden)
#         y_channel = self.y_channel(y_hidden)
#         x_hidden = y_channel * x_hidden
#         y_hidden = x_channel * y_hidden
#         #spatial
#         x_max = torch.max(x_hidden, dim=1, keepdim=True)[0]
#         x_avg = torch.mean(x_hidden, dim=1, keepdim=True)
#         x_spatial = torch.cat([x_max, x_avg], dim=1)

#         x_spatial = self.x_spatial(x_spatial)

#         y_max = torch.max(y_hidden, dim=1, keepdim=True)[0]
#         y_avg = torch.mean(y_hidden, dim=1, keepdim=True)
#         y_spatial = torch.cat([y_max, y_avg], dim=1)
#         y_spatial = self.y_spatial(y_spatial)
#         x_hidden = x_hidden * y_spatial
#         y_hidden = y_hidden * x_spatial

#         x = self.x_out(x_hidden) + x
#         y = self.y_out(y_hidden) + y
#         return x, y
