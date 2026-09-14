from torch import nn
from einops import rearrange
from monai.utils import ensure_tuple_rep


class PixelShuffle(nn.Module):
    def __init__(self, scale, spatial_dim=3):
        super().__init__()
        self.scale = ensure_tuple_rep(scale, spatial_dim)
        self.spatial_dim = spatial_dim

    def forward(self, x):
        if self.spatial_dim == 2:
            return rearrange(x, 'b (c s1 s2) h w -> b c (h s1) (w s2)',
                             s1=self.scale[0], s2=self.scale[1])
        return rearrange(x, 'b (c s1 s2 s3) d h w -> b c (d s1) (h s2) (w s3)',
                         s1=self.scale[0], s2=self.scale[1], s3=self.scale[2])
