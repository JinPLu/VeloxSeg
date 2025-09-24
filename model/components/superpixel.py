from torch import nn
from einops import rearrange

class PixelShuffle(nn.Module):

    def __init__(self, scale, spatial_dim=3):

        super().__init__()
        self.scale = scale
        self.spatial_dim = spatial_dim

    def forward(self, x):
        if self.spatial_dim == 2:
            return rearrange(x, 'b (c s1 s2) h w -> c (h s1) (w s2)', s1=self.scale, s2=self.scale)
        elif self.spatial_dim == 3:
            return rearrange(x, 'b (c s1 s2 s3) d h w -> b c (d s1) (h s2) (w s3)', s1=self.scale, s2=self.scale, s3=self.scale)
        else:
            raise ValueError("spatial_dim should be 2 or 3")