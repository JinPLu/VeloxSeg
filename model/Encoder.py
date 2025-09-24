from typing import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbed

# Core components for the VeloxSeg encoder architecture
from .components.attention_utils import LayerNorm     # Layer normalization
from .components.PWA import Transformer_BasicLayer    # Paired Window Attention blocks
from .components.conv_blocks import JLCLayer, DownConv  # Johnson-Lindenstrauss lemma-guided convolution & downsampling
from .components.common_function import get_conv, get_norm  # Utility functions


class Conv_Encoder(nn.Module):
    """
    Modal-Fusion Convolution Layer (Top Branch)
    
    Architecture Components:
    - Patch Embedding for input processing
    - JLC blocks (Group Gather → GC 1 to GC N → Group Scatter)
    - FFN (Feed-Forward Network) layers
    - Repeated x4 times as shown in the diagram
    
    Args:
        patch_size: Size of patches for embedding
        in_ch: Number of input channels (sum of all modalities)
        base_ch: Base number of channels
        depths: Number of JLC blocks at each level
        kernel_sizes: Kernel sizes for group convolutions (parallel kernels: [1,3,5])
        min_dim_group: Lower bound of group size for convolutions (JL-guided: [4,8,8,16])
        expansion_factor: Expansion factors for FFN
        dropout: Dropout rate
        spatial_dim: Spatial dimensions (2D or 3D)
    """
    def __init__(self, 
                patch_size: int = 4, 
                in_ch: int = 1, 
                base_ch: int = 16, 
                depths: Sequence[int] = [1, 1, 1, 1],
                kernel_sizes: Sequence[int] = [1, 3, 5],
                min_dim_group: Sequence[int] = [4, 8, 8, 16],
                expansion_factor: Sequence[int] = [3, 3, 2, 2],
                
                dropout: float = 0.0,
                spatial_dim: int = 3):
        super(Conv_Encoder, self).__init__()
            
        # Downsampling layers (Patch Embed blocks in diagram)
        self.down1 = DownConv(in_ch,       base_ch    , patch_size=patch_size, dim=spatial_dim)
        self.down2 = DownConv(base_ch,     base_ch * 2, patch_size=2, dim=spatial_dim)
        self.down3 = DownConv(base_ch * 2, base_ch * 4, patch_size=2, dim=spatial_dim)
        self.down4 = DownConv(base_ch * 4, base_ch * 8, patch_size=2, dim=spatial_dim)

        # JLC (Johnson-Lindenstrauss lemma-guided Convolution) blocks - core components of Modal-Fusion branch
        # Each JLC block contains: Group Gather → GC 1 to GC N → Group Scatter → Add & Norm → FFN → Add & Norm
        # Group sizes are determined by JL lemma: C_group ≥ c_JL * ε^(-2) * log N(M,v)
        # where N(M,v) is approximated as (M·v)^α for modality count M and volume ratio v
        groups = [base_ch * 2 ** i // min_dim_group[i] for i in range(4)]
        self.layer1 = JLCLayer(base_ch  , depths[0], kernel_sizes, groups[0], expansion_factor[0], 
                                   dropout=dropout, spatial_dim=spatial_dim)
        self.layer2 = JLCLayer(base_ch*2, depths[1], kernel_sizes, groups[1], expansion_factor[1], 
                                   dropout=dropout, spatial_dim=spatial_dim)
        self.layer3 = JLCLayer(base_ch*4, depths[2], kernel_sizes, groups[2], expansion_factor[2], 
                                   dropout=dropout, spatial_dim=spatial_dim)
        self.layer4 = JLCLayer(base_ch*8, depths[3], kernel_sizes, groups[3], expansion_factor[3], 
                                   dropout=dropout, spatial_dim=spatial_dim)

    
    def forward(self, x) -> Sequence[torch.Tensor]:
        # Level 1
        x = self.down1(x)
        enc1 = self.layer1(x)
        
        # Level 2
        x = self.down2(enc1)
        enc2 = self.layer2(x)
        
        # Level 3
        x = self.down3(enc2)
        enc3 = self.layer3(x)
        
        # Level 4
        x = self.down4(enc3)
        enc4 = self.layer4(x)
        
        return enc1, enc2, enc3, enc4


class Transformer_Encoder(nn.Module):
    """
    Modal-Cooperative Transformer Layer (Bottom Branch)
    
    Architecture Components:
    - Patch Embedding for input processing
    - PWA blocks (Linear Proj. → PW. Gather → MultiModal Grouped Attention x 1 → PW. Scatter → PW. Mixer)
    - FFN (Feed-Forward Network) layers
    - Repeated x4 times as shown in the diagram
    
    Args:
        input_size: Input spatial dimensions
        patch_size: Size of patches for embedding
        in_channels: Number of input channels per modality
        embed_dim: Embedding dimension
        depths: Number of PWA blocks at each level
        min_big_window_sizes: Window sizes for attention at each level
        min_small_window_sizes: Small window sizes for attention
        scale_factors: Downsampling factors
        num_heads: Number of attention heads
        min_dim_head: Lower bound of head size for PWA (JL-guided)
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        drop_path: Drop path rate
        ffn_expansion_ratio: FFN expansion ratios
        act_layer: Activation function
        norm_layer: Normalization layer type
        patch_norm: Whether to apply normalization after patch embedding
        qkv_bias: Whether to use bias in QKV projection
        spatial_dim: Spatial dimensions (2D or 3D)
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_channels: Sequence[int],
        embed_dim: int = 16,
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [6, 6, 6], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 2, 4],
        min_dim_head: Sequence[int] = [4, 8, 8, 16],
        ffn_expansion_ratio: Sequence[int] = [3, 3, 2, 2],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = LayerNorm,
        patch_norm: bool = False,
        qkv_bias: bool = True,
        spatial_dim: str = 3
    ) -> None:

        super(Transformer_Encoder, self).__init__()
        self.in_channels = in_channels
        self.num_modalities = len(in_channels)
        self.num_layers = len(depths)

        self.patch_size = patch_size
        # Patch embedding layers for each modality (Patch Embed blocks in diagram)
        self.patch_embeds = nn.ModuleList([PatchEmbed(
                                    patch_size      = self.patch_size,
                                    in_chans        = self.in_channels[m],
                                    embed_dim       = embed_dim,
                                    norm_layer      = norm_layer if patch_norm else None,
                                    spatial_dims    = spatial_dim,
                                    ) for m in range(self.num_modalities)])
        
        # Positional dropout
        self.pos_drop = nn.Dropout(p=proj_drop)
        # Drop path rates for stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.layers = nn.ModuleList()
        
        # Initialize PWA layers for each level (x4 repetitions in diagram)
        input_size = torch.tensor(input_size) // self.patch_size
        for i_layer in range(self.num_layers):
            self.layers.append(Transformer_BasicLayer(
                input_size              = input_size.tolist(),
                in_channels             = [int(embed_dim * 2**i_layer)] * self.num_modalities,
                depth                   = depths[i_layer],
                min_big_window_size     = min_big_window_sizes[i_layer],
                min_small_window_size   = min_small_window_sizes[i_layer],
                scale_factor            = scale_factors[i_layer],
                num_heads               = num_heads[i_layer],
                min_dim_head            = min_dim_head[i_layer],
                attn_drop               = attn_drop,
                proj_drop               = proj_drop,
                drop_path               = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                ffn_expansion_ratio     = ffn_expansion_ratio[i_layer],
                act_layer               = act_layer,
                norm_layer              = norm_layer,
                qkv_bias                = qkv_bias,
                do_downsample           = i_layer < self.num_layers - 1,
                dim                     = spatial_dim,
            ))
            # Downsample input size for next level
            input_size = input_size // 2

    def forward(self, xs) -> Sequence[torch.Tensor]:
        # Split input into separate modalities
        xs = torch.chunk(xs, self.num_modalities, dim=1)
        # Apply patch embedding to each modality
        xs = [self.patch_embeds[m](xs[m]) for m in range(self.num_modalities)]
        # Apply positional dropout
        xs = [self.pos_drop(xs[m]) for m in range(self.num_modalities)]

        # Process through 4 levels of PWA blocks (x4 repetitions in diagram)
        attn1, down = self.layers[0]([x.contiguous() for x in xs])
        attn2, down = self.layers[1]([d.contiguous() for d in down])
        attn3, down = self.layers[2]([d.contiguous() for d in down])
        attn4, _ = self.layers[3]([d.contiguous() for d in down])
        
        return attn1, attn2, attn3, attn4


class Encoder(nn.Module):
    """
    Main Encoder: Dual-Stream CNN-Transformer Architecture
    
    The dual-branch architecture enables efficient multimodal modeling:
    - Top Branch: Modal-Fusion Convolution Layer with JLC (robust local features)
    - Bottom Branch: Modal-Cooperative Transformer Layer with PWA (global context)
    - Cross-modal fusion via Add Operation (⊕) between branches
    - Modal Mixer for information exchange at multiple scales
    
    Key Benefits:
    - Avoids parameter explosion as modality count increases
    - Maximizes advantages and parallelism of both architectures
    - Enables complementary feature extraction and cross-modal interaction
    - Achieves low-cost but effective modal interaction (only +0.27 MParams, +0.09 GFLOPs)
    
    Args:
        input_size: Input spatial dimensions
        patch_size: Size of patches for embedding
        in_ch: Number of input channels per modality
        base_ch: Base number of channels for convolution branch
        conv_depths: Number of JLC blocks at each level in conv branch
        kernel_sizes: Kernel sizes for group convolutions (parallel: [1,3,5])
        min_dim_group: Lower bound of group size for convolutions (JL-guided: [4,8,8,16])
        conv_expansion_factor: Expansion factors for FFN in conv branch
        attn_base_ch: Base number of channels for attention branch
        depths: Number of PWA blocks at each level in attention branch
        min_big_window_sizes: Window sizes for attention at each level
        min_small_window_sizes: Small window sizes for attention
        min_dim_head: Lower bound of head size for PWA (JL-guided)
        scale_factors: Downsampling factors
        num_heads: Number of attention heads
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        drop_path: Drop path rate
        ffn_expansion_ratio: FFN expansion ratios
        act_layer: Activation function
        norm_layer: Normalization layer type
        patch_norm: Whether to apply normalization after patch embedding
        qkv_bias: Whether to use bias in QKV projection
        conv_drop: Dropout rate for convolution branch
        spatial_dim: Spatial dimensions (2D or 3D)
        output_attn: Whether to output attention features for knowledge distillation (SDKT)
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        in_ch: Sequence[int],
        base_ch: int = 16,
        
        conv_depths: Sequence[int] = [1, 1, 1, 1],
        kernel_sizes: Sequence[int] = [1, 3, 5],
        min_dim_group: Sequence[int] = [4, 8, 8, 16],
        conv_expansion_factor: Sequence[int] = [4, 4, 4, 4],
        
        attn_base_ch: int = 16,
        depths: Sequence[int] = [2, 2, 2, 2],
        min_big_window_sizes: Sequence[Sequence[int]] = [[3, 3, 3], [6, 6, 6], [3, 3, 3], [3, 3, 3]],
        min_small_window_sizes: Sequence[Sequence[int]] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        min_dim_head: Sequence[int] = [4, 8, 8, 16],
        scale_factors: Sequence[int] = [2, 2, 2, 2],
        num_heads: Sequence[int] = [1, 2, 4, 8],
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0,
        ffn_expansion_ratio: Sequence[int] = [4, 4, 4, 4],
        act_layer: str = "GELU",
        norm_layer = LayerNorm,
        patch_norm: bool = False,
        qkv_bias: bool = True,
        conv_drop: float = 0.0,
        spatial_dim: str = 3,
    ):

        super(Encoder, self).__init__()
        
        conv = get_conv(spatial_dim)
        norm = get_norm("IN", spatial_dim)
        
        self.in_channels = in_ch
        self.num_modalities = len(in_ch)
        
        # Bottom Branch: Modal-Cooperative Transformer Layer
        # Processes multimodal inputs through PWA blocks and provides Modal Mixer features
        self.encoder_attn = Transformer_Encoder(
            input_size              = input_size,
            patch_size              = patch_size,
            in_channels             = in_ch,
            embed_dim               = attn_base_ch,
            depths                  = depths,
            min_big_window_sizes    = min_big_window_sizes,
            min_small_window_sizes  = min_small_window_sizes,
            scale_factors           = scale_factors,
            num_heads               = num_heads,
            min_dim_head            = min_dim_head,
            attn_drop               = attn_drop,
            proj_drop               = proj_drop,
            drop_path               = drop_path,
            ffn_expansion_ratio     = ffn_expansion_ratio,
            act_layer               = act_layer,
            norm_layer              = norm_layer,
            patch_norm              = patch_norm,
            qkv_bias                = qkv_bias,
            spatial_dim             = spatial_dim
        )
        
        # Top Branch: Modal-Fusion Convolution Layer
        # Processes multimodal inputs through JLC blocks
        self.encoder_conv = Conv_Encoder(
            patch_size      = patch_size, 
            in_ch           = sum(in_ch), 
            base_ch         = base_ch, 
            
            depths          = conv_depths,
            kernel_sizes    = kernel_sizes,
            min_dim_group   = min_dim_group,
            expansion_factor= conv_expansion_factor,
            
            dropout         = conv_drop,
            spatial_dim     = spatial_dim
        )
        
        # Cross-modal fusion layers (Add Operation ⊕ in diagram)
        # These layers enable information exchange between transformer and convolution branches
        # Using 1x1 convolution as modal mixer to facilitate efficient modal interaction
        self.attn2conv_1 = nn.Sequential(conv(attn_base_ch     * self.num_modalities, base_ch    , 1, 1), norm(base_ch))
        self.attn2conv_2 = nn.Sequential(conv(attn_base_ch * 2 * self.num_modalities, base_ch * 2, 1, 1), norm(base_ch * 2))
        self.attn2conv_3 = nn.Sequential(conv(attn_base_ch * 4 * self.num_modalities, base_ch * 4, 1, 1), norm(base_ch * 4))
        self.attn2conv_4 = nn.Sequential(conv(attn_base_ch * 8 * self.num_modalities, base_ch * 8, 1, 1), norm(base_ch * 8))
        
    def forward(self, x) -> Sequence[torch.Tensor]:
        # Process through Modal-Cooperative Transformer Layer (bottom branch)
        attn1_, attn2_, attn3_, attn4_ = self.encoder_attn(x)
        
        # Convert attention features to convolution feature space for fusion
        attn1 = self.attn2conv_1(torch.cat(attn1_, dim=1))
        attn2 = self.attn2conv_2(torch.cat(attn2_, dim=1))
        attn3 = self.attn2conv_3(torch.cat(attn3_, dim=1))
        attn4 = self.attn2conv_4(torch.cat(attn4_, dim=1))
        
        # Cross-modal fusion via Add Operation (⊕) at each level
        # This implements the Modal Mixer mechanism from the framework diagram
        x = self.encoder_conv.down1(x) + attn1  # Level 1 fusion
        enc1 = self.encoder_conv.layer1(x)
        
        x = self.encoder_conv.down2(enc1) + attn2  # Level 2 fusion
        enc2 = self.encoder_conv.layer2(x)
        
        x = self.encoder_conv.down3(enc2) + attn3  # Level 3 fusion
        enc3 = self.encoder_conv.layer3(x)
        
        x = self.encoder_conv.down4(enc3) + attn4  # Level 4 fusion
        enc4 = self.encoder_conv.layer4(x)
        
        # Return attention features for knowledge distillation (SDKT) if needed
        if self.training:
            return [attn1_, attn2_, attn3_, attn4_], [enc1, enc2, enc3, enc4]
        else:
            return enc1, enc2, enc3, enc4
