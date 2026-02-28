"""
Image Dehazing Model Architectures.
Implementations of DehazeNet, AODNet, and other architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DehazeNet(nn.Module):
    """
    DehazeNet - Lightweight CNN for image dehazing.
    Reference: Li et al., "DehazeNet: An End-to-End System for Single Image Haze Removal"
    """
    
    def __init__(self, in_channels=3, depth=20):
        super(DehazeNet, self).__init__()
        self.depth = depth
        
        # Feature extraction
        self.feat_extract = nn.Conv2d(in_channels, 16, 3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ConvBlock(16, 16, 3, 1) for _ in range(depth)
        ])
        
        # Reconstruction
        self.feat_recon = nn.Conv2d(16, in_channels, 3, padding=1)
    
    def forward(self, x):
        # Feature extraction
        feat = self.feat_extract(x)
        res = feat
        
        # Residual learning
        for block in self.residual_blocks:
            feat = block(feat)
        
        # Add residual connection
        feat = feat + res
        
        # Reconstruction
        out = self.feat_recon(feat)
        
        return out


class AODNet(nn.Module):
    """
    All-in-One Dehazing Network (AODNet).
    Reference: Li et al., "AOD-Net: All-in-One Dehazing Network"
    
    Uses multi-scale features and joint transmission/airlight estimation.
    """
    
    def __init__(self, in_channels=3):
        super(AODNet, self).__init__()
        
        # Initial feature extraction
        self.relu = nn.ReLU(inplace=True)
        
        # Multi-scale feature extraction
        self.conv1_1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Reconstruction
        self.deconv3_1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(128, 64, 3, padding=1)
        
        self.deconv2_1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv2_3 = nn.Conv2d(64, 32, 3, padding=1)
        
        self.conv1_3 = nn.Conv2d(32, in_channels, 3, padding=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1_1(x))
        x1 = self.relu(self.conv1_2(x1))
        
        x2 = self.relu(self.conv2_1(x1))
        x2 = self.relu(self.conv2_2(x2))
        
        x3 = self.relu(self.conv3_1(x2))
        x3 = self.relu(self.conv3_2(x3))
        
        # Decoder with skip connections
        d3 = self.relu(self.deconv3_1(x3))
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.relu(self.conv3_3(d3))
        
        d2 = self.relu(self.deconv2_1(d3))
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.relu(self.conv2_3(d2))
        
        out = self.conv1_3(d2)
        
        # Skip connection: add back original input
        out = out + x
        
        return out


class PFFNet(nn.Module):
    """
    Progressive Fusion Framework Network (PFFNet).
    Uses progressive refinement with multiple outputs.
    """
    
    def __init__(self, in_channels=3, depth=8):
        super(PFFNet, self).__init__()
        self.depth = depth
        
        # Initial processing
        self.initial = nn.Conv2d(in_channels, 32, 3, padding=1)
        
        # Progressive blocks
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            block = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.blocks.append(block)
        
        # Output layers
        self.outputs = nn.ModuleList([
            nn.Conv2d(32, in_channels, 1) for _ in range(depth)
        ])
    
    def forward(self, x):
        feat = self.initial(x)
        
        # Progressive refinement
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            # Could return intermediate outputs for auxiliary supervision
        
        # Final output
        out = self.outputs[-1](feat)
        out = out + x  # Skip connection
        
        return out


class DarkChannelPrior(nn.Module):
    """
    Classical Dark Channel Prior (DCP) based approach.
    Non-neural baseline for comparison.
    """
    
    def __init__(self, patch_size=15, topk=0.001):
        super(DarkChannelPrior, self).__init__()
        self.patch_size = patch_size
        self.topk = topk
    
    def get_dark_channel(self, img):
        """Compute dark channel prior."""
        b, c, h, w = img.shape
        
        # Min pooling
        pad = self.patch_size // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        
        dark = torch.min(img_padded.unfold(2, self.patch_size, 1)
                                   .unfold(3, self.patch_size, 1),
                        dim=4)[0]
        
        return torch.min(dark, dim=1)[0]
    
    def forward(self, x):
        # This is a placeholder - actual DCP requires iterative optimization
        # For now, return input as-is
        return x


class DehazeFormer(nn.Module):
    """
    DehazeFormer - Transformer-based model for image dehazing.
    Uses multi-scale transformer blocks with efficient attention mechanisms.
    Reference: Song et al., "Vision Transformers for Image Restoration"
    """
    
    def __init__(self, in_channels=3, num_blocks=12, embed_dim=64, num_heads=8, 
                 mlp_ratio=4.0, window_size=8):
        super(DehazeFormer, self).__init__()
        
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, window_size)
            for _ in range(num_blocks)
        ])
        
        # Reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Feature extraction
        feat = self.initial_conv(x)
        residual = feat
        
        # Transformer processing
        for block in self.transformer_blocks:
            feat = block(feat)
        
        # Add residual connection
        feat = feat + residual
        
        # Reconstruction
        out = self.reconstruct(feat)
        
        # Skip connection: add back original input
        out = out + x
        
        return out


class TransformerBlock(nn.Module):
    """Multi-head self-attention transformer block with MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, window_size=8):
        super(TransformerBlock, self).__init__()
        
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Input: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape to (B, H, W, C) for windowed processing
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # Apply normalization and attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, H, W)
        x = x + attn_out
        
        # Apply normalization and MLP
        x_norm = self.norm2(x)
        x_flat = x_norm.reshape(B, -1, C)
        mlp_out = self.mlp(x_flat)
        mlp_out = mlp_out.reshape(B, H, W, C)
        x = x + mlp_out
        
        # Reshape back: (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        return x


class WindowAttention(nn.Module):
    """Efficient windowed multi-head self-attention."""
    
    def __init__(self, dim, window_size=8, num_heads=8):
        super(WindowAttention, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, H, W):
        # Input: (B, H, W, C)
        B, H, W, C = x.shape
        
        # Pad to be divisible by window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad = H + pad_h
        W_pad = W + pad_w
        
        # Reshape into windows
        x = x.reshape(B, H_pad // self.window_size, self.window_size, 
                     W_pad // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B * (H_pad // self.window_size) * (W_pad // self.window_size),
            self.window_size * self.window_size, C)
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(
            -1, self.window_size * self.window_size, C)
        out = self.proj(out)
        
        # Reshape back to image
        out = out.reshape(B, H_pad // self.window_size, W_pad // self.window_size,
                         self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)
        
        # Remove padding
        if pad_h > 0:
            out = out[:, :H, :, :]
        if pad_w > 0:
            out = out[:, :, :W, :]
        
        return out


def create_model(model_name: str = "AODNet", **kwargs):
    """Factory function to create models."""
    models = {
        "DehazeNet": DehazeNet,
        "AODNet": AODNet,
        "PFFNet": PFFNet,
        "DCP": DarkChannelPrior,
        "DehazeFormer": DehazeFormer
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sample input
    x = torch.randn(1, 3, 256, 256).to(device)
    
    # Test each model
    for model_name in ["AODNet", "DehazeNet", "PFFNet", "DehazeFormer"]:
        model = create_model(model_name).to(device)
        output = model(x)
        print(f"{model_name}: Input {x.shape} -> Output {output.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}\n")
