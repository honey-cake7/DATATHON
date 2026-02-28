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


def create_model(model_name: str = "AODNet", **kwargs):
    """Factory function to create models."""
    models = {
        "DehazeNet": DehazeNet,
        "AODNet": AODNet,
        "PFFNet": PFFNet,
        "DCP": DarkChannelPrior
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
    for model_name in ["AODNet", "DehazeNet", "PFFNet"]:
        model = create_model(model_name).to(device)
        output = model(x)
        print(f"{model_name}: Input {x.shape} -> Output {output.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}\n")
