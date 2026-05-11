import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class ResidualBlock3D(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        assert channels % num_groups == 0

        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, 3, 1, 1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        return x + out


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        avg = x.mean(dim=(2,3,4))
        maxv = x.amax(dim=(2,3,4))
        attn = self.mlp(avg) + self.mlp(maxv)
        attn = torch.sigmoid(attn).view(b, c, 1, 1, 1)
        return x * attn

class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maxv, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * attn

class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention3D(channels)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ResBlock3D_with_CBAM(nn.Module): 
    def __init__(self, in_channels=18, embed_dim=384, base_channels=64):
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # ---- Stage 1 ----
        self.stage1 = nn.Sequential(
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels),
            CBAM3D(base_channels)
        )

        # ---- Downsample ----
        self.downsample = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # ---- Stage 2 ----
        self.stage2 = nn.Sequential(
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels),
            CBAM3D(base_channels)
        )

        # ---- Channel lift ----
        self.proj = nn.Conv3d(base_channels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)        
        x = self.stage1(x)
        x = self.downsample(x) 
        x = self.stage2(x)
        x = self.proj(x)       
        return x
