import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels, hidden_dim):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2)  # need gamma and beta
        )

        self.out_channels = out_channels

    def forward(self, x, cond):
        # x: (B, C, D, H, W)
        # cond: (B, cond_dim)
        B, C, D, H, W = x.shape
        gamma_beta = self.film(cond)  # (B, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B, C), (B, C)
        # (B, C, 1, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # (B, C, 1, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return gamma, beta


class FiLMConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.film = FiLM(cond_dim=6, out_channels=out_channels, hidden_dim=128)
        self.gelu = nn.GELU()

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.bn(x)
        g, b = self.film(x, cond)
        x = g * x + b
        x = self.gelu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = FiLMConvBlock(in_channels, out_channels)
        self.enc2 = FiLMConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d((2, 2, 2))  # halves x y z

    def forward(self, x, cond):
        h = self.enc1(x, cond)
        h = self.enc2(h, cond)
        p = self.pool(h)
        return h, p


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.dec1 = FiLMConvBlock(out_channels*2, out_channels)
        self.dec2 = FiLMConvBlock(out_channels, out_channels)

    def forward(self, x, skip, cond):
        h = self.upconv(x)
        h = torch.cat([h, skip], axis=1)
        h = self.dec1(h, cond)
        h = self.dec2(h, cond)
        return h


class ScalarDecoder(nn.Module):
    def __init__(self, in_channels, out_scalars, hidden_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_scalars)
        )

    def forward(self, x):
        x = self.gap(x).flatten(1)  # (B, C)
        x = self.mlp(x)
        return x


class MultiHeadUNET(nn.Module):

    def __init__(self, in_channels=2, grid_out_channels=2, scalar_out_channels=3):
        super().__init__()
        self.e1 = Encoder(in_channels, 64)  # 32 32 8
        self.e2 = Encoder(64, 128)  # 16 16 4
        self.e3 = Encoder(128, 256)  # 8 8 2
        self.e4 = Encoder(256, 512)  # 4 4 1

        self.bottleneck = FiLMConvBlock(512, 1024)  # 4, 4, 1

        self.d1 = Decoder(1024, 512)  # 16 16 4
        self.d2 = Decoder(512, 256)  # 32 32 8
        self.d3 = Decoder(256, 128)  # 32 32 8
        self.d4 = Decoder(128, 64)  # 32 32 8

        self.output_conv = nn.Conv3d(
            64, grid_out_channels, kernel_size=1, padding=0)

        self.scalars_decoder = ScalarDecoder(
            1024, scalar_out_channels, hidden_dim=128)

    def forward(self, x, cond):
        x = F.pad(x, (0, 3, 0, 3, 0, 4))  # pad to 64 64 16

        skip1, x = self.e1(x, cond)
        skip2, x = self.e2(x, cond)
        skip3, x = self.e3(x, cond)
        skip4, x = self.e4(x, cond)

        bottleneck_out = self.bottleneck(x, cond)

        x = self.d1(bottleneck_out, skip4, cond)
        x = self.d2(x, skip3, cond)
        x = self.d3(x, skip2, cond)
        x = self.d4(x, skip1, cond)
        x = self.output_conv(x)
        x = x[:, :, :12, :61, :61]  # crop to 12 61 61

        scalar_out = self.scalars_decoder(bottleneck_out)
        return x, scalar_out
