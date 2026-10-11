import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, out_channels)
        self.enc2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d((2, 2, 2))  # halves x y z

    def forward(self, x):
        h = self.enc1(x)
        h = self.enc2(h)
        p = self.pool(h)
        return h, p


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(out_channels*2, out_channels)
        self.dec2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        h = self.upconv(x)
        h = torch.cat([h, skip], axis=1)
        h = self.dec1(h)
        h = self.dec2(h)
        return h


class UNET(nn.Module):

    def __init__(self, in_channels=3, out_channels=5):
        super().__init__()
        self.e1 = Encoder(in_channels, 64)  # 32 32 8
        self.e2 = Encoder(64, 128)  # 16 16 4
        self.e3 = Encoder(128, 256)  # 8 8 2
        self.e4 = Encoder(256, 512)  # 4 4 1

        self.bottleneck = ConvBlock(512, 1024)  # 4, 4, 1

        self.d1 = Decoder(1024, 512)  # 16 16 4
        self.d2 = Decoder(512, 256)  # 32 32 8
        self.d3 = Decoder(256, 128)  # 32 32 8
        self.d4 = Decoder(128, 64)  # 32 32 8
        self.output_conv = nn.Conv3d(
            64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 3, 0, 3, 0, 4))  # pad to 64 64 16

        skip1, x = self.e1(x)
        skip2, x = self.e2(x)
        skip3, x = self.e3(x)
        skip4, x = self.e4(x)

        x = self.bottleneck(x)

        x = self.d1(x, skip4)
        x = self.d2(x, skip3)
        x = self.d3(x, skip2)
        x = self.d4(x, skip1)
        x = self.output_conv(x)
        x = x[:, :, :12, :61, :61]  # crop to 12 61 61
        return x
