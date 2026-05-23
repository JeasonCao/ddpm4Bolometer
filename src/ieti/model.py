"""
model.py — 1D U-Net for waveform denoising.

Processes signals at 5 resolutions simultaneously (zoom in/out),
combining global context (slow drift) with local detail (sharp noise features)
via skip connections.

Single channel in/out — same model trained on both LMO and DLMO.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv1d → BatchNorm → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock + MaxPool1d(2) for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, skip_connection)."""
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + out_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net for waveform denoising.

    Architecture (default depth=4, base_channels=32):
        Encoder:
          Input        (B,   1, 32768)
          Enc0 Conv    (B,  32, 32768) → MaxPool → (B,  32, 16384)  [skip0]
          Enc1 Conv    (B,  64, 16384) → MaxPool → (B,  64,  8192)  [skip1]
          Enc2 Conv    (B, 128,  8192) → MaxPool → (B, 128,  4096)  [skip2]
          Enc3 Conv    (B, 256,  4096) → MaxPool → (B, 256,  2048)  [skip3]
        Bridge:
          Conv         (B, 512,  2048)
        Decoder:
          Up + cat(skip3) → (B, 256, 4096)
          Up + cat(skip2) → (B, 128, 8192)
          Up + cat(skip1) → (B,  64, 16384)
          Up + cat(skip0) → (B,  32, 32768)
        Output:
          Conv1d(32→1, k=1) → (B, 1, 32768)   [linear, no activation]

    ~7.85M parameters, ~520 MB VRAM for training (batch=8).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(DownBlock(ch, out_ch, kernel_size))
            ch = out_ch

        # Bridge
        bridge_ch = base_channels * (2 ** depth)
        self.bridge = ConvBlock(ch, bridge_ch, kernel_size)

        # Decoder
        self.decoders = nn.ModuleList()
        ch = bridge_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            self.decoders.append(UpBlock(ch, out_ch, kernel_size))
            ch = out_ch

        # Output
        self.output = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bridge
        x = self.bridge(x)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
