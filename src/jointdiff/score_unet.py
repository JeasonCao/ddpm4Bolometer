"""
Unconditional 1-channel score UNet for Joint Diffusion.

Reuses the UNet1D architecture from ddpm but with a single-channel input
(no conditioning signal). Each score network learns one distribution
independently: p(signal) or p(noise).

Usage:
    from src.jointdiff.score_unet import ScoreUNet1D
    model = ScoreUNet1D()
    eps_pred = model(x_t, t)  # (B, 1, L) -> (B, 1, L)
"""

import math
import torch
import torch.nn as nn

from src.ddpm.unet import (
    TimestepMLP, ResBlock1D, SelfAttention1D,
    Downsample1D, Upsample1D,
)


class ScoreUNet1D(nn.Module):
    """Unconditional 1D U-Net for score estimation.

    Same architecture as ddpm.UNet1D but with 1-channel input
    and no conditioning signal.

    Parameters
    ----------
    base_channels : int
        Channels at level 0.
    channel_mults : tuple
        Channel multipliers per level.
    emb_dim : int
        Timestep embedding dimension.
    attn_levels : tuple
        Which encoder levels get self-attention.
    """

    def __init__(
        self,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        emb_dim: int = 256,
        attn_levels: tuple = (),
    ):
        super().__init__()
        self.n_levels = len(channel_mults)
        channels = [base_channels * m for m in channel_mults]

        self.time_mlp = TimestepMLP(emb_dim)
        self.conv_in = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        self.enc_attns = nn.ModuleList()

        for i in range(self.n_levels):
            ch_in = channels[i - 1] if i > 0 else channels[0]
            ch_out = channels[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim),
                ResBlock1D(ch_out, ch_out, emb_dim),
            ]))
            if i in attn_levels:
                self.enc_attns.append(SelfAttention1D(ch_out))
            else:
                self.enc_attns.append(nn.Identity())
            if i < self.n_levels - 1:
                self.enc_downsamples.append(Downsample1D(ch_out))
            else:
                self.enc_downsamples.append(nn.Identity())

        # Bottleneck
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock1D(mid_ch, mid_ch, emb_dim)
        self.mid_attn = SelfAttention1D(mid_ch)
        self.mid_block2 = ResBlock1D(mid_ch, mid_ch, emb_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        for i in reversed(range(self.n_levels)):
            ch_out = channels[i]
            ch_in = ch_out + channels[i]
            ch_target = channels[i - 1] if i > 0 else channels[0]

            self.dec_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim),
                ResBlock1D(ch_out, ch_target, emb_dim),
            ]))
            if i in attn_levels:
                self.dec_attns.append(SelfAttention1D(ch_out))
            else:
                self.dec_attns.append(nn.Identity())
            if i > 0:
                self.dec_upsamples.append(Upsample1D(ch_target))
            else:
                self.dec_upsamples.append(nn.Identity())

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 1, kernel_size=3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, L) -- noisy signal at step t
        t : (B,) -- timestep indices (1..T)

        Returns
        -------
        eps_pred : (B, 1, L) -- predicted noise
        """
        t_emb = self.time_mlp(t)
        h = self.conv_in(x_t)

        skips = []
        for i in range(self.n_levels):
            for block in self.enc_blocks[i]:
                h = block(h, t_emb)
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < self.n_levels - 1:
                h = self.enc_downsamples[i](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for j, i in enumerate(reversed(range(self.n_levels))):
            skip = skips[i]
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            res1, res2 = self.dec_blocks[j]
            h = res1(h, t_emb)
            h = self.dec_attns[j](h)
            h = res2(h, t_emb)
            if i > 0:
                h = self.dec_upsamples[j](h)

        return self.conv_out(h)
