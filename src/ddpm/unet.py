"""
1D U-Net for DDPM denoising of bolometer pulse signals.

Architecture:
    Input: [x_t, x_tilde] concatenated -> 2 channels, 10000 samples
    Encoder: 4 levels (64 -> 128 -> 256 -> 512), stride-2 Conv1d
    Bottleneck: 512 channels with self-attention
    Decoder: mirror with skip connections
    Attention: at level 3 (1250 samples) and bottleneck (625 samples)
    Timestep: sinusoidal embedding -> MLP -> added into each ResBlock
    Output: 1 channel (predicted noise epsilon)

Usage:
    from src.ddpm.unet import UNet1D
    model = UNet1D()
    eps_pred = model(x_t, x_tilde, t)
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for a diffusion conditioning scalar.

    Two modes (selected at construction):
      - 'step':    input is an integer step index (1..T). No rescaling.
      - 'sqrt_ab': input is a continuous noise level sqrt(alpha_bar) in (0, 1].
                   Multiplied by `cond_scale` to spread small values across the
                   sinusoidal frequency grid (WaveGrad / DeScoD-ECG style).
    """

    def __init__(self, dim: int, cond_mode: str = 'step', cond_scale: float = 1000.0):
        super().__init__()
        assert cond_mode in ('step', 'sqrt_ab'), f"unknown cond_mode: {cond_mode}"
        self.dim = dim
        self.cond_mode = cond_mode
        self.cond_scale = cond_scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) — step indices (1..T) if cond_mode='step',
                   or continuous sqrt(alpha_bar) values if cond_mode='sqrt_ab'.

        Returns
        -------
        emb : (B, dim)
        """
        t_f = t.float()
        if self.cond_mode == 'sqrt_ab':
            t_f = t_f * self.cond_scale
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t_f.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class TimestepMLP(nn.Module):
    """Sinusoidal embedding -> MLP -> per-block conditioning vector."""

    def __init__(self, emb_dim: int = 256, cond_mode: str = 'step',
                 cond_scale: float = 1000.0):
        super().__init__()
        self.embed = SinusoidalTimestepEmbedding(emb_dim, cond_mode, cond_scale)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))  # (B, emb_dim)


class ScaleEmbedding(nn.Module):
    """Normalization scale max(|noisy|) -> conditioning vector.

    log10(scale) is mapped linearly so that ``log10_range`` (default 10 mV to
    10 V, the span of noise-only to MeV windows) lands on [-1, 1]; values
    outside are not clamped. Fourier features [x, sin(pi 2^k x), cos(pi 2^k x)]
    then feed a 2-layer MLP. Unlike the legacy UNet1DScaleCond (integer-step
    sinusoidal table on 10*ln(scale)), the frequencies are matched to the input
    range: the finest one repeats every ~0.09 decades (k = 5), coarser ones
    keep the encoding unique over the whole range.
    """

    def __init__(self, emb_dim: int = 256, n_freqs: int = 6,
                 log10_range: tuple = (-2.0, 1.0)):
        super().__init__()
        lo, hi = log10_range
        self.center = 0.5 * (lo + hi)
        self.half_width = 0.5 * (hi - lo)
        self.register_buffer(
            'freqs', math.pi * 2.0 ** torch.arange(n_freqs, dtype=torch.float32),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * n_freqs, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        """scale : (B,) max(|noisy|) in V -> (B, emb_dim)"""
        x = (torch.log10(scale.float().clamp(min=1e-8)) - self.center) / self.half_width
        args = x.unsqueeze(1) * self.freqs.unsqueeze(0)  # (B, n_freqs)
        feats = torch.cat([x.unsqueeze(1), torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(feats)


class ResBlock1D(nn.Module):
    """Residual block: Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU + skip.

    Conditioning (timestep [+ channel token] [+ scale]) enters in one of two ways:
      - film=False (default): a per-channel bias added after the first
        GroupNorm+SiLU. The embedding can only shift activations.
      - film=True (AdaGN, as in improved DDPM / ADM): the embedding is mapped
        through SiLU -> Linear to per-channel (gamma, beta), and the second
        GroupNorm becomes GN(h) * (1 + gamma) + beta. The embedding can then
        also scale features, i.e. modulate them multiplicatively -- needed for
        e.g. "suppress input detail when scale is small (noise-dominated)".
        The projection is zero-initialized so every block starts as a plain
        GroupNorm (gamma = beta = 0) and conditioning is learned as a delta.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 256, num_groups: int = 8,
                 film: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()
        self.film = film

        if film:
            # AdaGN projection -> (gamma, beta). Separate attribute name from
            # t_proj so additive and FiLM state dicts never load into each other.
            self.film_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_ch))
            nn.init.zeros_(self.film_proj[1].weight)
            nn.init.zeros_(self.film_proj[1].bias)
        else:
            # Timestep projection
            self.t_proj = nn.Linear(emb_dim, out_ch)

        # Residual connection (1x1 conv if channels change)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L)
        t_emb : (B, emb_dim) — combined conditioning embedding
        """
        h = self.act(self.norm1(self.conv1(x)))
        if self.film:
            gamma, beta = self.film_proj(t_emb).unsqueeze(-1).chunk(2, dim=1)
            h = self.act(self.norm2(self.conv2(h)) * (1 + gamma) + beta)
        else:
            # Add timestep embedding
            h = h + self.t_proj(t_emb).unsqueeze(-1)  # (B, out_ch, 1) broadcast
            h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the sequence dimension."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, L) -> (B, C, L)"""
        B, C, L = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # (B, C, L)
        return x + h


class Downsample1D(nn.Module):
    """Stride-2 convolution for downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Nearest-neighbor upsample + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net for conditional DDPM denoising.

    Parameters
    ----------
    in_channels : int
        Input channels (2 = x_t + x_tilde concatenated).
    out_channels : int
        Output channels (1 = predicted noise epsilon).
    base_channels : int
        Channels at level 0.
    channel_mults : tuple
        Channel multipliers per level.
    emb_dim : int
        Timestep embedding dimension.
    attn_levels : tuple
        Which encoder levels get self-attention (0-indexed).
    cond_mode : str
        'step' (default) — condition on integer diffusion step index 1..T.
        'sqrt_ab' — condition on continuous sqrt(alpha_bar) noise level
        (WaveGrad / DeScoD-ECG style).
    cond_scale : float
        Input multiplier applied before sinusoidal embedding when
        cond_mode='sqrt_ab'. Spreads the (0, 1] input across the
        sinusoidal frequency grid. Ignored when cond_mode='step'.
    n_channels : int or None
        If set, enable per-channel conditioning for the multi-channel
        foundation-model denoiser: a learned channel-token embedding (one row
        per channel) is added to the timestep embedding, so it reaches every
        ResBlock through the existing conditioning pathway. None (default)
        keeps the original single-channel behavior (no channel token). See
        ``src/ddpm/README.md`` (foundation-model section).
    scale_cond : bool
        If True, condition on the normalization scale max(|noisy|) via
        ``ScaleEmbedding``; its embedding is added to the timestep embedding
        and ``forward`` then requires ``scale``. Supersedes the separate
        ``UNet1DScaleCond`` class (kept only to load old checkpoints).
    film : bool
        If True, every ResBlock uses AdaGN (FiLM) conditioning instead of an
        additive bias; see ``ResBlock1D``. Independent of ``scale_cond``.
    scale_log10_range : tuple
        log10(scale [V]) interval mapped onto [-1, 1] by ``ScaleEmbedding``.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        emb_dim: int = 256,
        attn_levels: tuple = (),
        cond_mode: str = 'step',
        cond_scale: float = 1000.0,
        n_channels: int = None,
        scale_cond: bool = False,
        film: bool = False,
        scale_log10_range: tuple = (-2.0, 1.0),
    ):
        super().__init__()
        self.n_levels = len(channel_mults)
        self.cond_mode = cond_mode
        self.scale_cond = scale_cond
        self.film = film
        channels = [base_channels * m for m in channel_mults]  # [64, 128, 256, 512]

        # Timestep embedding
        self.time_mlp = TimestepMLP(emb_dim, cond_mode=cond_mode, cond_scale=cond_scale)

        # Optional per-channel conditioning (Tier-1 channel token). One learned
        # embedding row per channel, added to the timestep embedding. Zero-init
        # so an untrained / held-out channel starts from the channel-agnostic
        # ("null") behavior and fine-tuning only learns a delta on top of the
        # frozen backbone. Swap this module for FiLM / LoRA for higher tiers.
        self.n_channels = n_channels
        if n_channels is not None:
            self.channel_emb = nn.Embedding(n_channels, emb_dim)
            nn.init.zeros_(self.channel_emb.weight)
        else:
            self.channel_emb = None

        # Optional normalization-scale conditioning, added to the timestep
        # embedding like the channel token.
        self.scale_emb = (ScaleEmbedding(emb_dim, log10_range=scale_log10_range)
                          if scale_cond else None)

        # Initial projection
        self.conv_in = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        # ---- Encoder ----
        self.enc_blocks = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        self.enc_attns = nn.ModuleList()

        for i in range(self.n_levels):
            ch_in = channels[i - 1] if i > 0 else channels[0]
            ch_out = channels[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim, film=film),
                ResBlock1D(ch_out, ch_out, emb_dim, film=film),
            ]))
            if i in attn_levels:
                self.enc_attns.append(SelfAttention1D(ch_out))
            else:
                self.enc_attns.append(nn.Identity())

            if i < self.n_levels - 1:
                self.enc_downsamples.append(Downsample1D(ch_out))
            else:
                self.enc_downsamples.append(nn.Identity())  # no downsample at last level

        # ---- Bottleneck ----
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock1D(mid_ch, mid_ch, emb_dim, film=film)
        self.mid_attn = SelfAttention1D(mid_ch)
        self.mid_block2 = ResBlock1D(mid_ch, mid_ch, emb_dim, film=film)

        # ---- Decoder ----
        self.dec_blocks = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        for i in reversed(range(self.n_levels)):
            ch_out = channels[i]
            # Skip connection doubles input channels
            ch_in = ch_out + channels[i]  # concat with encoder skip

            if i > 0:
                ch_target = channels[i - 1]
            else:
                ch_target = channels[0]

            self.dec_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim, film=film),
                ResBlock1D(ch_out, ch_target, emb_dim, film=film),
            ]))

            if i in attn_levels:
                self.dec_attns.append(SelfAttention1D(ch_out))
            else:
                self.dec_attns.append(nn.Identity())

            if i > 0:
                self.dec_upsamples.append(Upsample1D(ch_target))
            else:
                self.dec_upsamples.append(nn.Identity())

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_tilde: torch.Tensor,
        t: torch.Tensor,
        channel_id: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, L) — noisy latent at step t
        x_tilde : (B, 1, L) — noisy observation (conditioning)
        t : (B,) — step indices (1..T) if cond_mode='step',
            continuous sqrt(alpha_bar) values if cond_mode='sqrt_ab'
        channel_id : (B,) long or None — channel index for the channel-token
            embedding. Ignored unless the model was built with n_channels.
        scale : (B,) or None — max(|noisy|) before normalization [V].
            Required if the model was built with scale_cond.

        Returns
        -------
        eps_pred : (B, 1, L) — predicted noise
        """
        # Concatenate input and condition
        x = torch.cat([x_t, x_tilde], dim=1)  # (B, 2, L)
        cond = self.time_mlp(t)  # (B, emb_dim)

        # Add the per-channel token (foundation-model conditioning)
        if self.channel_emb is not None and channel_id is not None:
            cond = cond + self.channel_emb(channel_id)

        # Add the normalization-scale embedding
        if self.scale_emb is not None:
            if scale is None:
                raise ValueError("model built with scale_cond=True needs `scale`")
            cond = cond + self.scale_emb(scale)

        # Initial conv
        h = self.conv_in(x)

        # Encoder — collect skip connections
        skips = []
        for i in range(self.n_levels):
            for block in self.enc_blocks[i]:
                h = block(h, cond)
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < self.n_levels - 1:
                h = self.enc_downsamples[i](h)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder
        for j, i in enumerate(reversed(range(self.n_levels))):
            skip = skips[i]
            # Match lengths if needed (from downsampling rounding)
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)

            res1, res2 = self.dec_blocks[j]
            h = res1(h, cond)
            h = self.dec_attns[j](h)
            h = res2(h, cond)

            if i > 0:
                h = self.dec_upsamples[j](h)

        return self.conv_out(h)


def build_ddpm_model(cond_mode: str = 'step', cond_scale: float = 1000.0,
                     scale_cond: bool = False, film: bool = False,
                     legacy_scale_cond: bool = False) -> nn.Module:
    """Construct the main-DDPM U-Net from CLI/config flags.

    ``legacy_scale_cond`` builds the old ``UNet1DScaleCond`` (additive
    10*ln(scale) sinusoidal embedding) so pre-existing ``--scale_cond``
    checkpoints such as ddpm_scale_cond/ still load; it cannot be combined
    with the new options.
    """
    if legacy_scale_cond:
        if scale_cond or film:
            raise ValueError("legacy scale conditioning cannot be combined with "
                             "scale_cond/film")
        from src.ddpm.unet_cond import UNet1DScaleCond
        return UNet1DScaleCond(cond_mode=cond_mode, cond_scale=cond_scale)
    return UNet1D(cond_mode=cond_mode, cond_scale=cond_scale,
                  scale_cond=scale_cond, film=film)
