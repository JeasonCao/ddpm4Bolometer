"""
1D U-Net with TFEM for TFCDiff denoising in DCT domain.

Architecture follows TFCDiff (Li et al., 2025):
    Input: [x_t, x_tilde] concatenated -> 2 channels, seq_len DCT coefficients
    Encoder: TFEM blocks (TFE + TFF) with detour downsampling
    Bottleneck: ResBlock + SelfAttention
    Decoder: ResBlock with skip connections + detour upsampling
    Timestep: FiLM conditioning on continuous sqrt(alpha_bar_t)
    Output: 1 channel (predicted noise in DCT domain)

Usage:
    from src.tfcdiff.unet import UNet
    model = UNet(seq_len=2400)
    eps_pred = model(x, noise_level)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Conv1d(nn.Conv1d):
    """Conv1d with Kaiming normal init and zero bias."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Linear(nn.Linear):
    """Linear with Kaiming normal init and zero bias."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal embedding for continuous noise level (sqrt_alpha_bar)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        """
        noise_level : (B,) or (B, 1) — continuous sqrt(alpha_bar_t)
        Returns: (B, dim)
        """
        noise_level = noise_level.view(-1, 1)
        half = self.dim // 2
        step = torch.arange(half, device=noise_level.device) / (half - 1)
        encoding = noise_level * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        return torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)


class FeatureWiseAffine(nn.Module):
    """FiLM: feature-wise affine transform conditioned on noise level."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.noise_func = Linear(in_channels, out_channels * 2)

    def forward(self, x, noise_embed):
        gamma, beta = self.noise_func(noise_embed).view(
            x.shape[0], -1, 1).chunk(2, dim=1)
        return (1 + gamma) * x + beta


class Block(nn.Module):
    """GroupNorm -> Swish -> Dropout -> Conv1d."""

    def __init__(self, dim, dim_out, groups=16, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            Conv1d(dim, dim_out, 3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    """Residual block with FiLM noise conditioning."""

    def __init__(self, dim, dim_out, noise_level_emb_dim=None,
                 dropout=0, norm_groups=16):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with depthwise conv normalization."""

    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.dw = Conv1d(in_channel, in_channel, kernel_size=3, stride=1,
                         padding=1, groups=in_channel, bias=False)
        self.qkv = Conv1d(in_channel, in_channel * 3, 1, bias=True)
        self.out = Conv1d(in_channel, in_channel, 1)

    def forward(self, x):
        batch, channel, seq_len = x.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.dw(self.norm(x))
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, seq_len)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum("bncl,bncL->bnlL", query, key) / math.sqrt(channel)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bnlL,bncL->bncl", attn, value)
        out = self.out(out.reshape(batch, channel, seq_len))
        return out + x


class ResnetBlockWithAttn(nn.Module):
    """ResnetBlock optionally followed by SelfAttention."""

    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None,
                 norm_groups=16, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim,
            norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# ── TFEM components ──────────────────────────────────────────────────────────

class TFE(nn.Module):
    """Temporal Feature Extraction: process DCT features via time-domain
    residual block.

    DCT coefficients -> zero-pad -> IDCT -> ResBlock -> DCT -> truncate.
    """

    def __init__(self, dim, dim_out, noise_level_emb_dim=None,
                 dropout=0, norm_groups=16, pad_ratio=2.6):
        super().__init__()
        self.pad_ratio = pad_ratio
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        _, _, seq_len = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        # Detour to time domain
        pad_len = int(seq_len * self.pad_ratio)
        h = F.pad(h, (0, pad_len), mode='constant', value=0)
        h = dct.idct(h, norm='ortho')
        h = self.block2(h)
        h = dct.dct(h, norm='ortho')
        h = h[:, :, :seq_len]
        return h + self.res_conv(x)


class TFF(nn.Module):
    """Temporal Feature Fusion: cross-domain attention.

    Compute Q, K, V in DCT domain, convert to time domain via IDCT,
    perform attention in time domain, convert back via DCT.
    """

    def __init__(self, in_channel, n_head=1, norm_groups=16, pad_ratio=2.6):
        super().__init__()
        self.n_head = n_head
        self.pad_ratio = pad_ratio
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.dw = Conv1d(in_channel, in_channel, kernel_size=3, stride=1,
                         padding=1, groups=in_channel, bias=False)
        self.qkv = Conv1d(in_channel, in_channel * 3, 1, bias=True)
        self.out = Conv1d(in_channel, in_channel, 1)

    def forward(self, x):
        batch, channel, seq_len = x.shape
        n_head = self.n_head
        head_dim = channel // n_head
        time_len = int(seq_len * (1 + self.pad_ratio))

        norm = self.dw(self.norm(x))
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, seq_len)
        query, key, value = qkv.chunk(3, dim=2)

        # Convert Q, K, V to time domain
        q_t = dct.idct(F.pad(query, (0, time_len - seq_len), mode='constant', value=0), norm='ortho')
        k_t = dct.idct(F.pad(key, (0, time_len - seq_len), mode='constant', value=0), norm='ortho')
        v_t = dct.idct(F.pad(value, (0, time_len - seq_len), mode='constant', value=0), norm='ortho')

        # Attention in time domain
        attn = torch.einsum("bncl,bncL->bnlL", q_t, k_t) / math.sqrt(channel)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bnlL,bncL->bncl", attn, v_t)
        # Back to DCT domain
        out = dct.dct(out, norm='ortho')
        out = out[:, :, :, :seq_len]
        out = self.out(out.reshape(batch, channel, seq_len))
        return out + x


class TFEM(nn.Module):
    """Temporal Feature Enhancement Mechanism: TFE + optional TFF."""

    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None,
                 norm_groups=16, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = TFE(
            dim, dim_out, noise_level_emb_dim,
            norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = TFF(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# ── Detour resampling ────────────────────────────────────────────────────────

class Downsample(nn.Module):
    """Detour downsample: DCT -> IDCT -> stride conv -> DCT."""

    def __init__(self, dim, pad_ratio=2.6):
        super().__init__()
        self.pad_ratio = pad_ratio
        self.norm = nn.GroupNorm(16, dim)
        self.conv = Conv1d(dim, dim, 3, stride=2, padding=1,
                           padding_mode='reflect')

    def forward(self, x):
        _, _, length = x.shape
        x = self.norm(x)
        x = dct.idct(
            F.pad(x, (0, int(length * self.pad_ratio)), mode='constant', value=0),
            norm='ortho')
        x = self.conv(x)
        x = dct.dct(x, norm='ortho')[:, :, :length // 2]
        return x


class Upsample(nn.Module):
    """Detour upsample: DCT -> IDCT -> nearest upsample -> conv -> DCT."""

    def __init__(self, dim, pad_ratio=2.6):
        super().__init__()
        self.pad_ratio = pad_ratio
        self.norm = nn.GroupNorm(16, dim)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv1d(dim, dim, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        _, _, length = x.shape
        x = self.norm(x)
        x = dct.idct(
            F.pad(x, (0, int(length * self.pad_ratio)), mode='constant', value=0),
            norm='ortho')
        x = self.up(x)
        x = dct.dct(x, norm='ortho')[:, :, :length * 2]
        return self.conv(x)


# ── Main U-Net ───────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """1D U-Net with TFEM for conditional diffusion denoising in DCT domain.

    Parameters
    ----------
    in_channel : int
        Input channels (2 = x_t + x_tilde concatenated).
    out_channel : int
        Output channels (1 = predicted noise).
    inner_channel : int
        Base channel count.
    channel_mults : tuple
        Channel multipliers per level.
    attn_res : tuple
        Sequence lengths at which to apply attention.
    res_blocks : int
        Number of residual blocks per level.
    dropout : float
        Dropout rate.
    norm_groups : int
        Groups for GroupNorm.
    seq_len : int
        Input sequence length (number of DCT coefficients).
    """

    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=64,
        norm_groups=16,
        channel_mults=(1, 2, 2, 2),
        attn_res=(600,),
        res_blocks=2,
        dropout=0,
        seq_len=2400,
    ):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            Linear(inner_channel, inner_channel * 4),
            Swish(),
            Linear(inner_channel * 4, inner_channel),
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = seq_len

        # ── Encoder ──
        downs = [Conv1d(in_channel, inner_channel, kernel_size=3,
                        padding=1, padding_mode='reflect')]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(TFEM(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        # ── Bottleneck ──
        self.mid = nn.ModuleList([
            ResnetBlockWithAttn(
                pre_channel, pre_channel,
                noise_level_emb_dim=noise_level_channel,
                norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlockWithAttn(
                pre_channel, pre_channel,
                noise_level_emb_dim=noise_level_channel,
                norm_groups=norm_groups, dropout=dropout, with_attn=False),
        ])

        # ── Decoder ──
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(ResnetBlockWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2
        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(16, pre_channel),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            Conv1d(pre_channel, default(out_channel, in_channel), 3,
                   padding=1, padding_mode='reflect'),
        )

    def forward(self, x, noise_level):
        """
        Parameters
        ----------
        x : (B, 2, L) — concatenated [x_tilde, x_t] in DCT domain
        noise_level : (B,) — continuous sqrt(alpha_bar_t)

        Returns
        -------
        eps_pred : (B, 1, L) — predicted noise in DCT domain
        """
        t = self.noise_level_mlp(noise_level)

        feats = []
        for layer in self.downs:
            if isinstance(layer, TFEM):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
