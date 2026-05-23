"""
TFCDiff diffusion process operating in DCT domain.

Handles:
  - Forward process in DCT space with continuous noise level sampling
  - Training loss (L1) between true and predicted noise
  - Reverse sampling with DCT->IDCT conversion
  - Multi-shot inference

Usage:
    from src.tfcdiff.diffusion import TFCDiffusion
    diffusion = TFCDiffusion(model, schedule, dct_len=2400, eta=27.0)
    loss = diffusion.training_loss(x_0, x_tilde)
    x_denoised = diffusion.sample(x_tilde)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

from src.tfcdiff.schedule import DiffusionSchedule


class TFCDiffusion(nn.Module):
    """Conditional diffusion for signal denoising in DCT domain."""

    def __init__(self, model: nn.Module, schedule: DiffusionSchedule,
                 dct_len: int = 2400, eta: float = 27.0,
                 signal_len: int = 10000, loss_type: str = 'l1'):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.T = schedule.T
        self.dct_len = dct_len
        self.eta = eta
        self.signal_len = signal_len
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss

    def to_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time-domain signal to truncated, scaled DCT coefficients.

        Parameters
        ----------
        x : (B, 1, L) — time-domain signal in [-1, 1]

        Returns
        -------
        d : (B, 1, dct_len) — scaled DCT coefficients
        """
        d = dct.dct(x, norm='ortho')
        d = d[:, :, :self.dct_len]
        d = d / self.eta
        return d

    def from_dct(self, d: torch.Tensor) -> torch.Tensor:
        """Convert scaled DCT coefficients back to time-domain signal.

        Parameters
        ----------
        d : (B, 1, dct_len) — scaled DCT coefficients

        Returns
        -------
        x : (B, 1, signal_len) — time-domain signal
        """
        d = d * self.eta
        d = F.pad(d, (0, self.signal_len - self.dct_len), mode='constant', value=0)
        x = dct.idct(d, norm='ortho')
        return x

    def q_sample(self, x_0: torch.Tensor,
                 continuous_sqrt_alpha_bar: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward process with continuous noise level.

        x_t = sqrt_alpha_bar * x_0 + sqrt(1 - alpha_bar) * eps

        Parameters
        ----------
        x_0 : (B, 1, dct_len) — clean DCT coefficients
        continuous_sqrt_alpha_bar : (B, 1, 1) — continuous noise level
        noise : (B, 1, dct_len) — optional pre-sampled noise

        Returns
        -------
        x_t : (B, 1, dct_len)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        return continuous_sqrt_alpha_bar * x_0 + \
            torch.sqrt(1.0 - continuous_sqrt_alpha_bar ** 2) * noise

    def training_loss(self, x_0: torch.Tensor,
                      x_tilde: torch.Tensor) -> torch.Tensor:
        """Compute loss for one training step.

        Parameters
        ----------
        x_0 : (B, 1, L) — clean signal in time domain, normalized to [-1, 1]
        x_tilde : (B, 1, L) — noisy observation in time domain, normalized

        Returns
        -------
        loss : scalar tensor
        """
        B = x_0.shape[0]
        device = x_0.device

        # Convert to DCT domain
        x_0_dct = self.to_dct(x_0)
        x_tilde_dct = self.to_dct(x_tilde)

        # Sample random timestep and continuous noise level
        t = np.random.randint(1, self.T + 1)
        continuous_sqrt_alpha_bar = torch.FloatTensor(
            np.random.uniform(
                self.schedule.sqrt_alpha_bar[t - 1].item(),
                self.schedule.sqrt_alpha_bar[min(t, self.T - 1)].item()
                if t < self.T else self.schedule.sqrt_alpha_bar[t - 1].item(),
                size=B
            )
        ).to(device)

        # Sample noise and create noisy latent
        noise = torch.randn_like(x_0_dct)
        x_t = self.q_sample(
            x_0_dct,
            continuous_sqrt_alpha_bar.view(-1, 1, 1),
            noise,
        )

        # Predict noise (model input: concatenated [x_tilde_dct, x_t])
        x_input = torch.cat([x_tilde_dct, x_t], dim=1)  # (B, 2, dct_len)
        eps_pred = self.model(x_input, continuous_sqrt_alpha_bar)

        return self.loss_fn(eps_pred, noise)

    @torch.no_grad()
    def p_mean_variance(self, x_t, t_idx, x_tilde_dct):
        """Compute reverse process mean and log variance at step t_idx.

        Parameters
        ----------
        x_t : (B, 1, dct_len)
        t_idx : int — 0-indexed timestep
        x_tilde_dct : (B, 1, dct_len)

        Returns
        -------
        mean, log_variance : (B, 1, dct_len) each
        """
        B = x_t.shape[0]
        device = x_t.device

        noise_level = self.schedule.sqrt_alpha_bar[t_idx].repeat(B).to(device)
        x_input = torch.cat([x_tilde_dct, x_t], dim=1)
        eps_pred = self.model(x_input, noise_level)

        # Compute x_0 estimate
        alpha_bar_t = self.schedule.alpha_bar[t_idx]
        sqrt_alpha_bar_t = self.schedule.sqrt_alpha_bar[t_idx]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bar[t_idx]

        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

        # Compute posterior mean
        alpha_t = self.schedule.alpha[t_idx]
        beta_t = self.schedule.beta[t_idx]

        coeff = beta_t / sqrt_one_minus_alpha_bar_t
        mean = (1.0 / self.schedule.sqrt_alpha[t_idx]) * (x_t - coeff * eps_pred)

        log_variance = self.schedule.posterior_log_variance[t_idx]

        return mean, log_variance

    @torch.no_grad()
    def sample(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """Reverse process: denoise x_tilde by running T steps in DCT domain,
        then convert back to time domain.

        Parameters
        ----------
        x_tilde : (B, 1, L) — noisy observation in time domain

        Returns
        -------
        x_0 : (B, 1, L) — denoised signal in time domain
        """
        device = x_tilde.device
        B = x_tilde.shape[0]

        x_tilde_dct = self.to_dct(x_tilde)

        # Start from pure noise in DCT domain
        x_t = torch.randn(B, 1, self.dct_len, device=device)

        for i in reversed(range(self.T)):
            mean, log_var = self.p_mean_variance(x_t, i, x_tilde_dct)
            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + noise * (0.5 * log_var).exp()
            else:
                x_t = mean

        # Convert back to time domain
        return self.from_dct(x_t)

    @torch.no_grad()
    def sample_multi_shot(self, x_tilde: torch.Tensor,
                          M: int = 10) -> torch.Tensor:
        """Multi-shot inference: run M independent reverse processes and average.

        Parameters
        ----------
        x_tilde : (B, 1, L) — noisy observation in time domain
        M : int — number of shots to average

        Returns
        -------
        x_0_avg : (B, 1, L) — averaged denoised signal in time domain
        """
        accum = torch.zeros_like(x_tilde)
        for _ in range(M):
            accum += self.sample(x_tilde)
        return accum / M
