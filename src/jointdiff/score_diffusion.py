"""
Score-based diffusion for unconditional score network training.

Standard DDPM forward process and denoising score matching loss.
Also provides Tweedie estimate and reverse step for inference.

Usage:
    from src.jointdiff.score_diffusion import ScoreDiffusion
    diffusion = ScoreDiffusion(model, schedule)
    loss = diffusion.score_loss(x_0)
"""

import torch
import torch.nn as nn

from src.ddpm.schedule import DiffusionSchedule


class ScoreDiffusion(nn.Module):
    """Unconditional DDPM for score matching."""

    def __init__(self, model: nn.Module, schedule: DiffusionSchedule):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.T = schedule.T

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1_ab * noise

    def score_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Denoising score matching loss (predict epsilon).

        Parameters
        ----------
        x_0 : (B, 1, L) -- clean signal

        Returns
        -------
        loss : scalar
        """
        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        eps_pred = self.model(x_t, t + 1)  # 1-indexed for model
        return nn.functional.mse_loss(eps_pred, noise)

    def tweedie_estimate(self, x_t: torch.Tensor, t_idx: int) -> torch.Tensor:
        """Compute E[x_0 | x_t] via Tweedie's formula.

        x0_hat = (x_t - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

        Parameters
        ----------
        x_t : (B, 1, L) -- noisy signal at step t
        t_idx : int -- 0-indexed timestep

        Returns
        -------
        x0_hat : (B, 1, L)
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_idx + 1, device=x_t.device, dtype=torch.long)
        eps_pred = self.model(x_t, t_batch)

        sqrt_ab = self.schedule.sqrt_alpha_bar[t_idx]
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t_idx]

        return (x_t - sqrt_1_ab * eps_pred) / sqrt_ab

    def reverse_step(self, x_t: torch.Tensor, t_idx: int):
        """One DDPM reverse step. Returns (x_{t-1}, x_mean, eps_pred).

        Parameters
        ----------
        x_t : (B, 1, L)
        t_idx : int -- 0-indexed (corresponds to going from t to t-1)

        Returns
        -------
        x_prev : (B, 1, L) -- x_{t-1} (stochastic)
        x_mean : (B, 1, L) -- deterministic mean
        eps_pred : (B, 1, L) -- predicted noise
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_idx + 1, device=x_t.device, dtype=torch.long)
        eps_pred = self.model(x_t, t_batch)

        alpha_t = self.schedule.alpha[t_idx]
        beta_t = self.schedule.beta[t_idx]
        sqrt_alpha = self.schedule.sqrt_alpha[t_idx]
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t_idx]

        coeff = beta_t / sqrt_1_ab
        x_mean = (1.0 / sqrt_alpha) * (x_t - coeff * eps_pred)

        if t_idx > 0:
            sigma = torch.sqrt(beta_t)
            x_prev = x_mean + sigma * torch.randn_like(x_t)
        else:
            x_prev = x_mean

        return x_prev, x_mean, eps_pred

    @torch.no_grad()
    def sample(self, shape, device) -> torch.Tensor:
        """Unconditional sampling (for QA / generation check)."""
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            x_t, _, _ = self.reverse_step(x_t, i)
        return x_t
