"""
Cold Diffusion forward and reverse processes for CDiffSD.

Replaces Gaussian noise with real noise from a noise library.
Model predicts x_0 directly (not epsilon).
Reverse uses deterministic Cold Diffusion update rule.

Usage:
    from src.cdiffsd.cold_diffusion import ColdDiffusion
    diffusion = ColdDiffusion(model, schedule)
    loss = diffusion.training_loss(x_0, x_tilde, n_real)
    x_denoised = diffusion.sample(x_tilde)
"""

import torch
import torch.nn as nn

from src.ddpm.schedule import DiffusionSchedule


class ColdDiffusion(nn.Module):
    """Conditional Cold Diffusion for signal denoising with real noise."""

    def __init__(self, model: nn.Module, schedule: DiffusionSchedule,
                 loss_type: str = 'l2'):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.T = schedule.T
        self.loss_fn = nn.functional.l1_loss if loss_type == 'l1' else nn.functional.mse_loss

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise_real: torch.Tensor) -> torch.Tensor:
        """Forward process: corrupt x_0 with real noise at level t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise_real

        Parameters
        ----------
        x_0 : (B, 1, L) -- clean signal
        t : (B,) -- timestep indices (0-indexed)
        noise_real : (B, 1, L) -- real noise from noise library

        Returns
        -------
        x_t : (B, 1, L)
        """
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        return sqrt_ab * x_0 + sqrt_1_ab * noise_real

    def training_loss(self, x_0: torch.Tensor, x_tilde: torch.Tensor,
                      noise_real: torch.Tensor) -> torch.Tensor:
        """Compute loss: predict x_0 from noisy x_t.

        Parameters
        ----------
        x_0 : (B, 1, L) -- clean signal
        x_tilde : (B, 1, L) -- noisy observation (conditioning)
        noise_real : (B, 1, L) -- real noise for forward corruption

        Returns
        -------
        loss : scalar tensor
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps (0-indexed)
        t = torch.randint(0, self.T, (B,), device=device)

        # Forward process with real noise
        x_t = self.q_sample(x_0, t, noise_real)

        # Model predicts x_0 (takes 1-indexed timesteps)
        x_0_pred = self.model(x_t, x_tilde, t + 1)

        return self.loss_fn(x_0_pred, x_0)

    @torch.no_grad()
    def sample(self, x_tilde: torch.Tensor, t_start: int = None,
               return_trajectory: bool = False) -> torch.Tensor:
        """Reverse process: Cold Diffusion deterministic update.

        Starts from x_tilde (the noisy observation) and iteratively
        refines it, following the CDiffSD paper.

        At each step:
            1. Predict x_0 from current x_t
            2. Estimate noise: n_est = (x_t - sqrt_ab_t * x_0_pred) / sqrt_1_ab_t
            3. Reconstruct: x_{t-1} = sqrt_ab_{t-1} * x_0_pred + sqrt_1_ab_{t-1} * n_est

        Parameters
        ----------
        x_tilde : (B, 1, L) -- noisy observation (conditioning + starting point)
        t_start : int or None
            Starting timestep for reverse process. If None, uses T.
            Lower t_start = fewer steps = faster inference for high-SNR signals.
        return_trajectory : bool

        Returns
        -------
        x_0 : (B, 1, L) -- denoised signal
        """
        device = x_tilde.device
        B, C, L = x_tilde.shape

        if t_start is None:
            t_start = self.T

        # Start from the noisy observation
        x_t = x_tilde.clone()

        trajectory = [x_t] if return_trajectory else None

        for i in reversed(range(t_start)):
            t_batch = torch.full((B,), i + 1, device=device, dtype=torch.long)

            # Predict x_0
            x_0_pred = self.model(x_t, x_tilde, t_batch)

            if i > 0:
                # Estimate noise from current state
                sqrt_ab = self.schedule.sqrt_alpha_bar[i]
                sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[i]
                n_est = (x_t - sqrt_ab * x_0_pred) / sqrt_1_ab

                # Reconstruct at previous noise level
                sqrt_ab_prev = self.schedule.sqrt_alpha_bar[i - 1]
                sqrt_1_ab_prev = self.schedule.sqrt_one_minus_alpha_bar[i - 1]
                x_t = sqrt_ab_prev * x_0_pred + sqrt_1_ab_prev * n_est
            else:
                x_t = x_0_pred

            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return x_t, trajectory
        return x_t

    @torch.no_grad()
    def sample_direct(self, x_tilde: torch.Tensor,
                      t_start: int = None) -> torch.Tensor:
        """Direct reconstruction: single-pass x_0 prediction (no iteration).

        As described in CDiffSD paper — apply the model once at timestep
        t_start to predict x_0 directly.

        Parameters
        ----------
        x_tilde : (B, 1, L) -- noisy observation
        t_start : int or None
            Timestep to use for prediction. If None, uses T.

        Returns
        -------
        x_0 : (B, 1, L)
        """
        device = x_tilde.device
        B = x_tilde.shape[0]

        if t_start is None:
            t_start = self.T

        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        return self.model(x_tilde, x_tilde, t_batch)

    @torch.no_grad()
    def sample_multi_shot(self, x_tilde: torch.Tensor,
                          M: int = 10,
                          t_start: int = None) -> torch.Tensor:
        """Multi-shot inference: run M reverse processes and average.

        Since Cold Diffusion reverse is deterministic from x_tilde,
        this is equivalent to a single shot. Kept for API compatibility
        with DDPM inference code.

        Parameters
        ----------
        x_tilde : (B, 1, L)
        M : int -- number of shots (all produce the same result here)
        t_start : int or None

        Returns
        -------
        x_0 : (B, 1, L)
        """
        return self.sample(x_tilde, t_start=t_start)
