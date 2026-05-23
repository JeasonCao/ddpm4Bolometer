"""
Joint reverse diffusion sampler with selectable guidance strategy.

Runs two independent score networks (signal + noise) simultaneously,
coupling them at each step via measurement consistency guidance.

Three guidance strategies available:
    projection — fast, crude (no model calls in guidance)
    dps        — moderate (autograd, equal weight)
    pigdm      — best (autograd, uncertainty-weighted)

Usage:
    from src.jointdiff.joint_sampler import JointSampler
    sampler = JointSampler(model_x, model_n, schedule, guidance='pigdm')
    x_0, n_0 = sampler.sample(y)
"""

import torch

from src.ddpm.schedule import DiffusionSchedule
from src.jointdiff.guidance import get_guidance, BaseGuidance


class JointSampler:
    """Joint reverse diffusion with configurable guidance.

    Parameters
    ----------
    model_x : ScoreUNet1D
        Trained signal score network.
    model_n : ScoreUNet1D
        Trained noise score network.
    schedule : DiffusionSchedule
        Noise schedule (shared by both networks).
    guidance : str or BaseGuidance
        Guidance strategy: 'projection', 'dps', or 'pigdm',
        or a pre-constructed BaseGuidance instance.
    alpha : float
        Noise coefficient in y = x + alpha * n.
    lambda_x : float
        Guidance step size for signal.
    lambda_n : float
        Guidance step size for noise.
    """

    def __init__(self, model_x, model_n, schedule: DiffusionSchedule,
                 guidance='pigdm',
                 alpha: float = 1.0,
                 lambda_x: float = 0.93,
                 lambda_n: float = 0.88):
        self.model_x = model_x
        self.model_n = model_n
        self.schedule = schedule
        self.T = schedule.T

        if isinstance(guidance, BaseGuidance):
            self.guidance = guidance
        else:
            self.guidance = get_guidance(
                guidance, schedule,
                alpha=alpha, lambda_x=lambda_x, lambda_n=lambda_n,
            )

    def _reverse_step(self, x_t, model, t_idx):
        """One DDPM reverse step (no grad)."""
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_idx + 1, device=x_t.device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x_t, t_batch)

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

        return x_prev

    def _needs_grad(self):
        """Check if guidance needs autograd (DPS and PiGDM do, Projection doesn't)."""
        from src.jointdiff.guidance import ProjectionGuidance
        return not isinstance(self.guidance, ProjectionGuidance)

    @torch.no_grad()
    def sample(self, y: torch.Tensor,
               return_trajectory: bool = False):
        """Joint reverse diffusion with guidance.

        Parameters
        ----------
        y : (B, 1, L) -- noisy observation (normalized)
        return_trajectory : bool

        Returns
        -------
        x_0 : (B, 1, L) -- denoised signal
        n_0 : (B, 1, L) -- separated noise
        """
        device = y.device
        B, C, L = y.shape

        x_t = torch.randn(B, 1, L, device=device)
        n_t = torch.randn(B, 1, L, device=device)

        trajectory = {'x': [x_t.cpu()], 'n': [n_t.cpu()]} if return_trajectory else None
        use_grad = self._needs_grad()

        for i in reversed(range(self.T)):
            # 1. Independent DDPM reverse steps
            x_t = self._reverse_step(x_t, self.model_x, i)
            n_t = self._reverse_step(n_t, self.model_n, i)

            # 2. Guidance step
            if use_grad:
                with torch.enable_grad():
                    x_t, n_t = self.guidance.joint_update(
                        y, x_t, n_t, self.model_x, self.model_n, i,
                    )
                if x_t.requires_grad:
                    x_t = x_t.detach()
                n_t = n_t.detach()
            else:
                x_t, n_t = self.guidance.joint_update(
                    y, x_t, n_t, self.model_x, self.model_n, i,
                )

            if return_trajectory:
                trajectory['x'].append(x_t.cpu())
                trajectory['n'].append(n_t.cpu())

        if return_trajectory:
            return x_t, n_t, trajectory
        return x_t, n_t
