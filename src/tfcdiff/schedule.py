"""
Quadratic noise schedule with SNR scaling for TFCDiff.

Extends the DDPM schedule with SNR-based rescaling (factor c) to better
preserve frequency-domain signal structure during diffusion.

Usage:
    from src.tfcdiff.schedule import DiffusionSchedule
    schedule = DiffusionSchedule(T=50, snr_scale=150.0)
"""

import numpy as np
import torch


class DiffusionSchedule:
    """Quadratic noise schedule with optional SNR scaling.

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    beta_1 : float
        Starting noise level.
    beta_T : float
        Ending noise level.
    snr_scale : float
        SNR scaling factor c. Set to 1.0 to disable.
    """

    def __init__(self, T: int = 50, beta_1: float = 1e-4, beta_T: float = 0.5,
                 snr_scale: float = 150.0):
        self.T = T

        # Quadratic schedule
        t = np.arange(1, T + 1, dtype=np.float64)
        sqrt_beta = (
            (T - t) / (T - 1) * beta_1 ** 0.5
            + (t - 1) / (T - 1) * beta_T ** 0.5
        )
        beta = sqrt_beta ** 2
        beta = np.clip(beta, 1e-8, 0.999)

        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)

        # SNR scaling: SNR'(t) = c * SNR(t), then recompute alpha_bar
        if snr_scale != 1.0:
            snr = alpha_bar / (1.0 - alpha_bar)
            scaled_snr = snr_scale * snr
            alpha_bar = scaled_snr / (1.0 + scaled_snr)

        # Store as float32 tensors
        self.alpha_bar = torch.tensor(alpha_bar, dtype=torch.float32)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # For reverse process, compute per-step alpha and beta from alpha_bar
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        self.alpha = self.alpha_bar / alpha_bar_prev
        self.beta = 1.0 - self.alpha
        self.sqrt_alpha = torch.sqrt(self.alpha)

        # Posterior variance: beta_tilde = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        self.posterior_variance = (
            (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar) * self.beta
        )
        self.posterior_variance[0] = self.beta[0]
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

    def to(self, device):
        """Move all tensors to device."""
        for attr in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                      'alpha', 'beta', 'sqrt_alpha',
                      'posterior_variance', 'posterior_log_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
