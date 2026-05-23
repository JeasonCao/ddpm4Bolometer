"""
Guidance strategies for Joint Diffusion measurement consistency.

Three methods to enforce y = beta*x + alpha*n during joint reverse diffusion,
from simplest to most sophisticated:

    Projection  — operate in noisy space, no model predictions needed
    DPS         — use Tweedie estimates, equal-weight gradient
    PiGDM       — use Tweedie estimates, uncertainty-weighted gradient (best)

Reference: Stevens et al., "Removing Structured Noise with Diffusion
Models", TMLR 2025.

Usage:
    from src.jointdiff.guidance import ProjectionGuidance, DPSGuidance, PiGDMGuidance
    guidance = PiGDMGuidance(schedule, alpha=1.0)
    x, n = guidance.joint_update(y, x_t, n_t, model_x, model_n, t_idx)
"""

import torch

from src.ddpm.schedule import DiffusionSchedule


class BaseGuidance:
    """Base class for joint diffusion guidance.

    Parameters
    ----------
    schedule : DiffusionSchedule
        Shared noise schedule.
    alpha : float
        Noise coefficient in y = beta*x + alpha*n.
    beta : float
        Signal coefficient.
    lambda_x : float
        Step size for signal update.
    lambda_n : float
        Step size for noise update.
    """

    def __init__(self, schedule: DiffusionSchedule,
                 alpha: float = 1.0, beta: float = 1.0,
                 lambda_x: float = 0.93, lambda_n: float = 0.88):
        self.schedule = schedule
        self.alpha = alpha
        self.beta = beta
        self.lambda_x = lambda_x
        self.lambda_n = lambda_n

    def r_t_squared(self, t_idx: int) -> float:
        """Posterior variance ratio: sigma_t^2 / (sigma_t^2 + 1)."""
        sigma_sq = 1.0 - self.schedule.alpha_bar[t_idx].item()
        return sigma_sq / (sigma_sq + 1.0)

    def _tweedie_detached(self, x_t, model, t_idx):
        """Tweedie estimate E[x_0|x_t] without gradient tracking."""
        with torch.no_grad():
            B = x_t.shape[0]
            t_batch = torch.full((B,), t_idx + 1, device=x_t.device, dtype=torch.long)
            eps_pred = model(x_t, t_batch)
            sqrt_ab = self.schedule.sqrt_alpha_bar[t_idx]
            sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t_idx]
            return (x_t - sqrt_1_ab * eps_pred) / sqrt_ab

    def _tweedie_with_grad(self, x_t, model, t_idx):
        """Tweedie estimate with autograd graph retained.

        Returns (x0_hat, x_t_grad) where x0_hat is connected to
        x_t_grad for backprop.
        """
        device = x_t.device
        B = x_t.shape[0]
        x_t_grad = x_t.detach().requires_grad_(True)
        t_batch = torch.full((B,), t_idx + 1, device=device, dtype=torch.long)
        eps_pred = model(x_t_grad, t_batch)
        sqrt_ab = self.schedule.sqrt_alpha_bar[t_idx]
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t_idx]
        x0_hat = (x_t_grad - sqrt_1_ab * eps_pred) / sqrt_ab
        return x0_hat, x_t_grad

    def _forward_diffuse(self, x_0, t_idx):
        """Forward-diffuse a clean signal to noise level t."""
        sqrt_ab = self.schedule.sqrt_alpha_bar[t_idx]
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t_idx]
        noise = torch.randn_like(x_0)
        return sqrt_ab * x_0 + sqrt_1_ab * noise

    @staticmethod
    def _solve_blend(y, x0_hat, n0_hat, reg: float = 1e-6):
        """Solve for optimal blend coefficients a, b minimizing ||y - a*x0 - b*n0||^2.

        Per-sample 2x2 least-squares solve. Returns (a, b) as tensors of
        shape (B, 1, 1) for broadcasting.

        Parameters
        ----------
        y, x0_hat, n0_hat : (B, 1, L)
        reg : float
            Tikhonov regularization to prevent ill-conditioning.

        Returns
        -------
        a, b : (B, 1, 1) blend coefficients
        """
        # Flatten to (B, L)
        yf = y.reshape(y.shape[0], -1)
        xf = x0_hat.reshape(x0_hat.shape[0], -1)
        nf = n0_hat.reshape(n0_hat.shape[0], -1)

        # Gram matrix entries: (B,)
        xx = (xf * xf).sum(dim=1)
        nn = (nf * nf).sum(dim=1)
        xn = (xf * nf).sum(dim=1)
        xy = (xf * yf).sum(dim=1)
        ny = (nf * yf).sum(dim=1)

        # Solve [xx xn; xn nn] [a; b] = [xy; ny] with regularization
        det = (xx + reg) * (nn + reg) - xn * xn
        a = ((nn + reg) * xy - xn * ny) / det
        b = ((xx + reg) * ny - xn * xy) / det

        return a.view(-1, 1, 1), b.view(-1, 1, 1)

    @staticmethod
    def _solve_blend_1d(r, n0_hat, reg: float = 1e-6):
        """Solve for scalar b minimizing ||r - b*n0||^2, per sample."""
        rf = r.reshape(r.shape[0], -1)
        nf = n0_hat.reshape(n0_hat.shape[0], -1)
        nn = (nf * nf).sum(dim=1) + reg
        rn = (rf * nf).sum(dim=1)
        b = rn / nn
        return b.view(-1, 1, 1)

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        raise NotImplementedError


class ProjectionGuidance(BaseGuidance):
    """Projection guidance — simplest, operates in noisy space.

    Forward-diffuses y to the current noise level, then pushes x and n
    toward satisfying the constraint. Does NOT use model predictions
    for the consistency step — only uses the noisy iterates directly.

    Cheapest (no extra model calls), but crudest corrections.
    """

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        """
        Parameters
        ----------
        y : (B, 1, L) -- observation
        x_t, n_t : (B, 1, L) -- current iterates
        model_x, model_n : not used (kept for interface compatibility)
        t_idx : int -- 0-indexed timestep

        Returns
        -------
        x_corrected, n_corrected : (B, 1, L)
        """
        a = self.alpha
        b = self.beta

        # Forward-diffuse y to current noise level
        y_t = self._forward_diffuse(y, t_idx)

        # Gradient of ||b*x + a*n - y_t||^2 w.r.t. x and n
        grad_x = -(b ** 2 * x_t - b * y_t + a * b * n_t)
        grad_n = -(a ** 2 * n_t - a * y_t + a * b * x_t)

        x_corrected = x_t + self.lambda_x * grad_x
        n_corrected = n_t + self.lambda_n * grad_n

        return x_corrected, n_corrected


class DPSGuidance(BaseGuidance):
    """DPS (Diffusion Posterior Sampling) guidance.

    Uses Tweedie estimates x0_hat, n0_hat to measure the consistency
    error ||y - a*x0_hat - b*n0_hat||, then backpropagates through the
    models via autograd.  Blend coefficients a, b are solved per-step
    via least-squares to handle scale mismatch between independently
    normalized score networks.
    """

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        """
        Parameters
        ----------
        y : (B, 1, L) -- observation
        x_t, n_t : (B, 1, L) -- current iterates
        model_x, model_n : score networks
        t_idx : int

        Returns
        -------
        x_corrected, n_corrected : (B, 1, L)
        """
        # --- Gradient for x (backprop through model_x) ---
        x0_hat, x_t_grad = self._tweedie_with_grad(x_t, model_x, t_idx)
        n0_hat_detached = self._tweedie_detached(n_t, model_n, t_idx)

        # Adaptive blend coefficients
        a, b = self._solve_blend(y, x0_hat.detach(), n0_hat_detached)

        norm_x = torch.linalg.norm(y - a * x0_hat - b * n0_hat_detached)
        grad_x = torch.autograd.grad(norm_x, x_t_grad)[0]

        # --- Gradient for n (backprop through model_n) ---
        n0_hat, n_t_grad = self._tweedie_with_grad(n_t, model_n, t_idx)
        x0_hat_detached = x0_hat.detach()

        # Re-solve with updated n0_hat
        a, b = self._solve_blend(y, x0_hat_detached, n0_hat.detach())

        norm_n = torch.linalg.norm(y - a * x0_hat_detached - b * n0_hat)
        grad_n = torch.autograd.grad(norm_n, n_t_grad)[0]

        x_corrected = x_t - self.lambda_x * grad_x
        n_corrected = n_t - self.lambda_n * grad_n

        return x_corrected, n_corrected


class PiGDMGuidance(BaseGuidance):
    """PiGDM (Pseudo-Inverse Guided Diffusion Model) guidance.

    Like DPS but weights corrections by posterior uncertainty:
    at early steps (high noise, uncertain estimates) corrections are
    gentle; at late steps (low noise, confident estimates) corrections
    are strong.  Blend coefficients a, b are solved per-step via
    least-squares to handle scale mismatch between independently
    normalized score networks.
    """

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        """
        Parameters
        ----------
        y : (B, 1, L) -- observation
        x_t, n_t : (B, 1, L) -- current iterates
        model_x, model_n : score networks
        t_idx : int

        Returns
        -------
        x_corrected, n_corrected : (B, 1, L)
        """
        r_sq = self.r_t_squared(t_idx)
        q_sq = self.r_t_squared(t_idx)  # same schedule for both
        sigma_t = r_sq + q_sq

        if sigma_t < 1e-10:
            return x_t, n_t

        # --- Gradient for x ---
        x0_hat, x_t_grad = self._tweedie_with_grad(x_t, model_x, t_idx)
        n0_hat_detached = self._tweedie_detached(n_t, model_n, t_idx)

        # Adaptive blend coefficients
        a, b = self._solve_blend(y, x0_hat.detach(), n0_hat_detached)

        residual_x = a * x0_hat + b * n0_hat_detached - y
        loss_x = (residual_x ** 2 / (2.0 * sigma_t)).sum()
        grad_x = torch.autograd.grad(loss_x, x_t_grad)[0]

        # --- Gradient for n ---
        n0_hat, n_t_grad = self._tweedie_with_grad(n_t, model_n, t_idx)
        x0_hat_detached = x0_hat.detach()

        # Re-solve with updated n0_hat
        a, b = self._solve_blend(y, x0_hat_detached, n0_hat.detach())

        residual_n = a * x0_hat_detached + b * n0_hat - y
        loss_n = (residual_n ** 2 / (2.0 * sigma_t)).sum()
        grad_n = torch.autograd.grad(loss_n, n_t_grad)[0]

        # Uncertainty-weighted corrections
        x_corrected = x_t - self.lambda_x * r_sq * grad_x
        n_corrected = n_t - self.lambda_n * q_sq * grad_n

        return x_corrected, n_corrected


class ResidualGuidance(BaseGuidance):
    """Guide Score_n to match a pre-computed residual r = y - x̂_0_ddpm.

    The observation `y` passed to joint_update should be the residual r,
    not the original noisy signal. This puts guidance entirely in the
    noise scale, avoiding the signal-dominance problem.
    """

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        """y here is the residual r, not the original observation."""
        r_sq = self.r_t_squared(t_idx)
        if r_sq < 1e-10:
            return x_t, n_t

        n0_hat, n_t_grad = self._tweedie_with_grad(n_t, model_n, t_idx)
        b = self._solve_blend_1d(y, n0_hat.detach())

        residual = b * n0_hat - y
        loss = (residual ** 2 / (2.0 * r_sq)).sum()
        grad_n = torch.autograd.grad(loss, n_t_grad)[0]

        n_corrected = n_t - self.lambda_n * r_sq * grad_n
        return x_t, n_corrected


class SignalFixedGuidance(BaseGuidance):
    """Guidance that leaves x_t untouched and only pushes n_t.

    Score_x runs unconditionally to produce a clean pulse. The guidance
    computes the residual y - a*x̂_0 and pushes n_t to explain it.
    This avoids injecting noise into the signal estimate when Score_n
    is too weak to carry its share.
    """

    def joint_update(self, y, x_t, n_t, model_x, model_n, t_idx):
        r_sq = self.r_t_squared(t_idx)
        q_sq = self.r_t_squared(t_idx)
        sigma_t = r_sq + q_sq

        if sigma_t < 1e-10:
            return x_t, n_t

        # Get x̂_0 estimate (detached — no gradient, x_t unchanged)
        x0_hat_detached = self._tweedie_detached(x_t, model_x, t_idx)

        # Get n̂_0 estimate with gradient
        n0_hat, n_t_grad = self._tweedie_with_grad(n_t, model_n, t_idx)

        # Solve blend coefficients
        a, b = self._solve_blend(y, x0_hat_detached, n0_hat.detach())

        # Only compute gradient for n_t
        residual = a * x0_hat_detached + b * n0_hat - y
        loss_n = (residual ** 2 / (2.0 * sigma_t)).sum()
        grad_n = torch.autograd.grad(loss_n, n_t_grad)[0]

        # x_t passes through unchanged
        n_corrected = n_t - self.lambda_n * q_sq * grad_n

        return x_t, n_corrected


def get_guidance(name: str, schedule: DiffusionSchedule, **kwargs) -> BaseGuidance:
    """Factory function to create guidance by name.

    Parameters
    ----------
    name : str
        One of 'projection', 'dps', 'pigdm'.
    schedule : DiffusionSchedule
    **kwargs : passed to guidance constructor (alpha, beta, lambda_x, lambda_n)
    """
    classes = {
        'projection': ProjectionGuidance,
        'dps': DPSGuidance,
        'pigdm': PiGDMGuidance,
        'signal_fixed': SignalFixedGuidance,
        'residual': ResidualGuidance,
    }
    if name not in classes:
        raise ValueError(f"Unknown guidance '{name}'. Choose from: {list(classes.keys())}")
    return classes[name](schedule, **kwargs)
