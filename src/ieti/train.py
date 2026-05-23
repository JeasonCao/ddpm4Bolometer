"""
train.py — training loop for 1D U-Net denoiser.

Supports multiple loss functions:
    'mse'             — standard L2 (stable, start here)
    'psd'             — ASD ratio loss (frequency-aware)
    'wavelet_psd'     — wavelet-based ASD ratio loss (stable + frequency-aware)
    'mse+psd'         — combined MSE + PSD ratio
    'mse+wavelet_psd' — combined MSE + wavelet PSD ratio
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import UNet1D


# ── LOSS FUNCTIONS ────────────────────────────────────────────────────────────

def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss."""
    return nn.functional.mse_loss(output, target)


def psd_ratio_loss(
    output: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """ASD ratio loss: mean over freq bins of sqrt(S[r,r] / S[h,h]).

    r = output - target (residual = what the network changed).
    S[x,x] = |FFT(x)|^2 (power spectral density via periodogram).
    """
    r = output - target
    S_rr = torch.abs(torch.fft.rfft(r, dim=-1)) ** 2
    S_hh = torch.abs(torch.fft.rfft(target, dim=-1)) ** 2
    ratio = torch.sqrt(S_rr / (S_hh + eps))
    return ratio.mean()


def wavelet_psd_ratio_loss(
    output: torch.Tensor, target: torch.Tensor,
    n_levels: int = 5, eps: float = 1e-10,
) -> torch.Tensor:
    """ASD ratio loss using Haar wavelet coefficient variance per level.

    More stable than FFT-based: each level pools many coefficients.
    Naturally adapted to 1/f noise structure.

    Uses manual Haar wavelet (no external dependency).
    """
    r = output - target
    loss = torch.tensor(0.0, device=output.device)

    r_current = r
    t_current = target

    for _ in range(n_levels):
        n = r_current.shape[-1]
        if n < 2:
            break

        # Haar detail coefficients: (x[::2] - x[1::2]) / sqrt(2)
        r_detail = (r_current[..., ::2] - r_current[..., 1::2]) / 1.4142
        t_detail = (t_current[..., ::2] - t_current[..., 1::2]) / 1.4142

        # PSD estimate = variance of detail coefficients
        S_rr = torch.var(r_detail, dim=-1) + eps
        S_hh = torch.var(t_detail, dim=-1) + eps

        loss = loss + torch.sqrt(S_rr / S_hh).mean()

        # Approximation coefficients for next level
        r_current = (r_current[..., ::2] + r_current[..., 1::2]) / 1.4142
        t_current = (t_current[..., ::2] + t_current[..., 1::2]) / 1.4142

    return loss / n_levels


def tv_loss(output: torch.Tensor) -> torch.Tensor:
    """Total variation loss: penalizes sample-to-sample jaggedness.

    TV(z) = mean(|z_{j+1} - z_j|)
    """
    return torch.abs(output[..., 1:] - output[..., :-1]).mean()


def combined_loss(
    output: torch.Tensor, target: torch.Tensor,
    loss_type: str = "mse", psd_weight: float = 0.5,
    tv_weight: float = 0.0,
) -> torch.Tensor:
    """Compute loss based on loss_type string.

    loss_type: 'mse' | 'psd' | 'wavelet_psd' | 'mse+psd' | 'mse+wavelet_psd'
    tv_weight: if > 0, adds TV regularization (for noisier N2N training)
    """
    if loss_type == "mse":
        loss = mse_loss(output, target)
    elif loss_type == "psd":
        loss = psd_ratio_loss(output, target)
    elif loss_type == "wavelet_psd":
        loss = wavelet_psd_ratio_loss(output, target)
    elif loss_type == "mse+psd":
        loss = mse_loss(output, target) + psd_weight * psd_ratio_loss(output, target)
    elif loss_type == "mse+wavelet_psd":
        loss = mse_loss(output, target) + psd_weight * wavelet_psd_ratio_loss(output, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if tv_weight > 0:
        loss = loss + tv_weight * tv_loss(output)

    return loss


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_epoch(
    model: UNet1D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "mse",
    psd_weight: float = 0.5,
    tv_weight: float = 0.0,
) -> float:
    """Train one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for noisy, target in loader:
        noisy = noisy.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = combined_loss(output, target, loss_type, psd_weight, tv_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(
    model: UNet1D,
    loader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
    psd_weight: float = 0.5,
    tv_weight: float = 0.0,
) -> float:
    """Validate one epoch. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for noisy, target in loader:
            noisy = noisy.to(device)
            target = target.to(device)
            output = model(noisy)
            loss = combined_loss(output, target, loss_type, psd_weight, tv_weight)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: UNet1D,
    train_dataset,
    val_dataset,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    lr_patience: int = 10,
    checkpoint_path: str = "checkpoints/unet1d_best.pt",
    loss_type: str = "mse",
    psd_weight: float = 0.5,
    tv_weight: float = 0.0,
    device: torch.device | None = None,
) -> dict:
    """Full training loop with scheduler and checkpointing.

    Returns dict with train_losses and val_losses lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_type, psd_weight, tv_weight)
        val_loss = validate_epoch(model, val_loader, device, loss_type, psd_weight, tv_weight)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{n_epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved best model (val_loss={val_loss:.6f})")

    # Save loss history alongside checkpoint
    loss_dir = os.path.dirname(checkpoint_path) or "."
    np.save(os.path.join(loss_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(loss_dir, "val_losses.npy"), np.array(val_losses))

    return {"train_losses": train_losses, "val_losses": val_losses}


def plot_loss_curves(
    train_losses_path: str = "checkpoints/train_losses.npy",
    val_losses_path: str = "checkpoints/val_losses.npy",
    output_path: str = "plots/loss_curves.png",
) -> None:
    """Plot train/val loss vs epoch and save to file."""
    import matplotlib.pyplot as plt

    train_losses = np.load(train_losses_path)
    val_losses = np.load(val_losses_path)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training QA: Loss vs Epoch")

    # Linear scale
    ax1.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax1.plot(epochs, val_losses, label="Val", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (linear)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(epochs, train_losses, label="Train", linewidth=1.5)
    ax2.semilogy(epochs, val_losses, label="Val", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss (log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def load_checkpoint(
    checkpoint_path: str, model: UNet1D, device: torch.device
) -> UNet1D:
    """Load saved state_dict into model. Returns model in eval mode."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model
