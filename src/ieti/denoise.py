"""
denoise.py — Sliding-window inference + overlap-add.

Two parallel modes:
  even_odd  — Even/odd split → UNet → interleave (half-rate)
  noisier   — Direct full-rate UNet inference (no noise added at inference)
"""

import numpy as np
import torch
from .model import UNet1D
from .periodic import (
    find_adjacent_quiet, estimate_noise_psd, apply_wiener_filter,
    detect_spike_frequencies, apply_sinusoidal_subtraction,
)

WINDOW_LEN = 32_768
HOP_LEN = WINDOW_LEN // 2  # 50% overlap


def denoise_stream(
    voltages: np.ndarray,
    model: UNet1D,
    rolling_med: np.ndarray,
    rolling_mad: np.ndarray,
    device: torch.device,
    window_len: int = WINDOW_LEN,
    hop_len: int = HOP_LEN,
    batch_size: int = 4,
    fs: float = 5000.0,
    use_periodic: bool = True,
    mad_threshold: float = 3.0,
    global_mean: float = 0.0,
    global_std: float = 1.0,
) -> np.ndarray:
    """Denoise using even/odd split + interleave.

    For each overlapping window:
    1. residual = voltages - rolling_med
    2. Find adjacent quiet → sinusoidal subtraction (3b) for periodic removal
    3. Normalize (mean/std from full cleaned window)
    4. Split even/odd (each window_len//2 samples)
    5. Batch both through model
    6. Interleave denoised_even + denoised_odd → window_len
    7. Denormalize (restore mean/std)
    8. Overlap-add with Hann blending

    Add rolling_med back at end.

    Parameters
    ----------
    voltages    : full voltage stream (float64)
    model       : trained UNet1D in eval mode
    rolling_med : precomputed rolling median (same length as voltages)
    rolling_mad : precomputed rolling MAD (same length as voltages)
    device      : torch device
    window_len  : window size (must match training)
    hop_len     : hop between windows (default 50% overlap)
    batch_size  : number of windows per batch (each produces 2 model inputs)
    fs          : sampling rate
    use_periodic: whether to apply sinusoidal subtraction periodic filtering
    mad_threshold: MAD threshold for quiet window detection

    Returns
    -------
    denoised : same shape as voltages, float64
    """
    model.eval()
    n = len(voltages)
    residual = voltages - rolling_med

    # Hann window for overlap-add (full window length)
    hann = np.hanning(window_len).astype(np.float64)

    # Output buffer and normalization buffer
    output = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    # Compute window start positions
    starts = list(range(0, n - window_len + 1, hop_len))
    if starts and starts[-1] + window_len < n:
        starts.append(n - window_len)

    # Process in batches
    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start:batch_start + batch_size]

        even_list = []
        odd_list = []
        means = []
        stds = []

        for s in batch_indices:
            w = residual[s:s + window_len].copy()

            # Stage 1: Sinusoidal subtraction (3b) periodic noise removal
            if use_periodic:
                quiet, _ = find_adjacent_quiet(
                    voltages, rolling_med, rolling_mad,
                    s, window_len, mad_threshold
                )
                if quiet is not None:
                    spike_freqs, _, _ = detect_spike_frequencies(
                        quiet, fs=fs, signal_window=w)
                    w = apply_sinusoidal_subtraction(
                        w, fs=fs, t_start_samples=s, spike_freqs=spike_freqs)

            # Per-window normalization
            w_mean = w.mean()
            w_std = w.std() + 1e-8
            w_norm = (w - w_mean) / w_std
            means.append(w_mean)
            stds.append(w_std)

            # Split even/odd
            even_list.append(w_norm[0::2])
            odd_list.append(w_norm[1::2])

        # Stack even and odd into single batch: [even_0, odd_0, even_1, odd_1, ...]
        all_inputs = []
        for e, o in zip(even_list, odd_list):
            all_inputs.append(e)
            all_inputs.append(o)

        batch_tensor = torch.from_numpy(
            np.stack(all_inputs).astype(np.float32)
        ).unsqueeze(1).to(device)  # (2*B, 1, HALF_LEN)

        # Forward pass
        with torch.no_grad():
            batch_out = model(batch_tensor).squeeze(1).cpu().numpy()  # (2*B, HALF_LEN)

        # Interleave + denormalize + overlap-add
        for i, s in enumerate(batch_indices):
            denoised_even = batch_out[2 * i]      # (HALF_LEN,)
            denoised_odd = batch_out[2 * i + 1]   # (HALF_LEN,)

            # Interleave back to full window length
            denoised_window = np.empty(window_len, dtype=np.float64)
            denoised_window[0::2] = denoised_even
            denoised_window[1::2] = denoised_odd

            # Denormalize with per-window stats
            denoised_window = denoised_window * stds[i] + means[i]

            # Overlap-add with Hann blending
            output[s:s + window_len] += denoised_window * hann
            norm[s:s + window_len] += hann

    # Normalize by sum of Hann windows (rectangular analysis + Hann synthesis)
    mask = norm > 1e-10
    output[mask] /= norm[mask]

    # Handle any edge samples not covered by windows
    if not mask.all():
        uncovered = ~mask
        output[uncovered] = residual[uncovered]

    # Add baseline back
    return output + rolling_med


def denoise_stream_noisier(
    voltages: np.ndarray,
    model: UNet1D,
    rolling_med: np.ndarray,
    rolling_mad: np.ndarray,
    device: torch.device,
    window_len: int = WINDOW_LEN,
    hop_len: int = HOP_LEN,
    batch_size: int = 4,
    fs: float = 5000.0,
    use_periodic: bool = True,
    mad_threshold: float = 3.0,
    global_mean: float = 0.0,
    global_std: float = 1.0,
) -> np.ndarray:
    """Denoise using noisier-trained model at full sample rate.

    At inference, no noise is added — the model is applied directly to the
    periodic-cleaned residual. The model was trained to map (signal + noise)
    → signal, regularized by TV loss.

    For each overlapping window:
    1. residual = voltages - rolling_med
    2. Find adjacent quiet → sinusoidal subtraction (3b) for periodic removal
    3. Normalize (mean/std)
    4. Run through model (full window, no split)
    5. Denormalize
    6. Overlap-add with Hann blending
    """
    model.eval()
    n = len(voltages)
    residual = voltages - rolling_med

    hann = np.hanning(window_len).astype(np.float64)
    output = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    starts = list(range(0, n - window_len + 1, hop_len))
    if starts and starts[-1] + window_len < n:
        starts.append(n - window_len)

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start:batch_start + batch_size]

        inputs = []
        means = []
        stds = []

        for s in batch_indices:
            w = residual[s:s + window_len].copy()

            # Periodic removal via sinusoidal subtraction (3b)
            if use_periodic:
                quiet, _ = find_adjacent_quiet(
                    voltages, rolling_med, rolling_mad,
                    s, window_len, mad_threshold
                )
                if quiet is not None:
                    spike_freqs, _, _ = detect_spike_frequencies(
                        quiet, fs=fs, signal_window=w)
                    w = apply_sinusoidal_subtraction(
                        w, fs=fs, t_start_samples=s, spike_freqs=spike_freqs)

            # Per-window normalization
            w_mean = w.mean()
            w_std = w.std() + 1e-8
            means.append(w_mean)
            stds.append(w_std)
            inputs.append((w - w_mean) / w_std)

        batch_tensor = torch.from_numpy(
            np.stack(inputs).astype(np.float32)
        ).unsqueeze(1).to(device)  # (B, 1, window_len)

        with torch.no_grad():
            batch_out = model(batch_tensor).squeeze(1).cpu().numpy()  # (B, window_len)

        for i, s in enumerate(batch_indices):
            denoised_window = batch_out[i].astype(np.float64)
            denoised_window = denoised_window * stds[i] + means[i]

            output[s:s + window_len] += denoised_window * hann
            norm[s:s + window_len] += hann

    mask = norm > 1e-10
    output[mask] /= norm[mask]

    if not mask.all():
        uncovered = ~mask
        output[uncovered] = residual[uncovered]

    return output + rolling_med
