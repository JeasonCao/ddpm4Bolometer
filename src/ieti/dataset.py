"""
dataset.py — Training datasets for Noise2Noise denoising.

Two parallel approaches:
  EvenOddN2NDataset  — even/odd sample split (half-rate, true N2N)
  NoisierN2NDataset  — add quiet-window noise to signal (full-rate, TV-regularized)

Preprocessing pipeline (must be applied before creating the dataset):
  Stage 1: Correlated drift removal (cross-channel, via drift.py)
           → voltage_arrays should be drift-corrected before passing in
  Stage 2: Rolling median/MAD baseline (via trigger.py)
           → pass rolling_meds and rolling_mads; subtracted per-window in __getitem__
  Stage 3: Periodic noise removal (sinusoidal subtraction 3b)
           → applied per-window inside __getitem__

Train/val split by time: time_range=(0.0, 0.8) for train, (0.8, 1.0) for val.
A gap of one window_len is enforced at the boundary to prevent leakage.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from .periodic import (
    find_adjacent_quiet, estimate_noise_psd, apply_wiener_filter,
    detect_spike_frequencies, apply_sinusoidal_subtraction,
)

WINDOW_LEN = 32_768   # legacy default for 5000 Hz
HALF_LEN = 16_384     # legacy default

# UNet has 5 levels of downsampling by 2 → needs divisibility by 32.
# Even/odd split halves the count → total must be divisible by 64.
UNET_DIVISOR = 64


def compute_window_lengths(
    sampling_rates: list[float],
    window_sec: float = 6.5,
) -> dict[float, int]:
    """Compute per-channel window lengths that satisfy:

    1. All channels cover the same time duration
    2. Sample counts maintain exact fs ratio (e.g. 5000:2000 = 5:2)
    3. Each count is divisible by 64 (32 for UNet downsampling × 2 for even/odd split)
    """
    from math import gcd
    from functools import reduce

    def lcm(a, b):
        return a * b // gcd(a, b)

    fs_ints = [int(round(fs)) for fs in sampling_rates]
    unique_fs = sorted(set(fs_ints))

    g = reduce(gcd, unique_fs)
    ratios = {fs: fs // g for fs in unique_fs}

    min_bases = [UNET_DIVISOR // gcd(r, UNET_DIVISOR) for r in ratios.values()]
    min_base = reduce(lcm, min_bases)

    target_base = window_sec * g
    k = max(1, round(target_base / min_base))
    base = min_base * k

    result = {}
    for fs in unique_fs:
        r = ratios[fs]
        n = base * r
        result[float(fs)] = n

    actual_sec = base / g
    print(f"  Window config: target={window_sec:.2f}s, actual={actual_sec:.4f}s")
    for fs in unique_fs:
        n = result[float(fs)]
        print(f"    fs={fs} Hz: {n} samples ({n/fs:.4f}s), "
              f"half={n//2}, ÷64={'OK' if n % 64 == 0 else 'FAIL'}")

    return result, actual_sec


def _remove_periodic_3b(window, quiet, fs, t_start):
    """Apply sinusoidal subtraction (3b) with coincidence spike detection."""
    spike_freqs, _, _ = detect_spike_frequencies(
        quiet, fs=fs, signal_window=window)
    cleaned = apply_sinusoidal_subtraction(
        window, fs=fs, t_start_samples=t_start, spike_freqs=spike_freqs)
    return cleaned


def _resolve_window_config(voltage_arrays, sampling_rates, window_sec, window_len, fs):
    """Shared logic to compute per-channel sampling rates and window lengths."""
    n_ch = len(voltage_arrays)

    if sampling_rates is not None:
        assert len(sampling_rates) == n_ch
        fs_list = [float(fs_i) for fs_i in sampling_rates]
        wl_map, actual_sec = compute_window_lengths(fs_list, window_sec)
        wl_list = [wl_map[float(int(round(fs_i)))] for fs_i in fs_list]
    elif window_len is not None:
        fs_list = [fs] * n_ch
        wl_list = [window_len] * n_ch
        actual_sec = window_len / fs
    else:
        fs_list = [fs] * n_ch
        wl_list = [WINDOW_LEN] * n_ch
        actual_sec = WINDOW_LEN / fs

    return fs_list, wl_list, actual_sec


def _build_window_list(voltage_arrays, rolling_meds, rolling_mads,
                       fs_list, wl_list, hop_ratio, mad_threshold,
                       time_range, rng):
    """Build list of (ch_idx, t_start) for windows within the time range.

    time_range : (frac_start, frac_end) as fractions of total length [0, 1].
    A gap of window_len samples is excluded at each boundary to prevent
    quiet-window leakage across splits.

    Uses precomputed quiet mask for fast filtering instead of calling
    find_adjacent_quiet per window.
    """
    windows = []
    frac_lo, frac_hi = time_range

    for ch_idx, (v, rm, rmad) in enumerate(
        zip(voltage_arrays, rolling_meds, rolling_mads)
    ):
        wl = wl_list[ch_idx]
        hop = max(1, int(wl * hop_ratio))
        n = len(v)

        # Compute sample boundaries with gap
        t_lo = int(frac_lo * n)
        t_hi = int(frac_hi * n)

        if frac_lo > 0:
            t_lo += wl
        if frac_hi < 1.0:
            t_hi -= wl

        t_lo = max(0, t_lo)
        t_hi = min(n - wl, t_hi)

        # Precompute quiet mask: True where |residual| < threshold * MAD
        residual = np.abs(v - rm)
        quiet_mask = residual < (mad_threshold * rmad)

        # For each candidate window, check if before or after has enough
        # contiguous quiet samples (at least wl // 4 for a useful estimate)
        min_quiet = max(256, wl // 4)
        n_candidates = 0
        for t in range(t_lo, t_hi, hop):
            n_candidates += 1
            has_quiet = False

            # Check full window before
            if t >= wl and quiet_mask[t - wl:t].all():
                has_quiet = True
            # Check full window after
            elif t + 2 * wl <= n and quiet_mask[t + wl:t + 2 * wl].all():
                has_quiet = True
            else:
                # Check for partial contiguous quiet run (before or after)
                # Scan backward from t
                if t >= min_quiet:
                    run_start = max(0, t - wl)
                    seg = quiet_mask[run_start:t]
                    # Count contiguous True from end
                    run = 0
                    for k in range(len(seg) - 1, -1, -1):
                        if seg[k]:
                            run += 1
                        else:
                            break
                    if run >= min_quiet:
                        has_quiet = True

                # Scan forward from t + wl
                if not has_quiet and t + wl < n:
                    seg_end = min(n, t + 3 * wl)
                    seg = quiet_mask[t + wl:seg_end]
                    run = 0
                    for k in range(len(seg)):
                        if seg[k]:
                            run += 1
                        else:
                            break
                    if run >= min_quiet:
                        has_quiet = True

            if has_quiet:
                windows.append((ch_idx, t))

        print(f"    ch{ch_idx}: {len(windows)} / {n_candidates} windows have quiet neighbours")

    rng.shuffle(windows)
    return windows


def _find_quiet_fast(residual, quiet_mask, t_start, wl, n):
    """Fast quiet-window finder using precomputed residual and quiet_mask.

    Returns quiet residual segment (may be shorter than wl) or None.
    """
    t_end = t_start + wl

    # Option 1: full window before
    before_start = t_start - wl
    if before_start >= 0 and quiet_mask[before_start:t_start].all():
        return residual[before_start:t_start].copy()

    # Option 2: full window after
    if t_end + wl <= n and quiet_mask[t_end:t_end + wl].all():
        return residual[t_end:t_end + wl].copy()

    # Option 3: longest contiguous quiet run before or after
    best = None

    if before_start >= 0:
        seg = quiet_mask[max(0, before_start):t_start]
        run = 0
        for k in range(len(seg) - 1, -1, -1):
            if seg[k]:
                run += 1
            else:
                break
        if run > 0:
            best = residual[t_start - run:t_start].copy()

    search_end = min(n, t_end + 2 * wl)
    if t_end < n:
        seg = quiet_mask[t_end:search_end]
        run = 0
        for k in range(len(seg)):
            if seg[k]:
                run += 1
            else:
                break
        if run > 0 and (best is None or run > len(best)):
            best = residual[t_end:t_end + run].copy()

    return best


def precompute_dataset(
    voltage_arrays: list[np.ndarray],
    rolling_meds: list[np.ndarray],
    rolling_mads: list[np.ndarray],
    output_dir: str,
    sampling_rates: list[float] | None = None,
    window_sec: float = 6.5,
    window_len: int | None = None,
    hop_ratio: float = 0.5,
    mad_threshold: float = 3.0,
    fs: float = 5000.0,
    use_periodic: bool = True,
    time_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
) -> str:
    """Precompute periodic-cleaned windows and save as .npy files.

    Saves to output_dir:
      cleaned_windows.npy  — (N, wl) float32, baseline-subtracted + periodic-cleaned
      quiet_noise.npy      — (N, wl) float32, periodic-cleaned quiet noise (for noisier)
      has_quiet.npy        — (N,) bool, whether quiet noise is valid
      meta.json            — fs, wl, n_windows, time_range, etc.

    Returns output_dir path.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    fs_list, wl_list, actual_sec = _resolve_window_config(
        voltage_arrays, sampling_rates, window_sec, window_len, fs)

    windows = _build_window_list(
        voltage_arrays, rolling_meds, rolling_mads,
        fs_list, wl_list, hop_ratio, mad_threshold, time_range, rng)

    wl = wl_list[0]
    n_win = len(windows)
    print(f"  Precomputing {n_win} windows (wl={wl})...")

    # Precompute residuals and quiet masks per channel (the key optimization)
    residuals = []
    quiet_masks = []
    for ch_idx, (v, rm, rmad) in enumerate(
        zip(voltage_arrays, rolling_meds, rolling_mads)
    ):
        res = v - rm
        residuals.append(res)
        quiet_masks.append(np.abs(res) < (mad_threshold * rmad))
    print(f"  Precomputed residuals and quiet masks")

    cleaned = np.empty((n_win, wl), dtype=np.float32)
    quiet_arr = np.empty((n_win, wl), dtype=np.float32)
    has_quiet = np.zeros(n_win, dtype=bool)

    for i, (ch_idx, t_start) in enumerate(windows):
        res = residuals[ch_idx]
        qm = quiet_masks[ch_idx]
        fs_ch = fs_list[ch_idx]
        wl_ch = wl_list[ch_idx]
        n_samples = len(res)

        # Baseline-subtracted window (just a slice of precomputed residual)
        window = res[t_start:t_start + wl_ch].copy()

        # Find quiet window using precomputed mask
        quiet = _find_quiet_fast(res, qm, t_start, wl_ch, n_samples)

        # Periodic removal on signal
        if use_periodic and quiet is not None:
            window = _remove_periodic_3b(window, quiet, fs_ch, t_start)
            # Remove periodic from quiet noise too
            quiet_spikes, _, _ = detect_spike_frequencies(quiet, fs=fs_ch)
            quiet = apply_sinusoidal_subtraction(
                quiet, fs=fs_ch, t_start_samples=t_start, spike_freqs=quiet_spikes)

        cleaned[i] = window.astype(np.float32)

        if quiet is not None:
            if len(quiet) < wl_ch:
                reps = (wl_ch // len(quiet)) + 1
                quiet = np.tile(quiet, reps)[:wl_ch]
            quiet_arr[i] = quiet[:wl_ch].astype(np.float32)
            has_quiet[i] = True

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_win} done")

    # Compute global normalization stats across all windows
    global_mean = float(cleaned.mean())
    global_std = float(cleaned.std()) + 1e-8
    print(f"  Global normalization: mean={global_mean:.6f}, std={global_std:.6f}")

    # Save
    np.save(os.path.join(output_dir, "cleaned_windows.npy"), cleaned)
    np.save(os.path.join(output_dir, "quiet_noise.npy"), quiet_arr)
    np.save(os.path.join(output_dir, "has_quiet.npy"), has_quiet)

    win_arr = np.array(windows, dtype=np.int64)
    np.save(os.path.join(output_dir, "window_list.npy"), win_arr)

    meta = {
        "n_windows": n_win,
        "window_len": wl,
        "fs": fs_list[0],
        "window_sec": actual_sec,
        "time_range": list(time_range),
        "mad_threshold": mad_threshold,
        "use_periodic": use_periodic,
        "hop_ratio": hop_ratio,
        "seed": seed,
        "global_mean": global_mean,
        "global_std": global_std,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = (cleaned.nbytes + quiet_arr.nbytes + has_quiet.nbytes) / 1e6
    print(f"  Saved {n_win} windows to {output_dir}/ ({size_mb:.0f} MB)")
    return output_dir


class EvenOddN2NDataset(Dataset):
    """True Noise2Noise via even/odd sample splitting.

    For each window:
    1. Normalize by global mean/std (shared across all windows)
    2. Split: even[0::2] -> input, odd[1::2] -> target

    Global normalization ensures overlap-add inference works correctly.
    Can be created from raw data or from precomputed .npy files.
    """

    def __init__(
        self,
        voltage_arrays: list[np.ndarray],
        rolling_meds: list[np.ndarray],
        rolling_mads: list[np.ndarray],
        sampling_rates: list[float] | None = None,
        window_sec: float = 6.5,
        window_len: int | None = None,
        hop_ratio: float = 0.5,
        mad_threshold: float = 3.0,
        fs: float = 5000.0,
        use_periodic: bool = True,
        time_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
    ):
        self.mad_threshold = mad_threshold
        self.use_periodic = use_periodic
        self.rng = np.random.default_rng(seed)

        self.voltage_arrays = voltage_arrays
        self.rolling_meds = rolling_meds
        self.rolling_mads = rolling_mads

        self.sampling_rates, self.window_lens, self.window_sec = \
            _resolve_window_config(voltage_arrays, sampling_rates,
                                   window_sec, window_len, fs)

        self.window_len = self.window_lens[0]
        self.fs = self.sampling_rates[0]

        self.windows = _build_window_list(
            voltage_arrays, rolling_meds, rolling_mads,
            self.sampling_rates, self.window_lens,
            hop_ratio, mad_threshold, time_range, self.rng)

        # First pass: collect all windows to compute global stats
        print(f"    Precomputing {len(self.windows)} even/odd pairs...")
        raw_windows = []
        for i, (ch_idx, t_start) in enumerate(self.windows):
            v = voltage_arrays[ch_idx]
            rm = rolling_meds[ch_idx]
            rmad = rolling_mads[ch_idx]
            wl = self.window_lens[ch_idx]
            fs_ch = self.sampling_rates[ch_idx]

            window = v[t_start:t_start + wl] - rm[t_start:t_start + wl]

            if use_periodic:
                quiet, _ = find_adjacent_quiet(v, rm, rmad, t_start, wl, mad_threshold)
                if quiet is not None:
                    window = _remove_periodic_3b(window, quiet, fs_ch, t_start)

            raw_windows.append(window.astype(np.float32))

            if (i + 1) % 2000 == 0:
                print(f"      {i + 1}/{len(self.windows)} done")

        # Per-window normalization and cache
        self._cache = []
        for w in raw_windows:
            w_mean = w.mean()
            w_std = w.std() + 1e-8
            w_norm = (w - w_mean) / w_std
            self._cache.append((w_norm[0::2], w_norm[1::2]))

        mem_mb = sum(e.nbytes + o.nbytes for e, o in self._cache) / 1e6
        print(f"    Cached {len(self._cache)} pairs ({mem_mb:.0f} MB)")

    @classmethod
    def from_precomputed(cls, precomputed_dir: str, **kwargs):
        """Load from precomputed .npy files (created by precompute_dataset).

        Uses per-window normalization (each window normalized by own mean/std).
        """
        obj = cls.__new__(cls)
        cleaned = np.load(os.path.join(precomputed_dir, "cleaned_windows.npy"), mmap_mode="r")
        win_list = np.load(os.path.join(precomputed_dir, "window_list.npy"))
        with open(os.path.join(precomputed_dir, "meta.json")) as f:
            meta = json.load(f)

        obj.window_len = meta["window_len"]
        obj.fs = meta["fs"]
        obj.sampling_rates = [meta["fs"]]
        obj.window_lens = [meta["window_len"]]
        obj.window_sec = meta["window_sec"]
        obj.windows = [tuple(w) for w in win_list.tolist()]

        print(f"    Loading {len(cleaned)} precomputed windows from {precomputed_dir}...")

        # Per-window normalization + split even/odd
        obj._cache = []
        for i in range(len(cleaned)):
            w = cleaned[i].astype(np.float32)
            w_mean = w.mean()
            w_std = w.std() + 1e-8
            w_norm = (w - w_mean) / w_std
            obj._cache.append((w_norm[0::2], w_norm[1::2]))

        mem_mb = sum(e.nbytes + o.nbytes for e, o in obj._cache) / 1e6
        print(f"    Loaded {len(obj._cache)} pairs ({mem_mb:.0f} MB)")
        return obj

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        even, odd = self._cache[idx]
        return (
            torch.from_numpy(even).unsqueeze(0),
            torch.from_numpy(odd).unsqueeze(0),
        )


class NoisierN2NDataset(Dataset):
    """Noisier-to-Noisy training via quiet-window noise addition.

    For each window:
    1. Find adjacent quiet window
    2. Apply sinusoidal subtraction (3b) to both signal and quiet windows
    3. Tile quiet noise to match signal length if shorter
    4. Input: signal + quiet_noise (noisier), Target: signal (original noisy)
    5. Normalize both with same mean/std

    Model operates at full sample rate. Paired with TV loss for regularization.
    """

    def __init__(
        self,
        voltage_arrays: list[np.ndarray],
        rolling_meds: list[np.ndarray],
        rolling_mads: list[np.ndarray],
        sampling_rates: list[float] | None = None,
        window_sec: float = 6.5,
        window_len: int | None = None,
        hop_ratio: float = 0.5,
        mad_threshold: float = 3.0,
        fs: float = 5000.0,
        use_periodic: bool = True,
        time_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
    ):
        self.mad_threshold = mad_threshold
        self.use_periodic = use_periodic
        self.rng = np.random.default_rng(seed)

        self.voltage_arrays = voltage_arrays
        self.rolling_meds = rolling_meds
        self.rolling_mads = rolling_mads

        self.sampling_rates, self.window_lens, self.window_sec = \
            _resolve_window_config(voltage_arrays, sampling_rates,
                                   window_sec, window_len, fs)

        self.window_len = self.window_lens[0]
        self.fs = self.sampling_rates[0]

        self.windows = _build_window_list(
            voltage_arrays, rolling_meds, rolling_mads,
            self.sampling_rates, self.window_lens,
            hop_ratio, mad_threshold, time_range, self.rng)

        # First pass: collect all windows and quiet noise
        print(f"    Precomputing {len(self.windows)} noisier pairs...")
        raw_signals = []
        raw_quiets = []
        for i, (ch_idx, t_start) in enumerate(self.windows):
            v = voltage_arrays[ch_idx]
            rm = rolling_meds[ch_idx]
            rmad = rolling_mads[ch_idx]
            wl = self.window_lens[ch_idx]
            fs_ch = self.sampling_rates[ch_idx]

            signal = v[t_start:t_start + wl] - rm[t_start:t_start + wl]
            quiet, _ = find_adjacent_quiet(v, rm, rmad, t_start, wl, mad_threshold)

            if use_periodic and quiet is not None:
                signal = _remove_periodic_3b(signal, quiet, fs_ch, t_start)
                quiet_spikes, _, _ = detect_spike_frequencies(quiet, fs=fs_ch)
                quiet = apply_sinusoidal_subtraction(
                    quiet, fs=fs_ch, t_start_samples=t_start, spike_freqs=quiet_spikes)

            if quiet is not None and len(quiet) < wl:
                reps = (wl // len(quiet)) + 1
                quiet = np.tile(quiet, reps)[:wl]

            raw_signals.append(signal.astype(np.float32))
            raw_quiets.append(quiet)

            if (i + 1) % 2000 == 0:
                print(f"      {i + 1}/{len(self.windows)} done")

        # Per-window normalization and cache
        self._cache = []
        for signal, quiet in zip(raw_signals, raw_quiets):
            s_mean = signal.mean()
            s_std = signal.std() + 1e-8
            target = (signal - s_mean) / s_std

            if quiet is not None:
                noisy_input = target + quiet.astype(np.float32) / s_std
            else:
                noisy_input = target.copy()

            self._cache.append((noisy_input.astype(np.float32), target.astype(np.float32)))

        mem_mb = sum(inp.nbytes + tgt.nbytes for inp, tgt in self._cache) / 1e6
        print(f"    Cached {len(self._cache)} pairs ({mem_mb:.0f} MB)")

    @classmethod
    def from_precomputed(cls, precomputed_dir: str, **kwargs):
        """Load from precomputed .npy files (created by precompute_dataset).

        Uses per-window normalization (each window normalized by own mean/std).
        """
        obj = cls.__new__(cls)
        cleaned = np.load(os.path.join(precomputed_dir, "cleaned_windows.npy"), mmap_mode="r")
        quiet_noise = np.load(os.path.join(precomputed_dir, "quiet_noise.npy"), mmap_mode="r")
        has_quiet = np.load(os.path.join(precomputed_dir, "has_quiet.npy"))
        win_list = np.load(os.path.join(precomputed_dir, "window_list.npy"))
        with open(os.path.join(precomputed_dir, "meta.json")) as f:
            meta = json.load(f)

        obj.window_len = meta["window_len"]
        obj.fs = meta["fs"]
        obj.sampling_rates = [meta["fs"]]
        obj.window_lens = [meta["window_len"]]
        obj.window_sec = meta["window_sec"]
        obj.windows = [tuple(w) for w in win_list.tolist()]

        print(f"    Loading {len(cleaned)} precomputed windows from {precomputed_dir}...")

        # Per-window normalization + build input/target pairs
        obj._cache = []
        for i in range(len(cleaned)):
            w = cleaned[i].astype(np.float32)
            w_mean = w.mean()
            w_std = w.std() + 1e-8
            target = (w - w_mean) / w_std

            if has_quiet[i]:
                quiet = quiet_noise[i].astype(np.float32)
                noisy_input = (target + quiet / w_std).astype(np.float32)
            else:
                noisy_input = target.copy()

            obj._cache.append((noisy_input, target.astype(np.float32)))

        mem_mb = sum(inp.nbytes + tgt.nbytes for inp, tgt in obj._cache) / 1e6
        print(f"    Loaded {len(obj._cache)} pairs ({mem_mb:.0f} MB)")
        return obj

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_input, target = self._cache[idx]
        return (
            torch.from_numpy(noisy_input).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0),
        )


def plot_dataset_examples(
    voltage_arrays: list[np.ndarray],
    rolling_meds: list[np.ndarray],
    rolling_mads: list[np.ndarray],
    dataset,
    fs: float = 5000.0,
    channel_names: list[str] | None = None,
    n_examples: int = 5,
    output_dir: str = "plots/dataset_qa",
) -> None:
    """QA visualization: for each example, plot input vs target.

    Works with both EvenOddN2NDataset and NoisierN2NDataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(len(voltage_arrays))]

    indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)

    for k, idx in enumerate(indices):
        ch_idx, t_start = dataset.windows[idx]

        inp_tensor, tgt_tensor = dataset[idx]
        inp = inp_tensor.squeeze().numpy()
        tgt = tgt_tensor.squeeze().numpy()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        fs_ch = dataset.sampling_rates[ch_idx]
        is_evenodd = isinstance(dataset, EvenOddN2NDataset)
        effective_fs = fs_ch / 2 if is_evenodd else fs_ch
        t_arr = np.arange(len(inp)) / effective_fs

        label_in = "even (input)" if is_evenodd else "noisy input"
        label_tgt = "odd (target)" if is_evenodd else "target"

        # Panel 1: Input and target overlaid
        axes[0].plot(t_arr, inp, color="tab:blue", linewidth=0.5, alpha=0.7, label=label_in)
        axes[0].plot(t_arr, tgt, color="tab:orange", linewidth=0.5, alpha=0.7, label=label_tgt)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].set_title(f"{channel_names[ch_idx]} — window {idx} (t={t_start/fs_ch:.2f}s)")
        axes[0].legend(fontsize=8)

        # Panel 2: Difference
        diff = inp - tgt
        axes[1].plot(t_arr, diff, color="tab:red", linewidth=0.5)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("input - target")
        axes[1].set_title(f"Difference (std={diff.std():.4f})")

        # Panel 3: PSDs
        from scipy.signal import welch as welch_fn
        f_i, psd_i = welch_fn(inp, fs=effective_fs, nperseg=min(2048, len(inp)))
        f_t, psd_t = welch_fn(tgt, fs=effective_fs, nperseg=min(2048, len(tgt)))
        axes[2].loglog(f_i[1:], psd_i[1:], label=f"{label_in} PSD", linewidth=0.8)
        axes[2].loglog(f_t[1:], psd_t[1:], label=f"{label_tgt} PSD", linewidth=0.8)
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("PSD")
        axes[2].set_title("Power spectral density")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"example_{k:02d}_{channel_names[ch_idx]}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {n_examples} QA plots to {output_dir}/")
