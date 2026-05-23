"""
trigger.py — adaptive trigger detection using rolling median + rolling MAD.

Replaces the broken fixed-baseline approach from easyTrigger.jl with a
rolling adaptive threshold that correctly handles the large thermal drift
present in LUCE IETI data.

Public API (backward-compatible names kept as wrappers):
    calculate_baseline_metrics, find_trigger_points,
    extract_signal_segment, process_triggered_signals
New API:
    rolling_median_chunked, rolling_mad_chunked, find_triggers_adaptive
"""

import numpy as np
from .convert import process_adc_file

try:
    import bottleneck as bn
    _HAS_BOTTLENECK = True
except ImportError:
    _HAS_BOTTLENECK = False


# ── ROLLING STATISTICS ────────────────────────────────────────────────────────

def rolling_median_chunked(
    voltages: np.ndarray,
    window: int,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Memory-efficient rolling median for large arrays.

    Uses bottleneck.move_median if available (fast, O(N log W)).
    Falls back to: downsample by stride → np.median → interpolate back.

    Parameters
    ----------
    voltages   : 1-D float64 array
    window     : rolling window size in samples (e.g. 60s * 5000 Hz = 300_000)
    chunk_size : unused in bottleneck path; kept for API consistency

    Returns
    -------
    rolling_med : same shape as voltages
    """
    if _HAS_BOTTLENECK:
        # bn.move_median: window centered via shift by window//2
        result = bn.move_median(voltages, window=window, min_count=1)
        # move_median is trailing — shift to approximate centered median
        half = window // 2
        result = np.roll(result, -half)
        # fill edges that wrapped
        result[-half:] = result[-(half + 1)]
        return result

    # Fallback: coarse median via stride_tricks then interpolate
    stride = max(1, window // 200)          # ~200 points per window at coarse scale
    coarse = voltages[::stride]
    w_coarse = max(3, window // stride)

    # Build sliding windows via stride_tricks
    n = len(coarse)
    half = w_coarse // 2
    padded = np.pad(coarse, half, mode="edge")
    shape = (n, w_coarse)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    coarse_med = np.median(windows, axis=1)

    # Interpolate back to full resolution
    x_coarse = np.arange(n) * stride
    x_full = np.arange(len(voltages))
    return np.interp(x_full, x_coarse, coarse_med)


def rolling_mad_chunked(
    voltages: np.ndarray,
    rolling_med: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling MAD (Median Absolute Deviation) given a precomputed rolling median.

    MAD = median(|x - rolling_med|) in a rolling window.
    Scaled by 1.4826 to match σ for Gaussian noise.
    """
    residuals = np.abs(voltages - rolling_med)
    mad = rolling_median_chunked(residuals, window)
    return mad * 1.4826


# ── ADAPTIVE TRIGGER ──────────────────────────────────────────────────────────

def find_triggers_adaptive(
    voltages: np.ndarray,
    fs: float,
    window_sec: float = 300.0,
    n_sigma: float = 6.0,
    refractory_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find trigger indices using rolling median + rolling MAD adaptive threshold.

    Triggers when: voltages[i] - rolling_med[i] > n_sigma * rolling_mad[i]

    Parameters
    ----------
    voltages       : 1-D voltage array
    fs             : sampling frequency in Hz
    window_sec     : rolling window size in seconds (default 60s)
    n_sigma        : threshold multiplier — tunable (default 6.0)
    refractory_sec : minimum time between triggers in seconds (default 1s)

    Returns
    -------
    trigger_indices : np.ndarray int64
    rolling_med     : np.ndarray float64 — baseline estimate
    rolling_mad     : np.ndarray float64 — noise estimate (σ-scaled)
    """
    window = int(window_sec * fs)
    refractory = int(refractory_sec * fs)

    rolling_med = rolling_median_chunked(voltages, window)
    rolling_mad = rolling_mad_chunked(voltages, rolling_med, window)

    residual = voltages - rolling_med
    threshold = n_sigma * rolling_mad

    trigger_indices = []
    i = 0
    while i < len(voltages):
        if residual[i] > threshold[i]:
            trigger_indices.append(i)
            i += refractory
        else:
            i += 1

    return np.array(trigger_indices, dtype=np.int64), rolling_med, rolling_mad


# ── SIGNAL SEGMENT EXTRACTION ─────────────────────────────────────────────────

def extract_signal_segment(
    times: np.ndarray,
    voltages: np.ndarray,
    trigger_index: int,
    pre_trigger_sec: float,
    post_trigger_sec: float,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a time window around a trigger index.

    Returns views (no copy) into times and voltages arrays.
    """
    pre_samples  = round(pre_trigger_sec * fs)
    post_samples = round(post_trigger_sec * fs)
    start = max(0, trigger_index - pre_samples)
    end   = min(len(voltages), trigger_index + post_samples)
    return times[start:end], voltages[start:end]


# ── BACKWARD-COMPATIBLE PUBLIC API ────────────────────────────────────────────
# These names are exported in __init__.py and used by explore.py.
# They wrap the new adaptive implementation.

def calculate_baseline_metrics(
    voltages: np.ndarray, baseline_period: int = 1000
) -> tuple[float, float]:
    """Legacy: mean and RMS of first N samples. Kept for backward compatibility."""
    actual = min(baseline_period, len(voltages))
    baseline = voltages[:actual]
    mean_val = float(np.mean(baseline))
    rms = float(np.sqrt(np.mean((baseline - mean_val) ** 2)))
    return mean_val, rms


def find_trigger_points(
    voltages: np.ndarray,
    threshold_upper: float,
    threshold_lower: float,
    fs: float,
    pre_trigger_sec: float,
    post_trigger_sec: float,
    num_triggers: int = 10,
) -> list[int]:
    """Legacy: fixed-threshold trigger. Kept for backward compatibility."""
    pre_samples  = round(pre_trigger_sec * fs)
    post_samples = round(post_trigger_sec * fs)

    trigger_indices = []
    marked = np.zeros(len(voltages), dtype=bool)

    for i, v in enumerate(voltages):
        if marked[i]:
            continue
        if threshold_lower < v < threshold_upper:
            continue
        trigger_indices.append(i)
        start = max(0, i - pre_samples)
        end   = min(len(voltages), i + post_samples)
        marked[start:end] = True
        if len(trigger_indices) >= num_triggers:
            break

    return trigger_indices


def process_triggered_signals(
    filename: str,
    max_samples: int = None,
    threshold_multiplier: float = 9,
    pre_trigger_sec: float = 0.03,
    post_trigger_sec: float = 0.07,
    num_triggers: int = 50,
) -> tuple[list[int], np.ndarray, np.ndarray] | None:
    """Legacy pipeline using adaptive trigger internally."""
    times, voltages, header = process_adc_file(filename, max_samples=max_samples)

    trigger_indices, _, _ = find_triggers_adaptive(
        voltages, header["fs"],
        n_sigma=threshold_multiplier,
        refractory_sec=pre_trigger_sec + post_trigger_sec,
    )
    trigger_indices = list(trigger_indices[:num_triggers])

    if not trigger_indices:
        return None
    return trigger_indices, times, voltages
