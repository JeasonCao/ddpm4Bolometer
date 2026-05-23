"""
periodic.py — Periodic noise removal (Stage 3 of pipeline).

Three options for removing periodic (narrowband) noise:
  3a. Wiener spectral filter — frequency-domain magnitude masking
  3b. Before/after sinusoidal fit — phase-coherent interpolation through pulse
  3c. Adaptive notch (LMS) — time-domain tracking of multiple sinusoids

Methods 3b and 3c auto-detect spike frequencies from quiet windows and perform
phase-aware subtraction, preserving signal content at noise frequencies.
"""

import numpy as np
from scipy.signal import welch, find_peaks


def find_adjacent_quiet(
    voltages: np.ndarray,
    rolling_med: np.ndarray,
    rolling_mad: np.ndarray,
    t_start: int,
    window_len: int,
    mad_threshold: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find adjacent quiet window for noise characterization.

    Returns (values, indices) of quiet residual samples adjacent to
    [t_start, t_start + window_len).  Always contiguous.

    Priority: full before -> full after -> longest contiguous quiet
    segment from before or after (may be shorter than window_len).
    Returns (None, None) if no quiet samples found.
    """
    t_end = t_start + window_len
    residual = voltages - rolling_med

    # Option 1: full window before
    before_start = t_start - window_len
    if before_start >= 0:
        seg = residual[before_start:t_start]
        mad_ref = rolling_mad[t_start]
        if np.max(np.abs(seg)) < mad_threshold * mad_ref:
            indices = np.arange(before_start, t_start)
            return seg.copy(), indices

    # Option 2: full window after
    after_end = t_end + window_len
    if after_end <= len(voltages):
        seg = residual[t_end:after_end]
        mad_ref = rolling_mad[t_end]
        if np.max(np.abs(seg)) < mad_threshold * mad_ref:
            indices = np.arange(t_end, after_end)
            return seg.copy(), indices

    # Option 3: longest contiguous quiet run from before or after
    best_seg = None
    best_idx = None

    # Scan before: find longest contiguous quiet run ending at t_start
    if before_start >= 0:
        seg = residual[before_start:t_start]
        mad_ref = rolling_mad[t_start]
        run = 0
        for k in range(len(seg) - 1, -1, -1):
            if abs(seg[k]) < mad_threshold * mad_ref:
                run += 1
            else:
                break
        if run > 0:
            start_k = t_start - run
            best_seg = residual[start_k:t_start].copy()
            best_idx = np.arange(start_k, t_start)

    # Scan after: find longest contiguous quiet run starting at t_end
    search_end = min(len(voltages), t_end + 2 * window_len)
    if t_end < len(voltages):
        seg = residual[t_end:search_end]
        mad_ref = rolling_mad[t_end]
        run = 0
        for k in range(len(seg)):
            if abs(seg[k]) < mad_threshold * mad_ref:
                run += 1
            else:
                break
        if run > 0 and (best_seg is None or run > len(best_seg)):
            best_seg = residual[t_end:t_end + run].copy()
            best_idx = np.arange(t_end, t_end + run)

    if best_seg is not None:
        return best_seg, best_idx

    return None, None


def find_adjacent_quiet_info(
    voltages: np.ndarray,
    rolling_med: np.ndarray,
    rolling_mad: np.ndarray,
    t_start: int,
    window_len: int,
    mad_threshold: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None, str, int, int]:
    """Like find_adjacent_quiet, but also returns source location info.

    Returns (data, indices, source, n_before, n_after) where:
    - data: quiet window values or None (always contiguous)
    - indices: absolute sample indices of each value, or None
    - source: 'before', 'after', 'partial_before', 'partial_after', or 'none'
    - n_before: number of samples taken from before region
    - n_after: number of samples taken from after region
    """
    t_end = t_start + window_len
    residual = voltages - rolling_med

    # Option 1: full window before
    before_start = t_start - window_len
    if before_start >= 0:
        seg = residual[before_start:t_start]
        mad_ref = rolling_mad[t_start]
        if np.max(np.abs(seg)) < mad_threshold * mad_ref:
            indices = np.arange(before_start, t_start)
            return seg.copy(), indices, "before", window_len, 0

    # Option 2: full window after
    after_end = t_end + window_len
    if after_end <= len(voltages):
        seg = residual[t_end:after_end]
        mad_ref = rolling_mad[t_end]
        if np.max(np.abs(seg)) < mad_threshold * mad_ref:
            indices = np.arange(t_end, after_end)
            return seg.copy(), indices, "after", 0, window_len

    # Option 3: longest contiguous quiet run from before or after
    best_seg = None
    best_idx = None
    best_source = "none"
    best_n_before = 0
    best_n_after = 0

    # Scan before: contiguous quiet run ending at t_start
    if before_start >= 0:
        seg = residual[before_start:t_start]
        mad_ref = rolling_mad[t_start]
        run = 0
        for k in range(len(seg) - 1, -1, -1):
            if abs(seg[k]) < mad_threshold * mad_ref:
                run += 1
            else:
                break
        if run > 0:
            start_k = t_start - run
            best_seg = residual[start_k:t_start].copy()
            best_idx = np.arange(start_k, t_start)
            best_source = "partial_before"
            best_n_before = run

    # Scan after: contiguous quiet run starting at t_end
    search_end = min(len(voltages), t_end + 2 * window_len)
    if t_end < len(voltages):
        seg = residual[t_end:search_end]
        mad_ref = rolling_mad[t_end]
        run = 0
        for k in range(len(seg)):
            if abs(seg[k]) < mad_threshold * mad_ref:
                run += 1
            else:
                break
        if run > 0 and (best_seg is None or run > len(best_seg)):
            best_seg = residual[t_end:t_end + run].copy()
            best_idx = np.arange(t_end, t_end + run)
            best_source = "partial_after"
            best_n_before = 0
            best_n_after = run

    if best_seg is not None:
        return best_seg, best_idx, best_source, best_n_before, best_n_after

    return None, None, "none", 0, 0


def estimate_noise_psd(
    quiet_window: np.ndarray, fs: float = 5000.0, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate noise PSD from a quiet window using Welch's method.

    By default nperseg equals the window length (raw periodogram), giving
    maximum frequency resolution so that narrow periodic spikes are captured
    at their true height — critical for the Wiener mask ratio S_nn/S_xx.

    Returns (freqs, psd) arrays.
    """
    if nperseg is None:
        nperseg = len(quiet_window)
    freqs, psd = welch(quiet_window, fs=fs, nperseg=min(nperseg, len(quiet_window)),
                       window="hann", scaling="density")
    return freqs, psd


def build_wiener_mask(
    signal_fft_mag2: np.ndarray,
    noise_psd_interp: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build Wiener suppression mask in frequency domain.

    H(f) = max(0, 1 - alpha * S_nn(f) / S_xx(f))

    Parameters
    ----------
    signal_fft_mag2 : |FFT(signal)|^2 at each frequency bin
    noise_psd_interp : noise PSD interpolated to match FFT bin frequencies
    alpha : oversubtraction factor (1.0 = standard Wiener)

    Returns
    -------
    mask : array of shape (n_fft_bins,), values in [0, 1]
    """
    mask = 1.0 - alpha * noise_psd_interp / (signal_fft_mag2 + 1e-30)
    return np.clip(mask, 0.0, 1.0)


def apply_wiener_filter(
    window: np.ndarray,
    noise_psd: np.ndarray,
    noise_freqs: np.ndarray,
    fs: float = 5000.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply Wiener spectral filter to a window.

    1. FFT the window
    2. Interpolate noise PSD to FFT frequency grid
    3. Build Wiener mask: H(f) = max(0, 1 - alpha * S_nn / |FFT|^2)
    4. Apply mask and IFFT back

    Parameters
    ----------
    window : time-domain signal (float64)
    noise_psd : noise PSD values from estimate_noise_psd
    noise_freqs : frequency values from estimate_noise_psd
    fs : sampling rate
    alpha : oversubtraction factor

    Returns
    -------
    cleaned : filtered window, same length as input (real-valued)
    """
    n = len(window)
    fft_vals = np.fft.rfft(window)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag2 = np.abs(fft_vals) ** 2

    # Interpolate noise PSD to match FFT frequency grid
    noise_psd_interp = np.interp(fft_freqs, noise_freqs, noise_psd)

    # Scale noise PSD from density to match periodogram magnitude
    # Welch returns one-sided PSD in V^2/Hz; |FFT|^2 = PSD * N * fs / 2
    noise_power = noise_psd_interp * n * fs / 2

    mask = build_wiener_mask(fft_mag2, noise_power, alpha)
    filtered_fft = fft_vals * mask

    return np.fft.irfft(filtered_fft, n=n)


# ═══════════════════════════════════════════════════════════════════════
# Shared: auto-detect spike frequencies from quiet window PSD
# ═══════════════════════════════════════════════════════════════════════

def _find_spectral_peaks(data, fs, min_freq, max_freq, prominence_factor):
    """Find local outlier peaks in log-amplitude spectrum.

    Returns (peak_bin_indices_in_full_fft, fft_freqs, fft_amp, fft_phase).
    """
    n = len(data)
    fft_vals = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_amp = np.abs(fft_vals) * 2 / n
    fft_phase = np.angle(fft_vals)

    mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
    idx_offset = np.argmax(mask)
    amp_region = fft_amp[mask]

    # Frequency-adaptive local outlier detection in log-amplitude space
    log_amp = np.log(amp_region + 1e-30)
    df = fft_freqs[1] - fft_freqs[0]

    min_half = max(5, int(2.0 / df))
    max_half = max(25, len(amp_region) // 50)
    half_wins = np.linspace(min_half, max_half, len(amp_region)).astype(int)

    n_bins = len(log_amp)
    local_med = np.empty(n_bins)
    local_resid = np.empty(n_bins)
    for i in range(n_bins):
        hw = half_wins[i]
        lo = max(0, i - hw)
        hi = min(n_bins, i + hw + 1)
        local_med[i] = np.median(log_amp[lo:hi])
        local_resid[i] = log_amp[i] - local_med[i]

    local_mad = np.empty(n_bins)
    abs_resid = np.abs(local_resid)
    for i in range(n_bins):
        hw = half_wins[i]
        lo = max(0, i - hw)
        hi = min(n_bins, i + hw + 1)
        local_mad[i] = np.median(abs_resid[lo:hi])
    local_mad = np.maximum(local_mad, 1e-10)

    deviation = local_resid / local_mad
    peaks, _ = find_peaks(deviation, height=prominence_factor)

    full_indices = peaks + idx_offset
    return full_indices, fft_freqs, fft_amp, fft_phase


def detect_spike_frequencies(
    quiet_window: np.ndarray,
    fs: float = 5000.0,
    prominence_factor: float = 5.0,
    min_freq: float = 0.5,
    max_freq: float | None = None,
    signal_window: np.ndarray | None = None,
    coincidence_tol_hz: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-detect periodic noise frequencies via coincidence between two spectra.

    Finds peaks that are local outliers in both the quiet window AND the signal
    window (if provided). Periodic noise appears in both; random fluctuations
    appear in only one.

    Parameters
    ----------
    quiet_window : noise-only time series (for peak detection + amplitudes)
    fs : sampling rate
    prominence_factor : peaks must exceed local median by this many local MADs
    min_freq : ignore frequencies below this (Hz)
    max_freq : ignore frequencies above this (Hz), default fs/2
    signal_window : pulse/signal time series for coincidence check. If None,
        only quiet_window peaks are used.
    coincidence_tol_hz : two peaks are coincident if within this tolerance (Hz)

    Returns
    -------
    spike_freqs : detected spike frequencies (Hz), refined via parabolic interp
    spike_amplitudes : amplitude at each spike (V, from quiet FFT)
    spike_phases : phase at each spike (radians, from quiet FFT)
    """
    if max_freq is None:
        max_freq = fs / 2

    # Find peaks in quiet window
    q_indices, fft_freqs, fft_amp, fft_phase = _find_spectral_peaks(
        quiet_window, fs, min_freq, max_freq, prominence_factor)

    if len(q_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # Coincidence: keep only peaks that also appear in signal window
    if signal_window is not None:
        s_indices, s_freqs, _, _ = _find_spectral_peaks(
            signal_window, fs, min_freq, max_freq, prominence_factor)

        if len(s_indices) > 0:
            s_peak_freqs = s_freqs[s_indices]
            keep = []
            for qi in q_indices:
                q_freq = fft_freqs[qi]
                # Check if any signal peak is within tolerance
                if np.min(np.abs(s_peak_freqs - q_freq)) <= coincidence_tol_hz:
                    keep.append(qi)
            q_indices = np.array(keep, dtype=int) if keep else np.array([], dtype=int)
        else:
            q_indices = np.array([], dtype=int)

    if len(q_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # Refine peak frequencies via parabolic interpolation on log-amplitude
    df = fft_freqs[1] - fft_freqs[0]
    refined_freqs = np.empty(len(q_indices))
    for i, k in enumerate(q_indices):
        if 1 <= k <= len(fft_amp) - 2:
            alpha = np.log(fft_amp[k - 1] + 1e-30)
            beta = np.log(fft_amp[k] + 1e-30)
            gamma = np.log(fft_amp[k + 1] + 1e-30)
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-30:
                delta = 0.5 * (alpha - gamma) / denom
                refined_freqs[i] = fft_freqs[k] + delta * df
            else:
                refined_freqs[i] = fft_freqs[k]
        else:
            refined_freqs[i] = fft_freqs[k]

    spike_amplitudes = fft_amp[q_indices]
    spike_phases = fft_phase[q_indices]

    return refined_freqs, spike_amplitudes, spike_phases


# ═══════════════════════════════════════════════════════════════════════
# 3b. Before/after sinusoidal fit with interpolation
# ═══════════════════════════════════════════════════════════════════════

def _fit_sinusoids(data: np.ndarray, freqs: np.ndarray, fs: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Fit multiple sinusoids to data via least-squares.

    Model: data(t) ≈ Σ_k [a_k·cos(2π·f_k·t) + b_k·sin(2π·f_k·t)]

    Returns (amplitudes, phases) for each frequency where:
        data_k(t) = amplitude_k · cos(2π·f_k·t + phase_k)
    """
    n = len(data)
    t = np.arange(n) / fs
    n_freq = len(freqs)

    # Build design matrix: [cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
    A = np.zeros((n, 2 * n_freq))
    for k, f in enumerate(freqs):
        omega_t = 2 * np.pi * f * t
        A[:, 2 * k] = np.cos(omega_t)
        A[:, 2 * k + 1] = np.sin(omega_t)

    # Least-squares solve
    coeffs, _, _, _ = np.linalg.lstsq(A, data, rcond=None)

    amplitudes = np.zeros(n_freq)
    phases = np.zeros(n_freq)
    for k in range(n_freq):
        a_k = coeffs[2 * k]
        b_k = coeffs[2 * k + 1]
        amplitudes[k] = np.sqrt(a_k**2 + b_k**2)
        phases[k] = np.arctan2(-b_k, a_k)  # cos(ωt + φ) = a·cos(ωt) + b·sin(ωt) when b = -A·sin(φ)

    return amplitudes, phases


def apply_sinusoidal_subtraction(
    window: np.ndarray,
    fs: float = 5000.0,
    t_start_samples: int = 0,
    spike_freqs: np.ndarray | None = None,
    quiet_ref: np.ndarray | None = None,
    prominence_factor: float = 5.0,
    min_freq: float = 0.5,
) -> np.ndarray:
    """Remove periodic noise by fitting sinusoids directly on the pulse window.

    Spike frequencies are detected from a quiet reference segment (or provided
    directly). Amplitude and phase are fitted on the pulse window itself — since
    the pulse is broadband, a narrow-band sinusoidal fit captures mostly the
    periodic noise, not the pulse signal.

    Parameters
    ----------
    window : pulse window (baseline-subtracted residual)
    fs : sampling rate
    t_start_samples : absolute sample index of window start (for phase coherence)
    spike_freqs : pre-detected spike frequencies (Hz). If None, auto-detects
        from quiet_ref.
    quiet_ref : quiet reference segment for spike detection when spike_freqs
        is None.
    prominence_factor : for spike detection (only when spike_freqs is None)
    min_freq : minimum frequency to detect (only when spike_freqs is None)

    Returns
    -------
    cleaned : window with periodic noise subtracted
    """
    # Detect spikes if not provided
    if spike_freqs is None:
        if quiet_ref is None:
            return window.copy()
        spike_freqs, _, _ = detect_spike_frequencies(
            quiet_ref, fs=fs, prominence_factor=prominence_factor, min_freq=min_freq)

    if len(spike_freqs) == 0:
        return window.copy()

    n_win = len(window)
    t_abs_win = np.arange(n_win) + t_start_samples

    # Fit amplitude and phase on the pulse window directly
    amp, phase = _fit_sinusoids_absolute(window, spike_freqs, t_abs_win, fs)

    # Build and subtract noise estimate
    t_win = t_abs_win / fs
    noise_estimate = np.zeros(n_win)
    for k, f in enumerate(spike_freqs):
        noise_estimate += amp[k] * np.cos(2 * np.pi * f * t_win + phase[k])

    return window - noise_estimate


def _fit_sinusoids_absolute(
    data: np.ndarray, freqs: np.ndarray, t_abs_samples: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sinusoids using absolute time references.

    Model: data(i) ≈ Σ_k [a_k·cos(2π·f_k·t_abs[i]/fs) + b_k·sin(2π·f_k·t_abs[i]/fs)]

    Returns (amplitudes, phases) where:
        noise_k(t) = amplitude_k · cos(2π·f_k·t + phase_k)
    """
    t = t_abs_samples / fs
    n = len(data)
    n_freq = len(freqs)

    A = np.zeros((n, 2 * n_freq))
    for k, f in enumerate(freqs):
        omega_t = 2 * np.pi * f * t
        A[:, 2 * k] = np.cos(omega_t)
        A[:, 2 * k + 1] = np.sin(omega_t)

    coeffs, _, _, _ = np.linalg.lstsq(A, data, rcond=None)

    amplitudes = np.zeros(n_freq)
    phases = np.zeros(n_freq)
    for k in range(n_freq):
        a_k = coeffs[2 * k]
        b_k = coeffs[2 * k + 1]
        amplitudes[k] = np.sqrt(a_k**2 + b_k**2)
        phases[k] = np.arctan2(-b_k, a_k)

    return amplitudes, phases


# ═══════════════════════════════════════════════════════════════════════
# 3c. Adaptive notch filter (multi-frequency LMS)
# ═══════════════════════════════════════════════════════════════════════

def apply_adaptive_notch(
    window: np.ndarray,
    fs: float = 5000.0,
    t_start_samples: int = 0,
    spike_freqs: np.ndarray | None = None,
    quiet_ref: np.ndarray | None = None,
    prominence_factor: float = 5.0,
    min_freq: float = 0.5,
) -> np.ndarray:
    """Remove periodic noise by fitting sinusoids directly on the pulse window.

    Uses least-squares (same as 3b) but kept as a separate entry point for
    API compatibility. Spike frequencies are detected from a quiet reference
    or provided directly.

    Parameters
    ----------
    window : pulse window (baseline-subtracted residual)
    fs : sampling rate
    t_start_samples : absolute sample index of window start
    spike_freqs : pre-detected spike frequencies (Hz). If None, auto-detects
        from quiet_ref or window.
    quiet_ref : quiet reference segment for spike detection when spike_freqs
        is None.
    prominence_factor : for spike detection (only when spike_freqs is None)
    min_freq : minimum frequency to detect (only when spike_freqs is None)

    Returns
    -------
    cleaned : window with periodic noise subtracted
    """
    # Detect spikes if not provided
    if spike_freqs is None:
        ref = quiet_ref if quiet_ref is not None else window
        spike_freqs, _, _ = detect_spike_frequencies(
            ref, fs=fs, prominence_factor=prominence_factor, min_freq=min_freq)

    if len(spike_freqs) == 0:
        return window.copy()

    # Delegate to sinusoidal subtraction (same underlying method)
    return apply_sinusoidal_subtraction(
        window, fs=fs, t_start_samples=t_start_samples, spike_freqs=spike_freqs)
