"""
drift.py — correlated drift removal between channels.

Uses low-pass filtering + linear regression to identify and subtract
the slow thermal drift component shared between LMO and DLMO.

At 1000 channels, this step generalizes to PCA common-mode rejection.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_lowpass(cutoff_hz: float = 0.03, fs: float = 5000.0, order: int = 2) -> np.ndarray:
    """Design a zero-phase Butterworth lowpass filter.

    Uses order=2 with sosfiltfilt (effective order 4 via forward-backward pass).
    Higher orders are numerically unstable at extreme normalized cutoffs.

    Returns SOS coefficients for use with sosfiltfilt.
    """
    return butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


def lowpass_filter(voltages: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply zero-phase lowpass filter. Returns same-length float64 array."""
    return sosfiltfilt(sos, voltages).astype(np.float64)


def fit_drift_predictor(
    ref_low: np.ndarray, target_low: np.ndarray
) -> tuple[float, float]:
    """Ordinary least squares: target_low ≈ slope * ref_low + intercept.

    Returns (slope, intercept).
    """
    slope, intercept = np.polyfit(ref_low, target_low, 1)
    return float(slope), float(intercept)


def remove_correlated_drift(
    lmo_voltages: np.ndarray,
    dlmo_voltages: np.ndarray,
    fs: float = 5000.0,
    cutoff_hz: float = 0.03,
    filter_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Remove the correlated slow drift between LMO and DLMO.

    Steps:
    1. Low-pass filter both channels to isolate slow drift
    2. Fit linear regression: dlmo_slow ≈ slope * lmo_slow + intercept
    3. Subtract slow components from both channels

    Safe for particle signals: pulse timescales (~0.1-2s) are well above
    the low-pass cutoff (~0.03 Hz = 33s period).

    Parameters
    ----------
    lmo_voltages   : LMO channel voltage array
    dlmo_voltages  : DLMO channel voltage array (same length)
    fs             : sampling frequency in Hz
    cutoff_hz      : low-pass cutoff frequency (from coherence analysis)
    filter_order   : Butterworth order (default 2, effective 4 via sosfiltfilt)

    Returns
    -------
    lmo_corrected  : LMO with its own slow drift removed
    dlmo_corrected : DLMO with correlated slow drift removed
    slope          : regression coefficient
    intercept      : regression intercept
    lmo_slow       : LMO slow drift component (for adding back later)
    dlmo_drift     : DLMO drift prediction (slope * lmo_slow + intercept)
    """
    sos = design_lowpass(cutoff_hz, fs, filter_order)
    lmo_slow = lowpass_filter(lmo_voltages, sos)
    dlmo_slow = lowpass_filter(dlmo_voltages, sos)

    slope, intercept = fit_drift_predictor(lmo_slow, dlmo_slow)

    lmo_corrected = lmo_voltages - lmo_slow
    dlmo_drift = slope * lmo_slow + intercept
    dlmo_corrected = dlmo_voltages - dlmo_drift

    return lmo_corrected, dlmo_corrected, slope, intercept, lmo_slow, dlmo_drift
