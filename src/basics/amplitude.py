"""Simple amplitude estimators for bolometer waveforms."""

import numpy as np


def max_baseline_amplitude(signal: np.ndarray,
                           baseline_frac: float = 0.05) -> float:
    """Estimate pulse amplitude as max(signal) − pre-trigger baseline.

    Parameters
    ----------
    signal        Waveform samples [V].
    baseline_frac Fraction of window used to estimate the pre-trigger baseline.

    Returns
    -------
    amplitude : float  [V]
    """
    n_pre    = max(1, int(len(signal) * baseline_frac))
    baseline = float(np.median(signal[:n_pre]))
    return float(signal.max()) - baseline
