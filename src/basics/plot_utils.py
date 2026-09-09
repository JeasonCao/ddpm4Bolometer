"""Shared utilities for plot scripts."""

import json
import os
from datetime import datetime

import h5py
import numpy as np
from scipy.optimize import curve_fit


def timestamped_path(path: str) -> str:
    """Insert a `_YYYYMMDD_HHMMSS` timestamp before the file extension, so
    repeated runs never overwrite a previous output."""
    root, ext = os.path.splitext(path)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{root}_{stamp}{ext}'


def suffix_from_model_dir(model_dir: str) -> str:
    """Build a plot filename suffix from a model output directory.

    Reads config.json and history.json in model_dir, and uses best_model.pt
    mtime as the training date.  Plot date is today.

    Example: '_l2_T50_ep72_tr20260605_pl20260606'
    """
    cfg_path     = os.path.join(model_dir, 'config.json')
    history_path = os.path.join(model_dir, 'history.json')
    model_path   = os.path.join(model_dir, 'best_model.pt')

    # --- loss and T from config.json ---
    loss_tag = 'unk'
    T_tag    = 'unk'
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        loss_tag = cfg.get('loss', 'unk')
        T_tag    = str(cfg.get('T', 'unk'))

    # --- best epoch from history.json (min val_loss) ---
    ep_tag = 'unk'
    if os.path.isfile(history_path):
        with open(history_path) as f:
            hist = json.load(f)
        val_losses = hist.get('val_loss', [])
        if val_losses:
            best_ep = int(val_losses.index(min(val_losses))) + 1  # 1-indexed
            ep_tag  = f'ep{best_ep}'

    # --- training date from best_model.pt mtime ---
    tr_tag = 'unk'
    if os.path.isfile(model_path):
        mtime  = os.path.getmtime(model_path)
        tr_tag = 'tr' + datetime.fromtimestamp(mtime).strftime('%Y%m%d')

    # --- plot date ---
    pl_tag = 'pl' + datetime.now().strftime('%Y%m%d')

    return f'_{loss_tag}_T{T_tag}_{ep_tag}_{tr_tag}_{pl_tag}'


def fit_energy_calibration(fit_files: list, which: str = 'noisy',
                           use_maxbase: bool = False) -> tuple:
    """Linear least-squares calibration: E [keV] = k * amp [V] + b.

    which     : 'noisy' or 'denoised' — which amplitude estimator to calibrate
    use_maxbase: use max-baseline amplitude instead of tri-exp fitted amplitude

    Returns (k [keV/V], b [keV]).
    Falls back to zero-intercept (b=0) when only one energy point is available.
    """
    if which == 'noisy':
        amp_key  = 'amp_maxbase_noisy'    if use_maxbase else 'amp_noisy'
        conv_key = None                   if use_maxbase else 'converged_noisy'
    else:
        amp_key  = 'amp_maxbase_denoised' if use_maxbase else 'amp_denoised'
        conv_key = None                   if use_maxbase else 'converged_denoised'

    amp_means, energies = [], []
    for path in fit_files:
        with h5py.File(path, 'r') as f:
            energy_kev = float(f.attrs['energy_kev'])
            amp  = f[amp_key][:]
            mask = np.ones(len(amp), dtype=bool) if use_maxbase else f[conv_key][:].astype(bool)
        amp_mean = np.nanmean(amp[mask])
        if np.isfinite(amp_mean) and amp_mean > 0:
            amp_means.append(amp_mean)
            energies.append(energy_kev)

    if len(energies) == 0:
        raise ValueError("No valid events found for energy calibration.")
    if len(energies) == 1:
        k = energies[0] / amp_means[0]
        return float(k), 0.0

    k, b = np.polyfit(amp_means, energies, 1)
    return float(k), float(b)


def logistic4(x, L, U, x0, k):
    """4-parameter logistic: L/U = lower/upper asymptote, x0 = inflection
    point, k = steepness. `x` is typically log(energy)."""
    return L + (U - L) / (1.0 + np.exp(-k * (x - x0)))


def fit_efficiency_curve(energies: np.ndarray, efficiencies: np.ndarray,
                         threshold_pct: float = 90.0):
    """Fit a 4-parameter logistic turn-on curve to an efficiency-vs-energy
    series, in log(energy) space.

    Efficiency curves are S-shaped and saturate at both ends — a polynomial
    fits this poorly (it either oscillates near a flat tail, since nothing
    forces it to flatten, or over/undershoots between sparse points). A
    logistic is monotonic and bounded by construction, matching the actual
    shape with no extra truncation logic needed.

    Returns (popt, log_e_min, log_e_max, threshold_energy_kev or None).
    popt = (L, U, x0, k); evaluate with `logistic4(logE, *popt)`.
    """
    log_e = np.log(np.asarray(energies, dtype=float))
    eff   = np.asarray(efficiencies, dtype=float)

    L0, U0 = float(eff.min()), float(eff.max())
    if U0 - L0 < 1e-6:
        U0 = L0 + 1.0
    mid = 0.5 * (L0 + U0)
    x0_0 = float(log_e[np.argmin(np.abs(eff - mid))])

    p0 = [L0, U0, x0_0, 1.0]
    # Bounds wide enough to cover curves that are already (near-)saturated
    # across the whole data range (e.g. reconstruction-after at low noise),
    # where the data has no real "low" asymptote to anchor L near 0.
    bounds = ([-20.0, -20.0, log_e.min() - 3.0, 1e-3],
             [110.0, 110.0, log_e.max() + 3.0, 50.0])
    try:
        popt, _ = curve_fit(logistic4, log_e, eff, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        popt = np.array(p0)

    L, U, x0, k = popt
    threshold = None
    if L < threshold_pct < U:
        log_e_thr = x0 - (1.0 / k) * np.log((U - L) / (threshold_pct - L) - 1.0)
        if log_e.min() - 3.0 <= log_e_thr <= log_e.max() + 3.0:
            threshold = float(np.exp(log_e_thr))

    return popt, float(log_e.min()), float(log_e.max()), threshold


def double_logistic(x, L, M, U, k1, x1, k2, x2):
    """Two-stage logistic: L -> M (transition at x1, rate k1) -> U
    (transition at x2, rate k2). Still monotonic and bounded (each stage
    is a logistic), but can fit a curve with a shallow rise followed by a
    much sharper jump — which a single 4-parameter logistic averages over
    and underfits in the middle."""
    return (L + (M - L) / (1.0 + np.exp(-k1 * (x - x1)))
              + (U - M) / (1.0 + np.exp(-k2 * (x - x2))))


def fit_efficiency_curve_double(energies: np.ndarray, efficiencies: np.ndarray,
                                threshold_pct: float = 90.0):
    """Fit a 7-parameter double-logistic to an efficiency-vs-energy series
    that has a visible two-stage shape (e.g. a slow rise followed by an
    abrupt jump) — see `double_logistic`. Use only when a single logistic
    visibly underfits; for clean single-stage S-curves prefer
    `fit_efficiency_curve`.

    Returns (popt, log_e_min, log_e_max, threshold_energy_kev or None).
    popt = (L, M, U, k1, x1, k2, x2); evaluate with `double_logistic(logE, *popt)`.
    """
    log_e = np.log(np.asarray(energies, dtype=float))
    eff   = np.asarray(efficiencies, dtype=float)

    # Initial guess: locate the single biggest jump between consecutive
    # (energy-sorted) points to seed the second transition (x2, M); seed the
    # first transition (x1) at the midpoint of the data preceding that jump.
    diffs = np.diff(eff)
    jump_idx = int(np.argmax(diffs))
    M0  = float(eff[jump_idx])
    x2_0 = float(0.5 * (log_e[jump_idx] + log_e[jump_idx + 1]))
    x1_0 = float(log_e[jump_idx // 2]) if jump_idx > 0 else float(log_e[0])
    L0, U0 = float(eff.min()), float(eff.max())

    p0 = [L0, M0, U0, 1.0, x1_0, 5.0, x2_0]
    bounds = ([-20.0, -20.0, -20.0, 1e-3, log_e.min() - 3.0, 1e-3, log_e.min() - 3.0],
             [110.0, 110.0, 110.0, 50.0, log_e.max() + 3.0, 50.0, log_e.max() + 3.0])
    try:
        popt, _ = curve_fit(double_logistic, log_e, eff, p0=p0, bounds=bounds, maxfev=30000)
    except (RuntimeError, ValueError):
        popt = np.array(p0)

    L, M, U, k1, x1, k2, x2 = popt
    threshold = None
    if L < threshold_pct < U:
        xs = np.linspace(log_e.min() - 3.0, log_e.max() + 3.0, 4000)
        vals = double_logistic(xs, *popt) - threshold_pct
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                from scipy.optimize import brentq
                log_e_thr = brentq(lambda x: double_logistic(x, *popt) - threshold_pct,
                                   xs[i], xs[i + 1])
                threshold = float(np.exp(log_e_thr))
                break

    return popt, float(log_e.min()), float(log_e.max()), threshold
