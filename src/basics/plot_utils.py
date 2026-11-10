"""Shared utilities for plot scripts."""

import glob
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


# 5 gamma-line-like calibration points, used by the figures that span the full
# energy range (resolution vs energy, reconstructed amplitude vs true energy).
# Deliberately excludes 3034 keV (the 0vbb ROI — real experiments don't
# calibrate on the signal region itself).
#
# 2026-09-22: this is NOT the point set for every figure. The efficiency
# figures (reconstruction efficiency, reconstruction + DT trigger) are plotted
# only up to 511 keV and are calibrated on the points inside that range, via
# the `calib_energies` argument below. That split is the project's long-
# standing convention and predates 2026-09-09; a 2026-09-10 change collapsed
# everything onto this one high-energy set, which was an over-correction to
# what was really a figure-label inconsistency (2.20 vs 2.22 keV shown for the
# same noise level) and was reverted on 2026-09-22.
#
# Why the split matters: the simulated detector response is mildly non-linear
# (slope drifts 774 -> 788 keV/V from 1 to 3034 keV), so a high-energy-only
# calibration leaves a +1 to +2% common-mode bias down at 1-50 keV. The
# efficiency cut is |dE/E| < 5%, so that bias eats a large part of the
# acceptance window and understates the measured efficiency by ~20% in
# threshold terms.
CALIB_ENERGIES_KEV = (239.0, 511.0, 727.0, 1461.0, 2615.0)

# 2026-09-22: the calibration slope that DEFINES the noise levels, used only
# for the "baseline resolution = ... keV" label on figures.
#
# The 1.382 mV noise level was chosen on 2026-09-12 by inverting
#     FWHM_baseline = 2.355 * noise_rms * k
# at k = 780.32 keV/V to land on CUORE's quoted 2.54 keV. Quoting the label
# with a k refit from each figure's own data instead makes it drift (782.06
# here, so the label read 2.55 keV) and invites exactly the question "we say
# we match CUORE's number but we print something 0.01 keV away from it" —
# which is the same class of problem as the 2.20-vs-2.22 inconsistency that
# was (over-)corrected on 2026-09-10.
#
# The label is a property of the noise level, fixed when that level was
# defined, so it takes the defining slope. It is NOT the calibration applied
# to any figure's data — those use CALIB_ENERGIES_KEV or the <=511 keV set.
#   1.382 mV -> 2.355 * 0.001382 * 780.32 = 2.54 keV
#   7.5   mV -> 2.355 * 0.0075   * 780.32 = 13.78 keV
CANONICAL_CALIB_K_KEV_PER_V = 780.32


def baseline_resolution_kev(noise_rms_v: float) -> float:
    """Baseline resolution FWHM in keV for a given noise RMS, at the canonical
    calibration slope. Single source of truth for every figure's label."""
    return 2.355 * noise_rms_v * CANONICAL_CALIB_K_KEV_PER_V


def calib_energies_below(fit_dir: str, max_energy: float):
    """Every energy point available in `fit_dir` at or below `max_energy`.

    This is the calibration point set for the efficiency figures: they are
    plotted over the same range they are calibrated on, so the response
    non-linearity is absorbed by the fit instead of showing up as a bias.
    """
    out = []
    for p in glob.glob(os.path.join(fit_dir, 'fit_efficiency_*keV.h5')):
        stem = os.path.basename(p)[len('fit_efficiency_'):-len('keV.h5')]
        try:
            E = float(stem)
        except ValueError:
            continue
        if E <= max_energy:
            out.append(E)
    return tuple(sorted(out))


def fit_global_calibration_zero_intercept(fit_dir: str, method: str = 'triexp',
                                          calib_energies=None):
    """Global multi-point calibration E = k*Amp, forced through the origin
    (b=0, so amp=0 <-> E=0 exactly — avoids the free-intercept version's
    failure mode where a small b fit from high-energy points only
    extrapolates to large, even negative, predicted energies at low
    amplitude/low energy).

    Uses DENOISED Gaussian-fit means (mirrors real experiments calibrating on
    their best/final-processing amplitude estimator), inverse-variance
    weighted by the Gaussian fit's own mu_err. Falls back to the raw denoised
    array's mean/SE when the Gaussian fit didn't converge for a given point.

    method: 'triexp' or 'maxbase'.
    calib_energies: point set to fit on. Defaults to CALIB_ENERGIES_KEV (the
        5 gamma-line-like high-energy points) for the full-range figures; the
        efficiency figures pass calib_energies_below(fit_dir, 511) instead.

    Returns (k [keV/V], residuals) where residuals is a tuple of
    (E_true, amp, E_pred, resid_keV) for the calibration points actually
    used, for logging/sanity-checking.
    """
    suf_d = 'triexp_denoised' if method == 'triexp' else 'maxbase_denoised'
    amp_key = 'amp_denoised' if method == 'triexp' else 'amp_maxbase_denoised'
    if calib_energies is None:
        calib_energies = CALIB_ENERGIES_KEV

    amps, errs, energies = [], [], []
    for E in calib_energies:
        cands = glob.glob(
            os.path.join(fit_dir, f'fit_efficiency_{int(round(E))}keV.h5'))
        if not cands:
            continue
        with h5py.File(cands[0], 'r') as f:
            converged = bool(f.attrs.get(f'gauss_converged_{suf_d}', False))
            if converged:
                mu  = float(f.attrs[f'gauss_mu_{suf_d}'])
                err = float(f.attrs[f'gauss_mu_err_{suf_d}'])
            else:
                arr = f[amp_key][:].astype(float)
                mu  = float(np.nanmean(arr))
                err = float(np.nanstd(arr) / np.sqrt(np.sum(np.isfinite(arr))))
        amps.append(mu); errs.append(max(err, 1e-12)); energies.append(E)

    if len(energies) < 2:
        raise ValueError(
            f'Not enough calibration points found in {fit_dir} for method={method} '
            f'(found {len(energies)}/{len(calib_energies)})')

    amps = np.array(amps); errs = np.array(errs); energies = np.array(energies)
    w = 1.0 / errs**2
    # Weighted least squares through the origin: k = sum(w*E*amp) / sum(w*amp^2)
    k = float(np.sum(w * energies * amps) / np.sum(w * amps**2))

    residuals = tuple(
        (float(E), float(amp), k * float(amp), k * float(amp) - float(E))
        for E, amp in zip(energies, amps)
    )
    return k, residuals


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
