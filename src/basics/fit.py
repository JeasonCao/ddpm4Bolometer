"""
Tri-exponential pulse fitting for low-temperature calorimeters.

Single-pulse model (Carrettoni & Vignati 2011, eq. 4.1) — one rise + two decays:

    v(t) = B + A * phi(t - t0; tau_r, alpha, tau_d1, tau_d2)
    phi(u; ...) = (-exp(-u/tau_r)
                   + alpha       * exp(-u/tau_d1)
                   + (1 - alpha) * exp(-u/tau_d2))   for u >= 0
                = 0                                   for u < 0

Real bolometer pulses have a fast (tau_d1) plus slow (tau_d2) decay component;
the single-decay biexponential underfits and leaves systematic residuals. The
prefactor A is NOT the peak amplitude — phi(u) has a unit-amplitude rise term
but a mixed decay, so the peak value of phi is computed numerically per pulse.

Fixed-vs-free conventions (matched to the inference QA pipeline):
    - Baseline B fixed = mean of first 1500 samples.
    - First-pulse onset t0 fixed = t0_fixed (default 1.5 s).
    - For pileup the two pulses share (tau_r, tau_d1, tau_d2); each has its own
      (A, alpha). Second-pulse onset t02 is bounded to
      [t0_fixed + dt_min, t0_fixed + dt_max]  (default 0.01-2.0 s).
    - chi2/ndf = SSR / (N - len(popt)), unweighted.

Public API
----------
triexp_single(t, B, A, t0, tau_r, alpha, tau_d1, tau_d2)            7 params
triexp_double(t, B, tau_r, tau_d1, tau_d2,                         10 params
             alpha1, A1, t01, alpha2, A2, t02)
fit_pulse(signal, n_pulses, fs=1000, t0_fixed=1.5,
          pileup_dt_range=(0.01, 2.0)) -> FitResult
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ── Parameter bounds ───────────────────────────────────────────────────────
_TAU_R_MIN,  _TAU_R_MAX  = 1e-3,   100e-3   # 1 - 100 ms
_TAU_D_MIN,  _TAU_D_MAX  = 10e-3,  1.0      # 10 ms - 1 s   (fast decay tau_d1)
_TAU_D2_MIN, _TAU_D2_MAX = 100e-3, 5.0      # 100 ms - 5 s  (slow decay tau_d2)
_ALPHA_MIN,  _ALPHA_MAX  = 0.01,   0.99
_A_MAX = 2.0                                # voltage prefactor max [V]

# Geometric-mean initial guesses
_TAU_R_GUESS  = float(np.sqrt(_TAU_R_MIN  * _TAU_R_MAX))
_TAU_D1_GUESS = float(np.sqrt(_TAU_D_MIN  * _TAU_D_MAX))
_TAU_D2_GUESS = float(np.sqrt(_TAU_D2_MIN * _TAU_D2_MAX))

_DEFAULT_FS         = 1000.0
_DEFAULT_T0_FIXED   = 1.5
_DEFAULT_DT_RANGE   = (0.01, 2.0)
_N_BASELINE_SAMPLES = 1500


# ── Model functions ────────────────────────────────────────────────────────

def _shape(dt, tau_r, alpha, tau_d1, tau_d2):
    """Unit-prefactor pulse shape; zero for dt < 0."""
    # Clamp to dt >= 0 inside the exp to avoid overflow on the masked-out
    # branch (np.where evaluates both branches before selecting).
    u = np.maximum(dt, 0.0)
    return np.where(
        dt >= 0,
        -np.exp(-u / tau_r)
        + alpha       * np.exp(-u / tau_d1)
        + (1.0 - alpha) * np.exp(-u / tau_d2),
        0.0,
    )


def triexp_single(t, B, A, t0, tau_r, alpha, tau_d1, tau_d2):
    """Single tri-exponential pulse on baseline B."""
    return B + A * _shape(t - t0, tau_r, alpha, tau_d1, tau_d2)


def triexp_double(t, B, tau_r, tau_d1, tau_d2,
                 alpha1, A1, t01, alpha2, A2, t02):
    """Pileup with shared (tau_r, tau_d1, tau_d2); per-pulse (A, alpha, t0)."""
    return (B
            + A1 * _shape(t - t01, tau_r, alpha1, tau_d1, tau_d2)
            + A2 * _shape(t - t02, tau_r, alpha2, tau_d1, tau_d2))


# ── Fit-result container ───────────────────────────────────────────────────

@dataclass
class FitResult:
    """Tri-exponential fit result.

    For shared-tau pileup the per-pulse tau_r/tau_d/tau_d2 lists each contain
    two identical entries (the shared values), so callers iterating
    ``zip(fr.tau_r, fr.tau_d)`` keep working.
    Reconstruct the curve with ``model_fn(t, *params)``.
    """
    success: bool = False
    n_pulses: int = 1
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    model_fn: Callable = None
    baseline: float = 0.0
    peak_amps:   List[float] = field(default_factory=list)
    onset_times: List[float] = field(default_factory=list)
    peak_times:  List[float] = field(default_factory=list)
    tau_r:       List[float] = field(default_factory=list)
    tau_d:       List[float] = field(default_factory=list)   # = tau_d1
    tau_d2:      List[float] = field(default_factory=list)
    alpha:       List[float] = field(default_factory=list)
    ssr: float = float('nan')
    ndf: int = 0
    chi2_per_ndf: float = float('nan')
    residual: np.ndarray = field(default_factory=lambda: np.array([]))
    message: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────

def _peak_info(tau_r, alpha, tau_d1, tau_d2) -> Tuple[float, float]:
    """Return (Δt at peak, shape value at peak) for a unit-prefactor pulse."""
    res = minimize_scalar(
        lambda dt: -_shape(dt, tau_r, alpha, tau_d1, tau_d2),
        bounds=(1e-7, min(10.0 * tau_d1, 3.0)),
        method='bounded',
    )
    return float(res.x), float(_shape(res.x, tau_r, alpha, tau_d1, tau_d2))


def peak_time(t0: float, tau_r: float, alpha: float,
              tau_d1: float, tau_d2: float) -> float:
    """Time of the maximum of triexp_single given onset t0 and tri-exp params."""
    dt_peak, _ = _peak_info(tau_r, alpha, tau_d1, tau_d2)
    return float(t0) + dt_peak


def _detect_peaks(signal, fs, n_peaks,
                  prom_frac: float = 0.10,
                  smooth_ms: float = 10.0,
                  min_dist_ms: float = 50.0) -> List[Tuple[float, float]]:
    """Top n_peaks peaks (time-ordered) of ``signal`` after Gaussian smoothing.

    Each peak is parabola-refined for sub-sample accuracy.
    Returns a list of (t_peak [s], value) tuples.
    """
    N = len(signal)
    t = np.arange(N) / fs
    sm = gaussian_filter1d(signal.astype(float),
                           sigma=max(smooth_ms * 1e-3 * fs, 0.5))
    prominence = max(prom_frac * (sm.max() - sm.min()), 1e-12)
    distance = max(1, int(min_dist_ms * 1e-3 * fs))
    idx, _ = find_peaks(sm, prominence=prominence, distance=distance)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(sm))])
    idx = np.sort(idx[np.argsort(sm[idx])[::-1][:n_peaks]])

    refined = []
    for i in idx:
        lo, hi = max(0, i - 5), min(N - 1, i + 5)
        c = np.polyfit(t[lo:hi + 1], signal[lo:hi + 1], 2)
        if c[0] < 0:
            tp = float(np.clip(-c[1] / (2 * c[0]), t[lo], t[hi]))
            refined.append((tp, float(np.polyval(c, tp))))
        else:
            refined.append((float(t[i]), float(signal[i])))
    return refined


def _baseline(signal, n_baseline=_N_BASELINE_SAMPLES):
    """Mean of the first ``n_baseline`` samples (capped at len/4)."""
    n = max(1, min(n_baseline,
                   len(signal) // 4 if len(signal) >= 4 else len(signal)))
    return float(np.mean(signal[:n]))


def _run_curve_fit(model, t, signal, p0, lo, hi):
    """Clip p0 into bounds, run curve_fit, return (popt, error_msg)."""
    p0 = list(np.clip(p0, lo, hi))
    try:
        popt, _ = curve_fit(
            model, t, signal, p0=p0, bounds=(lo, hi),
            maxfev=100_000, method='trf',
        )
    except Exception as exc:
        return None, str(exc)
    if not np.all(np.isfinite(popt)):
        return None, "NaN/Inf in fitted parameters"
    return popt, ""


# ── Result construction ────────────────────────────────────────────────────

def _populate_pulses(r: FitResult, pulses):
    """Fill per-pulse derived quantities from a list of tuples
    (A, t0, tau_r, alpha, tau_d1, tau_d2)."""
    for A, t0, tau_r, alpha, tau_d1, tau_d2 in pulses:
        dt_peak, shape_peak = _peak_info(tau_r, alpha, tau_d1, tau_d2)
        r.onset_times.append(float(t0))
        r.tau_r.append(float(tau_r))
        r.tau_d.append(float(tau_d1))
        r.tau_d2.append(float(tau_d2))
        r.alpha.append(float(alpha))
        r.peak_times.append(float(t0) + dt_peak)
        r.peak_amps.append(float(A) * shape_peak)


def _sort_by_peak_time(r: FitResult):
    """Sort all per-pulse lists by ascending peak time (pileup convention)."""
    if r.n_pulses < 2:
        return
    order = np.argsort(r.peak_times)
    for attr in ('peak_amps', 'onset_times', 'peak_times',
                 'tau_r', 'tau_d', 'tau_d2', 'alpha'):
        setattr(r, attr, [getattr(r, attr)[i] for i in order])


def _build_result(popt, model_fn, signal, fs, n_pulses,
                  success: bool, message: str = "") -> FitResult:
    """Pack a parameter vector + residual stats into a FitResult."""
    if n_pulses == 1:
        B, A, t0, tau_r, alpha, td1, td2 = popt
        pulses = [(A, t0, tau_r, alpha, td1, td2)]
    else:
        B, tr, td1, td2, a1, A1, t01, a2, A2, t02 = popt
        pulses = [(A1, t01, tr, a1, td1, td2),
                  (A2, t02, tr, a2, td1, td2)]

    r = FitResult(success=success, n_pulses=n_pulses,
                  params=np.asarray(popt), model_fn=model_fn,
                  baseline=float(B), message=message)
    _populate_pulses(r, pulses)
    _sort_by_peak_time(r)

    N = len(signal)
    t = np.arange(N) / fs
    residual = signal - model_fn(t, *popt)
    n_params = len(popt)
    ndf = N - n_params
    ssr = float(np.sum(residual ** 2))
    r.ssr = ssr
    r.ndf = ndf
    r.residual = residual
    r.chi2_per_ndf = (ssr / ndf) if (success and ndf > 0) else float('nan')
    return r


# ── Public API ─────────────────────────────────────────────────────────────

def fit_pulse(signal: np.ndarray,
              n_pulses: int = 1,
              fs: float = _DEFAULT_FS,
              t0_fixed: float = _DEFAULT_T0_FIXED,
              pileup_dt_range: tuple = _DEFAULT_DT_RANGE) -> FitResult:
    """Fit a tri-exponential pulse (1 or 2 pulses) to a 1-D bolometer waveform.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Voltage samples in physical units [V].
    n_pulses : int
        1 (single) or 2 (pileup).
    fs : float
        Sample rate [Hz]. Default 1 kHz.
    t0_fixed : float
        Fixed onset time of the first pulse [s]. Default 1.5 s.
    pileup_dt_range : tuple (dt_min, dt_max)
        Allowed onset separation for pileup pulses [s]. Default (0.01, 2.0).

    Notes
    -----
    Baseline B is fixed to the mean of the first 1500 samples. For pileup the
    two pulses share (tau_r, tau_d1, tau_d2); each gets its own (A, alpha).
    """
    if n_pulses not in (1, 2):
        raise ValueError(f"n_pulses must be 1 or 2, got {n_pulses}")

    N = len(signal)
    t = np.arange(N) / fs
    duration = N / fs
    B_fixed = _baseline(signal)

    # ── Single pulse: free (A, tau_r, alpha, tau_d1, tau_d2) ──
    if n_pulses == 1:
        pks = _detect_peaks(signal - B_fixed, fs, 1)
        v_pk = pks[0][1] if pks else float(np.max(signal) - B_fixed)
        A_init = max(v_pk, 1e-9)

        def _model(tt, A, tr, alpha, td1, td2):
            return triexp_single(tt, B_fixed, A, t0_fixed, tr, alpha, td1, td2)

        p0 = [A_init, _TAU_R_GUESS, 0.5,        _TAU_D1_GUESS, _TAU_D2_GUESS]
        lo = [0.0,    _TAU_R_MIN,   _ALPHA_MIN, _TAU_D_MIN,    _TAU_D2_MIN]
        hi = [_A_MAX, _TAU_R_MAX,   _ALPHA_MAX, _TAU_D_MAX,    _TAU_D2_MAX]
        popt_free, msg = _run_curve_fit(_model, t, signal, p0, lo, hi)

        if popt_free is None:
            A_, tr_, a_, td1_, td2_ = np.clip(p0, lo, hi)
            popt_full = np.array([B_fixed, A_, t0_fixed, tr_, a_, td1_, td2_])
            return _build_result(popt_full, triexp_single, signal, fs, 1,
                                 success=False, message=msg)
        A_, tr_, a_, td1_, td2_ = popt_free
        popt_full = np.array([B_fixed, A_, t0_fixed, tr_, a_, td1_, td2_])
        return _build_result(popt_full, triexp_single, signal, fs, 1,
                             success=True)

    # ── Pileup: free (tau_r, tau_d1, tau_d2, A1, alpha1, A2, alpha2, t02) ──
    pks = _detect_peaks(signal - B_fixed, fs, 2)
    while len(pks) < 2:
        last_t = pks[-1][0] if pks else t0_fixed + 0.3
        last_v = pks[0][1] * 0.5 if pks else 1e-9
        pks.append((min(last_t + 0.3, duration * 0.95), last_v))
    (tp1, v1), (tp2, v2) = pks[:2]

    dt_min, dt_max = pileup_dt_range
    t02_lo = t0_fixed + dt_min
    t02_hi = min(t0_fixed + dt_max, duration)
    if t02_hi <= t02_lo:
        t02_hi = t02_lo + 1e-3
    t02_init = float(np.clip(tp2 - _TAU_R_GUESS, t02_lo, t02_hi))

    A1_init = max(v1, 1e-9)
    A2_init = max(v2, 1e-9)

    def _model(tt, tr, td1, td2, a1, A1, a2, A2, t02):
        return triexp_double(tt, B_fixed, tr, td1, td2,
                            a1, A1, t0_fixed, a2, A2, t02)

    p0 = [_TAU_R_GUESS, _TAU_D1_GUESS, _TAU_D2_GUESS,
          0.5,        A1_init, 0.5,        A2_init, t02_init]
    lo = [_TAU_R_MIN,  _TAU_D_MIN,    _TAU_D2_MIN,
          _ALPHA_MIN, 0.0,     _ALPHA_MIN, 0.0,     t02_lo]
    hi = [_TAU_R_MAX,  _TAU_D_MAX,    _TAU_D2_MAX,
          _ALPHA_MAX, _A_MAX,  _ALPHA_MAX, _A_MAX,  t02_hi]
    popt_free, msg = _run_curve_fit(_model, t, signal, p0, lo, hi)

    if popt_free is None:
        tr_, td1_, td2_, a1_, A1_, a2_, A2_, t02_ = np.clip(p0, lo, hi)
        popt_full = np.array([B_fixed, tr_, td1_, td2_,
                              a1_, A1_, t0_fixed, a2_, A2_, t02_])
        return _build_result(popt_full, triexp_double, signal, fs, 2,
                             success=False, message=msg)
    tr_, td1_, td2_, a1_, A1_, a2_, A2_, t02_ = popt_free
    popt_full = np.array([B_fixed, tr_, td1_, td2_,
                          a1_, A1_, t0_fixed, a2_, A2_, t02_])
    return _build_result(popt_full, triexp_double, signal, fs, 2,
                         success=True)
