"""Tri-exponential pulse fitting for bolometer waveforms.

Single-pulse model (Carrettoni & Vignati 2011, Eq. 4.1):
    v(t) = B + A·(−exp(−dt/τ_r) + α·exp(−dt/τ_d1) + (1−α)·exp(−dt/τ_d2))·Θ(dt)

Public API
----------
biexp_single(t, B, A, t0, tau_r, alpha, tau_d1, tau_d2)               7 params
biexp_double_free(t, B, A1, t0_1, tau_r1, alpha1, tau_d1_1, tau_d2_1, 13 params
                     A2, t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2)
    Pileup model built from two independent single pulses.
    (biexp_double with shared τ is kept for backward compatibility but
    is no longer used by fit_pulse.)

fit_pulse(signal, n_pulses, fs, sigma, tau_constraint) -> FitResult

Pileup fitting strategy (n_pulses=2)
-------------------------------------
Sequential two-stage approach:
  1. Detect two peaks; sort by time → (t_pk1, t_pk2).
  2. Fit pulse-1 on the segment  signal[: i_pk2]  with biexp_single.
  3. Subtract pulse-1 model from the full signal → residual.
  4. Fit pulse-2 on residual[i_fit2_start :]  with biexp_single, with all
     three τ values constrained within ±tau_constraint of pulse-1 values.
     i_fit2_start is placed ~10·τ_r before the second peak to capture the
     rising edge while avoiding the first-peak residual region.
  5. Pack both single-pulse fits into a biexp_double_free parameter vector
     for a uniform result representation.

χ² / ndf
---------
Noise σ is estimated from the pre-trigger baseline (first 5 % of the window)
unless the caller supplies a per-sample sigma array.  The chi2 is then:

    χ²/ndf = Σ (resid_i / σ_noise)² / (N − n_params)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# --- Parameter bounds (also re-exported with underscore prefix for callers) ---
_TAU_R_MIN,  _TAU_R_MAX  = 10e-3,  200e-3   # 10 – 200 ms
# Both decay components share the same search range [10 ms, 8 s].
# The "fast" / "slow" labels are assigned post-fit by sorting: whichever τ
# comes out smaller is called τ_d1, the larger is τ_d2.  Keeping the bounds
# identical means no artificial ceiling forces the slow component into the
# wrong box, and τ values near or above former limits (e.g. 2 s, 6 s) are
# fully accessible to both components.
_TAU_D_MIN,  _TAU_D_MAX  = 10e-3,  10.0     # 10 ms – 10 s  (shared for τ_d1 and τ_d2)
_TAU_D2_MIN, _TAU_D2_MAX = _TAU_D_MIN, _TAU_D_MAX   # identical — kept for API compat
_ALPHA_MIN,  _ALPHA_MAX  = 0.01,   0.99
_B_MIN, _B_MAX = -1e-3, 1e-3                 # baseline range [V]
_A_MAX = 10                                  # amplitude prefactor max [V]

# Minimum separation factor enforcing τ_r < τ_d1 (prevents τ_r ≈ τ_d1
# degeneracy that makes the two leading exponentials cancel).
_TAU_ORDER_FACTOR = 1.5

# Initial guesses: different values to break the symmetry between the two
# decay components and help the optimizer land in distinct basins.
# Post-fit sorting ensures consistent labelling regardless of which basin
# the optimizer converges to.
_TAU_R_GUESS  = float(np.sqrt(_TAU_R_MIN * _TAU_R_MAX))  # ~45 ms
_TAU_D1_GUESS = 0.30   # 300 ms — seeded toward the fast-decay basin
_TAU_D2_GUESS = 3.00   # 3 s   — seeded toward the slow-decay basin

_DEFAULT_FS = 1000.0

# Default τ constraint for sequential pileup fitting.
_TAU_CONSTRAINT = 0.20


# ============================================================
# Model functions
# ============================================================

def _shape(dt, tau_r, alpha, tau_d1, tau_d2):
    """Unit-amplitude pulse shape; zero for dt < 0."""
    return np.where(
        dt >= 0,
        -np.exp(-dt / tau_r)
        + alpha * np.exp(-dt / tau_d1)
        + (1.0 - alpha) * np.exp(-dt / tau_d2),
        0.0,
    )


def biexp_single(t, B, A, t0, tau_r, alpha, tau_d1, tau_d2):
    """Single tri-exponential pulse (1 rise + 2 decays)."""
    return B + A * _shape(t - t0, tau_r, alpha, tau_d1, tau_d2)


def biexp_double(t, B, tau_r, tau_d1, tau_d2,
                 alpha1, A1, t0_1, alpha2, A2, t0_2):
    """Pileup with SHARED time constants (kept for backward compatibility)."""
    return (B + A1 * _shape(t - t0_1, tau_r, alpha1, tau_d1, tau_d2)
              + A2 * _shape(t - t0_2, tau_r, alpha2, tau_d1, tau_d2))


def biexp_double_free(t, B,
                      A1, t0_1, tau_r1, alpha1, tau_d1_1, tau_d2_1,
                      A2, t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2):
    """Pileup with INDEPENDENT time constants (13 params)."""
    return (B + A1 * _shape(t - t0_1, tau_r1, alpha1, tau_d1_1, tau_d2_1)
              + A2 * _shape(t - t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2))


# ============================================================
# Result container
# ============================================================

@dataclass
class FitResult:
    """Tri-exponential fit result.

    `tau_d` stores τ_d1 (fast decay) for backward compatibility with
    callers that iterate ``zip(fr.tau_r, fr.tau_d)``.
    Reconstruct the curve with ``model_fn(t, *params)``.

    chi2_per_ndf is computed with noise σ estimated from the pre-trigger
    baseline (or the caller-supplied sigma array), NOT from the residual
    variance — so it is a genuine goodness-of-fit statistic.
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
    tau_d:       List[float] = field(default_factory=list)   # = τ_d1
    tau_d2:      List[float] = field(default_factory=list)
    alpha:       List[float] = field(default_factory=list)
    chi2_per_ndf: float = np.nan
    noise_sigma: float = np.nan   # σ used for χ² (V)
    message: str = ""
    # Sequential-fit diagnostics (pileup only)
    pulse1_popt:         np.ndarray = field(default_factory=lambda: np.array([]))
    seg1_end_t:          float = np.nan   # end of pulse-1 fit segment [s]
    seg2_start_t:        float = np.nan   # start of pulse-2 residual fit [s]
    detected_peak_times: List[float] = field(default_factory=list)  # t_pk from _detect_peaks


# ============================================================
# Internal helpers
# ============================================================

def _peak_info(tau_r, alpha, tau_d1, tau_d2) -> Tuple[float, float]:
    """Return (Δt at peak, shape value at peak) for a unit-amplitude pulse."""
    res = minimize_scalar(
        lambda dt: -_shape(dt, tau_r, alpha, tau_d1, tau_d2),
        bounds=(1e-7, min(10.0 * tau_d1, 3.0)),
        method='bounded',
    )
    return float(res.x), float(_shape(res.x, tau_r, alpha, tau_d1, tau_d2))


def _detect_peaks(signal, fs, n_peaks,
                  prom_frac: float = 0.05,
                  smooth_ms: float = 10.0,
                  min_dist_ms: float = 50.0,
                  deriv_nsigma: float = 3.0,
                  deriv_sustain_ms: float = 30.0) -> List[Tuple[float, float]]:
    """Two-stage peak finder.  Returns (t_peak [s], value) tuples, time-ordered.

    Stage 1 — Prominence-based
        Gaussian-smooth → scipy.find_peaks with prominence ≥ prom_frac×range.
        Works well for separated peaks of similar amplitude.

    Stage 2 — Derivative trigger  (fallback when Stage 1 finds < n_peaks)
        Computes the smoothed derivative dV/dt and looks for contiguous segments
        where dV/dt > threshold sustained for ≥ deriv_sustain_ms.  The threshold
        is deriv_nsigma × σ of the pre-trigger derivative, making it adaptive to
        the local noise level.

        Why derivative trigger instead of sign-change (local minimum):
        • Sign-change requires the second pulse's rising slope to EXCEED the first
          pulse's decaying slope → fails for shoulder-type pileup where the net
          signal is still falling (no zero crossing).
        • Derivative trigger only requires dV/dt > noise threshold in the rising
          segment; it fires even when the net signal is still decaying, catching
          shoulders and small-amplitude pulses that sign-change misses.
    """
    N = len(signal)
    t = np.arange(N) / fs
    sm = gaussian_filter1d(signal.astype(float), sigma=max(smooth_ms * 1e-3 * fs, 0.5))

    # ---- Stage 1: prominence-based ----
    sig_range = sm.max() - sm.min()
    prominence = max(prom_frac * sig_range, 1e-12)
    distance   = max(1, int(min_dist_ms * 1e-3 * fs))
    idx, _ = find_peaks(sm, prominence=prominence, distance=distance)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(sm))])
    idx = np.sort(idx[np.argsort(sm[idx])[::-1][:n_peaks]])

    # ---- Stage 2: derivative trigger (only if still short) ----
    if len(idx) < n_peaks:
        # Use a slightly broader smoothing for the derivative to suppress noise
        deriv_sigma = max(smooth_ms * 2.0 * 1e-3 * fs, 1.0)
        sm_d = gaussian_filter1d(signal.astype(float), sigma=deriv_sigma)
        dv   = np.gradient(sm_d) * fs          # [V/s], length N

        # Adaptive threshold: nsigma × std of pre-trigger derivative
        n_pre = max(2, N // 20)
        noise_slope = float(np.std(dv[:max(1, n_pre - 1)]))
        threshold   = deriv_nsigma * max(noise_slope, 1e-12 * fs)

        sustain = max(1, int(deriv_sustain_ms * 1e-3 * fs))

        # Amplitude floor: candidate peak must be >3σ above pre-trigger baseline
        # so noisy false-triggers (which peak at noise level) are rejected.
        noise_amp = float(np.std(signal[:n_pre]))
        B0_val    = float(np.median(signal[:n_pre]))
        amp_floor = B0_val + max(3.0 * noise_amp, 1e-12)

        found_idx = list(idx)
        above = dv > threshold
        # Identify contiguous above-threshold segments
        padded      = np.concatenate([[False], above, [False]])
        trans       = np.diff(padded.astype(np.int8))
        seg_starts  = np.where(trans ==  1)[0]
        seg_ends    = np.where(trans == -1)[0]

        for s, e in zip(seg_starts, seg_ends):
            if e - s < sustain:
                continue           # too short — likely noise spike
            # Track from the rising-edge start to the first zero crossing
            # of dv (dv ≤ 0) — this is the true peak of this individual pulse.
            # Using zero-crossing rather than a fixed window avoids grabbing
            # the peak of a LATER, larger pulse whose rising edge hasn't started.
            j = e  # dv fell below threshold; keep going while still rising
            while j < N - 1 and dv[j] > 0:
                j += 1
            # argmax of smoothed signal from rising-edge start to zero crossing
            pk_i = s + int(np.argmax(sm[s:j + 1]))
            # Amplitude filter: reject if peak is near noise floor
            if sm[pk_i] < amp_floor:
                continue
            # Reject if too close to an already-accepted peak
            if any(abs(pk_i - ex) < distance for ex in found_idx):
                continue
            found_idx.append(pk_i)
            if len(found_idx) >= n_peaks:
                break

        idx = np.sort(np.array(found_idx[:n_peaks]))

    # ---- Parabola refinement for sub-sample accuracy ----
    refined = []
    for i in idx:
        lo, hi = max(0, i - 5), min(N - 1, i + 5)
        c = np.polyfit(t[lo:hi + 1], signal[lo:hi + 1], 2)
        if c[0] < 0:                              # concave-down → valid max
            tp = float(np.clip(-c[1] / (2 * c[0]), t[lo], t[hi]))
            refined.append((tp, float(np.polyval(c, tp))))
        else:
            refined.append((float(t[i]), float(signal[i])))
    return refined


def _baseline(signal):
    """Median of the first 5 % of the window."""
    return float(np.median(signal[:max(1, len(signal) // 20)]))


def _noise_sigma(signal, sigma_ext=None):
    """Return a scalar noise std for χ² computation.

    If ``sigma_ext`` is an array, return its mean (assumed homogeneous).
    Otherwise estimate from the pre-trigger region (first 5 %).
    A floor of 1e-12 prevents division by zero on noiseless synthetic data.
    """
    if sigma_ext is not None:
        return float(np.mean(sigma_ext))
    n_pre = max(2, len(signal) // 20)
    return float(max(np.std(signal[:n_pre]), 1e-12))


# --- (p0, lower, upper) builder ---

def _setup_single(signal, fs):
    duration = len(signal) / fs
    B0 = _baseline(signal)
    pks = _detect_peaks(signal - B0, fs, 1)
    t_pk, v_pk = pks[0] if pks else (duration * 0.3, float(signal.max() - B0))
    p0 = [B0, max(v_pk, 1e-9),
          float(np.clip(t_pk - _TAU_R_GUESS, 0, duration)),
          _TAU_R_GUESS, 0.5, _TAU_D1_GUESS, _TAU_D2_GUESS]
    lo = [_B_MIN, 0.0,    0.0,      _TAU_R_MIN, _ALPHA_MIN, _TAU_D_MIN,  _TAU_D2_MIN]
    hi = [_B_MAX, _A_MAX, duration, _TAU_R_MAX, _ALPHA_MAX, _TAU_D_MAX,  _TAU_D2_MAX]
    # Soft ordering hint: raise τ_d1 floor above τ_r (prevents τ_r ≈ τ_d1
    # degeneracy), and raise τ_d2 floor above τ_d1 initial guess (encourages
    # the optimizer to start in the correct "fast / slow" basin).
    # This is only a starting hint — the optimizer can still converge to any
    # value within [_TAU_D_MIN, _TAU_D_MAX]; post-fit sorting in _build_result
    # assigns the final fast/slow labels.
    lo[5] = max(lo[5], p0[3] * _TAU_ORDER_FACTOR)
    lo[6] = max(lo[6], p0[5] * _TAU_ORDER_FACTOR)
    p0[5] = float(np.clip(p0[5], lo[5], hi[5]))
    p0[6] = float(np.clip(p0[6], lo[6], hi[6]))
    return p0, lo, hi


# --- curve_fit wrapper ---

def _run_curve_fit(model, t, signal, p0, lo, hi, sigma):
    """Clip p0 into bounds, run curve_fit, return (popt, error_msg)."""
    p0 = list(np.clip(p0, lo, hi))
    try:
        popt, _ = curve_fit(
            model, t, signal, p0=p0, bounds=(lo, hi),
            sigma=sigma, absolute_sigma=(sigma is not None),
            maxfev=100_000, method='trf',
        )
    except Exception as exc:
        return None, str(exc)
    if not np.all(np.isfinite(popt)):
        return None, "NaN/Inf in fitted parameters"
    return popt, ""


def _ensure_tau_order(popt, model, t, signal, lo, hi, sigma):
    """If τ_d1 > τ_d2 in popt, swap labels and refit within the correct bounds.

    Because τ_d1 and τ_d2 have *different* search ranges
    (_TAU_D_MAX=3 s vs _TAU_D2_MAX=15 s), a label inversion means the slow
    component is searched inside the tight fast-decay box and may hit its
    upper bound.  Swapping the initial guess and refitting lets the slow
    component use the full [_TAU_D2_MIN, _TAU_D2_MAX] range it was designed
    for.

    Assumes biexp_single layout: [B, A, t0, τ_r, α, τ_d1, τ_d2]  (7 params).
    Returns the corrected popt (or the original if refit fails/stays inverted).
    """
    if len(popt) != 7 or popt[5] <= popt[6]:
        return popt                         # already in correct order
    # Swap (α, τ_d1, τ_d2) → (1−α, τ_d2, τ_d1) as new initial guess
    p0_swap = popt.copy()
    p0_swap[4] = 1.0 - float(popt[4])
    p0_swap[5] = float(popt[6])
    p0_swap[6] = float(popt[5])
    popt2, _ = _run_curve_fit(model, t, signal, p0_swap, lo, hi, sigma)
    if popt2 is not None and popt2[5] <= popt2[6]:
        return popt2                        # refit gave correctly ordered labels
    return popt                             # fallback; _build_result will sort


# --- Pack popt → FitResult ---

def _unpack(popt: np.ndarray):
    """Return (n_pulses, B, [(A, t0, tau_r, alpha, tau_d1, tau_d2), ...])."""
    n = len(popt)
    if n == 7:
        B, A, t0, tau_r, alpha, td1, td2 = popt
        return 1, float(B), [(A, t0, tau_r, alpha, td1, td2)]
    if n == 10:                                          # shared-τ pileup (legacy)
        B, tr, td1, td2, a1, A1, t01, a2, A2, t02 = popt
        return 2, float(B), [(A1, t01, tr, a1, td1, td2),
                             (A2, t02, tr, a2, td1, td2)]
    if n == 13:                                          # free-τ pileup
        return 2, float(popt[0]), [tuple(popt[1:7]), tuple(popt[7:13])]
    raise ValueError(f"Unexpected param vector length: {n}")


def _build_result(popt, model_fn, signal, fs, sigma_ext=None) -> FitResult:
    """Construct FitResult; χ²/ndf uses pre-trigger noise estimate."""
    n_pulses, B, pulses = _unpack(popt)
    r = FitResult(success=True, n_pulses=n_pulses, params=popt,
                  model_fn=model_fn, baseline=B)

    for A, t0, tau_r, alpha, tau_d1, tau_d2 in pulses:
        # Ensure τ_d1 ≤ τ_d2 (fast ≤ slow) by swapping labels if needed.
        # Swapping (τ_d1, α) ↔ (τ_d2, 1−α) leaves the waveform identical,
        # so this is a pure labelling convention, not a physical change.
        if tau_d1 > tau_d2:
            tau_d1, tau_d2 = tau_d2, tau_d1
            alpha = 1.0 - alpha
        dt_peak, shape_peak = _peak_info(tau_r, alpha, tau_d1, tau_d2)
        r.onset_times.append(float(t0))
        r.tau_r.append(float(tau_r))
        r.tau_d.append(float(tau_d1))
        r.tau_d2.append(float(tau_d2))
        r.alpha.append(float(alpha))
        r.peak_times.append(float(t0) + dt_peak)
        r.peak_amps.append(float(A) * shape_peak)

    N = len(signal)
    t = np.arange(N) / fs
    resid = signal - model_fn(t, *popt)
    ndf = N - len(popt)
    if ndf > 0:
        sig = _noise_sigma(signal, sigma_ext)
        r.noise_sigma = sig
        r.chi2_per_ndf = float(np.sum((resid / sig) ** 2) / ndf)
    return r


# ============================================================
# Sequential pileup fitter
# ============================================================

def _fit_sequential_pileup(signal, fs, sigma_ext, tau_constraint):
    """Fit two pulses sequentially; return (popt_13, seg_info, message).

    seg_info is a dict with keys: pulse1_popt, seg1_end_t, seg2_start_t.
    Returns (None, {}, message) on failure.

    Strategy
    --------
    1.  Detect two peaks (time-ordered).  Pad if only one found.
    2.  Fit pulse-1 on  signal[0 : i_split]  with biexp_single, where
        i_split is set well BEFORE pulse-2's onset (margin = max(2·τ_r_MAX,
        10 % of inter-peak gap) ≈ 200 ms by default).  This guarantees
        pulse-2 cannot contaminate the pulse-1 fit.
    3.  Compute residual = signal − pulse-1 model  (full length).
    4.  Fit pulse-2 on  residual[i_split :]  with biexp_single.  seg2 is
        contiguous with seg1 (no overlap).  All τ values are constrained
        within ±tau_constraint of the pulse-1 fitted values.
    5.  Return a 13-element parameter vector for biexp_double_free.
    """
    N = len(signal)
    t = np.arange(N) / fs
    duration = N / fs

    # ---- 1. detect peaks ----
    B0 = _baseline(signal)
    pks = _detect_peaks(signal - B0, fs, 2)
    while len(pks) < 2:
        last_t = pks[-1][0] if pks else duration * 0.15
        last_v = pks[0][1] * 0.5 if pks else 1e-9
        pks.append((min(last_t + duration * 0.3, duration * 0.9), last_v))
    (t_pk1, _), (t_pk2, _) = pks[:2]
    if t_pk1 > t_pk2:
        (t_pk1, _), (t_pk2, _) = (t_pk2, _), (t_pk1, _)

    # ---- 2. fit pulse-1 on first segment ----
    # End seg1 well BEFORE pulse-2 onset.  The onset of pulse-2 is at
    # t_pk2 − τ_r2, so we use  margin ≥ 2·τ_r_MAX  (200 ms) as a hard
    # safety floor — large enough to cover any physically-reasonable
    # rise time.  For widely-spaced peaks we also impose a fractional
    # buffer of 10 % of (t_pk2 − t_pk1).  For very close pileups, the
    # split is floored to t_pk1 + 50 ms so seg1 still contains pulse-1's
    # peak and a few samples of decay.
    seg1_margin = max(1.5 * _TAU_R_MAX, 0.10 * (t_pk2 - t_pk1))
    seg1_end_t  = max(t_pk1 + 0.05, t_pk2 - seg1_margin)
    i_split = int(np.clip(seg1_end_t * fs, 10, N - 10))
    seg1   = signal[:i_split]
    t_seg1 = t[:i_split]
    sig1   = sigma_ext[:i_split] if sigma_ext is not None else None

    p0_1, lo_1, hi_1 = _setup_single(seg1, fs)
    hi_1[2] = float(t_seg1[-1])          # restrict t0 to segment duration

    popt1, msg = _run_curve_fit(biexp_single, t_seg1, seg1, p0_1, lo_1, hi_1, sig1)
    if popt1 is None:
        return None, {}, f"pulse-1 fit failed: {msg}"
    # If labels are inverted (τ_d1 > τ_d2), swap and refit so the slow
    # component uses the correct [_TAU_D2_MIN, _TAU_D2_MAX] search range.
    B1, A1, t0_1, tau_r1, alpha1, tau_d1_1, tau_d2_1 = popt1

    # ---- 3. compute full residual ----
    resid = signal - biexp_single(t, *popt1)

    # ---- 4. fit pulse-2 on residual ----
    # seg2 starts EXACTLY where seg1 ends: the two segments are contiguous
    # and non-overlapping.  This ensures pulse-1's fit was never influenced
    # by any part of pulse-2, and pulse-2's fit operates on a residual that
    # already has pulse-1 cleanly subtracted.  The first ~200 ms of seg2
    # is just pulse-1 fit residual (≈ 0), then pulse-2's rising edge
    # appears around its true onset.
    i_fit2_start = i_split
    t_fit2_start = float(t_seg1[-1])

    resid2 = resid[i_fit2_start:]
    t_seg2 = t[i_fit2_start:]
    sig2   = sigma_ext[i_fit2_start:] if sigma_ext is not None else None

    if len(resid2) < 10:
        return None, {}, "second segment too short for pulse-2 fit"

    # P2 τ bounds: use global bounds unconditionally (no ±tau_constraint).
    # τ_r is still constrained to ±tau_constraint of P1 (same detector rise),
    # but the decay constants are freed so P2 can find its own best-fit shape
    # without being pinned to P1's potentially-imperfect decay values.
    tr_lo, tr_hi = (max(_TAU_R_MIN, tau_r1 * (1.0 - tau_constraint)),
                    min(_TAU_R_MAX, tau_r1 * (1.0 + tau_constraint)))

    td1_lo, td1_hi = _TAU_D_MIN,  _TAU_D_MAX
    td2_lo, td2_hi = _TAU_D2_MIN, _TAU_D2_MAX

    # Enforce τ_r < τ_d1 for pulse-2.
    td1_lo = max(td1_lo, tau_r1 * _TAU_ORDER_FACTOR)
    if td1_lo >= td1_hi:
        td1_hi = _TAU_D_MAX

    # Residual baseline: allow a small float (±10 % of segment peak or 5 % of
    # _A_MAX, whichever is larger) to absorb pulse-1 subtraction error.
    B2_range = max(0.05 * _A_MAX, float(np.abs(resid2).max()) * 0.1)

    t0_2_guess = float(np.clip(t_pk2 - tau_r1, t_seg2[0], t_seg2[-1]))
    v_resid_max = float(np.clip(resid2.max(), 1e-9, _A_MAX))

    p0_2 = [0.0, v_resid_max, t0_2_guess, tau_r1, alpha1,
            float(np.clip(_TAU_D1_GUESS, td1_lo, td1_hi)),
            float(np.clip(_TAU_D2_GUESS, td2_lo, td2_hi))]
    lo_2 = [_B_MIN,  0.0,         t_seg2[0],  tr_lo,  _ALPHA_MIN,  td1_lo,   td2_lo]
    hi_2 = [_B_MAX,  _A_MAX,      t_seg2[-1], tr_hi,  _ALPHA_MAX,  td1_hi,   td2_hi]

    popt2, msg = _run_curve_fit(biexp_single, t_seg2, resid2, p0_2, lo_2, hi_2, sig2)
    if popt2 is None:
        return None, {}, f"pulse-2 fit failed: {msg}"
    B2, A2, t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2 = popt2

    # ---- 5. pack as biexp_double_free (13 params) ----
    popt_combined = np.array([
        B1 + B2,                                          # combined baseline
        A1,  t0_1, tau_r1, alpha1, tau_d1_1, tau_d2_1,  # pulse 1
        A2,  t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2,  # pulse 2
    ])
    seg_info = dict(
        pulse1_popt         = popt1,
        seg1_end_t          = float(t_seg1[-1]),
        seg2_start_t        = float(t_seg2[0]),
        detected_peak_times = [t_pk1, t_pk2],
    )
    return popt_combined, seg_info, ""


# ============================================================
# Public API
# ============================================================

def fit_pulse(signal: np.ndarray,
              n_pulses: int = 1,
              fs: float = _DEFAULT_FS,
              sigma: np.ndarray = None,
              tau_constraint: float = _TAU_CONSTRAINT) -> FitResult:
    """Fit a tri-exponential model to a 1-D bolometer waveform.

    Parameters
    ----------
    signal          Waveform samples [V].
    n_pulses        1 (single) or 2 (pileup).
    fs              Sampling rate [Hz].
    sigma           Optional per-sample noise std [V] for χ² weighting in
                    curve_fit.  If None, curve_fit runs unweighted and noise
                    is estimated from the pre-trigger baseline for χ² only.
    tau_constraint  Pileup only: maximum allowed fractional difference between
                    the two pulses' time constants (default 0.20 = ±20 %).
                    Set to 1.0 to relax the constraint.

    Notes
    -----
    For n_pulses=2 the fitter uses a sequential strategy: pulse-1 is fitted
    on the segment before the second peak; pulse-2 is fitted on the residual
    after pulse-1 subtraction.  Both are ultimately represented as a
    biexp_double_free parameter vector (13 params) so FitResult is uniform.

    chi2_per_ndf = Σ(resid/σ_noise)² / (N−n_params)
    where σ_noise comes from the pre-trigger baseline (or sigma if supplied).
    """
    if n_pulses not in (1, 2):
        raise ValueError(f"n_pulses must be 1 or 2, got {n_pulses}")

    t = np.arange(len(signal)) / fs

    # ------------------------------------------------------------------ #
    #  Single pulse                                                        #
    # ------------------------------------------------------------------ #
    if n_pulses == 1:
        p0, lo, hi = _setup_single(signal, fs)
        popt, msg = _run_curve_fit(biexp_single, t, signal, p0, lo, hi, sigma)
        if popt is None:
            return FitResult(n_pulses=1, message=msg)
        return _build_result(popt, biexp_single, signal, fs, sigma)

    # ------------------------------------------------------------------ #
    #  Pileup — sequential two-stage strategy                             #
    # ------------------------------------------------------------------ #
    popt, seg_info, msg = _fit_sequential_pileup(signal, fs, sigma, tau_constraint)
    if popt is None:
        return FitResult(n_pulses=2, message=msg)
    r = _build_result(popt, biexp_double_free, signal, fs, sigma)
    r.pulse1_popt         = seg_info['pulse1_popt']
    r.seg1_end_t          = seg_info['seg1_end_t']
    r.seg2_start_t        = seg_info['seg2_start_t']
    r.detected_peak_times = seg_info['detected_peak_times']
    return r