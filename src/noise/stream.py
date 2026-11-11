"""
Continuous (multi-day) CUORE-like noise for the optimal-filter comparison streams.

generator.generate_noise() builds one 10 s window: every component is synthesised
at 10 kHz, the sum is Bessel-filtered and decimated to 1 kHz, and the window is
rescaled to its target RMS. A day of that at 10 kHz does not fit in memory, so this
module produces the same process directly at the 1 kHz output rate:

  * Gaussian part (colored 1/f^alpha + lin-f + white, white floor, mechanical
    resonances): one-sided PSD of each component before the filter, times the
    Bessel |H|^2, folded onto 0-500 Hz the way decimation aliases it, then drawn
    in one FFT per noise state.
  * Harmonics (50 Hz AC, ~1.4 Hz pulse tube): sinusoids evaluated on the 1 kHz
    grid with the Bessel complex gain applied (steady state of filtering a 10 kHz
    sinusoid; sampling on the 1 kHz grid reproduces the decimation aliasing).
  * Slow envelope: one smooth process below 0.2 Hz for the whole run, clipped to
    1 +- envelope_variation.

Below F_LOW the colored PSD is rolled off by f^2/(f^2+F_LOW^2). A 10 s training
window has no content below 0.1 Hz anyway; without the roll-off a multi-day 1/f
stream would wander far off baseline.

A noise "state" is one fixed parameter set. Adjacent states are joined with a
power-complementary (cos/sin) crossfade.
"""

import numpy as np
import scipy.fft as sfft
from scipy.signal import butter, sosfreqz

from src.basics.filters import make_bessel_sos
from src.noise.generator import sample_noise_params, F_INTERNAL

F_OUT = 1000.0
F_LOW = 0.1           # colored-noise roll-off [Hz]
ENV_FCUT = 0.2        # envelope band edge [Hz], as in envelope.py
ENV_STD = 1.0 / 3.0   # envelope shape std before clipping to [-1, 1]
T_WIN = 10.0          # RMS is defined as the std inside a 10 s window (mean removed), as in the eval sets
_BESSEL = make_bessel_sos(120.0, 6, F_INTERNAL)


def _in_window(f):
    """Fraction of a component at f that survives removing the mean of a T_WIN window."""
    return 1.0 - np.sinc(np.asarray(f, dtype=float) * T_WIN) ** 2


def _bessel_h(f):
    """Complex Bessel response at frequencies f [Hz] (10 kHz digital filter)."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    _, h = sosfreqz(_BESSEL, worN=f, fs=F_INTERNAL)
    return h


# ── State parameters ─────────────────────────────────────────────────────────

def _harmonic_set(rng, f0, n_harm, a_base, p_decay, log_rand=True):
    """Per-harmonic amplitudes and phases, drawn the way harmonics.py does."""
    ks = np.arange(1, n_harm + 1)
    ks = ks[ks * f0 < F_INTERNAL / 2]
    lr = rng.uniform(-0.5, 0.5, len(ks)) if log_rand else np.zeros(len(ks))
    return dict(f0=f0, k=ks, f=ks * f0, amp=a_base / ks ** p_decay * 10 ** lr,
                phase=rng.uniform(0, 2 * np.pi, len(ks)))


def random_state(rng, target_rms):
    """One noise state drawn from the noise-pool distribution (sample_noise_params)."""
    p = sample_noise_params(rng, noise_rms=target_rms)
    p['colored_factor'] = rng.uniform(0.3, 1.0)         # drawn inside generate_noise()
    p['ac'] = _harmonic_set(rng, 50.0, p['ac_n_harmonics'], p['ac_a_base'], p['ac_p_decay'])
    p['pt'] = _harmonic_set(rng, p['pt_freq'], p['pt_n_harmonics'], p['pt_a_base'], p['pt_p_decay'])
    p['target_rms'] = target_rms                       # every state sits at the run's RMS
    return p


def midpoint_state(rng, target_rms):
    """Every sampled parameter at the middle of its range; only phases are random.

    Resonances: 4 of them (the integer range is 3-6) at the base frequencies
    10/25/40/70 Hz, Q = 12.5, amplitude 0.3 x RMS. Per-harmonic log-random factors = 1.
    """
    r = target_rms
    alpha, f_cross = 1.0, 1.75
    p = {
        'target_rms': r, 'rms_scale': 1.0,
        'ac_a_base': 0.65 * r, 'ac_p_decay': 1.5, 'ac_n_harmonics': 20,
        'pt_freq': 1.4, 'pt_a_base': 0.5 * r, 'pt_p_decay': 1.5, 'pt_n_harmonics': 50,
        'alpha': alpha, 'f_cross': f_cross, 'a_pink': 1.0,
        'a_lin': 1.0 / f_cross ** (alpha + 1), 'a_white': 1.0 / 50.0 ** alpha * 1.25,
        'white_rms': 0.25 * r,
        'resonances': [dict(f_center=fc, Q=12.5, amplitude=0.3 * r) for fc in (10., 25., 40., 70.)],
        'envelope_variation': 0.1,
        'colored_factor': 0.65,
    }
    p['ac'] = _harmonic_set(rng, 50.0, 20, p['ac_a_base'], 1.5, log_rand=False)
    p['pt'] = _harmonic_set(rng, 1.4, 50, p['pt_a_base'], 1.5, log_rand=False)
    return p


# ── Spectra ──────────────────────────────────────────────────────────────────

def _pre_filter_psd(p, f):
    """One-sided PSD [V^2/Hz] of the Gaussian components before the Bessel filter."""
    f = np.asarray(f, dtype=float)
    fn = F_INTERNAL / 2
    # colored: shape normalised so its in-window variance is (factor * rms)^2, as generate_noise()
    # rescales each window's colored part by its own std
    fg = np.geomspace(1e-4, fn, 200001)
    shape = lambda x: ((p['a_pink'] / x ** p['alpha'] + p['a_lin'] * x + p['a_white'])
                       * x ** 2 / (x ** 2 + F_LOW ** 2))
    var_c = (p['colored_factor'] * p['target_rms']) ** 2
    s = np.where(f > 0, shape(np.maximum(f, 1e-12)), 0.0) * var_c / np.trapezoid(shape(fg) * _in_window(fg), fg)
    s = s + p['white_rms'] ** 2 / fn                       # white floor
    for res in p['resonances']:                            # 2nd-order Butterworth band-pass of white
        bw = res['f_center'] / res['Q']
        lo, hi = max(res['f_center'] - bw / 2, 0.1), min(res['f_center'] + bw / 2, fn - 1)
        if lo >= hi:
            continue
        sos = butter(2, [lo, hi], btype='band', fs=F_INTERNAL, output='sos')
        fb = np.linspace(0, fn, 400001)
        g2 = np.abs(sosfreqz(sos, worN=fb, fs=F_INTERNAL)[1]) ** 2
        s = s + res['amplitude'] ** 2 * np.interp(f, fb, g2) / np.trapezoid(g2, fb)
    return s


def output_psd(p, f_out):
    """One-sided PSD at the 1 kHz output: filtered, then folded by the x10 decimation."""
    f_out = np.asarray(f_out, dtype=float)
    m_max = int(F_INTERNAL // F_OUT)
    images = [f_out + m * F_OUT for m in range(m_max)] + [m * F_OUT - f_out for m in range(1, m_max + 1)]
    tot = np.zeros_like(f_out)
    for fa in images:                                       # every 10 kHz frequency landing on f_out
        ok = (fa > 0) & (fa <= F_INTERNAL / 2)
        if ok.any():
            tot[ok] += _pre_filter_psd(p, fa[ok]) * np.abs(_bessel_h(fa[ok])) ** 2
    return tot


def _psd_grid(p):
    """Output PSD on a grid fine enough to interpolate onto any rfft grid."""
    f = np.unique(np.concatenate([np.geomspace(1e-5, 1.0, 20001), np.arange(1.0, 500.0 + 1e-9, 0.002)]))
    return f, output_psd(p, f)


def _harmonics_out(hs):
    """Output-side amplitudes and phases (Bessel gain applied) of one harmonic set."""
    h = _bessel_h(hs['f'])
    return dict(f0=hs['f0'], k=hs['k'], f=hs['f'], amp=hs['amp'] * np.abs(h), phase=hs['phase'] + np.angle(h))


def finalize_state(p):
    """Scale Gaussian and harmonic parts so the state's RMS inside a 10 s window equals target_rms.

    Adds 'gauss_grid' (f, psd), 'harm' (output-side sets) and 'scale'.
    """
    f, s = _psd_grid(p)
    var_g = np.trapezoid(s * _in_window(f), f)
    harm = [_harmonics_out(p['ac']), _harmonics_out(p['pt'])]
    var_h = sum(0.5 * np.sum(h['amp'] ** 2 * _in_window(h['f'])) for h in harm)
    env_gain = 1.0 + p['envelope_variation'] ** 2 * ENV_STD ** 2
    scale = p['target_rms'] / np.sqrt((var_g + var_h) * env_gain)
    p['gauss_grid'] = (f, s * scale ** 2)
    p['harm'] = [dict(h, amp=h['amp'] * scale) for h in harm]
    p['scale'] = scale
    p['var_parts'] = dict(gauss=var_g * scale ** 2, harm=var_h * scale ** 2, env_gain=env_gain)
    return p


# ── Synthesis ────────────────────────────────────────────────────────────────

def gauss_block(rng, p, n):
    """n samples of the state's Gaussian part at 1 kHz (one FFT)."""
    nf = sfft.next_fast_len(n, real=True)
    f = sfft.rfftfreq(nf, 1.0 / F_OUT)
    s = np.interp(f, *p['gauss_grid'], left=0.0, right=0.0)
    s[0] = 0.0
    a = np.sqrt(s * nf * F_OUT / 4.0)                       # E|X_k|^2 = S nf fs / 2
    x = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * a
    x[-1] = x[-1].real if nf % 2 == 0 else x[-1]
    return sfft.irfft(x, n=nf, workers=-1)[:n]


def harmonics(p, i0, n, chunk=2_000_000):
    """Sum of the state's harmonics at global sample indices i0 .. i0+n-1.

    Each set is sum_k a_k sin(k w n + phi_k) = Im(sum_k c_k z^k), z = exp(i w n).
    If the fundamental divides the 1 kHz rate (50 Hz -> 20 samples) one period is tiled;
    otherwise the polynomial in z is evaluated by Horner's rule.
    """
    out = np.zeros(n)
    for h in p['harm']:
        c = h['amp'] * np.exp(1j * h['phase'])              # c_k for k = 1..K (contiguous)
        assert np.array_equal(h['k'], np.arange(1, len(c) + 1))
        period = F_OUT / h['f0']
        if abs(period - round(period)) < 1e-9:
            P = int(round(period))
            j = np.arange(P)
            one = np.sum(h['amp'][:, None] * np.sin(2 * np.pi * np.outer(h['k'], j) / P + h['phase'][:, None]), 0)
            out += one[(np.arange(i0, i0 + n) % P)]
            continue
        for c0 in range(0, n, chunk):
            idx = np.arange(i0 + c0, i0 + min(n, c0 + chunk), dtype=np.int64)
            z = np.exp(2j * np.pi * np.mod(idx * (h['f0'] / F_OUT), 1.0))
            acc = np.full(len(idx), c[-1])
            for ck in c[-2::-1]:
                acc = acc * z + ck
            out[c0:c0 + len(idx)] += (acc * z).imag
    return out


def envelope_shape(rng, n_total, rate=10.0):
    """Smooth band-limited process (< 0.2 Hz) for the whole run, std ENV_STD, clipped to [-1, 1].

    Returned at `rate` Hz; interpolate onto the 1 kHz grid with envelope_at().
    """
    m = int(np.ceil(n_total / F_OUT * rate)) + 2
    nf = sfft.next_fast_len(m, real=True)
    f = sfft.rfftfreq(nf, 1.0 / rate)
    band = (f > 0) & (f < ENV_FCUT)
    x = np.zeros(len(f), complex)
    x[band] = rng.standard_normal(band.sum()) + 1j * rng.standard_normal(band.sum())
    g = sfft.irfft(x, n=nf)[:m]
    g *= ENV_STD / g.std()
    return np.clip(g, -1.0, 1.0)


def envelope_at(env, i0, n, rate=10.0):
    t = np.arange(i0, i0 + n) / F_OUT
    return np.interp(t, np.arange(len(env)) / rate, env)


def crossfade_weights(n):
    """Power-complementary weights (old, new) over n samples."""
    x = (np.arange(n) + 0.5) / n
    return np.cos(0.5 * np.pi * x), np.sin(0.5 * np.pi * x)
