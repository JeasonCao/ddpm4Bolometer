"""
CUORE-like noise generator — combines all noise components.

Two noise families are tracked separately so we can ablate the DDPM
denoiser's behavior on each:

  * PERIODIC = AC line harmonics (50 Hz) + PT cooler harmonics (1.4 Hz)
               + mechanical resonances (narrowband bandpass-filtered noise)
  * WHITE    = colored noise (1/f^α + linear + white spectrum) + white floor

`generate_paired_noise()` produces three windows from the *same* component
draws (so they are paired):
  * periodic-only, rescaled to target_rms
  * white-only,    rescaled to target_rms
  * total = scale · (alpha·P + beta·W) with alpha² + beta² = 1
            (energy-preserving mix) and scale ~ U(0.5, 2.0).

Usage:
    from src.noise.generator import generate_paired_noise, sample_noise_params
    rng = np.random.default_rng(42)
    params = sample_noise_params(rng)
    periodic, white, total, meta = generate_paired_noise(rng, params)
"""

import numpy as np

from src.noise.harmonics import generate_harmonics
from src.noise.colored import generate_colored_noise
from src.noise.envelope import generate_slow_envelope
from src.noise.resonances import generate_resonances, sample_resonance_params
from src.basics.filters import apply_bessel_decimate

# Internal sample rate — must match pulse simulator
F_INTERNAL = 10000

# Component → noise-family bucket
PERIODIC_KEYS = ('ac', 'pt', 'resonance')
WHITE_KEYS = ('colored', 'white')
ALL_KEYS = PERIODIC_KEYS + WHITE_KEYS


def sample_noise_params(rng: np.random.Generator,
                        noise_rms: float = 7.5e-3) -> dict:
    """Sample randomized noise parameters for one window.

    target_rms is fixed (= noise_rms); window-level amplitude variation is
    introduced separately by the `scale` factor in generate_paired_noise().
    """
    target_rms = noise_rms

    # 1/f^alpha parameters
    alpha = rng.uniform(0.8, 1.2)
    f_cross = rng.uniform(0.5, 3.0)

    # PSD shape: a_pink/f^α + a_lin*f + a_white
    # At crossover: a_pink / f_cross^α = a_lin * f_cross
    a_pink = 1.0
    a_lin = 1.0 / f_cross ** (alpha + 1)
    # White floor ≈ pink level at 50 Hz, randomized
    a_white = a_pink / 50.0 ** alpha * rng.uniform(0.5, 2.0)

    return {
        # Overall
        'target_rms': target_rms,
        # AC harmonics (50 Hz) — same uniform range as PT for matched scale
        'ac_a_base': target_rms * rng.uniform(0.3, 1.0),
        'ac_p_decay': rng.uniform(1.0, 2.0),
        'ac_n_harmonics': 20,
        # PT cooler harmonics (1.4 Hz)
        'pt_freq': 1.4 + rng.uniform(-0.05, 0.05),
        'pt_a_base': target_rms * rng.uniform(0.3, 1.0),
        'pt_p_decay': rng.uniform(1.0, 2.0),
        'pt_n_harmonics': 50,
        # Colored noise
        'alpha': alpha,
        'f_cross': f_cross,
        'a_pink': a_pink,
        'a_lin': a_lin,
        'a_white': a_white,
        # White noise floor (separate from colored noise)
        'white_rms': target_rms * rng.uniform(0.1, 0.4),
        # Mechanical resonances
        'resonances': sample_resonance_params(rng, target_rms),
        # Envelope
        'envelope_variation': rng.uniform(0.0, 0.2),
        # Colored target (sampled here so paired calls use the same value)
        'colored_target': target_rms * rng.uniform(0.3, 1.0),
    }


def generate_components(rng: np.random.Generator,
                        params: dict,
                        duration: float = 10.0) -> dict:
    """Generate every noise component at the internal sample rate.

    Returns
    -------
    dict with keys 'ac', 'pt', 'colored', 'white', 'resonance' (each a
    1D array of length duration*F_INTERNAL) and 'envelope' (the shared
    multiplicative gain trace, also length duration*F_INTERNAL).
    """
    n_internal = int(duration * F_INTERNAL)

    ac = generate_harmonics(
        rng,
        f_fundamental=50.0,
        n_harmonics=params['ac_n_harmonics'],
        a_base=params['ac_a_base'],
        p_decay=params['ac_p_decay'],
        n_samples=n_internal,
        fs=F_INTERNAL,
    )

    pt = generate_harmonics(
        rng,
        f_fundamental=params['pt_freq'],
        n_harmonics=params['pt_n_harmonics'],
        a_base=params['pt_a_base'],
        p_decay=params['pt_p_decay'],
        n_samples=n_internal,
        fs=F_INTERNAL,
    )

    colored = generate_colored_noise(
        rng, n_internal, F_INTERNAL,
        alpha=params['alpha'],
        a_pink=params['a_pink'],
        a_lin=params['a_lin'],
        a_white=params['a_white'],
    )
    std_colored = np.std(colored)
    if std_colored > 0:
        colored = colored / std_colored * params['colored_target']

    white = rng.standard_normal(n_internal) * params['white_rms']

    resonance = generate_resonances(
        rng, n_internal, F_INTERNAL,
        resonances=params['resonances'],
    )

    envelope = generate_slow_envelope(
        rng, n_internal, F_INTERNAL,
        max_variation=params['envelope_variation'],
    )

    return {
        'ac': ac,
        'pt': pt,
        'colored': colored,
        'white': white,
        'resonance': resonance,
        'envelope': envelope,
    }


def _assemble(components: dict,
              keys,
              params: dict,
              duration: float = 10.0,
              f_sample: float = 1000.0,
              f_bessel: float = 120.0,
              bessel_order: int = 6) -> np.ndarray:
    """Sum the requested components, apply envelope + Bessel + decimate,
    and rescale to params['target_rms'].

    Both periodic and white subsets are passed through the same envelope
    (shared realization) and the same Bessel low-pass + decimation, so
    they are individually realistic and remain consistent under linear
    combination at the output rate.
    """
    summed = np.zeros_like(components['envelope'])
    for k in keys:
        summed = summed + components[k]
    summed = summed * components['envelope']

    n_out = int(duration * f_sample)
    out = apply_bessel_decimate(
        summed, F_INTERNAL, f_sample, f_bessel, bessel_order,
    )[:n_out]

    std = np.std(out)
    if std > 0:
        out = out / std * params['target_rms']
    return out


def generate_noise(rng: np.random.Generator,
                   params: dict,
                   duration: float = 10.0,
                   f_sample: float = 1000.0,
                   f_bessel: float = 120.0,
                   bessel_order: int = 6) -> np.ndarray:
    """Generate one combined-noise window (all components summed, no mix scale).

    Kept as a back-compat helper for QA scripts. New training/inference
    pipelines should use generate_paired_noise().
    """
    comps = generate_components(rng, params, duration=duration)
    return _assemble(comps, ALL_KEYS, params,
                     duration=duration, f_sample=f_sample,
                     f_bessel=f_bessel, bessel_order=bessel_order)


def generate_paired_noise(rng: np.random.Generator,
                          params: dict,
                          duration: float = 10.0,
                          f_sample: float = 1000.0,
                          f_bessel: float = 120.0,
                          bessel_order: int = 6,
                          scale_range: tuple = (0.5, 2.0)
                          ) -> tuple:
    """Generate paired (periodic, white, total) noise from one component draw.

    Mixing
    ------
    Energy-fraction r ~ U(0, 1) sets alpha = sqrt(r), beta = sqrt(1-r),
    so alpha² + beta² = 1 and RMS(alpha·P + beta·W) ≈ target_rms (P, W
    are uncorrelated). Then total = scale · (alpha·P + beta·W) with
    scale ~ U(scale_range), so RMS(total) ≈ scale · target_rms.

    Returns
    -------
    periodic : ndarray, shape (duration*f_sample,)
        Periodic-only window (AC + PT + resonances), rescaled to target_rms.
    white : ndarray, same shape
        White-only window (colored + white floor), rescaled to target_rms.
    total : ndarray, same shape
        Combined window scale · (alpha·periodic + beta·white).
    meta : dict
        {'alpha', 'beta', 'r', 'scale', 'target_rms'} — per-window metadata.
    """
    comps = generate_components(rng, params, duration=duration)

    asm_kwargs = dict(duration=duration, f_sample=f_sample,
                      f_bessel=f_bessel, bessel_order=bessel_order)
    periodic = _assemble(comps, PERIODIC_KEYS, params, **asm_kwargs)
    white = _assemble(comps, WHITE_KEYS, params, **asm_kwargs)

    r = float(rng.uniform(0.0, 1.0))
    alpha = float(np.sqrt(r))
    beta = float(np.sqrt(1.0 - r))
    scale = float(rng.uniform(*scale_range))

    total = scale * (alpha * periodic + beta * white)

    meta = {
        'alpha': alpha,
        'beta': beta,
        'r': r,
        'scale': scale,
        'target_rms': float(params['target_rms']),
    }
    return periodic, white, total, meta
