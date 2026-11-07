"""
Phenomenological tri-exponential pulse model (fast alternative to the ODE).

Single pulse (one rise + two decays, Carrettoni & Vignati 2011, eq. 4.1):

    V(t) = A [ -exp(-s/tau_r) + a1 exp(-s/tau_d1) + (1 - a1) exp(-s/tau_d2) ],
    s = t - t0,   V(t) = 0 for s < 0.

The bracket vanishes at s = 0, so the pulse is continuous at onset. This is the
same shape fitted by ``src.basics.fit`` (``_shape``), so templates generated
here and fits there share one parameterization.

Amplitude convention: the prefactor A is NOT the peak height (the bracket peaks
below 1). Each channel instead carries a sensitivity ``sens`` = peak voltage
per keV, and A = E * sens / phi_peak with phi_peak the numerical bracket
maximum. This keeps ``sens`` directly comparable to the ODE bank's peak/E
(0.4-1.6 mV/keV) and independent of the shape parameters.

Default parameter distributions come from free fits of this model to 120
single pulses of the 41C-parameter ODE bank (clean_v3): the fit residual RMS is
~2e-4 of the peak, so the model reproduces the ODE shapes closely. Fitted
medians: tau_r 85 ms, a1 0.82, tau_d1 0.49 s, tau_d2 3.4 s, 1.1 mV/keV.
The ODE pulses' effective onset sits ~6.7 ms after the nominal one (Bessel
group delay); no filter is applied here, so t0 is the nominal onset exactly.

Pileup is linear superposition, as in the ODE simulator. Unlike the ODE
(q-factor nonlinearity) the peak amplitude is exactly linear in energy.

Energy-dependent decay (optional, ``tau_energy_scaling``): real high-energy
pulses decay more slowly. Reference fits of two events give

    E          tau_r    tau_d1    tau_d2
    1.3 MeV    80 ms    261 ms    1.54 s
    15.3 MeV   32 ms    441 ms    1.78 s

Only the *relative* change is borrowed, not the absolute values (those come
from a different detector than our ODE-fitted defaults), and the reference
energies are mapped onto the generated spectrum range rather than taken
literally: the 1.3 MeV shape is used at E = 1 keV and the 15.3 MeV shape at
E = 5407 keV (the default E_min/E_max), interpolated linearly in between.
With the channel's sampled tau as the 1 keV value:

    tau_k(E) = tau_k^ch * [1 + (R_k - 1) * clip((E - 1)/(5407 - 1), 0, 1)]

with R = tau(15.3 MeV)/tau(1.3 MeV) (x1.69 tau_d1, x1.16 tau_d2). The literal
1.3-15.3 MeV range would leave ~93% of the spectrum unscaled and never reach
the upper point (the measured spectrum ends at 10 MeV); the mapping spreads
the full reference variation over the energies actually generated. Linear
rather than power-law, because a power law diverges toward low energy;
clamping outside [1, 5407] keV keeps every tau within the reference envelope. Only the decays scale (R_r = 1 by default) --
the reference rise speeds up 2.5x, but scaling it is left opt-in via
``TAU_SCALE_RATIO``. Since R_r <= 1 <= R_d1, R_d2 the ordering
tau_r < tau_d1 < tau_d2 is preserved. The peak is still E*sens: phi_peak is
recomputed per pulse, so the pulse integral grows faster than linearly.
In pileup each pulse uses the taus of its own energy.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class TriExpParams:
    """Per-channel tri-exponential pulse-shape parameters."""
    tau_r: float = 0.085     # [s] rise time constant
    a1: float = 0.82         # fast-decay fraction
    tau_d1: float = 0.49     # [s] fast decay time constant
    tau_d2: float = 3.4      # [s] slow decay time constant
    sens: float = 1.13e-3    # [V/keV] peak amplitude per unit energy
    # Not sampled: generation option, scale taus with energy (see module doc).
    tau_energy_scaling: bool = False


# Truncated Gaussians (mean, sigma, lower, upper), from fits to clean_v3.
TRIEXP_PARAM_DISTS = {
    'tau_r':  (0.085,   0.025,   0.02,   0.2),
    'a1':     (0.815,   0.03,    0.6,    0.95),
    'tau_d1': (0.49,    0.09,    0.2,    1.0),
    'tau_d2': (3.4,     0.4,     2.0,    5.0),
    'sens':   (1.13e-3, 0.3e-3,  0.3e-3, 2.5e-3),
}

TRIEXP_PARAM_NAMES = tuple(TRIEXP_PARAM_DISTS)

# Energy scaling of the time constants: the 1.3 MeV reference shape is used at
# E_LO and the 15.3 MeV one at E_HI [keV] (the default spectrum range); the
# ratios are tau(15.3 MeV) / tau(1.3 MeV) from the reference fits.
TAU_SCALE_E_LO = 1.0
TAU_SCALE_E_HI = 5407.0
TAU_SCALE_RATIO = {
    'tau_r':  1.0,            # reference: 32/80 = 0.4, not applied
    'tau_d1': 441.0 / 261.0,  # 1.69
    'tau_d2': 1.78 / 1.54,    # 1.16
}


def tau_scale(energy_kev: float, ratio: float) -> float:
    """Multiplicative tau factor: 1 below E_LO, linear to ``ratio`` at E_HI."""
    x = (energy_kev - TAU_SCALE_E_LO) / (TAU_SCALE_E_HI - TAU_SCALE_E_LO)
    return 1.0 + (ratio - 1.0) * float(np.clip(x, 0.0, 1.0))


def pulse_taus(params: 'TriExpParams', energy_kev: float) -> tuple:
    """(tau_r, tau_d1, tau_d2) of a pulse of this energy on this channel."""
    taus = (params.tau_r, params.tau_d1, params.tau_d2)
    if not params.tau_energy_scaling:
        return taus
    return tuple(tau * tau_scale(energy_kev, TAU_SCALE_RATIO[name])
                 for tau, name in zip(taus, ('tau_r', 'tau_d1', 'tau_d2')))


def triexp_shape(s, tau_r, a1, tau_d1, tau_d2):
    """Unit-prefactor bracket of V(t); zero for s < 0."""
    # Clamp inside the exp: np.where evaluates both branches.
    u = np.maximum(s, 0.0)
    return np.where(
        s >= 0,
        -np.exp(-u / tau_r)
        + a1 * np.exp(-u / tau_d1)
        + (1.0 - a1) * np.exp(-u / tau_d2),
        0.0,
    )


def _bracket_peak(tau_r, a1, tau_d1, tau_d2) -> float:
    res = minimize_scalar(
        lambda s: -triexp_shape(s, tau_r, a1, tau_d1, tau_d2),
        bounds=(1e-7, 10.0 * tau_d2), method='bounded',
    )
    return float(-res.fun)


def shape_peak(params: TriExpParams) -> float:
    """Maximum of the unit-prefactor bracket at the channel's base taus."""
    return _bracket_peak(params.tau_r, params.a1, params.tau_d1, params.tau_d2)


def sample_triexp_params(rng: np.random.Generator,
                         max_attempts: int = 1000) -> TriExpParams:
    """Draw channel parameters; resample until tau_r < tau_d1 < tau_d2."""
    for _ in range(max_attempts):
        params = TriExpParams()
        for name, (mean, sigma, lower, upper) in TRIEXP_PARAM_DISTS.items():
            while True:
                value = rng.normal(mean, sigma)
                if lower < value < upper:
                    break
            setattr(params, name, value)
        if params.tau_r < params.tau_d1 < params.tau_d2:
            return params
    raise RuntimeError(
        f"Could not sample ordered time constants in {max_attempts} attempts"
    )


def simulate_triexp_pileup(energies_kev: list, onsets: list,
                           params: TriExpParams,
                           duration: float = 4.0,
                           f_sample: float = 1000.0,
                           phi_peak: float = None) -> tuple:
    """Sum of tri-exponential pulses on one channel.

    ``phi_peak`` may be passed in to skip recomputing it per window; it is
    ignored with ``tau_energy_scaling``, where each pulse has its own taus.

    Returns
    -------
    t : ndarray, shape (n_samples,)   time [s]
    v : ndarray, shape (n_samples,)   voltage [V], positive-going
    """
    if phi_peak is None and not params.tau_energy_scaling:
        phi_peak = shape_peak(params)
    n_out = int(duration * f_sample)
    t = np.arange(n_out) / f_sample
    v = np.zeros(n_out)
    for E, t0 in zip(energies_kev, onsets):
        tau_r, tau_d1, tau_d2 = pulse_taus(params, E)
        peak = (_bracket_peak(tau_r, params.a1, tau_d1, tau_d2)
                if params.tau_energy_scaling else phi_peak)
        A = E * params.sens / peak
        v += A * triexp_shape(t - t0, tau_r, params.a1, tau_d1, tau_d2)
    return t, v


def simulate_triexp_pulse(energy_kev: float, params: TriExpParams,
                          duration: float = 4.0,
                          t_onset: float = 0.3,
                          f_sample: float = 1000.0,
                          phi_peak: float = None) -> tuple:
    """Single tri-exponential pulse; same signature style as simulate_pulse."""
    return simulate_triexp_pileup([energy_kev], [t_onset], params,
                                  duration=duration, f_sample=f_sample,
                                  phi_peak=phi_peak)
