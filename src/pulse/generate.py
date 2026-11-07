"""
Generate clean pulse dataset for DDPM training.

Outputs an HDF5 file with simulated CUORE-like pulse waveforms,
including single pulses and pileup events.

Two generation modes:

  * Flat dataset (``generate_dataset`` / ``--n_pulses``): detector parameters
    are resampled *per pulse* (legacy behavior, single output file).

  * Channel bank (``generate_channel_bank`` / ``--n_channels``): detector
    parameters are sampled *once per channel* and held fixed for every window
    in that channel's shard, while event energy and pileup still vary per
    window. Each channel is written as its own shard ``clean_NNN.h5`` whose
    attributes record the channel number and its fixed detector parameters.
    This is the data layout used by the multi-channel foundation-model
    denoiser (see src/ddpm/README.md).

Two pulse models (``--model``), usable with either mode:

  * ``ode`` (default): nonlinear electro-thermal ODE (src/pulse/simulator.py).
  * ``triexp``: analytic tri-exponential template (src/pulse/triexp.py),
    V(t) = A[-exp(-s/tau_r) + a1 exp(-s/tau_d1) + (1-a1) exp(-s/tau_d2)].
    Orders of magnitude faster; shape parameters replace detector parameters
    as the channel identity. ``--tau_energy_scaling`` makes the decay
    constants grow with energy (see triexp.py); each pulse's actual taus are
    stored per window (``tau_r_1``, ``tau_d1_1``, ..., ``tau_d2_2``).

Usage:
    # Legacy flat dataset
    python -u -m src.pulse.generate --n_pulses 100 --output data/clean_pulses.h5

    # Same, with the tri-exponential template instead of the ODE
    python -u -m src.pulse.generate --model triexp --n_pulses 100 \
        --output data/clean_triexp.h5

    # Channel bank: 100 channels x 200 windows each
    python -u -m src.pulse.generate --n_channels 100 --windows_per_channel 200 \
        --output_dir /media/Disk_YIN/yunshancheng/cuore/channel_bank/clean
"""

import argparse
import os
import time

import h5py
import numpy as np

from src.pulse.simulator import (
    DetectorParams, sample_params, find_equilibrium, is_valid_equilibrium,
    simulate_pulse, simulate_pileup,
)
from src.pulse.spectrum import EnergySpectrum
from src.pulse.triexp import (
    TRIEXP_PARAM_NAMES, sample_triexp_params, pulse_taus,
    simulate_triexp_pulse, simulate_triexp_pileup,
)

TAU_NAMES = ('tau_r', 'tau_d1', 'tau_d2')

PULSE_MODELS = ('ode', 'triexp')

# Detector parameters that define a channel's pulse-shape identity. These are
# fixed per channel and recorded in each shard's attributes.
DETECTOR_PARAM_NAMES = (
    'R0', 'T0', 'lambda0', 'q', 'g_ec', 'a_ec', 'C_p', 'T_base',
)


def param_names(model: str) -> tuple:
    """Names of the per-channel parameters recorded for a pulse model."""
    return TRIEXP_PARAM_NAMES if model == 'triexp' else DETECTOR_PARAM_NAMES


def sample_channel(rng: np.random.Generator, model: str = 'ode',
                   tau_energy_scaling: bool = False):
    """Sample one channel's parameters for the given pulse model.

    Returns (params, eq); ``eq`` is None for ``triexp`` (no equilibrium).
    ``tau_energy_scaling`` only applies to ``triexp``.
    """
    if model == 'triexp':
        params = sample_triexp_params(rng)
        params.tau_energy_scaling = tau_energy_scaling
        return params, None
    return sample_valid_params(rng)


def sample_valid_params(rng: np.random.Generator, max_attempts: int = 200):
    """Sample detector parameters until they yield a valid equilibrium.

    Returns
    -------
    (params, eq) : (DetectorParams, EquilibriumState)
    """
    for _ in range(max_attempts):
        params = sample_params(rng)
        try:
            eq = find_equilibrium(params)
        except Exception:
            continue
        if is_valid_equilibrium(params, eq):
            return params, eq
    raise RuntimeError(
        f"Could not find valid detector parameters in {max_attempts} attempts"
    )


def simulate_window(rng: np.random.Generator,
                    params: DetectorParams,
                    eq,
                    spectrum: EnergySpectrum,
                    duration: float,
                    f_sample: float,
                    n_samples: int,
                    pileup_fraction: float,
                    pileup_dt_min: float,
                    pileup_dt_max: float,
                    onset: float,
                    model: str = 'ode') -> dict:
    """Simulate a single window (one or two pulses) for fixed detector params.

    Only event content (energy, pileup) varies here; the detector response is
    fully determined by ``params``/``eq``. ``model='triexp'`` expects
    ``TriExpParams`` (``eq`` ignored) and never fails, so the ODE fallbacks
    below only ever trigger for ``ode``.

    Returns
    -------
    dict with keys: waveform, energy_1, energy_2, onset_1, onset_2, is_pileup
    """
    do_pileup = rng.random() < pileup_fraction
    t1 = onset
    E1 = spectrum.sample(rng, 1)[0]
    E2 = np.nan
    t2 = np.nan

    if model == 'triexp':
        if do_pileup:
            E2 = spectrum.sample(rng, 1)[0]
            t2 = t1 + rng.uniform(pileup_dt_min, pileup_dt_max)
            _, v = simulate_triexp_pileup([E1, E2], [t1, t2], params,
                                          duration=duration, f_sample=f_sample)
        else:
            _, v = simulate_triexp_pulse(E1, params, duration=duration,
                                         t_onset=t1, f_sample=f_sample)
        win = {
            'waveform': v[:n_samples],
            'energy_1': E1,
            'energy_2': E2,
            'onset_1': t1,
            'onset_2': t2,
            'is_pileup': bool(do_pileup),
        }
        for k, E in ((1, E1), (2, E2)):
            taus = pulse_taus(params, E) if np.isfinite(E) else (np.nan,) * 3
            for name, tau in zip(TAU_NAMES, taus):
                win[f'{name}_{k}'] = tau
        return win

    if do_pileup:
        E2 = spectrum.sample(rng, 1)[0]
        dt = rng.uniform(pileup_dt_min, pileup_dt_max)
        t2 = t1 + dt
        try:
            _, v = simulate_pileup(
                [E1, E2], [t1, t2], params, eq,
                duration=duration, f_sample=f_sample,
            )
        except RuntimeError as e:
            print(f"  ODE failed for pileup ({e}), falling back to single pulse")
            do_pileup = False
            E2 = np.nan
            t2 = np.nan

    if not do_pileup:
        try:
            _, v = simulate_pulse(
                E1, params, eq,
                duration=duration, t_onset=t1, f_sample=f_sample,
            )
        except RuntimeError as e:
            print(f"  ODE failed ({e}), writing zeros")
            v = np.zeros(n_samples)

    return {
        'waveform': v[:n_samples],
        'energy_1': E1,
        'energy_2': E2,
        'onset_1': t1,
        'onset_2': t2,
        'is_pileup': bool(do_pileup),
    }


def _alloc_window_arrays(n: int, n_samples: int, model: str = 'ode') -> dict:
    """Pre-allocate the per-window output arrays shared by both modes.

    ``triexp`` additionally gets per-pulse time constants (NaN for an absent
    second pulse).
    """
    arrays = {
        'waveforms': np.zeros((n, n_samples), dtype=np.float64),
        'energies_1': np.zeros(n, dtype=np.float64),
        'energies_2': np.full(n, np.nan, dtype=np.float64),
        'onsets_1': np.zeros(n, dtype=np.float64),
        'onsets_2': np.full(n, np.nan, dtype=np.float64),
        'is_pileup': np.zeros(n, dtype=bool),
    }
    if model == 'triexp':
        for name in TAU_NAMES:
            for k in (1, 2):
                arrays[f'{name}_{k}'] = np.full(n, np.nan, dtype=np.float64)
    return arrays


def _store_window(arrays: dict, i: int, win: dict):
    arrays['waveforms'][i] = win['waveform']
    for key in arrays:
        if key != 'waveforms':
            arrays[key][i] = win[key.replace('energies', 'energy')
                                    .replace('onsets', 'onset')]


def _write_window_datasets(f: h5py.File, arrays: dict):
    f.create_dataset('waveforms', data=arrays['waveforms'], compression='gzip')
    for key, data in arrays.items():
        if key != 'waveforms':
            f.create_dataset(key, data=data)


def generate_dataset(n_pulses: int, output_file: str,
                     duration: float = 10.0,
                     f_sample: float = 1000.0,
                     pileup_fraction: float = 0.3,
                     pileup_dt_min: float = 0.01,
                     pileup_dt_max: float = 2.0,
                     onset: float = 1.5,
                     seed: int = 42,
                     E_min: float = 1.0,
                     E_max: float = 5407.0,
                     model: str = 'ode',
                     tau_energy_scaling: bool = False):
    """Generate a flat clean-pulse dataset (detector params resampled per pulse).

    Legacy single-file mode. See ``generate_channel_bank`` for the multi-channel
    layout used by the foundation-model denoiser.
    """
    rng = np.random.default_rng(seed)
    spectrum = EnergySpectrum(E_min=E_min, E_max=E_max)
    n_samples = int(duration * f_sample)

    arrays = _alloc_window_arrays(n_pulses, n_samples, model)
    # Per-pulse detector parameters (vary in this mode)
    names = param_names(model)
    par = {name: np.zeros(n_pulses) for name in names}
    eq_V = np.zeros(n_pulses); eq_Te = np.zeros(n_pulses)
    eq_Tc = np.zeros(n_pulses); eq_Tt = np.zeros(n_pulses)

    t_start = time.time()
    for i in range(n_pulses):
        params, eq = sample_channel(rng, model, tau_energy_scaling)
        win = simulate_window(
            rng, params, eq, spectrum, duration, f_sample, n_samples,
            pileup_fraction, pileup_dt_min, pileup_dt_max, onset, model,
        )
        _store_window(arrays, i, win)

        for name in names:
            par[name][i] = getattr(params, name)
        if eq is not None:
            eq_V[i] = eq.V_bol; eq_Te[i] = eq.T_e
            eq_Tc[i] = eq.T_c; eq_Tt[i] = eq.T_t

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (n_pulses - i - 1) / rate if rate > 0 else 0
        e1 = win['energy_1']
        e2_str = f", E2={win['energy_2']:.1f} keV" if win['is_pileup'] else ""
        print(f"  [{i+1}/{n_pulses}] E1={e1:.1f} keV{e2_str}"
              f"  ({rate:.1f} pulses/s, ETA {eta:.0f}s)")

    print(f"\nSaving to {output_file} ...")
    with h5py.File(output_file, 'w') as f:
        f.attrs['f_sample'] = f_sample
        f.attrs['duration'] = duration
        f.attrs['n_samples'] = n_samples
        f.attrs['n_pulses'] = n_pulses
        f.attrs['pileup_fraction'] = pileup_fraction
        f.attrs['seed'] = seed
        f.attrs['pulse_model'] = model
        f.attrs['tau_energy_scaling'] = tau_energy_scaling

        _write_window_datasets(f, arrays)

        grp = f.create_group('params')
        for name in names:
            grp.create_dataset(name, data=par[name])

        if model == 'ode':
            grp = f.create_group('equilibrium')
            grp.create_dataset('V_bol', data=eq_V)
            grp.create_dataset('T_e', data=eq_Te)
            grp.create_dataset('T_c', data=eq_Tc)
            grp.create_dataset('T_t', data=eq_Tt)

    elapsed = time.time() - t_start
    print(f"Done. {n_pulses} pulses in {elapsed:.1f}s "
          f"({n_pulses/elapsed:.1f} pulses/s)")


def generate_channel_shard(channel_id: int, n_windows: int, output_file: str,
                           rng: np.random.Generator,
                           spectrum: EnergySpectrum,
                           duration: float, f_sample: float,
                           pileup_fraction: float,
                           pileup_dt_min: float, pileup_dt_max: float,
                           onset: float, seed: int,
                           model: str = 'ode',
                           tau_energy_scaling: bool = False):
    """Generate one channel's clean-pulse shard with FIXED detector params.

    The detector parameters (and derived equilibrium) are sampled once here and
    held fixed for all ``n_windows`` windows; only event energy/pileup vary.
    The fixed parameters and channel number are stored in the shard attributes.
    """
    n_samples = int(duration * f_sample)

    # One parameter draw defines this channel.
    params, eq = sample_channel(rng, model, tau_energy_scaling)

    arrays = _alloc_window_arrays(n_windows, n_samples, model)
    t_start = time.time()
    for i in range(n_windows):
        win = simulate_window(
            rng, params, eq, spectrum, duration, f_sample, n_samples,
            pileup_fraction, pileup_dt_min, pileup_dt_max, onset, model,
        )
        _store_window(arrays, i, win)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_windows - i - 1) / rate if rate > 0 else 0
            print(f"    ch{channel_id:03d} [{i+1}/{n_windows}] "
                  f"({rate:.1f} win/s, ETA {eta:.0f}s)")

    with h5py.File(output_file, 'w') as f:
        f.attrs['f_sample'] = f_sample
        f.attrs['duration'] = duration
        f.attrs['n_samples'] = n_samples
        f.attrs['n_windows'] = n_windows
        f.attrs['pileup_fraction'] = pileup_fraction
        f.attrs['seed'] = seed
        # Channel identity metadata
        f.attrs['channel_id'] = channel_id
        f.attrs['pulse_model'] = model
        f.attrs['tau_energy_scaling'] = tau_energy_scaling
        for name in param_names(model):
            f.attrs[f'param_{name}'] = getattr(params, name)
        if eq is not None:
            f.attrs['eq_V_bol'] = eq.V_bol
            f.attrs['eq_T_e'] = eq.T_e
            f.attrs['eq_T_c'] = eq.T_c
            f.attrs['eq_T_t'] = eq.T_t

        _write_window_datasets(f, arrays)

    return params, eq


def generate_channel_bank(n_channels: int, windows_per_channel: int,
                          output_dir: str,
                          duration: float = 10.0,
                          f_sample: float = 1000.0,
                          pileup_fraction: float = 0.3,
                          pileup_dt_min: float = 0.01,
                          pileup_dt_max: float = 2.0,
                          onset: float = 1.5,
                          base_seed: int = 42,
                          E_min: float = 1.0,
                          E_max: float = 5407.0,
                          model: str = 'ode',
                          tau_energy_scaling: bool = False):
    """Generate a bank of per-channel clean-pulse shards.

    Each channel gets an independent RNG stream (spawned from ``base_seed``) and
    is written to ``output_dir/clean_NNN.h5``. Detector params are fixed within a
    channel; energy/pileup vary per window. Resumable: existing shards are
    skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    spectrum = EnergySpectrum(E_min=E_min, E_max=E_max)
    child_seqs = np.random.SeedSequence(base_seed).spawn(n_channels)

    t_start = time.time()
    for c in range(n_channels):
        output_file = os.path.join(output_dir, f'clean_{c:03d}.h5')
        if os.path.exists(output_file):
            print(f"Channel {c} already exists, skipping.")
            continue

        seed = int(child_seqs[c].generate_state(1)[0])
        rng = np.random.default_rng(child_seqs[c])
        print(f"\n{'='*60}")
        print(f"Channel {c}/{n_channels}: {windows_per_channel} windows, "
              f"seed {seed}")
        print(f"{'='*60}")
        generate_channel_shard(
            c, windows_per_channel, output_file, rng, spectrum,
            duration, f_sample, pileup_fraction,
            pileup_dt_min, pileup_dt_max, onset, seed, model,
            tau_energy_scaling,
        )

    elapsed = time.time() - t_start
    print(f"\nAll {n_channels} channel shards complete in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean CUORE-like pulse dataset for DDPM training."
    )
    # Mode selection: --n_channels triggers the channel-bank layout.
    parser.add_argument('--n_channels', type=int, default=None,
                        help='If set, generate a channel bank (one shard per '
                             'channel, fixed detector params per channel).')
    parser.add_argument('--windows_per_channel', type=int, default=200,
                        help='Windows per channel (channel-bank mode)')
    parser.add_argument('--output_dir', type=str,
                        default='data/channel_bank/clean',
                        help='Output directory for channel shards '
                             '(channel-bank mode)')

    # Legacy flat-dataset args
    parser.add_argument('--n_pulses', type=int, default=100,
                        help='Number of pulse windows (flat mode)')
    parser.add_argument('--output', type=str, default='data/clean_pulses.h5',
                        help='Output HDF5 file path (flat mode)')

    # Shared args
    parser.add_argument('--model', type=str, default='ode',
                        choices=PULSE_MODELS,
                        help='Pulse model: electro-thermal ODE or analytic '
                             'tri-exponential template')
    parser.add_argument('--tau_energy_scaling', action='store_true',
                        help='triexp only: decay constants grow linearly with '
                             'energy from the 1.3 MeV reference shape at 1 keV '
                             'to the 15.3 MeV one at 5407 keV')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Window duration [s]')
    parser.add_argument('--f_sample', type=float, default=1000.0,
                        help='Output sampling rate [Hz]')
    parser.add_argument('--pileup_fraction', type=float, default=0.3,
                        help='Fraction of pileup windows')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (flat mode) / base seed (channel bank)')
    args = parser.parse_args()
    if args.tau_energy_scaling and args.model != 'triexp':
        parser.error('--tau_energy_scaling requires --model triexp')

    if args.n_channels is not None:
        generate_channel_bank(
            n_channels=args.n_channels,
            windows_per_channel=args.windows_per_channel,
            output_dir=args.output_dir,
            duration=args.duration,
            f_sample=args.f_sample,
            pileup_fraction=args.pileup_fraction,
            base_seed=args.seed,
            model=args.model,
            tau_energy_scaling=args.tau_energy_scaling,
        )
    else:
        generate_dataset(
            n_pulses=args.n_pulses,
            output_file=args.output,
            duration=args.duration,
            f_sample=args.f_sample,
            pileup_fraction=args.pileup_fraction,
            seed=args.seed,
            model=args.model,
            tau_energy_scaling=args.tau_energy_scaling,
        )


if __name__ == '__main__':
    main()
