"""
Generate evaluation datasets for DDPM denoising assessment.

Two modes:
  resolution  -- fixed-template signal + 10 000 random noise windows,
                 repeated for each specified energy.  Used for energy-
                 resolution vs SNR analysis.
  efficiency  -- 5 000 events at a single energy with noise RMS set to
                 signal_amplitude / snr_factor.  Used for small-signal
                 reconstruction-efficiency analysis.

Usage
-----
# Resolution dataset (default energies)
python -u scripts/generate_eval_dataset.py resolution \
    --noise_dir /home/wsl_0vbb/DDPM4bolometer/simu_data/noise \
    --output_dir /home/wsl_0vbb/DDPM4bolometer/eval_data/resolution

# Efficiency dataset at 100 keV, 2-sigma condition
python -u scripts/generate_eval_dataset.py efficiency \
    --noise_dir /home/wsl_0vbb/DDPM4bolometer/simu_data/noise \
    --output_dir /home/wsl_0vbb/DDPM4bolometer/eval_data/efficiency \
    --energy_kev 100 --snr_factor 2.0 --n_events 5000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import h5py
import numpy as np

from src.pulse.simulator import (DetectorParams, find_equilibrium, simulate_pulse,
                                  sample_params, is_valid_equilibrium)

# Default energy points [keV]
DEFAULT_ENERGIES = [583.0, 1461.0, 2528.0, 2615.0, 3034.0]

F_SAMPLE = 1000.0
DURATION = 10.0
ONSET    = 1.5          # pulse onset [s]
_DRIFT_THRESHOLD = -50.0  # V — last sample below this flags divergence


def _find_stable_params(energies_kev: list, seed: int = 9999,
                        max_attempts: int = 200) -> tuple:
    """Sample detector parameters until a set produces drift-free pulses at all energies."""
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        params = sample_params(rng)
        eq = find_equilibrium(params)
        if not is_valid_equilibrium(params, eq):
            continue
        stable = True
        for E in energies_kev:
            try:
                _, v = simulate_pulse(E, params, eq,
                                      duration=DURATION, t_onset=ONSET, f_sample=F_SAMPLE)
                if v[-1] < _DRIFT_THRESHOLD or not np.isfinite(v).all():
                    stable = False
                    break
            except Exception:
                stable = False
                break
        if stable:
            print(f"  Found stable params after {attempt+1} attempts (seed={seed})")
            return params, eq
    raise RuntimeError(f"Could not find stable parameters in {max_attempts} attempts")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_noise_pool(noise_dir: str):
    """Load all noise waveforms and their parameters from the noise shard directory."""
    h5_files = sorted(
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith('.h5')
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {noise_dir}")

    waveform_list = []
    param_lists = {}

    for path in h5_files:
        with h5py.File(path, 'r') as f:
            waveform_list.append(f['waveforms'][:])
            for key in f['params']:
                param_lists.setdefault(key, []).append(f['params'][key][:])

    waveforms = np.concatenate(waveform_list, axis=0)
    params = {k: np.concatenate(v, axis=0) for k, v in param_lists.items()}
    print(f"Noise pool: {waveforms.shape[0]} windows from {len(h5_files)} shards.")
    return waveforms, params


def _simulate_template(energy_kev: float, params, eq) -> np.ndarray:
    """Simulate a single clean pulse at the given energy."""
    _, v = simulate_pulse(energy_kev, params, eq,
                          duration=DURATION, t_onset=ONSET, f_sample=F_SAMPLE)
    n_samples = int(DURATION * F_SAMPLE)
    return v[:n_samples].astype(np.float64)


def _signal_amplitude(clean: np.ndarray) -> float:
    """Peak amplitude above pre-trigger baseline."""
    n_pre    = max(1, len(clean) // 20)
    baseline = float(np.median(clean[:n_pre]))
    return float(clean.max()) - baseline


def _snr_db(signal_amp: float, noise_rms: float) -> float:
    if noise_rms <= 0:
        return np.inf
    return 20.0 * np.log10(signal_amp / noise_rms)


def _noise_fracs(params_row: dict) -> dict:
    """Approximate fractional power of each noise component.

    Uses squared amplitude proxies — an approximation because different
    components have different spectral shapes.  Stored alongside raw params
    so users can recalculate if needed.
    """
    ac    = float(params_row['ac_a_base']) ** 2
    pt    = float(params_row['pt_a_base']) ** 2
    white = float(params_row['white_rms']) ** 2
    total = ac + pt + white
    if total == 0:
        return dict(ac_frac=0.0, pt_frac=0.0, white_frac=0.0, colored_frac=0.0)
    return dict(
        ac_frac=ac / total,
        pt_frac=pt / total,
        white_frac=white / total,
        colored_frac=0.0,   # colored (1/f) power not easily separated here
    )


def _save_dataset(output_path: str,
                  energy_kev: float,
                  clean_template: np.ndarray,
                  noisy_waveforms: np.ndarray,
                  snr_db: np.ndarray,
                  noise_rms_arr: np.ndarray,
                  noise_params: dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.attrs['energy_kev']  = energy_kev
        f.attrs['n_events']    = noisy_waveforms.shape[0]
        f.attrs['f_sample']    = F_SAMPLE
        f.attrs['duration']    = DURATION
        f.attrs['onset_s']     = ONSET
        pass  # detector params stored separately if needed

        f.create_dataset('clean_template',  data=clean_template,   compression='gzip')
        f.create_dataset('waveforms_noisy', data=noisy_waveforms,  compression='gzip')
        f.create_dataset('snr_db',          data=snr_db)
        f.create_dataset('noise_rms',       data=noise_rms_arr)

        grp = f.create_group('noise_params')
        for key, arr in noise_params.items():
            grp.create_dataset(key, data=arr)

    print(f"  Saved {noisy_waveforms.shape[0]} events → {output_path}")


# ── Resolution mode ───────────────────────────────────────────────────────────

def generate_resolution(noise_dir: str, output_dir: str,
                        energies: list, n_events: int, seed: int):
    """One HDF5 per energy: fixed template + n_events random noise windows."""
    rng = np.random.default_rng(seed)
    noise_pool, noise_param_pool = _load_noise_pool(noise_dir)
    n_pool = noise_pool.shape[0]

    os.makedirs(output_dir, exist_ok=True)

    print("Finding stable detector parameters...")
    det_params, det_eq = _find_stable_params(energies, seed=seed)

    for energy in energies:
        print(f"\nEnergy: {energy:.0f} keV")
        t0 = time.time()

        print("  Simulating clean template...")
        clean = _simulate_template(energy, det_params, det_eq)
        sig_amp = _signal_amplitude(clean)
        print(f"  Signal amplitude: {sig_amp*1e3:.2f} mV")

        # Sample noise indices (with replacement if pool smaller than n_events)
        idx = rng.choice(n_pool, size=n_events, replace=(n_pool < n_events))

        noisy      = np.zeros((n_events, len(clean)), dtype=np.float64)
        snr_arr    = np.zeros(n_events, dtype=np.float32)
        rms_arr    = np.zeros(n_events, dtype=np.float32)
        np_params  = {k: np.zeros(n_events, dtype=np.float32)
                      for k in noise_param_pool}
        np_params['ac_frac']      = np.zeros(n_events, dtype=np.float32)
        np_params['pt_frac']      = np.zeros(n_events, dtype=np.float32)
        np_params['white_frac']   = np.zeros(n_events, dtype=np.float32)
        np_params['colored_frac'] = np.zeros(n_events, dtype=np.float32)

        for i, ni in enumerate(idx):
            noise_w         = noise_pool[ni]
            noisy[i]        = clean + noise_w
            rms             = float(np.std(noise_w))
            rms_arr[i]      = rms
            snr_arr[i]      = _snr_db(sig_amp, rms)

            row = {k: noise_param_pool[k][ni] for k in noise_param_pool}
            for k in noise_param_pool:
                np_params[k][i] = row[k]
            for k, v in _noise_fracs(row).items():
                np_params[k][i] = v

        out_path = os.path.join(output_dir, f'eval_{energy:.0f}keV.h5')
        _save_dataset(out_path, energy, clean, noisy, snr_arr, rms_arr, np_params)
        print(f"  Done in {time.time()-t0:.1f}s")


# ── Efficiency mode ───────────────────────────────────────────────────────────

def generate_efficiency(noise_dir: str, output_dir: str,
                        energy_kev: float, snr_factor: float,
                        n_events: int, seed: int,
                        noise_rms_mv: float = None):
    """n_events at energy_kev with fixed noise level.

    If noise_rms_mv is given (in mV), noise is scaled to that absolute RMS.
    Otherwise noise_RMS = signal_amplitude / snr_factor.
    """
    rng = np.random.default_rng(seed)
    noise_pool, noise_param_pool = _load_noise_pool(noise_dir)

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    print("  Finding stable detector parameters...")
    det_params, det_eq = _find_stable_params([energy_kev], seed=seed)

    print("  Simulating clean template...")
    clean   = _simulate_template(energy_kev, det_params, det_eq)
    sig_amp = _signal_amplitude(clean)

    if noise_rms_mv is not None:
        target_noise_rms = noise_rms_mv * 1e-3
        print(f"\nEfficiency dataset: {energy_kev:.0f} keV, "
              f"fixed noise_rms={noise_rms_mv:.2f} mV")
    else:
        target_noise_rms = sig_amp / snr_factor
        print(f"\nEfficiency dataset: {energy_kev:.0f} keV, SNR-factor={snr_factor}")

    print(f"  Signal amplitude: {sig_amp*1e3:.2f} mV")
    print(f"  Target noise RMS: {target_noise_rms*1e3:.2f} mV  "
          f"(SNR ~ {_snr_db(sig_amp, target_noise_rms):.1f} dB)")

    # Sample noise windows and rescale each to target_noise_rms
    idx = rng.choice(noise_pool.shape[0], size=n_events,
                     replace=(noise_pool.shape[0] < n_events))

    noisy     = np.zeros((n_events, len(clean)), dtype=np.float64)
    snr_arr   = np.zeros(n_events, dtype=np.float32)
    rms_arr   = np.zeros(n_events, dtype=np.float32)
    np_params = {k: np.zeros(n_events, dtype=np.float32)
                 for k in noise_param_pool}
    np_params['ac_frac']      = np.zeros(n_events, dtype=np.float32)
    np_params['pt_frac']      = np.zeros(n_events, dtype=np.float32)
    np_params['white_frac']   = np.zeros(n_events, dtype=np.float32)
    np_params['colored_frac'] = np.zeros(n_events, dtype=np.float32)
    np_params['scale_factor'] = np.zeros(n_events, dtype=np.float32)

    for i, ni in enumerate(idx):
        noise_w  = noise_pool[ni]
        raw_rms  = float(np.std(noise_w))
        if raw_rms > 0:
            scale   = target_noise_rms / raw_rms
            noise_w = noise_w * scale
        else:
            scale = 1.0
        noisy[i]  = clean + noise_w
        actual_rms = float(np.std(noise_w))
        rms_arr[i] = actual_rms
        snr_arr[i] = _snr_db(sig_amp, actual_rms)

        row = {k: noise_param_pool[k][ni] for k in noise_param_pool}
        for k in noise_param_pool:
            np_params[k][i] = row[k]
        for k, v in _noise_fracs(row).items():
            np_params[k][i] = v
        np_params['scale_factor'][i] = scale

    out_path = os.path.join(output_dir, f'eval_efficiency_{energy_kev:.0f}keV.h5')
    _save_dataset(out_path, energy_kev, clean, noisy, snr_arr, rms_arr, np_params)
    print(f"Done in {time.time()-t0:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation datasets")
    sub = parser.add_subparsers(dest='mode', required=True)

    # ---- resolution subcommand ----
    p_res = sub.add_parser('resolution', help='Energy-resolution evaluation dataset')
    p_res.add_argument('--noise_dir', required=True)
    p_res.add_argument('--output_dir', required=True)
    p_res.add_argument('--energies', nargs='+', type=float, default=DEFAULT_ENERGIES,
                       help='Energy points in keV')
    p_res.add_argument('--n_events', type=int, default=10000)
    p_res.add_argument('--seed', type=int, default=0)

    # ---- efficiency subcommand ----
    p_eff = sub.add_parser('efficiency', help='Small-signal reconstruction-efficiency dataset')
    p_eff.add_argument('--noise_dir', required=True)
    p_eff.add_argument('--output_dir', required=True)
    p_eff.add_argument('--energy_kev', type=float, default=100.0)
    p_eff.add_argument('--snr_factor', type=float, default=2.0,
                       help='noise_RMS = signal_amplitude / snr_factor (ignored if --noise_rms set)')
    p_eff.add_argument('--noise_rms', type=float, default=None,
                       help='Fixed absolute noise RMS in mV (overrides --snr_factor)')
    p_eff.add_argument('--n_events', type=int, default=5000)
    p_eff.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    if args.mode == 'resolution':
        generate_resolution(args.noise_dir, args.output_dir,
                            args.energies, args.n_events, args.seed)
    else:
        generate_efficiency(args.noise_dir, args.output_dir,
                            args.energy_kev, args.snr_factor,
                            args.n_events, args.seed,
                            noise_rms_mv=args.noise_rms)


if __name__ == '__main__':
    main()
