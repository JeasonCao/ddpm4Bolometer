"""
Batch pulse fitting on eval datasets (noisy and denoised waveforms).

Reads an eval HDF5 that already contains waveforms_noisy and waveforms_denoised
(produced by run_batch_inference.py), fits each waveform with the tri-exponential
model from src/basics/fit.py, and writes a fit-results HDF5 with one row per event.

Usage
-----
python -u scripts/run_fitting.py \
    --input  /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/resolution/eval_583keV.h5 \
    --output /mnt/d/storage/research/DDPM4bolometer/fit_results/fit_583keV.h5 \
    --workers 8

# All files in a directory
python -u scripts/run_fitting.py \
    --input_dir  /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/resolution \
    --output_dir /mnt/d/storage/research/DDPM4bolometer/fit_results \
    --workers 8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import time
import multiprocessing as mp
import h5py
import numpy as np

from src.basics.fit import fit_pulse
from src.basics.amplitude import max_baseline_amplitude

F_SAMPLE = 1000.0


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a0, b0) / denom)

# ── Convergence criteria (user-configurable via CLI) ──────────────────────────
DEFAULT_CHI2_MAX   = 10.0   # chi2/ndf upper limit
DEFAULT_AMP_MIN    = 1e-5   # [V] minimum sensible amplitude
DEFAULT_AMP_MAX    = 9.9    # [V] maximum sensible amplitude (< _A_MAX=10)
DEFAULT_AMP_DEV    = 0.20   # fractional deviation from true amplitude for efficiency


# ── Single-event fitting (runs in worker process) ─────────────────────────────

def _fit_one(args):
    """Fit a single waveform. Returns a flat dict of scalars."""
    waveform, label, chi2_max, amp_min, amp_max, clean_template = args
    t0 = time.time()
    fr = fit_pulse(waveform, n_pulses=1, fs=F_SAMPLE)
    elapsed = time.time() - t0

    # Pearson r between fitted model curve and clean template.
    # NaN for failed fits (counts as "not matched" in efficiency analysis).
    fit_pearsonr = np.nan
    if fr.success and fr.model_fn is not None and len(clean_template) == len(waveform):
        t_arr = np.arange(len(waveform)) / F_SAMPLE
        model_wf = fr.model_fn(t_arr, *fr.params)
        fit_pearsonr = _pearson_r(model_wf, clean_template)

    # ── stored parameters ──────────────────────────────────────────────────────
    # Convention (biexp_single, single-pulse):
    #   v(t) = B + A_raw * (-exp(-dt/τ_r) + α·exp(-dt/τ_d1) + (1-α)·exp(-dt/τ_d2))
    # After fitting, τ_d1 ≤ τ_d2 is always enforced (fast ≤ slow).
    # Therefore:
    #   alpha  = coefficient of t_fall1  (fast decay component, τ_d1)
    #   1-alpha = coefficient of t_fall2 (slow decay component, τ_d2)
    # A_raw is the raw amplitude prefactor (not the physical peak amplitude;
    # physical peak = amp = A_raw * shape_peak).
    # With {B, A_raw, onset, t_rise, alpha, t_fall1, t_fall2} the full waveform
    # can be reconstructed as biexp_single(t, B, A_raw, onset, t_rise, alpha, t_fall1, t_fall2).
    s = fr.success
    result = {
        f'success_{label}':       s,
        f'amp_{label}':           float(fr.peak_amps[0])   if s else np.nan,  # physical peak [V]
        f'amp_raw_{label}':       float(fr.params[1])      if s else np.nan,  # A prefactor
        f'baseline_{label}':      float(fr.baseline)       if s else np.nan,  # B [V]
        f'alpha_{label}':         float(fr.alpha[0])       if s else np.nan,  # coeff of t_fall1
        f'chi2_{label}':          float(fr.chi2_per_ndf)   if s else np.nan,
        f't_rise_{label}':        float(fr.tau_r[0])       if s else np.nan,
        f't_fall1_{label}':       float(fr.tau_d[0])       if s else np.nan,  # fast (≤ t_fall2)
        f't_fall2_{label}':       float(fr.tau_d2[0])      if s else np.nan,  # slow
        f'onset_{label}':         float(fr.onset_times[0]) if s else np.nan,
        f'amp_maxbase_{label}':   max_baseline_amplitude(waveform),  # NOTE 2026-09-09: fixed baseline_frac bug, see true_amp comment below
        f'fit_pearsonr_{label}':  fit_pearsonr,
        f'fit_time_{label}':      elapsed,
    }

    # Convergence flag: success AND chi2 and amplitude in range
    converged = (
        fr.success
        and np.isfinite(fr.chi2_per_ndf)
        and fr.chi2_per_ndf <= chi2_max
        and fr.peak_amps
        and amp_min <= fr.peak_amps[0] <= amp_max
    )
    result[f'converged_{label}'] = converged
    return result


def _fit_pair(args):
    """Fit one noisy and one denoised waveform. Used as the Pool work unit."""
    idx, w_noisy, w_denoised, chi2_max, amp_min, amp_max, clean_template = args
    r_noisy    = _fit_one((w_noisy,    'noisy',    chi2_max, amp_min, amp_max, clean_template))
    r_denoised = _fit_one((w_denoised, 'denoised', chi2_max, amp_min, amp_max, clean_template))
    combined = {'idx': idx}
    combined.update(r_noisy)
    combined.update(r_denoised)
    return combined


# ── Main fitting routine ──────────────────────────────────────────────────────

def fit_dataset(input_path: str, output_path: str,
                workers: int,
                chi2_max: float, amp_min: float, amp_max: float,
                amp_dev: float, denoised_field: str = 'waveforms_denoised'):
    # NOTE 2026-09-16: denoised_field added to support comparing multiple
    # denoising methods stored side-by-side in the same eval HDF5 (e.g. the
    # 1.382mV 4-way run: waveforms_denoised_cold / _cold5shot / _warm5shot /
    # _eta1_1shot) without needing separate copies of the noisy/clean/etc
    # fields for each. Default keeps existing single-field callers unchanged.

    with h5py.File(input_path, 'r') as f:
        if denoised_field not in f:
            raise KeyError(f"{denoised_field} not found in {input_path}. "
                           "Run the appropriate inference script first.")
        energy_kev  = float(f.attrs.get('energy_kev', np.nan))
        w_noisy     = f['waveforms_noisy'][:]
        w_denoised  = f[denoised_field][:]
        # Use actual denoised count (may be truncated via --max_events)
        n_events    = min(len(w_noisy), len(w_denoised))
        snr_db      = f['snr_db'][:n_events]
        noise_rms   = f['noise_rms'][:n_events]
        noise_params = {k: f['noise_params'][k][:n_events] for k in f['noise_params']}

    with h5py.File(input_path, 'r') as f:
        clean_template = f['clean_template'][:]
    # NOTE 2026-09-09: previously called as max_baseline_amplitude(clean_template,
    # F_SAMPLE) -- F_SAMPLE (1000.0) was being passed into the `baseline_frac`
    # slot (expects a fraction in [0,1], default 0.05), which silently made the
    # "baseline" the median of the WHOLE waveform instead of just the
    # pre-trigger portion. See scripts/archive/run_fitting_pre_baseline_fix_20260909.py
    # for the full bug writeup. Fixed: use the default baseline_frac=0.05.
    true_amp = max_baseline_amplitude(clean_template)

    print(f"Fitting {n_events} events ({os.path.basename(input_path)})  "
          f"energy={energy_kev:.0f} keV  workers={workers}")
    t_start = time.time()

    work_items = [
        (i, w_noisy[i], w_denoised[i], chi2_max, amp_min, amp_max, clean_template)
        for i in range(n_events)
    ]

    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_fit_pair, work_items)
    else:
        results = [_fit_pair(item) for item in work_items]

    elapsed = time.time() - t_start
    print(f"  Fitting done: {elapsed:.1f}s  ({n_events/elapsed:.1f} events/s)")

    # ── Pack results into arrays ──────────────────────────────────────────────
    def arr(key, dtype=np.float32):
        return np.array([r[key] for r in results], dtype=dtype)

    amp_noisy    = arr('amp_noisy')
    amp_denoised = arr('amp_denoised')

    # Amplitude deviation from truth (for efficiency computation)
    dev_noisy    = np.abs(amp_noisy    - true_amp) / max(true_amp, 1e-12)
    dev_denoised = np.abs(amp_denoised - true_amp) / max(true_amp, 1e-12)

    converged_noisy    = arr('converged_noisy',    dtype=bool)
    converged_denoised = arr('converged_denoised', dtype=bool)

    efficient_noisy    = converged_noisy    & (dev_noisy    <= amp_dev)
    efficient_denoised = converged_denoised & (dev_denoised <= amp_dev)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.attrs['energy_kev']   = energy_kev
        f.attrs['n_events']     = n_events
        f.attrs['true_amp_V']   = true_amp
        f.attrs['chi2_max']     = chi2_max
        f.attrs['amp_min']      = amp_min
        f.attrs['amp_max']      = amp_max
        f.attrs['amp_dev_max']  = amp_dev

        f.create_dataset('energy_true',   data=np.full(n_events, energy_kev, dtype=np.float32))
        f.create_dataset('snr_db',        data=snr_db)
        f.create_dataset('noise_rms',     data=noise_rms)

        for label in ('noisy', 'denoised'):
            f.create_dataset(f'amp_{label}',           data=arr(f'amp_{label}'))
            f.create_dataset(f'amp_raw_{label}',       data=arr(f'amp_raw_{label}'))
            f.create_dataset(f'baseline_{label}',      data=arr(f'baseline_{label}'))
            f.create_dataset(f'alpha_{label}',         data=arr(f'alpha_{label}'))
            f.create_dataset(f'amp_maxbase_{label}',   data=arr(f'amp_maxbase_{label}'))
            f.create_dataset(f'chi2_{label}',          data=arr(f'chi2_{label}'))
            f.create_dataset(f't_rise_{label}',        data=arr(f't_rise_{label}'))
            f.create_dataset(f't_fall1_{label}',       data=arr(f't_fall1_{label}'))
            f.create_dataset(f't_fall2_{label}',       data=arr(f't_fall2_{label}'))
            f.create_dataset(f'onset_{label}',         data=arr(f'onset_{label}'))
            f.create_dataset(f'success_{label}',       data=arr(f'success_{label}', bool))
            f.create_dataset(f'converged_{label}',     data=arr(f'converged_{label}', bool))
            f.create_dataset(f'fit_pearsonr_{label}',  data=arr(f'fit_pearsonr_{label}'))

        f.create_dataset('amp_dev_noisy',       data=dev_noisy.astype(np.float32))
        f.create_dataset('amp_dev_denoised',    data=dev_denoised.astype(np.float32))
        f.create_dataset('efficient_noisy',     data=efficient_noisy)
        f.create_dataset('efficient_denoised',  data=efficient_denoised)

        grp = f.create_group('noise_params')
        for k, v in noise_params.items():
            grp.create_dataset(k, data=v)

    # Quick summary
    c_n = converged_noisy.mean() * 100
    c_d = converged_denoised.mean() * 100
    e_n = efficient_noisy.mean()  * 100
    e_d = efficient_denoised.mean()* 100
    print(f"  Convergence:  noisy={c_n:.1f}%  denoised={c_d:.1f}%")
    print(f"  Efficiency:   noisy={e_n:.1f}%  denoised={e_d:.1f}%")
    print(f"  Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _infer_subdir(path: str) -> str:
    """Guess fit-results subfolder from input path (resolution / efficiency / …)."""
    p = path.lower()
    if 'efficiency' in p:
        return 'efficiency'
    if 'resolution' in p:
        return 'resolution'
    return 'misc'


def main():
    parser = argparse.ArgumentParser(description="Batch pulse fitting on eval datasets")

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input',     help='Single eval HDF5')
    input_grp.add_argument('--input_dir', help='Directory of eval HDF5 files')

    output_grp = parser.add_mutually_exclusive_group()
    output_grp.add_argument('--output',     help='Output fit-results HDF5')
    output_grp.add_argument('--output_dir', help='Output directory for fit-results HDF5s')

    parser.add_argument('--workers',  type=int,   default=max(1, os.cpu_count() - 1))
    parser.add_argument('--chi2_max', type=float, default=DEFAULT_CHI2_MAX)
    parser.add_argument('--amp_min',  type=float, default=DEFAULT_AMP_MIN)
    parser.add_argument('--amp_max',  type=float, default=DEFAULT_AMP_MAX)
    parser.add_argument('--amp_dev',  type=float, default=DEFAULT_AMP_DEV,
                        help='Max fractional amplitude deviation for efficiency flag')
    parser.add_argument('--denoised_field', default='waveforms_denoised',
                        help='HDF5 dataset name to fit as the "denoised" waveform '
                             '(2026-09-16: for comparing multiple methods stored '
                             'side-by-side, e.g. waveforms_denoised_cold / '
                             '_cold5shot / _warm5shot / _eta1_1shot). Give each '
                             'method its own --output_dir, or repeat runs will '
                             'overwrite each other\'s output files.')

    args = parser.parse_args()

    if args.input:
        files = [args.input]
        if args.output:
            out_files = [args.output]
        elif args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.input))[0]
            out_files = [os.path.join(args.output_dir, f'fit_{base[5:]}.h5')]
        else:
            # derive subfolder name from input path (resolution / efficiency / etc.)
            subdir = _infer_subdir(args.input)
            default_out_dir = os.path.join(_RESULTS_DIR, 'fit_results', subdir)
            os.makedirs(default_out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.input))[0]
            out_files = [os.path.join(default_out_dir, f'fit_{base[5:]}.h5')]
    else:
        files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith('.h5')
        )
        subdir = _infer_subdir(args.input_dir)
        out_dir = args.output_dir or os.path.join(_RESULTS_DIR, 'fit_results', subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_files = [
            os.path.join(out_dir, 'fit_' + os.path.basename(p)[5:])
            for p in files
        ]

    for inp, out in zip(files, out_files):
        fit_dataset(inp, out,
                    workers=args.workers,
                    chi2_max=args.chi2_max,
                    amp_min=args.amp_min,
                    amp_max=args.amp_max,
                    amp_dev=args.amp_dev,
                    denoised_field=args.denoised_field)

    print("\nAll done.")


if __name__ == '__main__':
    main()
