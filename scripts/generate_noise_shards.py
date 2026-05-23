"""
Generate paired noise windows: periodic-only, white-only, and total
(combined with random energy mix and overall scale).

Output layout
-------------
    OUTPUT_DIR/
      noise_periodic/noise_NNN.h5   # AC + PT + resonances, RMS = target_rms
      noise_white/noise_NNN.h5      # colored + white floor, RMS = target_rms
      noise_total/noise_NNN.h5      # scale * (alpha*P + beta*W) + meta arrays

Per-window indexing is matched across the three folders: window i of
shard NNN in noise_periodic and the same in noise_white combine, with
that window's metadata, into the same window of noise_total.

Usage:
    python -u scripts/generate_noise_shards.py
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import h5py
import numpy as np

from src.noise.generator import (
    generate_paired_noise,
    sample_noise_params,
)

DURATION = 10.0
F_SAMPLE = 1000.0


def _write_shard(path, waveforms, params_dict, mix_meta=None, *, seed):
    """Write one H5 shard. mix_meta (alpha/beta/r/scale arrays) only for total."""
    n_windows, n_samples = waveforms.shape
    with h5py.File(path, 'w') as f:
        f.attrs['f_sample'] = F_SAMPLE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = n_samples
        f.attrs['n_windows'] = n_windows
        f.attrs['seed'] = seed

        f.create_dataset('waveforms', data=waveforms, compression='gzip')

        grp = f.create_group('params')
        for k, v in params_dict.items():
            grp.create_dataset(k, data=v)

        if mix_meta is not None:
            mgrp = f.create_group('mix')
            for k, v in mix_meta.items():
                mgrp.create_dataset(k, data=v)


def generate_noise_shard(n_windows, periodic_path, white_path, total_path,
                         seed):
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION * F_SAMPLE)

    wave_periodic = np.zeros((n_windows, n_samples), dtype=np.float64)
    wave_white = np.zeros((n_windows, n_samples), dtype=np.float64)
    wave_total = np.zeros((n_windows, n_samples), dtype=np.float64)

    # Per-window scalar params (logged identically in periodic/white/total)
    par_target_rms = np.zeros(n_windows)
    par_alpha_color = np.zeros(n_windows)  # 1/f^α exponent (not the mix α)
    par_f_cross = np.zeros(n_windows)
    par_ac_a_base = np.zeros(n_windows)
    par_pt_a_base = np.zeros(n_windows)
    par_white_rms = np.zeros(n_windows)
    par_envelope_var = np.zeros(n_windows)
    par_n_resonances = np.zeros(n_windows, dtype=int)

    # Mix metadata (only stored on noise_total)
    mix_alpha = np.zeros(n_windows)
    mix_beta = np.zeros(n_windows)
    mix_r = np.zeros(n_windows)
    mix_scale = np.zeros(n_windows)

    t_start = time.time()
    for i in range(n_windows):
        params = sample_noise_params(rng)
        periodic, white, total, meta = generate_paired_noise(
            rng, params, duration=DURATION, f_sample=F_SAMPLE,
        )
        wave_periodic[i] = periodic
        wave_white[i] = white
        wave_total[i] = total

        par_target_rms[i] = params['target_rms']
        par_alpha_color[i] = params['alpha']
        par_f_cross[i] = params['f_cross']
        par_ac_a_base[i] = params['ac_a_base']
        par_pt_a_base[i] = params['pt_a_base']
        par_white_rms[i] = params['white_rms']
        par_envelope_var[i] = params['envelope_variation']
        par_n_resonances[i] = len(params['resonances'])

        mix_alpha[i] = meta['alpha']
        mix_beta[i] = meta['beta']
        mix_r[i] = meta['r']
        mix_scale[i] = meta['scale']

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_windows - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_windows}] "
                  f"({rate:.1f} windows/s, ETA {eta:.0f}s)")

    params_dict = {
        'target_rms': par_target_rms,
        'alpha': par_alpha_color,
        'f_cross': par_f_cross,
        'ac_a_base': par_ac_a_base,
        'pt_a_base': par_pt_a_base,
        'white_rms': par_white_rms,
        'envelope_variation': par_envelope_var,
        'n_resonances': par_n_resonances,
    }
    mix_meta = {
        'alpha': mix_alpha,
        'beta': mix_beta,
        'r': mix_r,
        'scale': mix_scale,
    }

    print(f"Saving periodic → {periodic_path}")
    _write_shard(periodic_path, wave_periodic, params_dict, seed=seed)
    print(f"Saving white    → {white_path}")
    _write_shard(white_path, wave_white, params_dict, seed=seed)
    print(f"Saving total    → {total_path}")
    _write_shard(total_path, wave_total, params_dict,
                 mix_meta=mix_meta, seed=seed)

    elapsed = time.time() - t_start
    print(f"Done. {n_windows} paired windows in {elapsed:.1f}s "
          f"({n_windows/elapsed:.1f} windows/s)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate paired noise shards")
    parser.add_argument('--output_dir', type=str,
                        default='/media/Disk_YIN/yunshancheng/cuore/noise_v2',
                        help='Root dir; will contain noise_periodic/, noise_white/, noise_total/ '
                             '(or split= prefixes)')
    parser.add_argument('--split', type=str, default='',
                        help='Optional subdir under each of {periodic,white,total} '
                             '(e.g. "test"); default writes directly to noise_*/')
    parser.add_argument('--n_shards', type=int, default=10)
    parser.add_argument('--windows_per_shard', type=int, default=10000)
    parser.add_argument('--base_seed', type=int, default=7777)
    args = parser.parse_args()

    periodic_dir = os.path.join(args.output_dir, 'noise_periodic', args.split)
    white_dir = os.path.join(args.output_dir, 'noise_white', args.split)
    total_dir = os.path.join(args.output_dir, 'noise_total', args.split)

    for d in (periodic_dir, white_dir, total_dir):
        os.makedirs(d, exist_ok=True)

    for shard in range(args.n_shards):
        fname = f'noise_{shard:03d}.h5'
        periodic_path = os.path.join(periodic_dir, fname)
        white_path = os.path.join(white_dir, fname)
        total_path = os.path.join(total_dir, fname)

        if (os.path.exists(periodic_path)
                and os.path.exists(white_path)
                and os.path.exists(total_path)):
            print(f"Shard {shard} already complete, skipping.")
            continue

        seed = args.base_seed + shard * 1000
        print(f"\n{'='*60}")
        print(f"Generating shard {shard}/{args.n_shards}")
        print(f"  Windows: {args.windows_per_shard}, seed: {seed}")
        print(f"{'='*60}")

        generate_noise_shard(args.windows_per_shard,
                             periodic_path, white_path, total_path,
                             seed=seed)

    print(f"\nAll {args.n_shards} shards complete.")
