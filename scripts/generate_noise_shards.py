"""
Generate legacy single-file noise shards by default, or paired noise shards.
The paired mode saves periodic-only, white-only, and total waveforms.

Output layout
-------------
    OUTPUT_DIR/
      noise_periodic/noise_NNN.h5   # AC + PT + resonances, RMS = target_rms
      noise_white/noise_NNN.h5      # colored + white floor, RMS = target_rms
      noise_total/noise_NNN.h5      # scale * (alpha*P + beta*W) + meta arrays

Per-window indexing is matched across the three folders: window i of
shard NNN in noise_periodic and the same in noise_white combine, with
that window's metadata, into the same window of noise_total.

Paired output has two layouts:
  * Flat: noise parameters are resampled *per window*; each shard
    is just a chunk of i.i.d. windows.
  * Channel bank (``--n_channels``): noise parameters are sampled *once per
    shard* (= per channel) and held fixed for all its windows, so each shard
    has a stationary noise PSD identity. The per-window mix (alpha/beta), the
    overall scale, and the random realization still vary "in their current
    way". Each shard's attributes record the channel number and the fixed
    noise parameters. Shard NNN here is channel NNN, paired with clean shard
    clean_NNN.h5 from src.pulse.generate --n_channels.

Usage:
    # Legacy noise (default)
    python -u scripts/generate_noise_shards.py --output_dir /path/to/noise

    # Paired flat noise
    python -u scripts/generate_noise_shards.py --mode paired --output_dir /path/to/paired

    # Channel bank: 100 channels x 200 windows each
    python -u scripts/generate_noise_shards.py --n_channels 100 \
        --windows_per_channel 200 \
        --output_dir /media/Disk_YIN/yunshancheng/cuore/channel_bank/noise
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import h5py
import numpy as np

from src.noise.generator import (
    generate_noise,
    generate_paired_noise,
    sample_noise_params,
    sample_paired_noise_params,
)

DURATION = 10.0
F_SAMPLE = 1000.0

# Scalar noise parameters that define a channel's PSD identity (logged as shard
# metadata in channel-bank mode). 'resonances' is variable-length so we log its
# count only.
CHANNEL_PARAM_KEYS = (
    'target_rms', 'alpha', 'f_cross', 'ac_a_base', 'ac_p_decay',
    'pt_freq', 'pt_a_base', 'pt_p_decay', 'white_rms', 'envelope_variation',
)


def _write_shard(path, waveforms, params_dict, mix_meta=None, *, seed,
                 channel_id=None, fixed_params=None,
                 fixed_r=None, fixed_scale=None):
    """Write one H5 shard. mix_meta (alpha/beta/r/scale arrays) only for total.

    If ``channel_id`` is given (channel-bank mode), the channel number, the
    fixed noise parameters, and the fixed mix ``r``/``scale`` are recorded in
    the shard attributes.
    """
    n_windows, n_samples = waveforms.shape
    with h5py.File(path, 'w') as f:
        f.attrs['f_sample'] = F_SAMPLE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = n_samples
        f.attrs['n_windows'] = n_windows
        f.attrs['seed'] = seed

        if channel_id is not None:
            f.attrs['channel_id'] = channel_id
            if fixed_params is not None:
                for k in CHANNEL_PARAM_KEYS:
                    f.attrs[f'param_{k}'] = fixed_params[k]
                f.attrs['param_n_resonances'] = len(fixed_params['resonances'])
            if fixed_r is not None:
                f.attrs['param_mix_r'] = fixed_r
            if fixed_scale is not None:
                f.attrs['param_mix_scale'] = fixed_scale

        f.create_dataset('waveforms', data=waveforms, compression='gzip')

        grp = f.create_group('params')
        for k, v in params_dict.items():
            grp.create_dataset(k, data=v)

        if mix_meta is not None:
            mgrp = f.create_group('mix')
            for k, v in mix_meta.items():
                mgrp.create_dataset(k, data=v)


def generate_noise_shard(n_windows, periodic_path, white_path, total_path,
                         seed, channel_id=None, fixed_params=None,
                         fixed_r=None, fixed_scale=None):
    """Generate one legacy or paired noise shard.

    With periodic_path=None, write only the legacy total_path file.

    If ``fixed_params`` is given, those noise parameters are used for every
    window (channel-bank mode); otherwise parameters are resampled per window
    (flat mode). When ``fixed_r``/``fixed_scale`` are given, the periodic/white
    balance and overall amplitude are also held fixed per channel; only the
    random realization varies per window. In flat mode (both None) the mix and
    scale are drawn per window by ``generate_paired_noise``.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION * F_SAMPLE)

    paired = periodic_path is not None
    wave_total = np.zeros((n_windows, n_samples), dtype=np.float64)
    if paired:
        wave_periodic = np.zeros_like(wave_total)
        wave_white = np.zeros_like(wave_total)

    # Per-window scalar params (in channel mode these are constant per shard,
    # but we still log them per window so the file format is identical).
    par_target_rms = np.zeros(n_windows)
    par_alpha_color = np.zeros(n_windows)  # 1/f^α exponent (not the mix α)
    par_f_cross = np.zeros(n_windows)
    par_ac_a_base = np.zeros(n_windows)
    par_pt_a_base = np.zeros(n_windows)
    par_white_rms = np.zeros(n_windows)
    par_envelope_var = np.zeros(n_windows)
    par_n_resonances = np.zeros(n_windows, dtype=int)

    if paired:
        mix_alpha = np.zeros(n_windows)
        mix_beta = np.zeros(n_windows)
        mix_r = np.zeros(n_windows)
        mix_scale = np.zeros(n_windows)

    t_start = time.time()

    for i in range(n_windows):
        if paired:
            params = fixed_params if fixed_params is not None else sample_paired_noise_params(rng)
            periodic, white, total, meta = generate_paired_noise(
                rng, params, duration=DURATION, f_sample=F_SAMPLE,
                r=fixed_r, scale=fixed_scale,
            )
            wave_periodic[i] = periodic
            wave_white[i] = white
            wave_total[i] = total
            mix_alpha[i] = meta['alpha']
            mix_beta[i] = meta['beta']
            mix_r[i] = meta['r']
            mix_scale[i] = meta['scale']
        else:
            params = sample_noise_params(rng)
            wave_total[i] = generate_noise(rng, params, duration=DURATION, f_sample=F_SAMPLE)

        par_target_rms[i] = params['target_rms']
        par_alpha_color[i] = params['alpha']
        par_f_cross[i] = params['f_cross']
        par_ac_a_base[i] = params['ac_a_base']
        par_pt_a_base[i] = params['pt_a_base']
        par_white_rms[i] = params['white_rms']
        par_envelope_var[i] = params['envelope_variation']
        par_n_resonances[i] = len(params['resonances'])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_windows - i - 1) / rate if rate > 0 else 0
            tag = f"ch{channel_id:03d} " if channel_id is not None else ""
            print(f"  {tag}[{i+1}/{n_windows}] "
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
    if paired:
        mix_meta = {'alpha': mix_alpha, 'beta': mix_beta, 'r': mix_r, 'scale': mix_scale}
        kw = dict(seed=seed, channel_id=channel_id, fixed_params=fixed_params,
                  fixed_r=fixed_r, fixed_scale=fixed_scale)
        _write_shard(periodic_path, wave_periodic, params_dict, **kw)
        _write_shard(white_path, wave_white, params_dict, **kw)
        _write_shard(total_path, wave_total, params_dict, mix_meta=mix_meta, **kw)
    else:
        _write_shard(total_path, wave_total, params_dict, seed=seed)

    elapsed = time.time() - t_start
    print(f"Saved {n_windows} windows to {total_path} in {elapsed:.1f}s")


def _shard_paths(periodic_dir, white_dir, total_dir, idx):
    fname = f'noise_{idx:03d}.h5'
    return (os.path.join(periodic_dir, fname),
            os.path.join(white_dir, fname),
            os.path.join(total_dir, fname))


def run_flat(args, output_dir, paired):
    for shard in range(args.n_shards):
        if paired:
            periodic_path, white_path, total_path = _shard_paths(
                *paired_output_dirs(output_dir, args.split), shard,
            )
            paths = (periodic_path, white_path, total_path)
        else:
            total_path = os.path.join(output_dir, args.split, f'noise_{shard:03d}.h5')
            periodic_path, white_path = None, None
            paths = (total_path,)
        if all(os.path.exists(path) for path in paths):
            print(f"Shard {shard} already complete, skipping.")
            continue
        seed = args.base_seed + shard * 1000
        print(f"Generating shard {shard}/{args.n_shards} (windows: {args.windows_per_shard}, seed: {seed})")
        generate_noise_shard(args.windows_per_shard,
                             periodic_path, white_path, total_path, seed=seed)

    print(f"\nAll {args.n_shards} shards complete.")


def paired_output_dirs(output_dir, split):
    return tuple(os.path.join(output_dir, name, split)
                 for name in ('noise_periodic', 'noise_white', 'noise_total'))


def run_channel_bank(args, periodic_dir, white_dir, total_dir):
    """One shard per channel, noise params fixed per channel."""
    child_seqs = np.random.SeedSequence(args.base_seed).spawn(args.n_channels)
    for c in range(args.n_channels):
        periodic_path, white_path, total_path = _shard_paths(
            periodic_dir, white_dir, total_dir, c)
        if all(os.path.exists(p) for p in (periodic_path, white_path, total_path)):
            print(f"Channel {c} already complete, skipping.")
            continue

        # Independent stream per channel; the noise-param draw is fixed here,
        # along with the mix balance r and overall scale (channel identity).
        param_rng = np.random.default_rng(child_seqs[c])
        fixed_params = sample_paired_noise_params(param_rng)
        fixed_r = float(param_rng.uniform(0.0, 1.0))
        fixed_scale = float(param_rng.uniform(0.5, 2.0))
        seed = int(child_seqs[c].generate_state(1)[0])

        print(f"\n{'='*60}")
        print(f"Channel {c}/{args.n_channels}  "
              f"(windows: {args.windows_per_channel}, seed: {seed}, "
              f"r={fixed_r:.3f}, scale={fixed_scale:.3f})")
        print(f"{'='*60}")
        generate_noise_shard(args.windows_per_channel,
                             periodic_path, white_path, total_path,
                             seed=seed, channel_id=c, fixed_params=fixed_params,
                             fixed_r=fixed_r, fixed_scale=fixed_scale)

    print(f"\nAll {args.n_channels} channel shards complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate noise shards")
    parser.add_argument('--mode', choices=('legacy', 'paired'),
                        help='Legacy single-file shards by default; paired generates three noise families')
    parser.add_argument('--output_dir', type=str,
                        help='Output root; paired mode creates noise_periodic, noise_white and noise_total')
    parser.add_argument('--split', type=str, default='',
                        help='Optional output subdirectory, e.g. test')
    # Channel-bank mode
    parser.add_argument('--n_channels', type=int, default=None,
                        help='If set, generate a channel bank: one shard per '
                             'channel with fixed noise params per channel.')
    parser.add_argument('--windows_per_channel', type=int, default=200,
                        help='Windows per channel (channel-bank mode)')
    # Flat mode
    parser.add_argument('--n_shards', type=int, default=10)
    parser.add_argument('--windows_per_shard', type=int, default=10000)
    parser.add_argument('--base_seed', type=int, default=7777)
    args = parser.parse_args()

    # The existing --n_channels option explicitly requests a paired channel bank.
    paired = args.mode == 'paired' or (args.mode is None and args.n_channels is not None)
    if args.n_channels is not None and not paired:
        parser.error('--n_channels requires paired mode')
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ('/media/Disk_YIN/yunshancheng/cuore/noise_v2' if paired
                      else '/mnt/d/storage/research/DDPM4bolometer/simu_data/noise')

    output_dirs = paired_output_dirs(output_dir, args.split) if paired else (os.path.join(output_dir, args.split),)
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    if args.n_channels is not None:
        run_channel_bank(args, *output_dirs)
    else:
        run_flat(args, output_dir, paired)
