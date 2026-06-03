"""Run DDPM denoising over the test set and store everything as HDF5.

Flow
----
1. Read clean test + noise_total test.
2. Pair clean[i] ↔ noise[i] deterministically (i↔i, no permutation).
3. Build noisy = clean[i] + noise[i], normalize by max(|noisy|), run
   **deterministic 1-shot** DDPM (stochastic=False).
4. Write one output shard that mirrors the clean/noise layout and copies
   the per-window metadata from clean[i] and noise[i].

Output H5 layout
----------------
    /waveforms        (N, L) float32  — 1-shot DDPM (deterministic)
    /waveforms_clean  (N, L) float32  — copy of clean
    /waveforms_noisy  (N, L) float32  — clean + noise_total
    /scale            (N,)   float32  — per-window normalization (max(|noisy|))

    /is_pileup, /energies_1, /energies_2, /onsets_1, /onsets_2  — from clean
    /clean/params/...        — detector parameters (R0, T0, lambda0, ...)
    /clean/equilibrium/...   — equilibrium state per pulse
    /noise/mix/{alpha,beta,r,scale}  — per-window mix metadata
    /noise/params/...        — noise component params (target_rms, alpha-color, ...)

Attributes
----------
    f_sample, duration, n_samples, n_windows
    model_path, T, beta_1, beta_T, cond_mode, sampler
    source_clean, source_noise

Usage
-----
    python3 -u scripts/denoise_test_set.py \\
        --clean_h5 /media/.../clean_v3/clean_low/test/clean_000.h5 \\
        --noise_h5 /media/.../noise_v2/noise_total/test/noise_000.h5 \\
        --model_path /media/.../ddpm_v3_low/best_model.pt \\
        --output /media/.../ddpm_v3_low/denoised_test_000.h5 \\
        --batch_size 64
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion


def copy_group(src, dst):
    """Shallow-copy every leaf dataset in `src` into `dst` (preserves layout)."""
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            dst.create_dataset(name, data=obj[()], compression='gzip')
    src.visititems(_visit)


def denoise_batch(diffusion, x_noisy_norm, device):
    """Return 1-shot deterministic denoised batch, shape (B, L) normalized."""
    x = torch.from_numpy(x_noisy_norm.astype(np.float32)).unsqueeze(1).to(device)
    with torch.no_grad():
        x_single = diffusion.sample(x, stochastic=False).squeeze(1)
    return x_single.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean_h5', required=True)
    p.add_argument('--noise_h5', required=True,
                   help='noise_total h5 (must contain /mix/ group)')
    p.add_argument('--model_path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--beta_1', type=float, default=1e-4)
    p.add_argument('--beta_T', type=float, default=0.5)
    p.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_limit', type=int, default=None,
                   help='Stop after this many windows (default: all)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1,
                                 beta_T=args.beta_T).to(device)
    print(f'Schedule: T={args.T}, beta_T={args.beta_T}, '
          f'alpha_bar_T={schedule.alpha_bar[-1].item():.4g}')
    model = UNet1D(cond_mode=args.cond_mode).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=args.cond_mode)

    with h5py.File(args.clean_h5, 'r') as cf, h5py.File(args.noise_h5, 'r') as nf:
        n_clean = int(cf.attrs.get('n_windows', cf.attrs.get('n_pulses')))
        n_noise = int(nf.attrs.get('n_windows', nf.attrs.get('n_pulses')))
        n_pair = min(n_clean, n_noise)
        if args.n_limit is not None:
            n_pair = min(n_pair, args.n_limit)
        f_sample = float(cf.attrs['f_sample'])
        duration = float(cf.attrs['duration'])
        L = int(cf.attrs['n_samples'])
        print(f'Pairing {n_pair} windows (clean={n_clean}, noise={n_noise}); L={L}.')

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with h5py.File(args.output, 'w') as out:
            # Attributes
            out.attrs['f_sample'] = f_sample
            out.attrs['duration'] = duration
            out.attrs['n_samples'] = L
            out.attrs['n_windows'] = n_pair
            out.attrs['model_path'] = args.model_path
            out.attrs['T'] = args.T
            out.attrs['beta_1'] = args.beta_1
            out.attrs['beta_T'] = args.beta_T
            out.attrs['cond_mode'] = args.cond_mode
            out.attrs['sampler'] = 'ddpm_deterministic'
            out.attrs['source_clean'] = args.clean_h5
            out.attrs['source_noise'] = args.noise_h5

            chunk = (1, L)
            ds_single = out.create_dataset(
                'waveforms', shape=(n_pair, L), dtype='float32',
                chunks=chunk, compression='gzip')
            ds_clean = out.create_dataset(
                'waveforms_clean', shape=(n_pair, L), dtype='float32',
                chunks=chunk, compression='gzip')
            ds_noisy = out.create_dataset(
                'waveforms_noisy', shape=(n_pair, L), dtype='float32',
                chunks=chunk, compression='gzip')
            ds_scale = out.create_dataset(
                'scale', shape=(n_pair,), dtype='float32')

            # Copy per-window metadata from clean (sliced to n_pair)
            for key in ('is_pileup', 'energies_1', 'energies_2',
                        'onsets_1', 'onsets_2'):
                if key in cf:
                    out.create_dataset(key, data=cf[key][:n_pair],
                                       compression='gzip')
            clean_grp = out.create_group('clean')
            params_grp = clean_grp.create_group('params')
            for k in cf['params']:
                params_grp.create_dataset(k, data=cf[f'params/{k}'][:n_pair],
                                          compression='gzip')
            eq_grp = clean_grp.create_group('equilibrium')
            for k in cf['equilibrium']:
                eq_grp.create_dataset(k, data=cf[f'equilibrium/{k}'][:n_pair],
                                      compression='gzip')

            # Copy noise mix + noise params (sliced to n_pair)
            noise_grp = out.create_group('noise')
            mix_grp = noise_grp.create_group('mix')
            for k in nf['mix']:
                mix_grp.create_dataset(k, data=nf[f'mix/{k}'][:n_pair],
                                       compression='gzip')
            np_grp = noise_grp.create_group('params')
            for k in nf['params']:
                np_grp.create_dataset(k, data=nf[f'params/{k}'][:n_pair],
                                      compression='gzip')

            # Run denoising in batches
            c_wfs = cf['waveforms']
            n_wfs = nf['waveforms']
            t_start = time.time()
            for start in range(0, n_pair, args.batch_size):
                end = min(start + args.batch_size, n_pair)
                clean_b = c_wfs[start:end].astype(np.float64)
                noise_b = n_wfs[start:end].astype(np.float64)
                noisy_b = clean_b + noise_b

                # Per-window normalization
                scale_b = np.max(np.abs(noisy_b), axis=1)
                scale_b = np.maximum(scale_b, 1e-12)
                inv = 1.0 / scale_b
                noisy_n = noisy_b * inv[:, None]

                single_n = denoise_batch(diffusion, noisy_n, device)

                # Un-normalize back to physical units
                single_p = single_n * scale_b[:, None]

                ds_single[start:end] = single_p.astype(np.float32)
                ds_clean[start:end] = clean_b.astype(np.float32)
                ds_noisy[start:end] = noisy_b.astype(np.float32)
                ds_scale[start:end] = scale_b.astype(np.float32)

                elapsed = time.time() - t_start
                rate = end / elapsed
                eta = (n_pair - end) / rate if rate > 0 else 0
                print(f'  [{end}/{n_pair}] '
                      f'rate={rate:.1f} win/s ETA={eta:.0f}s', flush=True)

    print(f'\nWrote {args.output} ({n_pair} windows).')


if __name__ == '__main__':
    main()
