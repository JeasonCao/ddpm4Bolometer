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
    /clean/params/...        — detector parameters (R0, T0, lambda0, ...), or
                               tri-exp shape parameters for --model triexp data
    /clean/equilibrium/...   — equilibrium state per pulse (ODE data only)
    /tau_r_1, /tau_d1_1, ... — per-pulse time constants (tri-exp data only)
    /noise/mix/{alpha,beta,r,scale}  — per-window mix metadata (paired noise only)
    /noise/params/...        — noise component params (target_rms, alpha-color, ...)

Attributes
----------
    f_sample, duration, n_samples, n_windows
    model_path, T, beta_1, beta_T, cond_mode, sampler
    scale_cond, scale_emb, film, pulse_model
    source_clean, source_noise

Usage
-----
    python3 -u scripts/denoise_test_set.py \\
        --clean_h5 /media/.../clean_v3/clean_low/test/clean_000.h5 \\
        --noise_h5 /media/.../noise_v2/noise_total/test/noise_000.h5 \\
        --model_path /media/.../ddpm_v3_low/best_model.pt \\
        --output /media/.../ddpm_v3_low/denoised_test_000.h5 \\
        --batch_size 64

    Scale-embedding / FiLM models: add the training flags, e.g. --scale_emb --film
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
from src.ddpm.unet import build_ddpm_model
from src.ddpm.diffusion import GaussianDiffusion


def copy_group(src, dst):
    """Shallow-copy every leaf dataset in `src` into `dst` (preserves layout)."""
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            dst.create_dataset(name, data=obj[()], compression='gzip')
    src.visititems(_visit)


def denoise_batch(diffusion, x_noisy_norm, device, scale=None):
    """Return 1-shot deterministic denoised batch, shape (B, L) normalized.

    ``scale`` (B,) max(|noisy|) in V is passed only to scale-conditioned models.
    """
    x = torch.from_numpy(x_noisy_norm.astype(np.float32)).unsqueeze(1).to(device)
    sc = (torch.from_numpy(scale.astype(np.float32)).to(device)
          if scale is not None else None)
    with torch.no_grad():
        x_single = diffusion.sample(x, scale=sc, stochastic=False).squeeze(1)
    return x_single.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean_h5', required=True)
    p.add_argument('--noise_h5', required=True,
                   help='Legacy noise shard or paired noise_total HDF5')
    p.add_argument('--model_path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--beta_1', type=float, default=1e-4)
    p.add_argument('--beta_T', type=float, default=0.5)
    p.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    p.add_argument('--scale_cond', action='store_true',
                   help='Legacy UNet1DScaleCond checkpoint')
    p.add_argument('--scale_emb', action='store_true',
                   help='UNet1D with scale embedding (must match training)')
    p.add_argument('--film', action='store_true',
                   help='UNet1D with AdaGN/FiLM ResBlocks (must match training)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0,
                   help='Seeds x_T of the reverse process (deterministic DDPM '
                        'still starts from random noise); makes reruns identical')
    p.add_argument('--n_limit', type=int, default=None,
                   help='Stop after this many windows (default: all)')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1,
                                 beta_T=args.beta_T).to(device)
    print(f'Schedule: T={args.T}, beta_T={args.beta_T}, '
          f'alpha_bar_T={schedule.alpha_bar[-1].item():.4g}')
    model = build_ddpm_model(cond_mode=args.cond_mode, scale_cond=args.scale_emb,
                             film=args.film,
                             legacy_scale_cond=args.scale_cond).to(device)
    use_scale = args.scale_cond or args.scale_emb
    print(f'Model: {type(model).__name__}, legacy_scale_cond={args.scale_cond}, '
          f'scale_emb={args.scale_emb}, film={args.film}')
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
            out.attrs['seed'] = args.seed
            out.attrs['scale_cond'] = args.scale_cond
            out.attrs['scale_emb'] = args.scale_emb
            out.attrs['film'] = args.film
            out.attrs['pulse_model'] = cf.attrs.get('pulse_model', 'ode')
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
                        'onsets_1', 'onsets_2',
                        'tau_r_1', 'tau_d1_1', 'tau_d2_1',
                        'tau_r_2', 'tau_d1_2', 'tau_d2_2'):
                if key in cf:
                    out.create_dataset(key, data=cf[key][:n_pair],
                                       compression='gzip')
            clean_grp = out.create_group('clean')
            params_grp = clean_grp.create_group('params')
            for k in cf['params']:
                params_grp.create_dataset(k, data=cf[f'params/{k}'][:n_pair],
                                          compression='gzip')
            if 'equilibrium' in cf:  # ODE data only
                eq_grp = clean_grp.create_group('equilibrium')
                for k in cf['equilibrium']:
                    eq_grp.create_dataset(k, data=cf[f'equilibrium/{k}'][:n_pair],
                                          compression='gzip')

            # Copy noise mix + noise params (sliced to n_pair)
            noise_grp = out.create_group('noise')
            if 'mix' in nf:
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

                single_n = denoise_batch(diffusion, noisy_n, device,
                                         scale_b if use_scale else None)

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
