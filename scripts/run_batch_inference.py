"""
Batch inference on evaluation datasets (DDIM or DDPM).

Reads an eval HDF5 produced by generate_eval_dataset.py, runs the trained
DDPM model on all noisy waveforms in GPU batches, and writes the denoised
waveforms back into the same file (or a copy).

Usage
-----
# DDIM single-shot (eta=0)
python -u scripts/run_batch_inference.py \
    --model_path /mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt \
    --input_dir /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/resolution \
    --sampler ddim --seed 42

# DDPM 1-shot (stochastic, fast validation)
python -u scripts/run_batch_inference.py \
    --model_path /mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt \
    --input     /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/resolution/eval_583keV.h5 \
    --sampler ddpm --shots 1

# Scale embedding + FiLM, Gaussian start and no per-step noise (float32)
python -u scripts/run_batch_inference.py --model_path /path/to/best_model.pt \
    --input /path/to/writable_eval_copy.h5 --scale_emb --film --sampler ddpm --no_noise
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime
import h5py
import numpy as np
import torch

from src.ddpm.unet import build_ddpm_model
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion


def load_model(model_path: str, device: torch.device,
               scale_cond: bool = False,
               T: int = 50, beta_1: float = 1e-4, beta_T: float = 0.5,
               cond_mode: str = 'step', *, cond_scale: float = 1000.0,
               scale_emb: bool = False, film: bool = False):
    schedule = DiffusionSchedule(T=T, beta_1=beta_1, beta_T=beta_T).to(device)
    model = build_ddpm_model(cond_mode=cond_mode, cond_scale=cond_scale,
                             scale_cond=scale_emb, film=film,
                             legacy_scale_cond=scale_cond).to(device)
    state = torch.load(model_path, map_location=device)
    # best_model.pt stores a bare state_dict; checkpoint_XXX.pt stores a dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=cond_mode)
    return diffusion


@torch.no_grad()
def denoise_batch(diffusion, noisy_np: np.ndarray,
                  device: torch.device,
                  batch_size: int = 32,
                  shots: int = 1,
                  sampler: str = 'ddim',
                  eta: float = 0.0,
                  seed: int = 42, *, use_scale: bool = False,
                  no_noise: bool = False) -> np.ndarray:
    """Denoise a (N, L) array of noisy waveforms.

    Returns denoised (N, L) float32 array.
    sampler='ddim' (default): deterministic, no per-step noise injection.
    sampler='ddpm': shots>1 averages independent samples. When no_noise is
    set, each shot starts from Gaussian noise and uses the posterior mean
    at every reverse step in float32.
    """
    torch.manual_seed(seed)

    N, L = noisy_np.shape
    denoised = np.zeros((N, L), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = noisy_np[start:end].astype(np.float32)          # (B, L)

        # Per-window normalisation (same as training dataset)
        scale_vals = np.abs(batch).max(axis=1, keepdims=True)   # (B, 1)
        scale_vals = np.where(scale_vals == 0, 1.0, scale_vals)
        batch_norm = batch / scale_vals                          # (B, L) in [-1,1]

        x_tilde = torch.from_numpy(batch_norm).unsqueeze(1).to(device)  # (B,1,L)

        scale = (torch.from_numpy(scale_vals[:, 0]).to(device=device, dtype=torch.float32)
                 if use_scale else None)
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=device.type == 'cuda' and not no_noise)
        with amp_ctx:
            if sampler == 'ddim':
                x_out = diffusion.ddim_sample(x_tilde, scale=scale, eta=eta)
            elif shots == 1:
                x_out = diffusion.sample(x_tilde, scale=scale, stochastic=not no_noise)
            else:
                x_out = diffusion.sample_multi_shot(
                    x_tilde, scale=scale, M=shots, stochastic=not no_noise,
                )

        # Denormalise
        x_out_np = x_out.squeeze(1).cpu().numpy()               # (B, L)
        denoised[start:end] = x_out_np * scale_vals

        if (start // batch_size) % 10 == 0:
            pct = 100.0 * end / N
            print(f"  [{end}/{N}]  {pct:.0f}%")

    return denoised


def process_file(h5_path: str, diffusion, device: torch.device,
                 batch_size: int, shots: int, overwrite: bool,
                 sampler: str, eta: float, seed: int,
                 max_events: int = 0, *, use_scale: bool = False,
                 no_noise: bool = False, model_metadata=None):
    with h5py.File(h5_path, 'r') as f:
        if 'waveforms_denoised' in f and not overwrite:
            print(f"  Already denoised, skipping: {h5_path}")
            return
        limit = max_events if max_events > 0 else None
        noisy = f['waveforms_noisy'][:limit]

    N = noisy.shape[0]
    label = f'DDIM(eta={eta})' if sampler == 'ddim' else f'DDPM {shots}-shot'
    if no_noise:
        label += ' (no per-step noise, float32)'
    print(f"Processing {os.path.basename(h5_path)}  "
          f"({N} events, {label}, batch={batch_size})")
    t0 = time.time()
    denoised = denoise_batch(diffusion, noisy, device, batch_size,
                             shots=shots, sampler=sampler, eta=eta, seed=seed,
                             use_scale=use_scale, no_noise=no_noise)
    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s  ({N/elapsed:.1f} events/s)")

    with h5py.File(h5_path, 'a') as f:
        if 'waveforms_denoised' in f:
            del f['waveforms_denoised']
        f.create_dataset('waveforms_denoised', data=denoised, compression='gzip')
        f.attrs['inference_sampler']    = sampler
        f.attrs['inference_shots']      = shots if sampler == 'ddpm' else 1
        f.attrs['inference_eta']        = eta
        f.attrs['inference_no_noise']   = no_noise
        f.attrs['inference_amp']        = device.type == 'cuda' and not no_noise
        if model_metadata is not None:
            f.attrs.update(model_metadata)
        f.attrs['inference_seed']       = seed
        f.attrs['inference_batch_size'] = batch_size
        f.attrs['inference_timestamp']  = datetime.now().strftime('%Y%m%d')
    print(f"  Saved waveforms_denoised → {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch DDPM inference on eval datasets")
    # Model
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--scale_cond', action='store_true',
                        help='Legacy scale-conditioned U-Net checkpoint')
    parser.add_argument('--scale_emb', action='store_true',
                        help='New scale embedding; match training')
    parser.add_argument('--film', action='store_true',
                        help='AdaGN/FiLM ResBlocks; match training')
    parser.add_argument('--T',        type=int,   default=50)
    parser.add_argument('--beta_1',   type=float, default=1e-4)
    parser.add_argument('--beta_T',   type=float, default=0.5)
    parser.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    parser.add_argument('--cond_scale', type=float, default=1000.0)
    # Data
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input',     help='Single eval HDF5 file')
    input_grp.add_argument('--input_dir', help='Directory of eval HDF5 files')
    # Inference
    parser.add_argument('--sampler',    default='ddim', choices=['ddim', 'ddpm'],
                        help='ddim: deterministic (recommended); ddpm: stochastic')
    parser.add_argument('--no_noise', action='store_true',
                        help='DDPM posterior mean from a Gaussian start, in float32')
    parser.add_argument('--eta',        type=float, default=0.0,
                        help='DDIM eta: 0=deterministic, 1=DDPM-equivalent (ignored for ddpm)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--shots',      type=int, default=1,
                        help='Number of DDPM samples to average (ignored for ddim)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--overwrite',  action='store_true',
                        help='Re-run even if waveforms_denoised already exists')
    parser.add_argument('--max_events', type=int, default=0,
                        help='Truncate each file to this many events (0 = all)')

    args = parser.parse_args()
    if args.no_noise and args.sampler != 'ddpm':
        parser.error('--no_noise requires --sampler ddpm')
    if args.scale_cond and (args.scale_emb or args.film):
        parser.error('--scale_cond is incompatible with --scale_emb and --film')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    diffusion = load_model(
        args.model_path, device,
        scale_cond=args.scale_cond,
        T=args.T, beta_1=args.beta_1, beta_T=args.beta_T,
        cond_mode=args.cond_mode, cond_scale=args.cond_scale,
        scale_emb=args.scale_emb, film=args.film,
    )
    print("Model loaded.")

    if args.input:
        files = [args.input]
    else:
        files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith('.h5')
        )

    model_metadata = {
        'inference_model_path': os.path.abspath(args.model_path),
        'inference_T': args.T,
        'inference_beta_1': args.beta_1,
        'inference_beta_T': args.beta_T,
        'inference_cond_mode': args.cond_mode,
        'inference_cond_scale': args.cond_scale,
        'inference_legacy_scale_cond': args.scale_cond,
        'inference_scale_emb': args.scale_emb,
        'inference_film': args.film,
    }
    for path in files:
        process_file(path, diffusion, device, args.batch_size, args.shots,
                     args.overwrite, args.sampler, args.eta, args.seed,
                     args.max_events, use_scale=args.scale_cond or args.scale_emb,
                     no_noise=args.no_noise, model_metadata=model_metadata)

    print("\nAll done.")


if __name__ == '__main__':
    main()
