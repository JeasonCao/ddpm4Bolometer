"""
Batch DDPM inference on evaluation datasets.

Reads an eval HDF5 produced by generate_eval_dataset.py, runs the trained
DDPM model on all noisy waveforms in GPU batches, and writes the denoised
waveforms back into the same file (or a copy).

Usage
-----
# 1-shot (fast, for validation)
python -u scripts/run_batch_inference.py \
    --model_path /home/wsl_0vbb/DDPM4bolometer/model_output/best_model.pt \
    --input     /home/wsl_0vbb/DDPM4bolometer/eval_data/resolution/eval_583keV.h5 \
    --shots 1

# 10-shot (accurate, for final results)
python -u scripts/run_batch_inference.py \
    --model_path /home/wsl_0vbb/DDPM4bolometer/model_output/best_model.pt \
    --input_dir /home/wsl_0vbb/DDPM4bolometer/eval_data/resolution \
    --shots 10 --batch_size 32
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import h5py
import numpy as np
import torch

from src.ddpm.unet import UNet1D
from src.ddpm.unet_cond import UNet1DScaleCond
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion


def load_model(model_path: str, device: torch.device,
               scale_cond: bool = False,
               T: int = 50, beta_1: float = 1e-4, beta_T: float = 0.05,
               cond_mode: str = 'step'):
    schedule = DiffusionSchedule(T=T, beta_1=beta_1, beta_T=beta_T, device=device)
    model_cls = UNet1DScaleCond if scale_cond else UNet1D
    model = model_cls(cond_mode=cond_mode).to(device)
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
                  shots: int = 1) -> np.ndarray:
    """Denoise a (N, L) array of noisy waveforms.

    Returns denoised (N, L) float32 array.
    """
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
        scale_t  = torch.from_numpy(scale_vals[:, 0]).to(device)         # (B,)

        if shots == 1:
            x_out = diffusion.sample(x_tilde)                   # (B,1,L)
        else:
            x_out = diffusion.sample_multi_shot(x_tilde, M=shots)

        # Denormalise
        x_out_np = x_out.squeeze(1).cpu().numpy()               # (B, L)
        denoised[start:end] = x_out_np * scale_vals

        if (start // batch_size) % 10 == 0:
            pct = 100.0 * end / N
            print(f"  [{end}/{N}]  {pct:.0f}%")

    return denoised


def process_file(h5_path: str, diffusion, device: torch.device,
                 batch_size: int, shots: int, overwrite: bool):
    with h5py.File(h5_path, 'r') as f:
        if 'waveforms_denoised' in f and not overwrite:
            print(f"  Already denoised, skipping: {h5_path}")
            return
        noisy = f['waveforms_noisy'][:]

    energy = None
    with h5py.File(h5_path, 'r') as f:
        energy = float(f.attrs.get('energy_kev', -1))

    N = noisy.shape[0]
    print(f"Processing {os.path.basename(h5_path)}  "
          f"({N} events, {shots}-shot, batch={batch_size})")
    t0 = time.time()
    denoised = denoise_batch(diffusion, noisy, device, batch_size, shots)
    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s  ({N/elapsed:.1f} events/s)")

    with h5py.File(h5_path, 'a') as f:
        if 'waveforms_denoised' in f:
            del f['waveforms_denoised']
        f.create_dataset('waveforms_denoised', data=denoised, compression='gzip')
        f.attrs['inference_shots']      = shots
        f.attrs['inference_batch_size'] = batch_size
    print(f"  Saved waveforms_denoised → {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch DDPM inference on eval datasets")
    # Model
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--scale_cond', action='store_true')
    parser.add_argument('--T',        type=int,   default=50)
    parser.add_argument('--beta_1',   type=float, default=1e-4)
    parser.add_argument('--beta_T',   type=float, default=0.05)
    parser.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    # Data
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input',     help='Single eval HDF5 file')
    input_grp.add_argument('--input_dir', help='Directory of eval HDF5 files')
    # Inference
    parser.add_argument('--shots',      type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--overwrite',  action='store_true',
                        help='Re-run even if waveforms_denoised already exists')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    diffusion = load_model(
        args.model_path, device,
        scale_cond=args.scale_cond,
        T=args.T, beta_1=args.beta_1, beta_T=args.beta_T,
        cond_mode=args.cond_mode,
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

    for path in files:
        process_file(path, diffusion, device, args.batch_size, args.shots, args.overwrite)

    print("\nAll done.")


if __name__ == '__main__':
    main()
