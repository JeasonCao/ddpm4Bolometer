"""
Focused comparison of two DDPM models on a small set of pileup indices.
Same logic as compare_models_pileup_qa.py but plots a few rows at large size
so jitter/dip features are actually visible.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import PulseNoiseDataset
from src.basics.fit import fit_pulse, triexp_double


def load_model(model_path, T, device):
    cfg_path = os.path.join(os.path.dirname(model_path), 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    schedule = DiffusionSchedule(T=T,
                                 beta_1=cfg.get('beta_1', 1e-4),
                                 beta_T=cfg.get('beta_T', 0.05)).to(device)
    model = UNet1D(cond_mode=cfg.get('cond_mode', 'step'),
                   cond_scale=cfg.get('cond_scale', 1000.0)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diff = GaussianDiffusion(model, schedule, cond_mode=cfg.get('cond_mode', 'step'))
    return diff, cfg


def run_single_shot(diffusion, x_noisy_dev, sample_seed):
    g = torch.Generator(device=x_noisy_dev.device)
    g.manual_seed(sample_seed)
    B, C, L = x_noisy_dev.shape
    x_t = torch.randn(B, 1, L, device=x_noisy_dev.device, generator=g)
    with torch.no_grad():
        for i in reversed(range(diffusion.T)):
            cond = torch.full((B,), i + 1, device=x_noisy_dev.device, dtype=torch.long)
            eps_pred = diffusion._predict_noise(x_t, x_noisy_dev, cond, None)
            beta_t = diffusion.schedule.beta[i]
            coeff = beta_t / diffusion.schedule.sqrt_one_minus_alpha_bar[i]
            x_t = (1.0 / diffusion.schedule.sqrt_alpha[i]) * (x_t - coeff * eps_pred)
    return x_t.squeeze().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_a', required=True)
    p.add_argument('--label_a', default='A')
    p.add_argument('--model_b', required=True)
    p.add_argument('--label_b', default='B')
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--noise_dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--init_seed', type=int, default=12345)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    diff_a, cfg_a = load_model(args.model_a, args.T, device)
    diff_b, cfg_b = load_model(args.model_b, args.T, device)
    print(f"A ({args.label_a}): beta_T={cfg_a.get('beta_T')}, loss={cfg_a.get('loss')}")
    print(f"B ({args.label_b}): beta_T={cfg_b.get('beta_T')}, loss={cfg_b.get('loss')}")

    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)

    fs = 1000.0
    n_rows = len(args.indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.0 * n_rows), squeeze=False)

    for row, idx in enumerate(args.indices):
        x_clean, x_noisy, scale = dataset[int(idx)]
        scale = scale.item()
        x_noisy_dev = x_noisy.unsqueeze(0).to(device)
        sample_seed = args.init_seed + int(idx)

        x_a = run_single_shot(diff_a, x_noisy_dev, sample_seed) * scale
        x_b = run_single_shot(diff_b, x_noisy_dev, sample_seed) * scale
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale

        sigs = {'Clean': x_clean_np, 'Noisy': x_noisy_np,
                args.label_a: x_a, args.label_b: x_b}
        fits = {k: fit_pulse(v, 2) for k, v in sigs.items()}
        t = np.arange(len(x_clean_np)) / fs

        for col, (name, sig) in enumerate(sigs.items()):
            ax = axes[row, col]
            ax.plot(t, sig * 1e3, 'k-', lw=0.5, alpha=0.8, label='data')
            fr = fits[name]
            ax.plot(t, triexp_double(t, *fr.params) * 1e3, 'tab:blue', lw=1.0,
                    label='fit')
            ax.set_title(
                f"{name}  (idx={idx}, scale={scale*1e3:.1f} mV)\n"
                f"chi2/ndf={fr.chi2_per_ndf:.2e}  "
                f"tau_r={fr.tau_r[0]*1e3:.1f} ms  "
                f"tau_d1={fr.tau_d[0]*1e3:.0f} ms",
                fontsize=9)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('mV')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Difference panel: bT05 minus l1 (or whatever a/b are) — same column count, so add as overlay
        # Add a small inset-style residual on the rightmost (label_b) panel
        ax = axes[row, 3]
        diff = (x_b - x_a) * 1e3  # mV
        ax2 = ax.twinx()
        ax2.plot(t, diff, color='tab:red', lw=0.4, alpha=0.5,
                 label=f'{args.label_b}-{args.label_a}')
        ax2.set_ylabel(f'{args.label_b} - {args.label_a} [mV]',
                       color='tab:red', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=7)
        ax2.legend(loc='lower right', fontsize=7)

    fig.suptitle(
        f"Same-init deterministic comparison\n"
        f"A={args.label_a} (beta_T={cfg_a.get('beta_T')}, {cfg_a.get('loss')})  "
        f"B={args.label_b} (beta_T={cfg_b.get('beta_T')}, {cfg_b.get('loss')})  "
        f"init_seed={args.init_seed}", fontsize=12)
    plt.tight_layout()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
