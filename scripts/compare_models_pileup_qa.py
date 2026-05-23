"""
Compare two DDPM models on the same deterministic pileup QA samples.

For each model, loads its training schedule (beta_1, beta_T, cond_mode) from
the config.json next to best_model.pt, so the reverse process matches training.
Runs deterministic 1-shot inference (no per-step noise) and fits each output
with triexp_double, plotting side-by-side overlays for the same indices.

Initial x_T is drawn from a fixed torch seed so both models see the same
starting random tensor for each sample — the only differences in the outputs
come from the network weights and the schedule.

Usage:
    python3 -u scripts/compare_models_pileup_qa.py \\
        --model_a /media/Disk_YIN/yunshancheng/cuore/ddpm_l1_low/best_model.pt \\
        --label_a l1_bT05e-2 \\
        --model_b /media/Disk_YIN/yunshancheng/cuore/ddpm_bT05_low/best_model.pt \\
        --label_b l2_bT0.5 \\
        --clean_dir /media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/test/clean_000.h5 \\
        --noise_dir /media/Disk_YIN/yunshancheng/cuore/noise/test/noise_000.h5 \\
        --output plots/pileup_fit_qa/compare_models.png \\
        --n_pileup 30
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
from src.basics.fit import (
    fit_pulse, triexp_double,
    _TAU_R_MIN, _TAU_R_MAX, _TAU_D_MIN, _TAU_D_MAX,
)


def rail_hit(fr, tol_frac=0.01):
    if not fr.success:
        return True, "curve_fit failed"
    tol_r = tol_frac * (_TAU_R_MAX - _TAU_R_MIN)
    tol_d = tol_frac * (_TAU_D_MAX - _TAU_D_MIN)
    for i, (tr, td) in enumerate(zip(fr.tau_r, fr.tau_d)):
        if tr - _TAU_R_MIN < tol_r:
            return True, f"P{i+1} tau_r at lower bound"
        if _TAU_R_MAX - tr < tol_r:
            return True, f"P{i+1} tau_r at upper bound"
        if td - _TAU_D_MIN < tol_d:
            return True, f"P{i+1} tau_d at lower bound"
        if _TAU_D_MAX - td < tol_d:
            return True, f"P{i+1} tau_d at upper bound"
    return False, ""


def load_model_with_schedule(model_path, T, device):
    """Load best_model.pt + read sibling config.json for schedule params."""
    cfg_path = os.path.join(os.path.dirname(model_path), 'config.json')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    beta_1 = cfg.get('beta_1', 1e-4)
    beta_T = cfg.get('beta_T', 0.05)
    cond_mode = cfg.get('cond_mode', 'step')
    cond_scale = cfg.get('cond_scale', 1000.0)
    print(f"[{model_path}]")
    print(f"  beta_1={beta_1}, beta_T={beta_T}, cond_mode={cond_mode}, "
          f"cond_scale={cond_scale}, loss={cfg.get('loss', '?')}")

    schedule = DiffusionSchedule(T=T, beta_1=beta_1, beta_T=beta_T).to(device)
    model = UNet1D(cond_mode=cond_mode, cond_scale=cond_scale).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=cond_mode)
    return diffusion, cfg


def run_single_shot(diffusion, x_noisy_dev, sample_seed):
    """Deterministic 1-shot: fix torch seed so x_T is identical across models."""
    g = torch.Generator(device=x_noisy_dev.device)
    g.manual_seed(sample_seed)
    # Mirror diffusion.sample but with controlled initial noise
    B, C, L = x_noisy_dev.shape
    x_t = torch.randn(B, 1, L, device=x_noisy_dev.device, generator=g)

    with torch.no_grad():
        for i in reversed(range(diffusion.T)):
            if diffusion.cond_mode == 'sqrt_ab':
                cond = diffusion.schedule.sqrt_alpha_bar[i].expand(B)
            else:
                cond = torch.full((B,), i + 1, device=x_noisy_dev.device,
                                  dtype=torch.long)
            eps_pred = diffusion._predict_noise(x_t, x_noisy_dev, cond, None)
            beta_t = diffusion.schedule.beta[i]
            coeff = beta_t / diffusion.schedule.sqrt_one_minus_alpha_bar[i]
            mean = (1.0 / diffusion.schedule.sqrt_alpha[i]) * (x_t - coeff * eps_pred)
            x_t = mean  # no_noise (deterministic)
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
    p.add_argument('--n_pileup', type=int, default=30)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--seed', type=int, default=2026,
                   help='Seed for selecting pileup indices (matches debug script)')
    p.add_argument('--init_seed', type=int, default=12345,
                   help='Base seed for initial x_T; both models use the same seed per sample')
    p.add_argument('--highlight', type=int, nargs='*', default=[4517, 7579],
                   help='Sample indices to mark as user-flagged')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    diff_a, cfg_a = load_model_with_schedule(args.model_a, args.T, device)
    diff_b, cfg_b = load_model_with_schedule(args.model_b, args.T, device)

    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    with h5py.File(args.clean_dir, 'r') as f:
        is_pileup_all = f['is_pileup'][:]

    pool = np.where(is_pileup_all)[0]
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(pool, size=min(args.n_pileup, len(pool)), replace=False)
    print(f"Selected {len(indices)} pileup samples from pool of {len(pool)} (seed={args.seed})")

    entries = []
    for k, idx in enumerate(indices):
        x_clean, x_noisy, scale = dataset[int(idx)]
        scale = scale.item()
        x_noisy_dev = x_noisy.unsqueeze(0).to(device)
        sample_seed = args.init_seed + int(idx)

        x_a = run_single_shot(diff_a, x_noisy_dev, sample_seed) * scale
        x_b = run_single_shot(diff_b, x_noisy_dev, sample_seed) * scale
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale

        # Residuals against clean — exposes jitter/dips numerically
        res_a = x_a - x_clean_np
        res_b = x_b - x_clean_np
        rms_a = float(np.sqrt(np.mean(res_a ** 2)))
        rms_b = float(np.sqrt(np.mean(res_b ** 2)))
        # Peak-region (around the two onsets) residual: 1.4-3.5 s (samples 1400..3500)
        # captures rising-edge jitter and the second-peak dip you noted at idx 4517
        peak_slice = slice(1400, 3500)
        rms_a_peak = float(np.sqrt(np.mean(res_a[peak_slice] ** 2)))
        rms_b_peak = float(np.sqrt(np.mean(res_b[peak_slice] ** 2)))

        fits = {
            'clean': fit_pulse(x_clean_np, 2),
            'noisy': fit_pulse(x_noisy_np, 2),
            'a':     fit_pulse(x_a, 2),
            'b':     fit_pulse(x_b, 2),
        }
        bad = {src: rail_hit(fits[src]) for src in ['noisy', 'a', 'b']}
        entries.append(dict(
            idx=int(idx), scale=scale, fits=fits,
            signals={'clean': x_clean_np, 'noisy': x_noisy_np, 'a': x_a, 'b': x_b},
            residuals={'a': res_a, 'b': res_b},
            rms={'a': rms_a, 'b': rms_b, 'a_peak': rms_a_peak, 'b_peak': rms_b_peak},
            bad=bad,
        ))
        flag_a = bad['a'][1] or 'ok'
        flag_b = bad['b'][1] or 'ok'
        marker = '  <<' if int(idx) in args.highlight else ''
        print(f"  [{k+1:2d}/{len(indices)}] idx={int(idx):5d}  "
              f"RMS {args.label_a}={rms_a*1e3:5.2f} mV ({rms_a_peak*1e3:5.2f} peak)  "
              f"{args.label_b}={rms_b*1e3:5.2f} mV ({rms_b_peak*1e3:5.2f} peak)  "
              f"fitA={flag_a:25s} fitB={flag_b:25s}{marker}")

    # Plot: rows = samples, cols = clean, noisy, model_a, model_b, residual
    n_rows = len(entries)
    fig, axes = plt.subplots(n_rows, 5, figsize=(25, 2.6 * n_rows), squeeze=False)
    col_keys = ['clean', 'noisy', 'a', 'b']
    col_names = {'clean': 'Clean', 'noisy': 'Noisy',
                 'a': args.label_a, 'b': args.label_b}
    fs = 1000.0

    for row, e in enumerate(entries):
        t = np.arange(len(e['signals']['clean'])) / fs
        is_highlight = e['idx'] in args.highlight
        for col, src in enumerate(col_keys):
            ax = axes[row, col]
            sig = e['signals'][src]
            fr = e['fits'][src]
            ax.plot(t, sig * 1e3, 'k-', lw=0.5, alpha=0.7, label='data')
            fit_curve = triexp_double(t, *fr.params)
            is_bad, why = (False, '')
            if src in e['bad']:
                is_bad, why = e['bad'][src]
            fit_color = 'red' if is_bad else 'tab:blue'
            ax.plot(t, fit_curve * 1e3, color=fit_color, lw=0.9, label='fit')
            status = f'X {why}' if is_bad else 'ok'
            title_color = 'red' if is_bad else 'black'
            ax.set_title(f"{col_names[src]}  chi2/ndf={fr.chi2_per_ndf:.2e}\n{status}",
                         fontsize=8, color=title_color)
            ax.set_xlabel('Time [s]', fontsize=7)
            ax.set_ylabel('mV', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            if is_highlight:
                for spine in ax.spines.values():
                    spine.set_color('orange')
                    spine.set_linewidth(2.0)
            if col == 0:
                tag = f"idx={e['idx']}"
                if is_highlight:
                    tag += '\n(flagged)'
                ax.text(-0.25, 0.5, tag,
                        transform=ax.transAxes, fontsize=9,
                        rotation=90, va='center', ha='center',
                        fontweight='bold',
                        color='orange' if is_highlight else 'black')

        # Residual column: (denoised - clean) for both models overlaid
        ax = axes[row, 4]
        ax.axhline(0, color='black', lw=0.4)
        ax.plot(t, e['residuals']['a'] * 1e3, color='tab:blue', lw=0.5,
                alpha=0.85, label=args.label_a)
        ax.plot(t, e['residuals']['b'] * 1e3, color='tab:red', lw=0.5,
                alpha=0.85, label=args.label_b)
        ax.set_title(
            f"residual = denoised - clean\n"
            f"RMS  A={e['rms']['a']*1e3:.2f}  B={e['rms']['b']*1e3:.2f} mV  "
            f"(peak A={e['rms']['a_peak']*1e3:.2f}  B={e['rms']['b_peak']*1e3:.2f})",
            fontsize=8)
        ax.set_xlabel('Time [s]', fontsize=7)
        ax.set_ylabel('mV', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)
        if is_highlight:
            for spine in ax.spines.values():
                spine.set_color('orange')
                spine.set_linewidth(2.0)

    # Aggregate residual stats
    rms_a_all = np.array([e['rms']['a'] for e in entries])
    rms_b_all = np.array([e['rms']['b'] for e in entries])
    rms_a_pk = np.array([e['rms']['a_peak'] for e in entries])
    rms_b_pk = np.array([e['rms']['b_peak'] for e in entries])
    print(f"\nAggregate over {len(entries)} pileup samples:")
    print(f"  full-window RMS residual  "
          f"{args.label_a}: median={np.median(rms_a_all)*1e3:.3f} mV, "
          f"mean={rms_a_all.mean()*1e3:.3f} mV")
    print(f"                            "
          f"{args.label_b}: median={np.median(rms_b_all)*1e3:.3f} mV, "
          f"mean={rms_b_all.mean()*1e3:.3f} mV")
    print(f"  peak-region RMS (1.4-3.5s) "
          f"{args.label_a}: median={np.median(rms_a_pk)*1e3:.3f} mV, "
          f"mean={rms_a_pk.mean()*1e3:.3f} mV")
    print(f"                            "
          f"{args.label_b}: median={np.median(rms_b_pk)*1e3:.3f} mV, "
          f"mean={rms_b_pk.mean()*1e3:.3f} mV")
    n_b_worse = int(np.sum(rms_b_all > rms_a_all))
    print(f"  windows where {args.label_b} has higher RMS than {args.label_a}: "
          f"{n_b_worse}/{len(entries)}")

    fig.suptitle(
        f"Same-init deterministic comparison: {args.label_a} vs {args.label_b}  "
        f"(n={len(entries)} pileup, init_seed={args.init_seed})",
        fontsize=12)
    plt.tight_layout()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
