"""
Warm-start DDIM inference experiment (added 2026-09-10).

Compares cold-start ddim_sample() (t_start = T-1, the production default)
against ddim_sample_warmstart() at several smaller t_start values, on a
subset of real eval waveforms. Evaluates:
  - PSD high-frequency floor (mean power above --hf_cutoff Hz) — the
    original motivation: does warm start reduce it?
  - Fidelity vs the clean template (SNR, MSE) — sanity check that warm
    start with too-small t_start doesn't just leave real noise un-removed
  - Wall-clock time per event — warm start with small t_start runs fewer
    reverse steps, so should also be faster

Does NOT touch any production eval/fit_results files — reads directly from
the raw eval HDF5s (waveforms_noisy + clean_template) and writes its own
output under results/analysis/warmstart/.

Usage
-----
python scripts/infer_warmstart.py \
    --model_path /mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt \
    --eval_files /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/efficiency_7p5mV/eval_efficiency_15keV.h5 \
                 /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/efficiency_1p2mV/eval_efficiency_1461keV.h5 \
    --n_events 100 \
    --t_starts 5 10 15 20 30 49 \
    --output_dir results/analysis/warmstart
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import re
import json
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.unet import UNet1D
from src.ddpm.unet_cond import UNet1DScaleCond
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.inference import compute_snr, compute_mse


def load_model(model_path, device, scale_cond=False, T=50, beta_1=1e-4, beta_T=0.5,
               cond_mode='step'):
    schedule = DiffusionSchedule(T=T, beta_1=beta_1, beta_T=beta_T).to(device)
    model_cls = UNet1DScaleCond if scale_cond else UNet1D
    model = model_cls(cond_mode=cond_mode).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=cond_mode)
    return diffusion, schedule


def choose_warmstart_t(schedule, sigma_normalized):
    """Analytical first-guess t_start: the 0-indexed step whose
    sqrt(1-alpha_bar[t]) is closest to the observed (per-window-normalized)
    noise std. Approximate (SDEdit-style matching ignores the sqrt(alpha_bar)
    shrinkage of the signal term) -- meant as a sweep starting point, not a
    precise answer; the sweep itself is what determines the real answer."""
    target = torch.sqrt(1.0 - schedule.alpha_bar).cpu().numpy()
    idx = int(np.argmin(np.abs(target - sigma_normalized)))
    return idx


def hf_floor_power(waveform, f_sample=1000.0, hf_cutoff=100.0):
    """Mean PSD power at frequencies >= hf_cutoff Hz (the 'floor' metric)."""
    n = len(waveform)
    freqs = np.fft.rfftfreq(n, d=1.0 / f_sample)
    psd = np.abs(np.fft.rfft(waveform)) ** 2 / n
    mask = freqs >= hf_cutoff
    return float(np.mean(psd[mask])) if mask.any() else float('nan')


def denoise_one_config(diffusion, x_tilde_batch, t_start, steps, eta, device):
    """x_tilde_batch: (B, L) numpy, ALREADY per-window normalized to [-1,1]."""
    x = torch.from_numpy(x_tilde_batch).unsqueeze(1).to(device)
    t0 = time.time()
    if t_start >= diffusion.T - 1:
        out = diffusion.ddim_sample(x, eta=eta)
    else:
        out = diffusion.ddim_sample_warmstart(x, t_start=t_start, steps=steps, eta=eta)
    elapsed = time.time() - t0
    return out.squeeze(1).cpu().numpy(), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--eval_files', nargs='+', required=True)
    ap.add_argument('--n_events', type=int, default=100)
    ap.add_argument('--t_starts', type=int, nargs='+', default=[5, 10, 15, 20, 30, 49])
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--hf_cutoff', type=float, default=100.0)
    ap.add_argument('--f_sample', type=float, default=1000.0)
    ap.add_argument('--output_dir', default='results/analysis/warmstart')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')
    diffusion, schedule = load_model(args.model_path, device)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for eval_path in args.eval_files:
        m = re.search(r'(\d+)keV', eval_path)
        energy_kev = float(m.group(1)) if m else float('nan')
        noise_tag = '1p2mV' if '1p2mV' in eval_path else ('7p5mV' if '7p5mV' in eval_path else '?')

        with h5py.File(eval_path, 'r') as f:
            clean = f['clean_template'][:]
            noisy = f['waveforms_noisy'][:args.n_events].astype(np.float32)
            noise_rms = float(np.median(f['noise_rms'][:args.n_events]))

        # Per-window normalization, same convention as run_batch_inference.py
        scale_vals = np.abs(noisy).max(axis=1, keepdims=True)
        scale_vals = np.where(scale_vals == 0, 1.0, scale_vals)
        noisy_norm = (noisy / scale_vals).astype(np.float32)

        sigma_normalized_typical = noise_rms / np.median(scale_vals)
        t_guess = choose_warmstart_t(schedule, sigma_normalized_typical)
        print(f'\n=== {os.path.basename(eval_path)}  (E={energy_kev} keV, {noise_tag}) ===')
        print(f'  noise_rms={noise_rms:.5f} V, median scale={np.median(scale_vals):.5f} V, '
              f'sigma_normalized~{sigma_normalized_typical:.4f} -> analytical t_start guess = {t_guess}')

        t_start_list = sorted(set(args.t_starts + [t_guess, schedule.T - 1]))

        for t_start in t_start_list:
            is_coldstart = (t_start >= schedule.T - 1)
            steps = schedule.T if is_coldstart else (t_start + 1)
            out_norm, elapsed = denoise_one_config(
                diffusion, noisy_norm, t_start if not is_coldstart else schedule.T - 1,
                steps, args.eta, device)
            out = out_norm * scale_vals  # denormalize back to volts

            hf_floors, snrs, mses = [], [], []
            for i in range(out.shape[0]):
                hf_floors.append(hf_floor_power(out[i], args.f_sample, args.hf_cutoff))
                snrs.append(compute_snr(clean, out[i]))
                mses.append(compute_mse(clean, out[i]))

            row = dict(
                eval_file=os.path.basename(eval_path), energy_kev=energy_kev,
                noise_tag=noise_tag, t_start=t_start, is_coldstart=is_coldstart,
                steps=steps, n_events=out.shape[0],
                hf_floor_mean=float(np.mean(hf_floors)),
                hf_floor_median=float(np.median(hf_floors)),
                snr_db_mean=float(np.mean(snrs)),
                mse_mean=float(np.mean(mses)),
                wall_time_s=elapsed, time_per_event_ms=1000.0 * elapsed / out.shape[0],
            )
            all_rows.append(row)
            tag = 'COLD' if is_coldstart else f't_start={t_start:2d}'
            print(f'  [{tag:12s}] steps={steps:3d}  HF_floor={row["hf_floor_mean"]:.3e}  '
                  f'SNR={row["snr_db_mean"]:6.2f} dB  MSE={row["mse_mean"]:.3e}  '
                  f'{row["time_per_event_ms"]:.1f} ms/event')

    # ── save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'warmstart_sweep.csv')
    keys = list(all_rows[0].keys())
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in all_rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    print(f'\nSaved -> {csv_path}')

    # ── comparison plot ─────────────────────────────────────────────────────
    configs = sorted(set((r['eval_file'], r['energy_kev'], r['noise_tag']) for r in all_rows))
    fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4.2 * len(configs)), squeeze=False)
    for row_i, (ef, E, nt) in enumerate(configs):
        sub = [r for r in all_rows if r['eval_file'] == ef]
        sub.sort(key=lambda r: r['t_start'])
        ts = [r['t_start'] for r in sub]
        cold_idx = [i for i, r in enumerate(sub) if r['is_coldstart']][0]

        ax = axes[row_i][0]
        ax.plot(ts, [r['hf_floor_mean'] for r in sub], 'o-', color='tomato')
        ax.axhline(sub[cold_idx]['hf_floor_mean'], color='gray', ls='--', label='cold-start')
        ax.set_xlabel('t_start'); ax.set_ylabel(f'HF floor (>{args.hf_cutoff:.0f}Hz) power')
        ax.set_title(f'{E:.0f} keV, {nt} — HF floor'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row_i][1]
        ax.plot(ts, [r['snr_db_mean'] for r in sub], 'o-', color='steelblue')
        ax.axhline(sub[cold_idx]['snr_db_mean'], color='gray', ls='--', label='cold-start')
        ax.set_xlabel('t_start'); ax.set_ylabel('SNR (dB) vs clean')
        ax.set_title(f'{E:.0f} keV, {nt} — fidelity'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row_i][2]
        ax.plot(ts, [r['time_per_event_ms'] for r in sub], 'o-', color='seagreen')
        ax.axhline(sub[cold_idx]['time_per_event_ms'], color='gray', ls='--', label='cold-start')
        ax.set_xlabel('t_start'); ax.set_ylabel('ms / event')
        ax.set_title(f'{E:.0f} keV, {nt} — speed'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, 'warmstart_sweep.png')
    fig.savefig(plot_path, dpi=140)
    print(f'Saved -> {plot_path}')


if __name__ == '__main__':
    main()
