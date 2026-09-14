"""
Warm-start vs cold-start denoising, time domain + PSD, for a few high-energy
(high-SNR) events (added 2026-09-10) -- reuses the plotting layout from
plot_waveform_psd_example.py's plot_row(), extended to overlay both
cold-start and warm-start denoised curves for direct comparison.

Runs inference itself (unlike plot_waveform_psd_example.py, which reads a
pre-existing 'waveforms_denoised' field) using the same load_model() /
per-window-normalization convention as scripts/infer_warmstart.py.

Usage
-----
python scripts/plot_warmstart_comparison.py \
    --model_path /mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt \
    --eval_file /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/efficiency_7p5mV/eval_efficiency_1461keV.h5 \
    --t_start_warm 3 \
    --n_events 3 \
    --output results/analysis/warmstart/warmstart_vs_cold_1461keV_7p5mV.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import h5py
import numpy as np
import torch
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.infer_warmstart import load_model

F_SAMPLE = 1000.0


def denoise(diffusion, noisy_batch, device, t_start=None, eta=0.0):
    """noisy_batch: (B, L) raw volts. Returns (B, L) denoised, raw volts."""
    scale_vals = np.abs(noisy_batch).max(axis=1, keepdims=True)
    scale_vals = np.where(scale_vals == 0, 1.0, scale_vals)
    x = torch.from_numpy((noisy_batch / scale_vals).astype(np.float32)).unsqueeze(1).to(device)
    with torch.no_grad():
        if t_start is None or t_start >= diffusion.T - 1:
            out = diffusion.ddim_sample(x, eta=eta)
        else:
            out = diffusion.ddim_sample_warmstart(x, t_start=t_start, steps=t_start + 1, eta=eta)
    return out.squeeze(1).cpu().numpy() * scale_vals


def plot_row_compare(ax_t, ax_p, energy_kev, clean, noisy, denoised_cold, denoised_warm,
                     snr_db, t_start_warm, fs=F_SAMPLE):
    t = np.arange(len(clean)) / fs

    # ── time domain ──────────────────────────────────────────────────────────
    ax_t.plot(t, noisy,          color='steelblue', lw=0.6, alpha=0.55, label='Noisy')
    ax_t.plot(t, denoised_cold,  color='gray',       lw=1.1, alpha=0.9, label='Denoised (cold-start)')
    ax_t.plot(t, denoised_warm,  color='tomato',     lw=1.1, alpha=0.9, label=f'Denoised (warm-start, t={t_start_warm})')
    ax_t.plot(t, clean,          color='black',      lw=1.0, ls='--', alpha=0.6, label='True (noiseless)')
    ax_t.set_xlabel('Time (s)', fontsize=9)
    ax_t.set_ylabel('Amplitude (V)', fontsize=9)
    ax_t.set_title(f'{energy_kev:.0f} keV — time domain (SNR = {snr_db:.1f} dB)', fontsize=10)
    ax_t.legend(fontsize=7.5, loc='upper right')
    ax_t.grid(alpha=0.3)

    # ── PSD ──────────────────────────────────────────────────────────────────
    noise_only = noisy - clean
    nperseg = min(2048, len(clean))
    for wf, color, label, lw in [
        (clean,          'black',     'True (noiseless)', 1.0),
        (noisy,          'steelblue', 'Noisy',             0.9),
        (denoised_cold,  'gray',      'Denoised (cold-start)', 1.1),
        (denoised_warm,  'tomato',    f'Denoised (warm-start, t={t_start_warm})', 1.1),
        (noise_only,     'lightgray', 'Injected noise',    0.7),
    ]:
        f_psd, p_psd = welch(wf, fs=fs, nperseg=nperseg)
        mask = f_psd > 0
        ax_p.loglog(f_psd[mask], p_psd[mask], color=color, lw=lw, alpha=0.85, label=label)
    ax_p.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_p.set_ylabel(r'PSD (V$^2$/Hz)', fontsize=9)
    ax_p.set_title(f'{energy_kev:.0f} keV — power spectral density', fontsize=10)
    ax_p.legend(fontsize=7, loc='lower left')
    ax_p.grid(alpha=0.3, which='both')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--eval_file', required=True)
    ap.add_argument('--t_start_warm', type=int, required=True,
                    help='Empirically-optimal warm-start t_start for this energy/noise combo')
    ap.add_argument('--n_events', type=int, default=3)
    ap.add_argument('--event_start', type=int, default=0)
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    diffusion, schedule = load_model(args.model_path, device)

    with h5py.File(args.eval_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        clean = f['clean_template'][:]
        idx = slice(args.event_start, args.event_start + args.n_events)
        noisy_batch = f['waveforms_noisy'][idx].astype(np.float32)
        snr_batch = f['snr_db'][idx]

    denoised_cold = denoise(diffusion, noisy_batch, device, t_start=None, eta=args.eta)
    denoised_warm = denoise(diffusion, noisy_batch, device, t_start=args.t_start_warm, eta=args.eta)

    n = args.n_events
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.75 * n), squeeze=False)
    for i in range(n):
        plot_row_compare(axes[i, 0], axes[i, 1], energy_kev, clean,
                         noisy_batch[i], denoised_cold[i], denoised_warm[i],
                         float(snr_batch[i]), args.t_start_warm)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
