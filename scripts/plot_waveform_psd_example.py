"""
Paper figure: qualitative denoising example, time domain + PSD.

2 rows x 2 columns:
  row 1 — low-energy example   (left: time domain, right: PSD)
  row 2 — high-energy example  (left: time domain, right: PSD)

Each time-domain panel overlays: true clean pulse, noisy waveform, denoised
waveform. Each PSD panel (via scipy.signal.welch, log-log) overlays: clean,
noisy, denoised, and the injected noise realization (noisy - clean).

Single-pulse only (no pile-up) — uses the existing efficiency eval datasets,
which are already single deterministic-template + additive-noise, no
pile-up mixture, so no filtering needed.

Usage
-----
python scripts/plot_waveform_psd_example.py \
    --eval_file_low  /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_7p5mV/eval_efficiency_15keV.h5 \
    --eval_file_high /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_7p5mV/eval_efficiency_1461keV.h5 \
    --output results/plots/reconstruction_quality/waveform_psd_example.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import h5py
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F_SAMPLE = 1000.0


def load_event(eval_file: str, event_idx: int):
    with h5py.File(eval_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        clean = f['clean_template'][:]
        noisy = f['waveforms_noisy'][event_idx]
        if 'waveforms_denoised' not in f:
            raise KeyError(
                f"{eval_file} has no 'waveforms_denoised' field — "
                f"run run_batch_inference.py on it first.")
        denoised = f['waveforms_denoised'][event_idx]
        snr_db = float(f['snr_db'][event_idx])
    return energy_kev, clean, noisy, denoised, snr_db


def plot_row(ax_t, ax_p, energy_kev, clean, noisy, denoised, snr_db, fs=F_SAMPLE):
    t = np.arange(len(clean)) / fs

    # ── time domain ──────────────────────────────────────────────────────────
    ax_t.plot(t, noisy,    color='steelblue', lw=0.7, alpha=0.7, label='Noisy')
    ax_t.plot(t, denoised, color='tomato',    lw=1.1, alpha=0.9, label='Denoised')
    ax_t.plot(t, clean,    color='black',     lw=1.0, ls='--', alpha=0.6, label='True (noiseless)')
    ax_t.set_xlabel('Time (s)', fontsize=9)
    ax_t.set_ylabel('Amplitude (V)', fontsize=9)
    ax_t.set_title(f'{energy_kev:.0f} keV — time domain (SNR = {snr_db:.1f} dB)', fontsize=10)
    ax_t.legend(fontsize=7.5, loc='upper right')
    ax_t.grid(alpha=0.3)

    # ── PSD ──────────────────────────────────────────────────────────────────
    noise_only = noisy - clean
    nperseg = min(2048, len(clean))
    for wf, color, label in [
        (clean,      'black',     'True (noiseless)'),
        (noisy,      'steelblue', 'Noisy'),
        (denoised,   'tomato',    'Denoised'),
        (noise_only, 'gray',      'Injected noise'),
    ]:
        f_psd, p_psd = welch(wf, fs=fs, nperseg=nperseg)
        # avoid log(0) on the DC bin
        mask = f_psd > 0
        ax_p.loglog(f_psd[mask], p_psd[mask], color=color, lw=1.0, alpha=0.85, label=label)
    ax_p.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_p.set_ylabel(r'PSD (V$^2$/Hz)', fontsize=9)
    ax_p.set_title(f'{energy_kev:.0f} keV — power spectral density', fontsize=10)
    ax_p.legend(fontsize=7.5, loc='lower left')
    ax_p.grid(alpha=0.3, which='both')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_file_low', required=True,
                    help='Denoised eval HDF5 for the low-energy example')
    ap.add_argument('--eval_file_high', required=True,
                    help='Denoised eval HDF5 for the high-energy example')
    ap.add_argument('--event_idx', type=int, default=0,
                    help='Event index within each file (SNR is ~constant '
                         'across events at fixed noise RMS, so any index works)')
    ap.add_argument('--output', default=os.path.join(
        _RESULTS_DIR, 'plots', 'reconstruction_quality', 'waveform_psd_example.png'))
    args = ap.parse_args()

    e_lo, clean_lo, noisy_lo, denoised_lo, snr_lo = load_event(args.eval_file_low, args.event_idx)
    e_hi, clean_hi, noisy_hi, denoised_hi, snr_hi = load_event(args.eval_file_high, args.event_idx)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    plot_row(axes[0, 0], axes[0, 1], e_lo, clean_lo, noisy_lo, denoised_lo, snr_lo)
    plot_row(axes[1, 0], axes[1, 1], e_hi, clean_hi, noisy_hi, denoised_hi, snr_hi)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
