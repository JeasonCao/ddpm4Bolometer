"""Paper figure: denoising-algorithm examples.

Reads denoised waveforms + paired clean/noisy/metadata from a single
denoised-test HDF5 produced by ``scripts/denoise_test_set.py`` — no GPU
or model load needed.

4 rows x 2 columns:
  - left  : time domain, noisy + 1-shot denoised  (mV)
  - right : PSD, noisy + clean + noise + 1-shot denoised + denoised-baseline
            (0–BASELINE_T_END s of the denoised waveform).
            Units mV^2/Hz, log-log.

The denoised-baseline trace shows the floor of the residual after
denoising, isolated to the pre-pulse window so the pulse itself doesn't
dominate the PSD.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


DENOISED_H5 = '/media/Disk_YIN/yunshancheng/cuore/ddpm_v3_low/denoised_test_000.h5'

# Pre-pulse baseline window (pulse onset fixed at 1.5 s; use the leading
# 0-1.5 s slice to measure the denoised baseline residual PSD).
BASELINE_T_END = 1.5  # seconds


def pick_index(is_pu, peak_amps_mV, pileup, offset, min_amp_mV=None,
               max_amp_mV=None):
    """Pick the offset-th index matching (pileup, optional amp window)."""
    mask = (is_pu == pileup)
    if min_amp_mV is not None:
        mask &= (peak_amps_mV >= min_amp_mV)
    if max_amp_mV is not None:
        mask &= (peak_amps_mV < max_amp_mV)
    pool = np.where(mask)[0]
    if len(pool) <= offset:
        raise ValueError(
            f'not enough candidates (pileup={pileup}, '
            f'amp=[{min_amp_mV}, {max_amp_mV}] mV): '
            f'have {len(pool)}, need offset {offset}'
        )
    return int(pool[offset])


def compute_psd(x, fs, nperseg_cap=2048):
    nperseg = min(nperseg_cap, len(x))
    f, p = welch(x, fs=fs, nperseg=nperseg, detrend='constant')
    return f[1:], p[1:]  # drop DC for log-log


def plot_row(ax_time, ax_psd, clean_mV, noise_mV, den_mV, fs, title,
             show_legend=False):
    t = np.arange(clean_mV.size) / fs
    noisy_mV = clean_mV + noise_mV

    ax_time.plot(t, noisy_mV, color='tab:red', lw=0.4, label='observation',
                 alpha=0.8)
    ax_time.plot(t, den_mV, color='tab:purple', lw=1.0, label='denoised')
    ax_time.set_xlabel('time (s)')
    ax_time.set_ylabel('amplitude (mV)')
    ax_time.set_title(title)
    if show_legend:
        ax_time.legend(loc='upper right', fontsize=8, frameon=False)

    f_c, p_c = compute_psd(clean_mV, fs)
    f_n, p_n = compute_psd(noise_mV, fs)
    f_o, p_o = compute_psd(noisy_mV, fs)
    f_d, p_d = compute_psd(den_mV, fs)
    # Pre-pulse window: denoised baseline residual (0 to BASELINE_T_END s).
    n_base = int(BASELINE_T_END * fs)
    den_base_mV = den_mV[:n_base]
    f_db, p_db = compute_psd(den_base_mV, fs, nperseg_cap=512)
    ax_psd.loglog(f_o, p_o, color='tab:red', lw=1.3, label='observation',
                  alpha=0.8)
    ax_psd.loglog(f_n, p_n, color='tab:gray', lw=0.8, label='noise',
                  alpha=0.8)
    ax_psd.loglog(f_c, p_c, color='tab:blue', lw=0.8, label='clean')
    ax_psd.loglog(f_d, p_d, color='tab:purple', lw=0.8, label='denoised')
    ax_psd.loglog(f_db, p_db, color='tab:green', lw=0.9,
                  linestyle='--',
                  label=f'denoised baseline (0–{BASELINE_T_END:g} s)')
    ax_psd.set_xlabel('frequency (Hz)')
    ax_psd.set_ylabel('PSD (mV$^2$/Hz)')
    if show_legend:
        ax_psd.legend(loc='lower left', fontsize=7, frameon=False)
    ax_psd.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoised_h5', default=DENOISED_H5,
                   help='Path to denoised_test_000.h5')
    p.add_argument('--out',
                   default='/home/yunshan/cuore/plots/paper/denoising_examples.pdf')
    args = p.parse_args()

    with h5py.File(args.denoised_h5, 'r') as f:
        fs = float(f.attrs['f_sample'])
        is_pu = f['is_pileup'][:]
        wfs_clean = f['waveforms_clean']
        wfs_noisy = f['waveforms_noisy']
        wfs_den = f['waveforms']

        # Peak-minus-baseline amplitude in mV for selection (per window).
        # Use waveforms_clean to characterize amplitude — the clean pulse
        # peak is what we want to rank by.
        n = wfs_clean.shape[0]
        peak_amp_mV = np.zeros(n, dtype=np.float64)
        bl = wfs_clean[:, :200].astype(np.float64)
        bl_med = np.median(bl, axis=1)
        pk = wfs_clean[:, :].max(axis=1)
        peak_amp_mV[:] = (pk - bl_med) * 1000.0

        # Layout: single low-amp, single mid-amp, pileup low-amp, pileup mid.
        # Row 4 cap: keep moderate-amplitude pileup so the PSD isn't
        # dominated by spectral-leakage wiggles from a very large transient
        # (welch with nperseg=2048 over a 10000-sample window only gets
        # ~9 averaging segments; sharper/larger pulses → more visible
        # periodogram variance riding on top of the noise floor).
        rows = [
            (0, 0, 0.0,    60.0, 'example 1: single, low amp'),
            (0, 0, 60.0,   None, 'example 2: single, higher amp'),
            (1, 0, 0.0,    60.0, 'example 3: pileup, low amp'),
            (1, 0, 100.0, 130.0, 'example 4: pileup, higher amp'),
        ]
        idxs = []
        for pu, offset, amin, amax, _title in rows:
            idxs.append(pick_index(is_pu, peak_amp_mV, pu, offset,
                                   min_amp_mV=amin, max_amp_mV=amax))

        fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.2 * len(rows)))
        for row_i, ((pu, off, amin, amax, title), idx) in enumerate(zip(rows, idxs)):
            clean_V = wfs_clean[idx].astype(np.float64)
            noisy_V = wfs_noisy[idx].astype(np.float64)
            den_V = wfs_den[idx].astype(np.float64)
            noise_V = noisy_V - clean_V
            clean_mV = clean_V * 1000.0
            noise_mV = noise_V * 1000.0
            den_mV = den_V * 1000.0
            print(f'row {row_i}: idx={idx} pileup={bool(pu)} '
                  f'clean_amp={peak_amp_mV[idx]:.1f} mV — {title}')
            plot_row(axes[row_i, 0], axes[row_i, 1],
                     clean_mV, noise_mV, den_mV, fs, title,
                     show_legend=(row_i == 0))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    # Also save PNG sibling for easy preview.
    png_out = os.path.splitext(args.out)[0] + '.png'
    fig.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')
    print(f'wrote {png_out}')


if __name__ == '__main__':
    main()
