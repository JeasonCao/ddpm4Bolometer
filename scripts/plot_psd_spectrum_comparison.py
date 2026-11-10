"""
2026-09-16: actual PSD (power vs frequency) comparison for the 1.382mV 4-way
method comparison -- complements plot_1p382mV_method_comparison.py's HF-floor
plot (which collapses the whole spectrum to one scalar per energy). This one
shows the spectral SHAPE: where in frequency space each method's noise floor
differs, and how the pulse signal itself sits above it.

Picks a handful of representative energies, averages the Welch PSD over a
random subsample of events per energy (a single event's PSD is too noisy to
read), and overlays clean / noisy / all 4 denoised methods on log-log axes.

Usage
-----
python -u scripts/plot_psd_spectrum_comparison.py
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

DEFAULT_EVAL_DIR = ('/mnt/d/storage/research/DDPM4bolometer/'
                     'eval_data_holdout_denoised/efficiency_1p382mV')
DEFAULT_ENERGIES = (15.0, 511.0, 2615.0)

_METHOD_FIELD = {
    'cold':       'waveforms_denoised_cold',
    'cold5shot':  'waveforms_denoised_cold5shot',
    'warm5shot':  'waveforms_denoised_warm5shot',
    'eta1_1shot': 'waveforms_denoised_eta1_1shot',
}
_METHOD_LABEL = {
    'cold':       'Cold-start (η=0, 1-shot) — prod.',
    'cold5shot':  'Cold-start (η=1, 5-shot avg)',
    'warm5shot':  'Warm-start (η=1, 5-shot avg)',
    'eta1_1shot': 'η=1, 1-shot',
}
_METHOD_COLOR = {
    'cold':       '#4e79a7',
    'cold5shot':  '#59a14f',
    'warm5shot':  '#e15759',
    'eta1_1shot': '#f28e2b',
}


def _mean_psd(waveforms, fs, nperseg):
    """waveforms: (n, L). Returns (freqs, mean_psd) -- averaged in POWER
    (linear) space across events, not in dB, so it stays a true mean floor."""
    psds = []
    for w in waveforms:
        f_psd, p_psd = welch(w, fs=fs, nperseg=nperseg)
        psds.append(p_psd)
    return f_psd, np.mean(psds, axis=0)


def plot_one_energy(eval_dir, energy_kev, n_sample, output_dir, hf_cutoff=100.0):
    path = os.path.join(eval_dir, f'eval_efficiency_{int(round(energy_kev))}keV.h5')
    if not os.path.exists(path):
        print(f'WARNING: not found: {path}'); return

    with h5py.File(path, 'r') as f:
        fs = float(f.attrs.get('f_sample', 1000.0))
        n = f['waveforms_noisy'].shape[0]
        idx = np.sort(np.random.RandomState(0).choice(n, min(n_sample, n), replace=False))
        clean = f['clean_template'][:]
        noisy = f['waveforms_noisy'][idx]
        denoised = {}
        for m, field in _METHOD_FIELD.items():
            if field in f:
                denoised[m] = f[field][idx]
        L = clean.shape[0]

    nperseg = min(2048, L)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    f_clean, p_clean = welch(clean, fs=fs, nperseg=nperseg)
    ax.plot(f_clean, p_clean, color='black', lw=1.3, ls='--', alpha=0.8,
            label='Clean template (noiseless)')

    f_noisy, p_noisy = _mean_psd(noisy, fs, nperseg)
    ax.plot(f_noisy, p_noisy, color='#9aa7b0', lw=1.3, alpha=0.9,
            label='Before denoising (noisy)')

    for m in ('cold', 'cold5shot', 'warm5shot', 'eta1_1shot'):
        if m not in denoised:
            continue
        f_m, p_m = _mean_psd(denoised[m], fs, nperseg)
        ax.plot(f_m, p_m, color=_METHOD_COLOR[m], lw=1.5, label=_METHOD_LABEL[m])

    ax.axvline(hf_cutoff, color='gray', ls=':', lw=0.9, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(f_clean[1], fs / 2)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('PSD (V² / Hz)', fontsize=10)
    ax.set_title(
        f'{energy_kev:.0f} keV — PSD, mean of {min(n_sample, n)} events '
        f'(1.382 mV, dashed line at {hf_cutoff:.0f} Hz = the HF-floor cutoff)',
        fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'psd_spectrum_{int(round(energy_kev))}keV_1p382mV.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', default=DEFAULT_EVAL_DIR)
    parser.add_argument('--energies', type=float, nargs='+', default=list(DEFAULT_ENERGIES))
    parser.add_argument('--n_sample', type=int, default=200)
    parser.add_argument('--output_dir', default=os.path.join(
        _RESULTS_DIR, 'plots', 'method_comparison_1p382mV'))
    parser.add_argument('--hf_cutoff', type=float, default=100.0)
    args = parser.parse_args()

    for E in args.energies:
        plot_one_energy(args.eval_dir, E, args.n_sample, args.output_dir, args.hf_cutoff)

    print('\nAll done.')


if __name__ == '__main__':
    main()
