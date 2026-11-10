"""OF vs DDPM: 15 keV energy spectrum on the same events, both noise levels (one panel each).
OF only returns an amplitude, so only the energy resolution is compared (no waveform comparison).

Events are the old 17-point eval set (eval_data_holdout), where DDPM outputs already exist
(1.382 mV: no_noise sampler; 7.5 mV: zero-init, the only one run there). The OF is the one in the
old Octopus (octoreflow_gio) OptimumFilter module, reproduced in numpy and checked against Octopus
on run 999996:
  - filter conj(S)/D with D = mean |X| of the noise windows (Octopus stores |X|, not |X|^2)
  - S = FFT of the clean 2615 keV template (the average pulse), noise windows = 2615 keV file
    waveforms minus its clean template (independent of the 15 keV events)
  - baseline = mean of the first 1.4 s subtracted, no windowing
  - delay scan over +-10 samples in 0.5-sample steps, keep the minimum chi2 (= largest amplitude)
Energies: DDPM from the tri-exponential fit amplitude, OF from amplitude x 2615 keV template
height; both converted with the same k = 780.32 keV/V.

Usage: python scripts/of_compare/plot_of_vs_ddpm_15keV.py
"""
import os, sys, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_gauss_fit import fit_gaussian

D = '/mnt/d/storage/research/DDPM4bolometer'
K = 780.32
FS = 1000.0
LEVELS = [
    ('1p382mV', '1.382 mV', f'{D}/nonoise_out/efficiency_1p382mV', 'waveforms_denoised_nonoise',
     f'{PROJ}/results/fit_results/1p382mV_nonoise', 'DDPM (no_noise)'),
    ('7p5mV', '7.5 mV', f'{D}/zeroinit_out/efficiency_7p5mV', 'waveforms_denoised_zeroinit',
     f'{PROJ}/results/fit_results/7p5mV_zeroinit', 'DDPM (zero-init)'),
]
C_DDPM, C_OF = '#c2603c', '#2f6f9f'


def bsub(x):
    return x - x[..., :1400].mean(-1, keepdims=True)


def of_amplitudes(X, S, Dn, shifts):
    """Octopus OptimumFilter: amplitude at the delay with minimum chi2. X: (n, nf) event FFTs."""
    H = np.conj(S) / Dn
    norm = np.real((H * S).sum())
    k = np.arange(S.size)
    P = np.exp(-2j * np.pi * np.outer(shifts, k) / (2 * (S.size - 1)))   # shift of the template by `shifts` samples
    A = np.real(X @ (H[None, :] * P).T) / norm                            # (n, n_shifts)
    # chi2(shift) = const - amp^2 * norm for this estimator, so min chi2 = max amplitude^2
    best = np.argmax(A ** 2, axis=1)
    return A[np.arange(len(A)), best], shifts[best]


def run_level(tag, den_dir, den_field):
    ev = f'{D}/eval_data_holdout/efficiency_{tag}'
    with h5py.File(f'{ev}/eval_efficiency_2615keV.h5') as f:
        T26 = f['clean_template'][:]
        noise = bsub(f['waveforms_noisy'][:] - T26)
    S = np.fft.rfft(T26)
    Dn = np.abs(np.fft.rfft(noise)).mean(0); del noise
    with h5py.File(f'{ev}/eval_efficiency_15keV.h5') as f:
        noisy = f['waveforms_noisy'][:]
    X = np.fft.rfft(bsub(noisy))
    shifts = np.arange(-10, 10.01, 0.5)
    a, dt = of_amplitudes(X, S, Dn, shifts)
    return dict(T26=T26, a=a, dt=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=f'{PROJ}/results/plots/OFcompare/of_vs_ddpm_15keV.png')
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, (tag, label, den_dir, den_field, fit_dir, ddpm_lab) in enumerate(LEVELS):
        r = run_level(tag, den_dir, den_field)
        A26 = r['T26'].max()
        E_of = r['a'] * A26 * K
        with h5py.File(f'{fit_dir}/fit_efficiency_15keV.h5') as f:
            amp = f['amp_denoised'][:]; ok = f['success_denoised'][:].astype(bool) & np.isfinite(amp)
        E_dd = amp[ok] * K
        g_dd, g_of = fit_gaussian(E_dd), fit_gaussian(E_of)
        print(f'{label}: DDPM sigma {g_dd["sigma"]:.4f} keV (std {E_dd.std():.4f}), mean {E_dd.mean():.3f} | '
              f'OF sigma {g_of["sigma"]:.4f} keV (std {E_of.std():.4f}), mean {E_of.mean():.3f}, delay {r["dt"].mean():.2f}+-{r["dt"].std():.2f} samples')

        # (4) energy spectrum
        ax = axes[col]
        allE = np.concatenate([E_dd, E_of])
        bins = np.linspace(np.percentile(allE, 0.2), np.percentile(allE, 99.8), 120)
        ax.hist(E_dd, bins=bins, histtype='step', color=C_DDPM, lw=1.5,
                label=f'{ddpm_lab}  FWHM = {2.355 * g_dd["sigma"]:.3f} keV')
        ax.hist(E_of, bins=bins, histtype='step', color=C_OF, lw=1.5,
                label=f'OF  FWHM = {2.355 * g_of["sigma"]:.3f} keV')
        ax.axvline(15, color='k', ls='--', lw=1.0, label='True energy')
        ax.set_title(f'{label} noise, 15 keV (same 5000 events)', fontsize=11)
        ax.set_xlabel('Reconstructed energy (keV)'); ax.set_ylabel('Counts'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    fig.savefig(a.output, dpi=150)
    print('Saved ->', a.output)


if __name__ == '__main__':
    main()
