"""
2026-09-22: PSD before/after denoising, shown in BOTH physical and normalized
units, for the zero-init production method only.

Why the normalized panel matters
--------------------------------
Inference normalizes each window by `scale = max|noisy|` (identical to
training, src/ddpm/dataset.py), runs the sampler, and multiplies the output
back by `scale`. The network's residual error is therefore a FIXED quantity in
NORMALIZED space; the multiply-back converts it into a physical error
proportional to the pulse amplitude. The two panels make that visible
directly: in physical units the post-denoising high-frequency floor rises with
energy, while in normalized units the same floors collapse onto one another.

Only `waveforms_denoised_zeroinit` is drawn -- that is the method this
production run used. (The 4-way cold/cold5shot/warm5shot/eta1_1shot overlay
lives in plot_psd_spectrum_comparison.py and is not part of the paper.)

Usage
-----
python -u scripts/plot_psd_normalized_zeroinit.py
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import sys, os
sys.path.insert(0, str(_REPO_ROOT))
_PROJECT_DIR = str(_REPO_ROOT)
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import h5py
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F_SAMPLE = 1000.0
DEFAULT_EVAL_DIR = ('/mnt/d/storage/research/DDPM4bolometer/'
                    'eval_data_holdout_denoised/efficiency_1p382mV')
DEFAULT_ENERGIES = (15.0, 239.0, 2615.0)
FIELD = 'waveforms_denoised_zeroinit'
COLORS = ('#2f6f9f', '#c2603c', '#4a8b62')


def _find(eval_dir, energy_kev):
    for p in sorted(glob.glob(os.path.join(eval_dir, 'eval_efficiency_*.h5'))):
        with h5py.File(p, 'r') as f:
            if abs(float(f.attrs['energy_kev']) - energy_kev) < 1.0:
                return p
    raise FileNotFoundError(f'no {energy_kev} keV file in {eval_dir}')


def _mean_psd(w):
    """Welch PSD averaged over events. w: (n_events, n_samples)."""
    f, p = welch(w, fs=F_SAMPLE, nperseg=min(4096, w.shape[1]), axis=-1)
    return f, p.mean(axis=0)


def load(eval_dir, energy_kev, n_sample, rng):
    path = _find(eval_dir, energy_kev)
    with h5py.File(path, 'r') as f:
        n = f['waveforms_noisy'].shape[0]
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
        noisy = f['waveforms_noisy'][idx].astype(np.float64)
        den = f[FIELD][idx].astype(np.float64)
        clean = f['clean_template'][:].astype(np.float64)

    # The exact per-window normalization used at inference time.
    scale = np.abs(noisy).max(axis=1, keepdims=True)

    out = {}
    for tag, arr in (('noisy', noisy), ('denoised', den)):
        out[f'f_{tag}'], out[f'p_{tag}'] = _mean_psd(arr)
        _, out[f'pn_{tag}'] = _mean_psd(arr / scale)
    out['f_clean'], out['p_clean'] = welch(clean, fs=F_SAMPLE, nperseg=min(4096, len(clean)))
    _, out['pn_clean'] = welch(clean / scale.mean(), fs=F_SAMPLE, nperseg=min(4096, len(clean)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', default=DEFAULT_EVAL_DIR)
    ap.add_argument('--energies', type=float, nargs='+', default=list(DEFAULT_ENERGIES))
    ap.add_argument('--n_sample', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default=os.path.join(
        _RESULTS_DIR, 'plots', 'paper_zeroinit', 'psd_normalized_zeroinit.png'))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = {E: load(args.eval_dir, E, args.n_sample, rng) for E in args.energies}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key_n, key_d, title, ylab) in zip(axes, [
            ('p_noisy', 'p_denoised', 'Physical units', r'PSD (V$^2$/Hz)'),
            ('pn_noisy', 'pn_denoised',
             r'Normalized units  ($w / \max|w_{\rm noisy}|$)', r'PSD (normalized$^2$/Hz)')]):
        for c, E in zip(COLORS, args.energies):
            d = data[E]
            ax.loglog(d['f_noisy'], d[key_n], color=c, lw=1.0, ls='--', alpha=0.55)
            ax.loglog(d['f_denoised'], d[key_d], color=c, lw=1.6,
                      label=f'{E:.0f} keV')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.3, which='both')

    # One shared explanation of the linestyle convention, stated once.
    axes[0].plot([], [], color='0.35', ls='--', lw=1.0, label='before (dashed)')
    axes[0].plot([], [], color='0.35', ls='-', lw=1.6, label='after (solid)')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[1].legend(fontsize=8, loc='lower left')

    tag = os.path.basename(args.eval_dir.rstrip('/')).replace('efficiency_', '')
    fig.suptitle('PSD before and after denoising (zero-init), '
                 f'{tag.replace("p", ".").replace("mV", "")} mV baseline noise',
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved -> {args.output}')

    # The quantitative version of what the right-hand panel shows.
    print('\nHigh-frequency floor (f >= 100 Hz), mean PSD:')
    print(f'{"E[keV]":>8} {"physical":>13} {"normalized":>13}')
    for E in args.energies:
        d = data[E]
        m = d['f_denoised'] >= 100.0
        print(f'{E:8.0f} {d["p_denoised"][m].mean():13.4e} {d["pn_denoised"][m].mean():13.4e}')


if __name__ == '__main__':
    main()
