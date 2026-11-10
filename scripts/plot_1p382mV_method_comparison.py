"""
2026-09-16: 4-way denoising-method comparison for the 1.382mV production run
(cold / cold5shot / warm5shot / eta1_1shot).

Background: 1.382mV is the noise level re-tuned so the baseline resolution
self-consistently hits CUORE's literature 2.54 keV. On top of the existing
cold-start-vs-warm-start-5shot comparison, local exploratory tests this
window found that eta=1.0 single-shot dramatically beats eta=0 (cold)
single-shot on both HF floor and FWHM, but also carries large run-to-run
(shot) noise -- large enough that its single-shot FWHM numbers are
significantly contaminated by that noise, not purely reconstruction
fidelity. This script produces the production-scale, all-17-energy
comparison of all 4 candidate methods so that decision can be made on real
numbers instead of the n=100-150 local spot-checks.

Two figures:
  1. Energy resolution (FWHM, keV and %) vs true energy -- one line per
     method, overlaid on the same axes, plus the noisy baseline. Reuses the
     fit_results HDF5s + Gaussian-fit calibration machinery from
     plot_reconstruction_quality.py (same global zero-intercept calibration
     convention per fit_dir) so numbers stay comparable to the 1.2mV/7.5mV
     figures.
  2. PSD high-frequency floor vs true energy -- computed directly from the
     denoised waveform arrays in eval_data_holdout_denoised (not from fit
     results), reusing hf_floor_power() from infer_warmstart.py.

Prerequisites
-------------
Run for each method first:
  python -u scripts/run_fitting.py --input_dir <eval_data_holdout_denoised>/efficiency_1p382mV \
      --denoised_field waveforms_denoised_<method> \
      --output_dir results/fit_results/efficiency_1p382mV_<method> --workers 11
  python -u scripts/run_gauss_fit.py --fit_dir results/fit_results/efficiency_1p382mV_<method>
for method in cold, cold5shot, warm5shot, eta1_1shot.

Usage
-----
python -u scripts/plot_1p382mV_method_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.plot_reconstruction_quality import _log_xlim, _save_fig
from scripts.run_gauss_fit import fit_gaussian
from scripts.infer_warmstart import hf_floor_power

# 2026-09-16 IMPORTANT: this script deliberately does NOT reuse
# plot_reconstruction_quality.py's _load_all()/_fit_global_calibration()
# (the ones driven by run_gauss_fit.py's precomputed gauss_*_{noisy,denoised}
# attrs), even though those exist and would be the obvious thing to reuse.
#
# Reason: run_gauss_fit.py's Gaussian fit is computed only over events that
# passed run_fitting.py's per-event tri-exp chi2/ndf gate (chi2_max=10,
# converged_denoised). That chi2 is normalized by a noise sigma estimated
# ADAPTIVELY from each waveform's own pre-trigger baseline (src/basics/fit.py,
# NOT the file's fixed noise_rms). For cold5shot/warm5shot/eta1_1shot, the
# post-denoising baseline is much quieter than plain cold-start, so the same
# absolute chi2_max=10 threshold -- fine for cold-start (~90% pass, matches
# the existing 1.2mV/7.5mV production fit_results) -- becomes far too strict:
# observed denoised-convergence collapsed to <10% (some energies <1%) for
# cold5shot, even though the amplitude estimate itself is excellent (median
# amp_dev ~0.1%, ~100% within the 20% ground-truth-deviation cut -- checked
# directly against fit_efficiency_511keV.h5 for efficiency_1p382mV_cold5shot).
# Gating the Gaussian-fit input on this collapsed, energy-dependent, and
# often sub-MIN_EVENTS(30) "converged" flag would make the resolution number
# for the cleaner methods come from a tiny, chi2-cherry-picked subsample --
# not a fair comparison, and not what "energy resolution" is supposed to mean
# in a synthetic eval where ground truth is available.
#
# Fix: below, fit_gaussian() is called directly on the FULL, ungated
# amp_noisy/amp_denoised arrays for every method (including cold, for
# consistency), and the zero-intercept calibration is replicated locally
# from those ungated Gaussian means instead of the gated file attrs. This
# only affects THIS 4-way comparison script -- the shared
# plot_reconstruction_quality.py / run_gauss_fit.py pipeline and the
# already-reported 1.2mV/7.5mV numbers are untouched (verified: those use
# plain cold-start, single-shot eta=0, whose chi2 convergence is normal,
# ~90-92%, unaffected by this issue).

CALIB_ENERGIES_KEV = (239.0, 511.0, 727.0, 1461.0, 2615.0)


def _load_all_ungated(fit_dir, method='triexp'):
    """Like plot_reconstruction_quality._load_all(), but the Gaussian fit
    (and the calibration built from it) runs on the FULL amp array, with no
    chi2-convergence gating. See the module docstring/comment above for why."""
    amp_key_n = 'amp_noisy' if method == 'triexp' else 'amp_maxbase_noisy'
    amp_key_d = 'amp_denoised' if method == 'triexp' else 'amp_maxbase_denoised'

    files = sorted(glob.glob(os.path.join(fit_dir, 'fit_efficiency_*.h5')))
    if not files:
        return None

    rows = []
    for path in files:
        with h5py.File(path, 'r') as f:
            E_true = float(f.attrs['energy_kev'])
            amp_n = f[amp_key_n][:].astype(float)
            amp_d = f[amp_key_d][:].astype(float)
        rn = fit_gaussian(amp_n)
        rd = fit_gaussian(amp_d)
        rows.append(dict(
            E_true=E_true,
            mu_n=rn['mu'], mu_err_n=rn['mu_err'],
            sigma_n=rn['sigma'], sig_err_n=rn['sigma_err'],
            mu_d=rd['mu'], mu_err_d=rd['mu_err'],
            sigma_d=rd['sigma'], sig_err_d=rd['sigma_err'],
        ))
    rows.sort(key=lambda r: r['E_true'])

    # Zero-intercept global calibration (same convention/formula as
    # src/basics/plot_utils.py:fit_global_calibration_zero_intercept),
    # sourced from this script's own ungated Gaussian means instead of the
    # gated file attrs.
    calib_rows = [r for r in rows if any(abs(r['E_true'] - E) < 1.0 for E in CALIB_ENERGIES_KEV)]
    if len(calib_rows) < 2:
        print(f'WARNING: only {len(calib_rows)} calibration points found in {fit_dir}')
        return None
    amps     = np.array([r['mu_d'] for r in calib_rows])
    raw_errs = np.array([r['mu_err_d'] for r in calib_rows])
    errs     = np.array([e if np.isfinite(e) and e > 0 else 1e6 for e in raw_errs])
    energies = np.array([r['E_true'] for r in calib_rows])
    w = 1.0 / errs**2
    k = float(np.sum(w * energies * amps) / np.sum(w * amps**2))

    def arr(key):
        return np.array([r[key] for r in rows])

    E_true = arr('E_true')
    return dict(
        k=k, E_true=E_true,
        mu_n=k * arr('mu_n'),    mu_err_n=k * arr('mu_err_n'),
        sigma_n=k * arr('sigma_n'), sig_err_n=k * arr('sig_err_n'),
        mu_d=k * arr('mu_d'),    mu_err_d=k * arr('mu_err_d'),
        sigma_d=k * arr('sigma_d'), sig_err_d=k * arr('sig_err_d'),
    )

_METHODS = ['cold', 'cold5shot', 'warm5shot', 'eta1_1shot']
_METHOD_LABEL = {
    'cold':       'Cold-start (eta=0, 1-shot)',
    'cold5shot':  'Cold-start (eta=1, 5-shot avg)',
    'warm5shot':  'Warm-start (eta=1, 5-shot avg)',
    'eta1_1shot': 'eta=1, 1-shot',
}
_METHOD_COLOR = {
    'cold':       'steelblue',
    'cold5shot':  'seagreen',
    'warm5shot':  'tomato',
    'eta1_1shot': 'darkorange',
}
_METHOD_FIELD = {
    'cold':       'waveforms_denoised_cold',
    'cold5shot':  'waveforms_denoised_cold5shot',
    'warm5shot':  'waveforms_denoised_warm5shot',
    'eta1_1shot': 'waveforms_denoised_eta1_1shot',
}

DEFAULT_FIT_BASE = os.path.join(_RESULTS_DIR, 'fit_results')
DEFAULT_EVAL_DIR = ('/mnt/d/storage/research/DDPM4bolometer/'
                     'eval_data_holdout_denoised/efficiency_1p382mV')


# ── energy resolution FWHM (keV and %) ───────────────────────────────────────

def plot_resolution_comparison(fit_base, method_key='triexp', output_dir=None, pct=False):
    output_dir = output_dir or os.path.join(_RESULTS_DIR, 'plots', 'method_comparison_1p382mV')
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    noisy_plotted = False
    E_true_ref = None
    for m in _METHODS:
        fit_dir = os.path.join(fit_base, f'efficiency_1p382mV_{m}')
        d = _load_all_ungated(fit_dir, method_key)
        if d is None:
            print(f'WARNING: no fit results in {fit_dir}, skipping {m}')
            continue
        E_true = d['E_true']
        E_true_ref = E_true
        fwhm_d = 2.355 * d['sigma_d']
        fwhm_err_d = 2.355 * d['sig_err_d']
        if pct:
            fwhm_d = fwhm_d / E_true * 100
            fwhm_err_d = fwhm_err_d / E_true * 100
        ax.errorbar(E_true, fwhm_d, yerr=fwhm_err_d, fmt='o-', ms=4, lw=1.4,
                    color=_METHOD_COLOR[m], elinewidth=0.8, capsize=2,
                    label=_METHOD_LABEL[m])
        if not noisy_plotted:
            fwhm_n = 2.355 * d['sigma_n']
            fwhm_err_n = 2.355 * d['sig_err_n']
            if pct:
                fwhm_n = fwhm_n / E_true * 100
                fwhm_err_n = fwhm_err_n / E_true * 100
            ax.errorbar(E_true, fwhm_n, yerr=fwhm_err_n, fmt='s--', ms=4, lw=1.2,
                        color='gray', elinewidth=0.8, capsize=2, alpha=0.7,
                        label='Before denoising (noisy)')
            noisy_plotted = True

    if E_true_ref is None:
        print('No data found for any method -- aborting resolution plot.')
        return

    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true_ref))
    ax.set_ylim(0, None)
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ylabel = 'Energy resolution FWHM / E  (%)' if pct else 'Energy resolution FWHM (keV)'
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(
        f'1.382 mV — 4-way denoising method comparison ({method_key})\n'
        f'FWHM = 2.355 × σ_Gauss (own per-fit_dir calibration), error bars from Gaussian fit',
        fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')
    tag = 'resolution_pct' if pct else 'resolution'
    _save_fig(fig, os.path.join(output_dir, f'{tag}_{method_key}_1p382mV_4way.png'))


# ── ground-truth amplitude-deviation efficiency ─────────────────────────────
# Fraction of events within amp_dev_max (default 20%) of the TRUE amplitude
# (known because this is a synthetic eval with ground truth). Unlike the
# tri-exp chi2/ndf "converged" gate (see module docstring), this is a direct
# ground-truth accuracy check and is NOT distorted by a cleaner method's
# lower residual noise, so it's a fair cross-method efficiency comparison.

def plot_amp_efficiency_comparison(fit_base, output_dir=None):
    output_dir = output_dir or os.path.join(_RESULTS_DIR, 'plots', 'method_comparison_1p382mV')
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    noisy_plotted = False
    E_true_ref = None
    for m in _METHODS:
        fit_dir = os.path.join(fit_base, f'efficiency_1p382mV_{m}')
        files = sorted(glob.glob(os.path.join(fit_dir, 'fit_efficiency_*.h5')))
        if not files:
            print(f'WARNING: no fit results in {fit_dir}, skipping {m}')
            continue
        rows = []
        for path in files:
            with h5py.File(path, 'r') as f:
                E_true = float(f.attrs['energy_kev'])
                dev_d = f['amp_dev_denoised'][:]
                dev_n = f['amp_dev_noisy'][:]
            rows.append(dict(E_true=E_true,
                             eff_d=float(np.mean(dev_d <= 0.20)),
                             eff_n=float(np.mean(dev_n <= 0.20))))
        rows.sort(key=lambda r: r['E_true'])
        E_true = np.array([r['E_true'] for r in rows])
        E_true_ref = E_true
        ax.plot(E_true, [r['eff_d'] for r in rows], 'o-', ms=4, lw=1.4,
                color=_METHOD_COLOR[m], label=_METHOD_LABEL[m])
        if not noisy_plotted:
            ax.plot(E_true, [r['eff_n'] for r in rows], 's--', ms=4, lw=1.2,
                    color='gray', alpha=0.7, label='Before denoising (noisy)')
            noisy_plotted = True

    if E_true_ref is None:
        print('No data found for any method -- aborting amp-efficiency plot.')
        return

    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true_ref))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel('Fraction within 20% of true amplitude', fontsize=10)
    ax.set_title(
        '1.382 mV — 4-way denoising method comparison: ground-truth amplitude efficiency\n'
        '(fraction of events with |amp_fit - amp_true| / amp_true ≤ 20%; NOT gated by tri-exp chi2)',
        fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, os.path.join(output_dir, 'amp_efficiency_1p382mV_4way.png'))


# ── PSD high-frequency floor ─────────────────────────────────────────────────

def plot_hf_floor_comparison(eval_dir, output_dir=None, hf_cutoff=100.0, n_sample=500):
    output_dir = output_dir or os.path.join(_RESULTS_DIR, 'plots', 'method_comparison_1p382mV')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(eval_dir, 'eval_efficiency_*.h5')))
    if not files:
        print(f'No eval files found in {eval_dir}')
        return

    rows = []
    for path in files:
        with h5py.File(path, 'r') as f:
            E_true = float(f.attrs['energy_kev'])
            fs = float(f.attrs.get('f_sample', 1000.0))
            n = f['waveforms_noisy'].shape[0]
            if n <= n_sample:
                idx = np.arange(n)
            else:
                idx = np.sort(np.random.RandomState(0).choice(n, n_sample, replace=False))
            row = {'E_true': E_true}
            noisy = f['waveforms_noisy'][idx]
            row['noisy'] = float(np.mean([hf_floor_power(w, fs, hf_cutoff) for w in noisy]))
            for m in _METHODS:
                field = _METHOD_FIELD[m]
                if field not in f:
                    row[m] = np.nan
                    continue
                wf = f[field][idx]
                row[m] = float(np.mean([hf_floor_power(w, fs, hf_cutoff) for w in wf]))
        rows.append(row)
    rows.sort(key=lambda r: r['E_true'])

    E_true = np.array([r['E_true'] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(E_true, [r['noisy'] for r in rows], 's--', ms=4, lw=1.2,
            color='gray', alpha=0.7, label='Before denoising (noisy)')
    for m in _METHODS:
        vals = [r[m] for r in rows]
        ax.plot(E_true, vals, 'o-', ms=4, lw=1.4, color=_METHOD_COLOR[m],
                label=_METHOD_LABEL[m])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_xlim(E_true))
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel(f'PSD high-frequency floor (mean power, f ≥ {hf_cutoff:.0f} Hz)', fontsize=10)
    ax.set_title(
        f'1.382 mV — 4-way denoising method comparison: PSD HF floor\n'
        f'(n≤{n_sample} events/point, random subsample, seed=0)',
        fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, os.path.join(output_dir, 'hf_floor_1p382mV_4way.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_base', default=DEFAULT_FIT_BASE)
    parser.add_argument('--eval_dir', default=DEFAULT_EVAL_DIR)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--hf_cutoff', type=float, default=100.0)
    parser.add_argument('--n_sample', type=int, default=500)
    args = parser.parse_args()

    for method_key in ('triexp', 'maxbase'):
        plot_resolution_comparison(args.fit_base, method_key, args.output_dir, pct=False)
        plot_resolution_comparison(args.fit_base, method_key, args.output_dir, pct=True)
    plot_amp_efficiency_comparison(args.fit_base, args.output_dir)
    plot_hf_floor_comparison(args.eval_dir, args.output_dir, args.hf_cutoff, args.n_sample)

    print('\nAll done.')


if __name__ == '__main__':
    main()
