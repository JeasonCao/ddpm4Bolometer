"""
Plot reconstruction quality: mean reconstructed energy vs true energy,
with Gaussian-fit error bars, linear fit, and a bias inset.

Calibration per energy point: k = E_true / true_amp_V  (template-based single-point)
  → E_reco = k × gauss_mu  (peak from Gaussian fit of amplitude distribution)

4 figures: 2 noise levels (1p2mV, 7p5mV) × 2 methods (triexp, maxbase).
X-axis is log scale.

Usage
-----
python scripts/plot_reconstruction_quality.py
python scripts/plot_reconstruction_quality.py \
    --fit_dirs results/fit_results/efficiency_1p2mV results/fit_results/efficiency_7p5mV \
    --output_dir results/plots/reconstruction_quality
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_NOISE_LABELS = {'1p2mV': '1.2 mV', '7p5mV': '7.5 mV'}

_METHOD_SUFFIX = {
    'triexp':  ('triexp_noisy',  'triexp_denoised'),
    'maxbase': ('maxbase_noisy', 'maxbase_denoised'),
}


# ── load Gaussian fit stats from one HDF5 ────────────────────────────────────

def _load_gauss_stats(path, method):
    """
    Returns dict with keys: E_true, k,
      mu_n, mu_err_n, sigma_n, sigma_err_n,
      mu_d, mu_err_d, sigma_d, sigma_err_d
    All amplitude values multiplied by k → energy units [keV].
    Falls back to nanmean/nanstd if Gaussian attrs are missing or not converged.
    """
    suf_n, suf_d = _METHOD_SUFFIX[method]

    with h5py.File(path, 'r') as f:
        E_true   = float(f.attrs['energy_kev'])
        true_amp = float(f.attrs['true_amp_V'])
        k = E_true / true_amp

        def _get(suffix):
            converged = bool(f.attrs.get(f'gauss_converged_{suffix}', False))
            if converged:
                mu      = float(f.attrs[f'gauss_mu_{suffix}'])
                mu_err  = float(f.attrs[f'gauss_mu_err_{suffix}'])
                sigma   = float(f.attrs[f'gauss_sigma_{suffix}'])
                sig_err = float(f.attrs[f'gauss_sigma_err_{suffix}'])
            else:
                # fallback: compute from raw array
                if 'noisy' in suffix:
                    key = 'amp_noisy' if method == 'triexp' else 'amp_maxbase_noisy'
                else:
                    key = 'amp_denoised' if method == 'triexp' else 'amp_maxbase_denoised'
                arr = f[key][:].astype(float)
                mu      = np.nanmean(arr)
                mu_err  = np.nanstd(arr) / np.sqrt(np.sum(np.isfinite(arr)))
                sigma   = np.nanstd(arr)
                sig_err = np.nan
            return mu, mu_err, sigma, sig_err

        mu_n, mu_err_n, sigma_n, sig_err_n = _get(suf_n)
        mu_d, mu_err_d, sigma_d, sig_err_d = _get(suf_d)

    return dict(
        E_true=E_true, k=k,
        mu_n     = k * mu_n,     mu_err_n  = k * mu_err_n,
        sigma_n  = k * sigma_n,  sig_err_n = k * sig_err_n,
        mu_d     = k * mu_d,     mu_err_d  = k * mu_err_d,
        sigma_d  = k * sigma_d,  sig_err_d = k * sig_err_d,
    )


def _load_all(fit_dir, method):
    files = sorted(glob.glob(os.path.join(fit_dir, 'fit_efficiency_*.h5')))
    if not files:
        return None
    rows = [_load_gauss_stats(p, method) for p in files]
    rows.sort(key=lambda r: r['E_true'])

    def arr(key):
        return np.array([r[key] for r in rows])

    return dict(
        E_true   = arr('E_true'),
        mu_n     = arr('mu_n'),    mu_err_n  = arr('mu_err_n'),
        sigma_n  = arr('sigma_n'), sig_err_n = arr('sig_err_n'),
        mu_d     = arr('mu_d'),    mu_err_d  = arr('mu_err_d'),
        sigma_d  = arr('sigma_d'), sig_err_d = arr('sig_err_d'),
    )


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {path}')


def _log_xlim(E_true):
    return E_true.min() * 0.6, E_true.max() * 1.6


def _pad_ylim(values, pad_frac=0.15, clip=80):
    """Symmetric-safe padded y-limits — avoids the sign-flip bug of `* 1.3`
    (which shrinks the range, and can crop data, when the min/max is positive)."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    span = hi - lo
    pad = span * pad_frac if span > 0 else max(abs(hi), 0.01) * 0.3
    ylo = max(lo - pad, -clip)
    yhi = min(hi + pad,  clip)
    return ylo, yhi


# ── reco energy vs true energy ────────────────────────────────────────────────

def plot_quality(fit_dir, method, output_path):
    noise_tag = os.path.basename(fit_dir).replace('efficiency_', '')
    noise_str = _NOISE_LABELS.get(noise_tag, noise_tag)
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'

    d = _load_all(fit_dir, method)
    if d is None:
        print(f'No files found in {fit_dir}'); return

    E_true  = d['E_true']
    mu_n    = d['mu_n']
    mu_d    = d['mu_d']
    mu_err_n = d['mu_err_n']
    mu_err_d = d['mu_err_d']
    # Use distribution σ as error bar (energy resolution), not fit precision of μ
    sigma_n = d['sigma_n']
    sigma_d = d['sigma_d']

    E_line = np.logspace(np.log10(E_true.min() * 0.7),
                         np.log10(E_true.max() * 1.4), 300)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Clip lower band edge above 0 — log y-scale can't show non-positive values
    ax.fill_between(E_true, np.clip(mu_n - sigma_n, 1e-6, None), mu_n + sigma_n,
                    color='steelblue', alpha=0.2, lw=0)
    ax.fill_between(E_true, np.clip(mu_d - sigma_d, 1e-6, None), mu_d + sigma_d,
                    color='tomato', alpha=0.2, lw=0)
    ax.errorbar(E_true, mu_n, yerr=sigma_n, fmt='o', ms=4,
                color='steelblue', elinewidth=0.8, capsize=2,
                label='Before denoising  (band/err = σ)')
    ax.errorbar(E_true, mu_d, yerr=sigma_d, fmt='o', ms=4,
                color='tomato', elinewidth=0.8, capsize=2,
                label='After denoising  (band/err = σ)')

    # Ideal diagonal
    ax.plot(E_line, E_line, color='black', lw=1.2, ls=':', label='Ideal (y = x)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    xlim = _log_xlim(E_true)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_xlabel('True energy (keV)', fontsize=11)
    ax.set_ylabel('Reconstructed energy (keV)', fontsize=11)
    ax.set_title(
        f'Reconstruction Quality — {method_label}\n'
        f'noise RMS = {noise_str},  error bars from Gaussian fit of amplitude dist.',
        fontsize=10.5
    )
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.3, which='both')

    # Inset: bias (%)
    bias_n = (mu_n - E_true) / E_true * 100
    bias_d = (mu_d - E_true) / E_true * 100
    bias_err_n = mu_err_n / E_true * 100
    bias_err_d = mu_err_d / E_true * 100

    axins = inset_axes(ax, width='40%', height='32%', loc='lower right',
                       bbox_to_anchor=(-0.02, 0.06, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.fill_between(E_true, bias_n - bias_err_n, bias_n + bias_err_n,
                       color='steelblue', alpha=0.25, lw=0)
    axins.fill_between(E_true, bias_d - bias_err_d, bias_d + bias_err_d,
                       color='tomato', alpha=0.25, lw=0)
    axins.errorbar(E_true, bias_n, yerr=bias_err_n, fmt='o-', ms=3, lw=1.2,
                   color='steelblue', elinewidth=0.7, capsize=1.5, label='Before')
    axins.errorbar(E_true, bias_d, yerr=bias_err_d, fmt='o-', ms=3, lw=1.2,
                   color='tomato', elinewidth=0.7, capsize=1.5, label='After')
    axins.axhline(0, color='black', ls='--', lw=0.9)

    all_bias = np.concatenate([bias_n[np.isfinite(bias_n)],
                                bias_d[np.isfinite(bias_d)]])
    ylo, yhi = _pad_ylim(all_bias)
    axins.set_ylim(ylo, yhi)
    axins.set_xscale('log')
    axins.set_xlim(*_log_xlim(E_true))
    axins.set_xlabel('True energy (keV)', fontsize=7.5)
    axins.set_ylabel('Bias (%)', fontsize=7.5)
    axins.tick_params(labelsize=7)
    axins.legend(fontsize=7, loc='upper right')
    axins.grid(alpha=0.3, which='both')
    axins.set_title('Bias = (reco − true) / true', fontsize=7.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output_path}')


# ── standalone bias figure ────────────────────────────────────────────────────

def plot_bias(fit_dir, method, output_path):
    noise_tag    = os.path.basename(fit_dir).replace('efficiency_', '')
    noise_str    = _NOISE_LABELS.get(noise_tag, noise_tag)
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'

    d = _load_all(fit_dir, method)
    if d is None:
        return

    E_true = d['E_true']
    bias_n     = (d['mu_n'] - E_true) / E_true * 100
    bias_d     = (d['mu_d'] - E_true) / E_true * 100
    bias_err_n = d['mu_err_n'] / E_true * 100
    bias_err_d = d['mu_err_d'] / E_true * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(E_true, bias_n, yerr=bias_err_n, fmt='o-', ms=4, lw=1.4,
                color='steelblue', elinewidth=0.8, capsize=2,
                label='Before denoising')
    ax.errorbar(E_true, bias_d, yerr=bias_err_d, fmt='o-', ms=4, lw=1.4,
                color='tomato', elinewidth=0.8, capsize=2,
                label='After denoising')
    ax.axhline(0, color='black', ls='--', lw=1.0, label='Bias = 0')

    all_bias = np.concatenate([bias_n[np.isfinite(bias_n)],
                                bias_d[np.isfinite(bias_d)]])
    ylo, yhi = _pad_ylim(all_bias)
    ax.set_ylim(ylo, yhi)
    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true))
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel('Bias (%)', fontsize=10)
    ax.set_title(
        f'Reconstruction Bias — {method_label}  |  noise RMS = {noise_str}\n'
        f'Bias = (reco − true) / true × 100,  error bars from Gaussian fit',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, output_path)


# ── energy resolution FWHM ────────────────────────────────────────────────────

def plot_resolution_fwhm(fit_dir, method, output_path):
    noise_tag    = os.path.basename(fit_dir).replace('efficiency_', '')
    noise_str    = _NOISE_LABELS.get(noise_tag, noise_tag)
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'

    d = _load_all(fit_dir, method)
    if d is None:
        return

    E_true  = d['E_true']
    fwhm_n  = 2.355 * d['sigma_n']
    fwhm_d  = 2.355 * d['sigma_d']
    fwhm_err_n = 2.355 * d['sig_err_n']
    fwhm_err_d = 2.355 * d['sig_err_d']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(E_true, fwhm_n, yerr=fwhm_err_n, fmt='o-', ms=4, lw=1.4,
                color='steelblue', elinewidth=0.8, capsize=2,
                label='Before denoising')
    ax.errorbar(E_true, fwhm_d, yerr=fwhm_err_d, fmt='o-', ms=4, lw=1.4,
                color='tomato', elinewidth=0.8, capsize=2,
                label='After denoising')
    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true))
    ax.set_ylim(0, None)
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel('Energy resolution FWHM (keV)', fontsize=10)
    ax.set_title(
        f'Energy Resolution — {method_label}  |  noise RMS = {noise_str}\n'
        f'FWHM = 2.355 × σ_Gauss,  error bars from Gaussian fit',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, output_path)


# ── relative energy resolution (%) ───────────────────────────────────────────

def plot_resolution_pct(fit_dir, method, output_path):
    noise_tag    = os.path.basename(fit_dir).replace('efficiency_', '')
    noise_str    = _NOISE_LABELS.get(noise_tag, noise_tag)
    method_label = 'Tri-exp fit' if method == 'triexp' else 'Max-baseline'

    d = _load_all(fit_dir, method)
    if d is None:
        return

    E_true = d['E_true']
    fwhm_pct_n     = 2.355 * d['sigma_n']     / E_true * 100
    fwhm_pct_d     = 2.355 * d['sigma_d']     / E_true * 100
    fwhm_pct_err_n = 2.355 * d['sig_err_n']   / E_true * 100
    fwhm_pct_err_d = 2.355 * d['sig_err_d']   / E_true * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(E_true, fwhm_pct_n, yerr=fwhm_pct_err_n, fmt='o-', ms=4, lw=1.4,
                color='steelblue', elinewidth=0.8, capsize=2,
                label='Before denoising')
    ax.errorbar(E_true, fwhm_pct_d, yerr=fwhm_pct_err_d, fmt='o-', ms=4, lw=1.4,
                color='tomato', elinewidth=0.8, capsize=2,
                label='After denoising')
    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true))
    ax.set_ylim(0, None)
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel('Energy resolution FWHM / E  (%)', fontsize=10)
    ax.set_title(
        f'Relative Energy Resolution — {method_label}  |  noise RMS = {noise_str}\n'
        f'FWHM(%) = 2.355 × σ_Gauss / E_true × 100,  error bars from Gaussian fit',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, output_path)


# ── Pearson r vs energy ───────────────────────────────────────────────────────

def plot_pearsonr(fit_dir, output_path):
    noise_tag = os.path.basename(fit_dir).replace('efficiency_', '')
    noise_str = _NOISE_LABELS.get(noise_tag, noise_tag)

    files = sorted(glob.glob(os.path.join(fit_dir, 'fit_efficiency_*.h5')))
    if not files:
        return

    rows = []
    for path in files:
        with h5py.File(path, 'r') as f:
            E_true = float(f.attrs['energy_kev'])
            r_n = f['fit_pearsonr_noisy'][:].astype(float)
            r_d = f['fit_pearsonr_denoised'][:].astype(float)
        rows.append({
            'E_true': E_true,
            'mean_n': np.nanmean(r_n),
            'std_n':  np.nanstd(r_n),
            'mean_d': np.nanmean(r_d),
            'std_d':  np.nanstd(r_d),
        })
    rows.sort(key=lambda r: r['E_true'])

    E_true = np.array([r['E_true'] for r in rows])
    mean_n = np.array([r['mean_n'] for r in rows])
    std_n  = np.array([r['std_n']  for r in rows])
    mean_d = np.array([r['mean_d'] for r in rows])
    std_d  = np.array([r['std_d']  for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(E_true, mean_n - std_n, mean_n + std_n,
                    color='steelblue', alpha=0.15)
    ax.fill_between(E_true, mean_d - std_d, mean_d + std_d,
                    color='tomato', alpha=0.15)
    ax.plot(E_true, mean_n, 'o-', ms=4, lw=1.4, color='steelblue',
            label='Before denoising')
    ax.plot(E_true, mean_d, 'o-', ms=4, lw=1.4, color='tomato',
            label='After denoising')
    ax.axhline(1.0, color='black', ls=':', lw=0.9, label='Perfect match (r = 1)')

    ax.set_xscale('log')
    ax.set_xlim(*_log_xlim(E_true))
    ax.set_ylim(max(0, np.nanmin(mean_n - std_n) - 0.05), 1.05)
    ax.set_xlabel('True energy (keV)', fontsize=10)
    ax.set_ylabel('Mean Pearson r', fontsize=10)
    ax.set_title(
        f'Fit–Template Pearson Correlation vs Energy\n'
        f'r(tri-exp fit curve, clean template)  |  noise RMS = {noise_str}\n'
        f'(NaN for failed fits excluded from mean; shaded band = ±1σ)',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    _save_fig(fig, output_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_dirs', nargs='+', default=None)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    fit_dirs = args.fit_dirs or [
        os.path.join(_RESULTS_DIR, 'fit_results', 'efficiency_1p2mV'),
        os.path.join(_RESULTS_DIR, 'fit_results', 'efficiency_7p5mV'),
    ]
    out_dir = args.output_dir or os.path.join(_RESULTS_DIR, 'plots',
                                               'reconstruction_quality')
    os.makedirs(out_dir, exist_ok=True)

    for fit_dir in fit_dirs:
        noise_tag = os.path.basename(fit_dir).replace('efficiency_', '')
        for method in ('triexp', 'maxbase'):
            plot_quality(fit_dir, method,
                         os.path.join(out_dir, f'reco_quality_{method}_{noise_tag}.png'))
            plot_bias(fit_dir, method,
                      os.path.join(out_dir, f'bias_{method}_{noise_tag}.png'))
            plot_resolution_fwhm(fit_dir, method,
                      os.path.join(out_dir, f'resolution_{method}_{noise_tag}.png'))
            plot_resolution_pct(fit_dir, method,
                      os.path.join(out_dir, f'resolution_pct_{method}_{noise_tag}.png'))
        plot_pearsonr(fit_dir,
                      os.path.join(out_dir, f'pearsonr_{noise_tag}.png'))

    print('\nAll done.')


if __name__ == '__main__':
    main()
