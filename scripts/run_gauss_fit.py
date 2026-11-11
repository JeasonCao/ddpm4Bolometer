"""
Fit a Gaussian to the amplitude distribution for each fit-results HDF5 file.

Reads fit-results produced by run_fitting.py and fits a Gaussian to the
amp_noisy / amp_denoised / amp_maxbase_noisy / amp_maxbase_denoised
distributions. Results are written back as new attributes in the same file.

New attributes added
--------------------
gauss_mu_triexp_{noisy,denoised}        : Gaussian peak position [V]
gauss_mu_err_triexp_{noisy,denoised}    : 1-sigma uncertainty on peak [V]
gauss_sigma_triexp_{noisy,denoised}     : Gaussian sigma [V]
gauss_sigma_err_triexp_{noisy,denoised} : 1-sigma uncertainty on sigma [V]
gauss_chi2ndf_triexp_{noisy,denoised}   : chi2/ndf of fit (goodness of fit)
gauss_converged_triexp_{noisy,denoised} : bool, fit converged?
gauss_n_triexp_{noisy,denoised}         : number of events used in fit
(same pattern for _maxbase_)

Usage
-----
# Single directory
python -u scripts/run_gauss_fit.py \
    --fit_dir results/fit_results/efficiency_1p2mV

# Multiple directories
python -u scripts/run_gauss_fit.py \
    --fit_dir results/fit_results/efficiency_1p2mV \
             results/fit_results/efficiency_7p5mV \
             results/fit_results/resolution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import warnings
import h5py
import numpy as np
from scipy.optimize import curve_fit

MIN_EVENTS = 30   # minimum events required to attempt a Gaussian fit


def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian(amps: np.ndarray) -> dict:
    """Fit a Gaussian to a 1-D array of amplitude values.

    Returns a dict with keys:
      mu, mu_err, sigma, sigma_err, chi2ndf, converged, n_events
    All in the same units as amps.
    """
    result = dict(mu=np.nan, mu_err=np.nan,
                  sigma=np.nan, sigma_err=np.nan,
                  chi2ndf=np.nan, converged=False, n_events=0)

    vals = amps[np.isfinite(amps)]
    result['n_events'] = len(vals)

    if len(vals) < MIN_EVENTS:
        return result

    mu0    = np.median(vals)
    sigma0 = np.std(vals)
    if sigma0 == 0:
        return result

    # Histogram over ±4σ around median
    lo = mu0 - 4.0 * sigma0
    hi = mu0 + 4.0 * sigma0
    n_bins = max(20, min(100, len(vals) // 10))
    counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Poisson uncertainties for chi2 (floor at 1 to avoid /0)
    sigma_counts = np.where(counts > 0, np.sqrt(counts), 1.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, pcov = curve_fit(
                gaussian, centers, counts,
                p0=[mu0, sigma0, counts.max()],
                sigma=sigma_counts,
                absolute_sigma=True,
                maxfev=10000,
            )
        perr = np.sqrt(np.diag(pcov))

        mu_fit    = popt[0]
        sigma_fit = abs(popt[1])
        mu_err    = perr[0]
        sigma_err = perr[1]

        # Chi2/ndf over bins with counts > 0
        mask = counts > 0
        residuals = (counts[mask] - gaussian(centers[mask], *popt)) / sigma_counts[mask]
        ndf = mask.sum() - 3
        chi2ndf = float(np.sum(residuals ** 2) / ndf) if ndf > 0 else np.nan

        result.update(dict(
            mu=float(mu_fit),
            mu_err=float(mu_err),
            sigma=float(sigma_fit),
            sigma_err=float(sigma_err),
            chi2ndf=chi2ndf,
            converged=True,
        ))

    except Exception:
        # Fall back: store raw statistics with large uncertainty flags
        result.update(dict(
            mu=float(mu0),
            mu_err=np.nan,
            sigma=float(sigma0),
            sigma_err=np.nan,
            chi2ndf=np.nan,
            converged=False,
        ))

    return result


def process_file(h5_path: str, overwrite: bool = False) -> None:
    """Fit Gaussian to amplitude distributions in one fit-results HDF5 file."""

    with h5py.File(h5_path, 'r') as f:
        energy_kev  = float(f.attrs['energy_kev'])
        amp_n       = f['amp_noisy'][:]
        amp_d       = f['amp_denoised'][:]
        conv_n      = f['converged_noisy'][:].astype(bool)
        conv_d      = f['converged_denoised'][:].astype(bool)
        amp_mb_n    = f['amp_maxbase_noisy'][:]
        amp_mb_d    = f['amp_maxbase_denoised'][:]
        already_done = 'gauss_mu_triexp_noisy' in f.attrs

    if already_done and not overwrite:
        print(f"  Skipping (already fitted): {os.path.basename(h5_path)}")
        return

    print(f"  Fitting: {os.path.basename(h5_path)}  ({energy_kev:.0f} keV)")

    res_triexp_n  = fit_gaussian(amp_n[conv_n])
    res_triexp_d  = fit_gaussian(amp_d[conv_d])
    res_maxbase_n = fit_gaussian(amp_mb_n)
    res_maxbase_d = fit_gaussian(amp_mb_d)

    with h5py.File(h5_path, 'a') as f:
        for suffix, res in [
            ('triexp_noisy',    res_triexp_n),
            ('triexp_denoised', res_triexp_d),
            ('maxbase_noisy',    res_maxbase_n),
            ('maxbase_denoised', res_maxbase_d),
        ]:
            f.attrs[f'gauss_mu_{suffix}']        = res['mu']
            f.attrs[f'gauss_mu_err_{suffix}']    = res['mu_err']
            f.attrs[f'gauss_sigma_{suffix}']     = res['sigma']
            f.attrs[f'gauss_sigma_err_{suffix}'] = res['sigma_err']
            f.attrs[f'gauss_chi2ndf_{suffix}']   = res['chi2ndf']
            f.attrs[f'gauss_converged_{suffix}'] = res['converged']
            f.attrs[f'gauss_n_{suffix}']         = res['n_events']

    print(f"    triexp  noisy   : mu={res_triexp_n['mu']:.4f} V  "
          f"sigma={res_triexp_n['sigma']:.4f} V  "
          f"chi2/ndf={res_triexp_n['chi2ndf']:.2f}  "
          f"converged={res_triexp_n['converged']}")
    print(f"    triexp  denoised: mu={res_triexp_d['mu']:.4f} V  "
          f"sigma={res_triexp_d['sigma']:.4f} V  "
          f"chi2/ndf={res_triexp_d['chi2ndf']:.2f}  "
          f"converged={res_triexp_d['converged']}")
    print(f"    maxbase noisy   : mu={res_maxbase_n['mu']:.4f} V  "
          f"sigma={res_maxbase_n['sigma']:.4f} V  "
          f"chi2/ndf={res_maxbase_n['chi2ndf']:.2f}")
    print(f"    maxbase denoised: mu={res_maxbase_d['mu']:.4f} V  "
          f"sigma={res_maxbase_d['sigma']:.4f} V  "
          f"chi2/ndf={res_maxbase_d['chi2ndf']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian fit to amplitude distributions in fit-results HDF5 files")
    parser.add_argument('--fit_dir', nargs='+', required=False,
                        default=[
                            os.path.join(_RESULTS_DIR, 'fit_results', 'efficiency_1p2mV'),
                            os.path.join(_RESULTS_DIR, 'fit_results', 'efficiency_7p5mV'),
                            os.path.join(_RESULTS_DIR, 'fit_results', 'resolution'),
                        ],
                        help='One or more fit_results directories to process')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-fit even if Gaussian results already exist')
    args = parser.parse_args()

    for fit_dir in args.fit_dir:
        if not os.path.isdir(fit_dir):
            print(f"WARNING: directory not found, skipping: {fit_dir}")
            continue

        files = sorted(
            os.path.join(fit_dir, f)
            for f in os.listdir(fit_dir)
            if f.endswith('.h5')
        )
        if not files:
            print(f"WARNING: no HDF5 files in {fit_dir}")
            continue

        print(f"\n=== {fit_dir} ({len(files)} files) ===")
        for path in files:
            process_file(path, overwrite=args.overwrite)

    print("\nAll done.")


if __name__ == '__main__':
    main()
