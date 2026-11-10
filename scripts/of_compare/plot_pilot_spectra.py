"""Pilot runs 999996/999997: energy spectra of DDPM vs Octopus OF on the same events, with Gaussian fits.

DDPM : tri-exponential fit amplitude of the no_noise output (results/fit_results/pilot_run{run}_nonoise)
OF   : old Octopus OptimumFilter amplitude with ideal triggers (OFcompare_local/run{run}/of_ideal.csv),
       matched to the true onsets as in analyze_local_of.py, delay-at-limit events dropped
Both are calibrated per peak (E x amp / mean amp), so the peaks sit at the true energy and only the
widths are compared. Each method gets its own binning; histograms are normalised to unit area.
Gaussian fit: least squares on the histogram within +-3 robust sigma (1.4826 MAD) of the median.

Usage: python scripts/of_compare/plot_pilot_spectra.py   (rows for runs whose fit results are missing are skipped)
"""
import os, sys, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np, h5py
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/mnt/d/storage/research/DDPM4bolometer'
RUNS = [(999996, '1.382 mV'), (999997, '7.5 mV')]
ENERGIES = [15, 2615]
C_DDPM, C_OF = '#c2603c', '#2f6f9f'
OUT = f'{PROJ}/results/plots/OFcompare/pilot_spectra_of_vs_ddpm.png'


def of_amplitudes(run):
    rows = list(csv.DictReader(open(f'{D}/OFcompare_local/run{run}/of_ideal.csv')))
    col = lambda k: np.array([r[k] == 'True' if r[k] in ('True', 'False') else float(r[k]) for r in rows])
    ts = col('module_triggersample').astype(np.int64); isn = col('module_isnoise').astype(bool)
    good = col('optimumfilter_good').astype(bool); amp = col('optimumfilter_amplitude')
    lim = col('optimumfilter_Chi2AtLowerLimit').astype(bool) | col('optimumfilter_Chi2AtUpperLimit').astype(bool)
    ev = list(csv.DictReader(open(f'{D}/OFcompare_truth/run{run}/truth/events.csv')))
    on = np.array([int(r['onset_sample']) for r in ev]); E = np.array([int(r['energy_keV']) for r in ev])
    sig = np.flatnonzero(~isn & good & ~lim)
    j = np.clip(np.searchsorted(on, ts[sig]), 1, len(on) - 1)
    near = np.where(np.abs(ts[sig] - on[j - 1]) < np.abs(ts[sig] - on[j]), j - 1, j)
    ok = np.abs(ts[sig] - on[near]) <= 300
    return {e: amp[sig[ok]][E[near[ok]] == e] for e in ENERGIES}


def ddpm_amplitudes(run):
    out = {}
    for e in ENERGIES:
        with h5py.File(f'{PROJ}/results/fit_results/pilot_run{run}_nonoise/fit_efficiency_{e}keV.h5') as f:
            a = f['amp_denoised'][:].astype(float); ok = f['success_denoised'][:].astype(bool) & np.isfinite(a)
        out[e] = a[ok]
    return out


def gauss(x, mu, s, A):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2)


def spectrum(ax, x, color, label):
    """Unit-area histogram on its own binning + Gaussian fit within +-3 robust sigma."""
    med = np.median(x); rs = 1.4826 * np.median(np.abs(x - med))
    edges = np.linspace(med - 6 * rs, med + 6 * rs, 81)
    h, _ = np.histogram(x, edges); w = edges[1] - edges[0]
    dens = h / (len(x) * w); c = 0.5 * (edges[1:] + edges[:-1])
    m = np.abs(c - med) < 3 * rs
    p, cov = curve_fit(gauss, c[m], dens[m], p0=[med, rs, dens.max()],
                       sigma=np.sqrt(np.maximum(h[m], 1)) / (len(x) * w), absolute_sigma=True)
    mu, s = p[0], abs(p[1]); ds = np.sqrt(cov[1, 1])
    ax.stairs(dens, edges, color=color, lw=1.4,
              label=f'{label}: Gaussian $\\sigma$ = {s:.4f} $\\pm$ {ds:.4f} keV (std {x.std():.4f}, N = {len(x)})')
    xx = np.linspace(mu - 4 * s, mu + 4 * s, 400)
    ax.plot(xx, gauss(xx, *p[:2], p[2]), color=color, ls='--', lw=1.0)
    return s, x.std()


def main():
    runs = [(r, l) for r, l in RUNS if os.path.exists(f'{PROJ}/results/fit_results/pilot_run{r}_nonoise/fit_efficiency_2615keV.h5')]
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 4.6 * len(runs)), squeeze=False)
    for row, (run, lab) in enumerate(runs):
        of, dd = of_amplitudes(run), ddpm_amplitudes(run)
        for colm, e in enumerate(ENERGIES):
            ax = axes[row, colm]
            s_dd, _ = spectrum(ax, dd[e] / dd[e].mean() * e, C_DDPM, 'DDPM (no_noise)')
            s_of, _ = spectrum(ax, of[e] / of[e].mean() * e, C_OF, 'Octopus OF')
            print(f'run {run} {e} keV: DDPM sigma {s_dd:.4f}, OF sigma {s_of:.4f} keV, ratio {s_dd / s_of:.1f}')
            ax.axvline(e, color='k', lw=0.6, ls=':')
            ax.set_xlim(e - 6 * 1.4826 * np.median(np.abs(dd[e] / dd[e].mean() * e - e)),
                        e + 6 * 1.4826 * np.median(np.abs(dd[e] / dd[e].mean() * e - e)))
            ax.set_title(f'run {run} ({lab} noise), {e} keV, same events', fontsize=11)
            ax.set_xlabel('Energy, calibrated per peak (keV)'); ax.set_ylabel('Probability density (1/keV)')
            ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    fig.text(0.5, 0.005, 'Dashed: Gaussian fit within $\\pm$3 robust $\\sigma$ of the median. Histograms normalised to unit area, own binning per method.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
