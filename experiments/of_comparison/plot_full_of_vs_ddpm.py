"""Full runs: energy resolution vs energy, Octopus OF (new stream, ideal triggers) vs DDPM zero-init (old eval set).

OF   : OFcompare_local/run{run}/of_ideal.csv from run_full_of_chain.sh, signal triggers matched to the
       true onsets within +-300 samples, delay-at-limit events dropped (counted in the table)
DDPM : tri-exp fit amplitude of the zero-init output on the old eval set (results/fit_results/{level}_zeroinit)
Raw  : tri-exp fit on the noisy windows of the same old eval set (amp_noisy), for reference
The two data sets are different events (old set: per-window noise draw; new stream: daily noise
states), so this is a per-energy comparison of methods at the same noise level, not event by event.
All amplitudes are calibrated per peak (E x amp / mean amp). Width = Gaussian fit within +-3 robust
sigma (1.4826 MAD) of the median, as in plot_pilot_spectra.py; the plain std is also reported.
Per-peak calibration hides a biased mean, so mean_keV_gain2615 (mean amplitude converted with the
2615 keV gain of the same method) is reported as well: a method that pulls low-energy outputs towards
a typical pulse shows up there, not in sigma.

Usage: python experiments/of_comparison/plot_full_of_vs_ddpm.py --run 999999
Output: results/plots/OFcompare/full_run{run}_resolution_vs_energy.png and .json next to it
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, csv, json, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np, h5py
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/mnt/d/storage/research/DDPM4bolometer'
LEVEL = {999998: ('1p382mV', '1.382 mV'), 999999: ('7p5mV', '7.5 mV')}
C_DDPM, C_OF, C_RAW = '#c2603c', '#2f6f9f', '#7f7f7f'


def of_amplitudes(run):
    rows = list(csv.DictReader(open(f'{D}/OFcompare_local/run{run}/of_ideal.csv')))
    col = lambda k: np.array([r[k] == 'True' if r[k] in ('True', 'False') else float(r[k]) for r in rows])
    ts = col('module_triggersample').astype(np.int64); isn = col('module_isnoise').astype(bool)
    good = col('optimumfilter_good').astype(bool); amp = col('optimumfilter_amplitude')
    lim = col('optimumfilter_Chi2AtLowerLimit').astype(bool) | col('optimumfilter_Chi2AtUpperLimit').astype(bool)
    ev = list(csv.DictReader(open(f'{D}/OFcompare_truth/run{run}/truth/events.csv')))
    on = np.array([int(r['onset_sample']) for r in ev]); E = np.array([int(r['energy_keV']) for r in ev])
    sig = np.flatnonzero(~isn & good)
    j = np.clip(np.searchsorted(on, ts[sig]), 1, len(on) - 1)
    near = np.where(np.abs(ts[sig] - on[j - 1]) < np.abs(ts[sig] - on[j]), j - 1, j)
    ok = np.abs(ts[sig] - on[near]) <= 300
    s, e, l = sig[ok], E[near[ok]], lim[sig[ok]]
    out = {int(k): dict(amp=amp[s[(e == k) & ~l]], n_true=int((E == k).sum()), n_at_limit=int(((e == k) & l).sum()))
           for k in sorted(set(E.tolist()))}
    return out, amp[isn & good]


def fit_amplitudes(level, field):
    out = {}
    for fn in os.listdir(f'{PROJ}/results/fit_results/{level}_zeroinit'):
        e = int(fn.split('_')[-1].replace('keV.h5', ''))
        with h5py.File(f'{PROJ}/results/fit_results/{level}_zeroinit/{fn}') as f:
            a = f[f'amp_{field}'][:].astype(float); ok = f[f'success_{field}'][:].astype(bool) & np.isfinite(a)
        out[e] = dict(amp=a[ok], n_true=len(a))
    return out


def gauss(x, mu, s, A):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2)


def width(x):
    """Gaussian sigma (+-error) within +-3 robust sigma of the median, and the plain std."""
    med = np.median(x); rs = 1.4826 * np.median(np.abs(x - med))
    edges = np.linspace(med - 6 * rs, med + 6 * rs, 81)
    h, _ = np.histogram(x, edges); w = edges[1] - edges[0]
    c = 0.5 * (edges[1:] + edges[:-1]); m = np.abs(c - med) < 3 * rs
    dens = h / (len(x) * w)
    try:
        p, cov = curve_fit(gauss, c[m], dens[m], p0=[med, rs, dens.max()],
                           sigma=np.sqrt(np.maximum(h[m], 1)) / (len(x) * w), absolute_sigma=True)
        return abs(p[1]), float(np.sqrt(cov[1, 1])), float(x.std())
    except RuntimeError:
        return float('nan'), float('nan'), float(x.std())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, default=999999)
    a = ap.parse_args()
    level, lab = LEVEL[a.run]
    of, noise = of_amplitudes(a.run)
    dd, raw = fit_amplitudes(level, 'denoised'), fit_amplitudes(level, 'noisy')
    energies = sorted(set(of) & set(dd))
    table = {}
    g2615 = {name: src[2615]['amp'].mean() / 2615 for name, src in (('OF', of), ('DDPM_zeroinit', dd), ('raw_triexp', raw))}
    for e in energies:
        row = {}
        for name, src in (('OF', of), ('DDPM_zeroinit', dd), ('raw_triexp', raw)):
            x = src[e]['amp']
            s, ds, sd = width(x / x.mean() * e)
            row[name] = dict(sigma_keV=s, sigma_err_keV=ds, std_keV=sd, n=len(x), n_true=src[e]['n_true'],
                             mean_keV_gain2615=float(x.mean() / g2615[name]))
            if name == 'OF':
                row[name]['n_delay_at_limit'] = src[e]['n_at_limit']
        table[e] = row
        print(f"{e:5d} keV  OF {row['OF']['sigma_keV']:.4f} (std {row['OF']['std_keV']:.4f}, N {row['OF']['n']})   "
              f"DDPM {row['DDPM_zeroinit']['sigma_keV']:.4f} (std {row['DDPM_zeroinit']['std_keV']:.4f}, N {row['DDPM_zeroinit']['n']})   "
              f"raw {row['raw_triexp']['sigma_keV']:.4f}   mean@2615gain OF {row['OF']['mean_keV_gain2615']:.2f} "
              f"DDPM {row['DDPM_zeroinit']['mean_keV_gain2615']:.2f} raw {row['raw_triexp']['mean_keV_gain2615']:.2f}")
    # OF on the noise windows (no delay scan there), converted with the gain of each peak
    gains = {e: of[e]['amp'].mean() / e for e in energies}
    # same width definition as the peaks (Gaussian fit); width() wants a positive peak, so shift by a constant
    noise_keV = {e: float(width(noise / g - (noise / g).mean() + 100)[0]) for e, g in gains.items()}
    noise_std_keV = {e: float(noise.std() / g) for e, g in gains.items()}
    print('OF noise windows: Gaussian sigma ' + ', '.join(f'{noise_keV[e]:.4f}' for e in (energies[0], energies[-1]))
          + ' keV, std ' + ', '.join(f'{noise_std_keV[e]:.4f}' for e in (energies[0], energies[-1])) + ' keV (gain of lowest / highest peak)')

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    E = np.array(energies, float)
    get = lambda k, q: np.array([table[e][k][q] for e in energies])
    for k, c, l, mk in (('OF', C_OF, f'Octopus OF, new stream run {a.run}', 'o'),
                        ('DDPM_zeroinit', C_DDPM, 'DDPM zero-init + tri-exp, old eval set', 's'),
                        ('raw_triexp', C_RAW, 'Tri-exp on noisy windows, old eval set', '^')):
        ax.errorbar(E, get(k, 'sigma_keV'), yerr=get(k, 'sigma_err_keV'), color=c, marker=mk, ms=5, lw=1.2, capsize=2, label=l)
    ax.plot(E, [noise_keV[e] for e in energies], color=C_OF, ls=':', lw=1.0, label='OF baseline resolution (noise windows, Gaussian $\\sigma$)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('Gaussian $\\sigma$ (keV)')
    ax.set_title(f'Energy resolution vs energy, {lab} noise', fontsize=11); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    axr.plot(E, get('DDPM_zeroinit', 'sigma_keV') / get('OF', 'sigma_keV'), color=C_DDPM, marker='s', ms=5)
    axr.axhline(1, color='k', lw=0.6, ls=':'); axr.set_yscale('log')
    axr.set_ylabel('$\\sigma_{DDPM}$ / $\\sigma_{OF}$'); axr.set_xlabel('Energy (keV)'); axr.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out = f'{PROJ}/results/plots/OFcompare/full_run{a.run}_resolution_vs_energy'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + '.png', dpi=150)
    json.dump(dict(run=a.run, level=level, peaks=table, of_noise_sigma_keV=noise_keV, of_noise_std_keV=noise_std_keV), open(out + '.json', 'w'), indent=1)
    print('Saved ->', out + '.png')


if __name__ == '__main__':
    main()
