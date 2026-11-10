"""Energy resolution vs energy for four methods: raw, raw + Bessel, DDPM zero-init, DDPM zero-init + Bessel.

Bessel = the simulator chain (x10 upsample, 6th-order 120 Hz Bessel at 10 kHz, decimate), scripts/apply_bessel_to_ddpm.py.
Fits: results/fit_results/{level}_zeroinit (raw = amp_noisy, DDPM = amp_denoised),
      {level}_noisy_bessel (raw + Bessel), {level}_zeroinit_bessel (DDPM + Bessel), denoised columns.
Events: tri-exp fit success only, no chi2/ndf cut (the cut removes most DDPM + Bessel events at high
energy because its sigma comes from the filtered pre-trigger). Width as in plot_full_of_vs_ddpm.py:
per-peak calibration, Gaussian fit within +-3 robust sigma.

--amp maxbase uses the max-baseline amplitude (window max minus median of the first 0.5 s, no fit,
all events) instead of the tri-exp fit amplitude.

Usage: python scripts/plot_bessel_resolution.py [--amp triexp|maxbase]
Output: results/plots/bessel/bessel_resolution_vs_energy{,_maxbase}.png and .json
"""
import os, sys, json, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, f'{PROJ}/scripts/of_compare')
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_full_of_vs_ddpm import width

EN = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034]
LEVELS = [('1p382mV', '1.382 mV'), ('7p5mV', '7.5 mV')]
SER = [('raw', 'Raw (no denoising)', '#9a9a9a', '^', 'zeroinit', 'noisy'),
       ('rawB', 'Raw + Bessel', '#2a9d8f', 'v', 'noisy_bessel', 'denoised'),
       ('dd', 'DDPM', '#2f6f9f', 's', 'zeroinit', 'denoised'),
       ('ddB', 'DDPM + Bessel', '#c2603c', 'o', 'zeroinit_bessel', 'denoised')]


def sigma(level, tag, col, E, amp):
    with h5py.File(f'{PROJ}/results/fit_results/{level}_{tag}/fit_efficiency_{E}keV.h5') as f:
        if amp == 'maxbase':
            a = f[f'amp_maxbase_{col}'][:].astype(float); ok = np.isfinite(a)
        else:
            a = f[f'amp_{col}'][:].astype(float); ok = f[f'success_{col}'][:].astype(bool) & np.isfinite(a)
    x = a[ok]; s, ds, _ = width(x / x.mean() * E)
    return s, ds, int(ok.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--amp', choices=['triexp', 'maxbase'], default='triexp')
    amp = ap.parse_args().amp
    res = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for j, (level, lab) in enumerate(LEVELS):
        res[level] = {k: [sigma(level, tag, col, E, amp) for E in EN] for k, _, _, _, tag, col in SER}
        ax, axr = axes[0, j], axes[1, j]
        for k, l, c, mk, _, _ in SER:
            s = np.array([r[0] for r in res[level][k]]); ds = np.array([r[1] for r in res[level][k]])
            ax.errorbar(EN, s, yerr=ds, color=c, marker=mk, ms=5, lw=1.2, capsize=2, label=l,
                        ls='--' if k.endswith('B') else '-', mfc='none' if k.endswith('B') else c)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.3, which='both')
        ax.set_title(f'{lab} noise', fontsize=11); ax.set_ylabel('Gaussian $\\sigma$ (keV)')
        if j == 0: ax.legend(fontsize=9)
        g = lambda k: np.array([r[0] for r in res[level][k]])
        axr.plot(EN, g('rawB') / g('raw'), color='#2a9d8f', marker='v', ms=5, label='(Raw + Bessel) / Raw')
        axr.plot(EN, g('ddB') / g('dd'), color='#c2603c', marker='o', ms=5, label='(DDPM + Bessel) / DDPM')
        axr.axhline(1, color='k', lw=0.6, ls=':'); axr.grid(alpha=0.3, which='both')
        axr.set_ylabel('$\\sigma$ ratio'); axr.set_xlabel('Energy (keV)')
        if j == 0: axr.legend(fontsize=8)
    sel = 'tri-exp fit amplitude, fit success, no $\\chi^2$ cut' if amp == 'triexp' else 'max-baseline amplitude, all events'
    fig.suptitle(f'Energy resolution vs energy: effect of a Bessel filter ({sel})', fontsize=12)
    fig.tight_layout()
    out = f'{PROJ}/results/plots/bessel/bessel_resolution_vs_energy' + ('_maxbase' if amp == 'maxbase' else '')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + '.png', dpi=150)
    json.dump(dict(energies=EN, amplitude=amp, selection='fit success, no chi2 cut' if amp == 'triexp' else 'all events',
                   sigma_keV={lv: {k: [dict(sigma=r[0], err=r[1], n=r[2]) for r in v] for k, v in d.items()} for lv, d in res.items()}),
              open(out + '.json', 'w'), indent=1)
    print('Saved ->', out + '.png')


if __name__ == '__main__':
    main()
