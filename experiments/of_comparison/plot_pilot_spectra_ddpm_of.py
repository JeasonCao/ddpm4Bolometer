"""Pilot runs 999996/999997: energy spectra of three chains on the same events, with Gaussian fits.

  DDPM + tri-exp        : results/fit_results/pilot_run{run}_nonoise (all 5000 events)
  DDPM + Octopus OF     : OFcompare_ddpm/run{run}/of_ideal.csv (build_ddpm_stream.py + run_ddpm_of_chain.sh),
                          evaluation half B only (half A made the residual blocks / ANPS)
  Octopus OF on raw data: OFcompare_local/run{run}/of_ideal.csv (all events)
Per-peak calibration and Gaussian fit as in plot_pilot_spectra.py.

Usage: python experiments/of_comparison/plot_pilot_spectra_ddpm_of.py
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.of_comparison.plot_pilot_spectra import of_amplitudes, ddpm_amplitudes, spectrum, RUNS, ENERGIES

D = '/mnt/d/storage/research/DDPM4bolometer'
OUT = f'{PROJ}/results/plots/OFcompare/pilot_spectra_ddpm_octopus_of.png'


def ddpm_of_amplitudes(run):
    rows = list(csv.DictReader(open(f'{D}/OFcompare_ddpm/run{run}/of_ideal.csv')))
    idx = {int(float(r['module_triggersample'])): r for r in rows}
    out = {}
    for e in ENERGIES:
        sel = [idx[int(r['onset_sample'])] for r in csv.DictReader(open(f'{D}/OFcompare_ddpm/run{run}/truth/events.csv'))
               if int(r['energy_keV']) == e and r['half'] == 'B']
        out[e] = np.array([float(r['optimumfilter_amplitude']) for r in sel
                           if r['optimumfilter_good'] == 'True' and r['optimumfilter_Chi2AtLowerLimit'] == 'False'
                           and r['optimumfilter_Chi2AtUpperLimit'] == 'False'])
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (run, lab) in enumerate(RUNS):
        tri, dof, raw = ddpm_amplitudes(run), ddpm_of_amplitudes(run), of_amplitudes(run)
        for col, e in enumerate(ENERGIES):
            ax = axes[row, col]
            cal = lambda x: x / x.mean() * e
            s1, _ = spectrum(ax, cal(tri[e]), '#c2603c', 'DDPM + tri-exp')
            s2, _ = spectrum(ax, cal(dof[e]), '#5a9e4b', 'DDPM + Octopus OF (half B)')
            s3, _ = spectrum(ax, cal(raw[e]), '#2f6f9f', 'Octopus OF on raw data')
            print(f'run {run} {e} keV: tri-exp {s1:.4f}, DDPM+OF {s2:.4f}, raw OF {s3:.4f} keV')
            w = 6 * 1.4826 * np.median(np.abs(cal(tri[e]) - e))
            ax.set_xlim(e - w, e + w); ax.axvline(e, color='k', lw=0.6, ls=':')
            ax.set_title(f'run {run} ({lab} noise), {e} keV', fontsize=12)
            ax.set_xlabel('Energy, calibrated per peak (keV)'); ax.set_ylabel('Probability density (1/keV)')
            ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3); ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    fig.text(0.5, 0.005, 'Dashed: Gaussian fit within $\\pm$3 robust $\\sigma$ of the median. Unit-area histograms, own binning per method.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(OUT, dpi=150); print('Saved ->', OUT)


if __name__ == '__main__':
    main()
