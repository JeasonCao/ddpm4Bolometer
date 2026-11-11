"""Peak-height linearity of the eval-set clean templates, drawn like plot_reconstruction_quality.py.

Calibration as in plot_reconstruction_quality.py: zero-intercept E = k * peak, k fit from the
gamma-line-like points 239/511/727/1461/2615 keV (3034 keV excluded), then applied to all 17
energies. Clean templates have no noise, so the fit is unweighted and there are no error bars.
Main panel: reconstructed vs true energy (log-log) with y = x; inset: bias = (reco - true) / true.

Usage: python scripts/plot_template_peak_linearity.py
Output: results/plots/template_shape/template_peak_linearity.png and .json
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, glob, json
PROJ = str(_REPO_ROOT)
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

D = '/mnt/d/storage/research/DDPM4bolometer/eval_data_OFcompare/7p5mV'
CALIB = [239.0, 511.0, 727.0, 1461.0, 2615.0]

pk = {}
for fn in glob.glob(f'{D}/eval_efficiency_*keV.h5'):
    with h5py.File(fn) as f:
        pk[float(f.attrs['energy_kev'])] = float(f['clean_template'][:].max())
E = np.array(sorted(pk)); A = np.array([pk[e] for e in E])          # peak height, V
c = np.isin(E, CALIB)
k = (A[c] @ E[c]) / (A[c] @ A[c])                                    # keV per V, zero intercept
R = k * A; bias = (R - E) / E * 100

fig, ax = plt.subplots(figsize=(8, 6.5))
lim = (E.min() * 0.6, E.max() * 1.6)                                 # as _log_xlim in plot_reconstruction_quality.py
x = np.logspace(np.log10(E.min() * 0.7), np.log10(E.max() * 1.4), 300)
ax.errorbar(E, R, fmt='o', ms=4, color='tomato', label='Clean template, peak height')
ax.plot(x, x, color='black', lw=1.2, ls=':', label='Ideal (y = x)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('True energy (keV)', fontsize=11); ax.set_ylabel('Reconstructed energy (keV)', fontsize=11)
ax.set_title('Reconstruction Quality — Peak height of simulated clean templates (no noise)\n'
             'zero-intercept calibration on 239/511/727/1461/2615 keV', fontsize=10.5)
ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.3, which='both')

ai = inset_axes(ax, width='40%', height='32%', loc='lower right', bbox_to_anchor=(-0.02, 0.06, 1, 1), bbox_transform=ax.transAxes)
ai.errorbar(E, bias, fmt='o-', ms=3, lw=1.2, color='tomato', label='Peak height')
ai.axhline(0, color='black', ls='--', lw=0.9)
lo, hi = bias.min(), bias.max(); pad = (hi - lo) * 0.15                # as _pad_ylim
ai.set_ylim(lo - pad, hi + pad)
ai.set_xscale('log'); ai.set_xlim(*lim)
ai.set_xlabel('True energy (keV)', fontsize=7.5); ai.set_ylabel('Bias (%)', fontsize=7.5); ai.tick_params(labelsize=7)
ai.legend(fontsize=7, loc='upper right'); ai.grid(alpha=0.3, which='both'); ai.set_title('Bias = (reco − true) / true', fontsize=7.5)

out = f'{PROJ}/results/plots/template_shape/template_peak_linearity'
fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
json.dump(dict(calibration_keV=CALIB, k_keV_per_V=k, uV_per_keV=1e6 / k,
               rows=[dict(E_keV=e, peak_V=a, reco_keV=r, bias_pct=b) for e, a, r, b in zip(E, A, R, bias)]),
          open(out + '.json', 'w'), indent=1)
print(f'k = {k:.3f} keV/V ({1e6 / k:.2f} uV/keV)')
for e, r, b in zip(E, R, bias):
    print(f'{e:6g} keV  reco {r:9.3f} keV  bias {b:+.3f}%')
print('Saved ->', out + '.png')
