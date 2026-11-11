"""Slide figure for the evaluation ("validation") dataset: one fixed clean template per energy,
plus the same hold-out noise window rescaled to 1.382 mV and 7.5 mV RMS. Reads raw eval files only."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import h5py, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/'
OUT = 'results/plots/slides/validation_dataset_examples.png'
LEV = (('1p382mV', '1.382 mV'), ('7p5mV', '7.5 mV'))
ENS = (5, 50, 2615)
EV = 0
INK, MEAS = '#1d2328', '#1f62b3'
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif', 'axes.edgecolor': '#5a646d',
                     'xtick.color': '#5a646d', 'ytick.color': '#5a646d', 'axes.labelcolor': INK})
t = np.arange(10000) / 1000.0
fig, ax = plt.subplots(2, 3, figsize=(16, 5.6), sharex=True, gridspec_kw={'hspace': 0.28, 'wspace': 0.2})
for r, (lv, lvlab) in enumerate(LEV):
    for c, E in enumerate(ENS):
        with h5py.File(D + f'efficiency_{lv}/eval_efficiency_{E}keV.h5') as f:
            tmpl = f['clean_template'][:] * 1e3; noisy = f['waveforms_noisy'][EV] * 1e3
            snr = float(np.median(f['snr_db'][:]))
        a = ax[r, c]
        a.plot(t, noisy, lw=0.45, color=MEAS, alpha=0.75, label='noisy event')
        a.plot(t, tmpl, lw=1.8, color=INK, label='clean template')
        a.spines[['top', 'right']].set_visible(False); a.grid(alpha=0.25)
        a.set_title(f'{E} keV  ·  noise {lvlab}  ·  SNR {snr:.0f} dB', fontsize=13, loc='left')
        if E >= 1000:   # noise invisible at this scale: zoom on the tail
            ins = a.inset_axes([0.42, 0.38, 0.55, 0.55]); m = (t >= 5.0) & (t <= 5.2)
            rms = float(np.std(noisy[:1400] - tmpl[:1400]))
            ins.plot(t[m], noisy[m], lw=0.45, color=MEAS, alpha=0.75); ins.plot(t[m], tmpl[m], lw=1.8, color=INK)
            ins.set_ylim(tmpl[m].min() - 4 * rms, tmpl[m].max() + 4 * rms)
            ins.tick_params(labelsize=9); ins.set_title('zoom, 5.0–5.2 s', fontsize=10, loc='left')
            a.indicate_inset_zoom(ins, edgecolor='#8a939b')
        if c == 0: a.set_ylabel('amplitude (mV)')
        if r == 1: a.set_xlabel('time (s)')
ax[0, 0].legend(frameon=False, loc='upper right', fontsize=11)
fig.savefig(OUT, dpi=160, bbox_inches='tight', facecolor='white')
print('saved', OUT)
