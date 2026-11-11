"""Overview figure of one OFcompare run: a 5 min stretch, one 15 keV window, quiet-window PSD."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, json, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.dirname(__file__)); os.chdir(PROJ)
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import generate_of_stream as GEN
from verify_of_stream import read_bin, volts

ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True); a = ap.parse_args()
root = os.path.join(GEN.TRUTH, f'run{a.run}'); raw_dir = os.path.join(GEN.STAGE, f'run{a.run}', 'raw')
info = json.load(open(os.path.join(root, 'truth', 'run_info.json')))
ev = np.loadtxt(os.path.join(root, 'truth', 'events.csv'), delimiter=',', skiprows=1)
nz = np.loadtxt(os.path.join(root, 'truth', 'noise_windows.csv'), delimiter=',', skiprows=1)
_, _, _, d = read_bin(os.path.join(raw_dir, info['partials'][0]))
x = volts(d, 10.0)
t0 = 3600 * 1000; L = 300 * 1000                          # 5 min starting 1 h into the run
seg = x[t0:t0 + L]; tt = np.arange(t0, t0 + L) / 1000 / 60

fig = plt.figure(figsize=(12, 7.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.22)
ax = fig.add_subplot(gs[0, :])
ax.plot(tt, seg * 1e3, lw=0.4, color='#1F5FA8')
for i, e, o in ev[:, :3]:
    if t0 <= o < t0 + L:
        ax.axvspan((o - 1500) / 6e4, (o + 8500) / 6e4, color='#D2622A', alpha=0.08, lw=0)
        ax.text(o / 6e4, seg.max() * 1e3 * 1.02, f'{int(e)}', fontsize=7, ha='center', color='#D2622A')
for j, t in nz[:, :2]:
    if t0 <= t < t0 + L:
        ax.axvspan((t - 1500) / 6e4, (t + 8500) / 6e4, color='#11845A', alpha=0.15, lw=0)
ax.set_yscale('symlog', linthresh=20 * info['rms'] * 1e3); ax.set_xlabel('time in run [min]'); ax.set_ylabel('V [mV]')
ax.set_title(f'run {a.run} ({info["level"]}, {info["noise"]} noise): 5 min of partial 000. '
             'Orange: 10 s event windows (label = keV); green: quiet noise windows', loc='left', fontsize=10)
# one 15 keV window
ax = fig.add_subplot(gs[1, 0])
k = np.nonzero((ev[:, 1] == 15) & (ev[:, 4] == 0))[0][0]; o = int(ev[k, 2])
w = x[o - 1500:o + 8500]
import h5py
with h5py.File(os.path.join(root, 'templates', 'templates.h5')) as f: tm = f['15keV'][:10000]
ax.plot(np.arange(10000) / 1000, w * 1e3, lw=0.4, color='#1F5FA8', label='stream window')
ax.plot(np.arange(10000) / 1000, tm * 1e3, lw=1.0, color='k', ls='--', label='injected 15 keV template')
ax.set_xlabel('time in window [s] (onset at 1.5 s)'); ax.set_ylabel('V [mV]'); ax.legend(frameon=False, fontsize=8)
ax.set_title(f'event {int(ev[k, 0])}, 15 keV', loc='left', fontsize=10)
# PSD of quiet windows
ax = fig.add_subplot(gs[1, 1])
ws = np.array([x[int(t) - 1500:int(t) + 8500] for t in nz[:, 1] if nz[nz[:, 1] == t, 3][0] == 0 and t + 8500 < len(x)])
hw = np.hanning(10000); X = np.fft.rfft((ws - ws.mean(1, keepdims=True)) * hw, axis=1)
psd = 2 * np.mean(np.abs(X) ** 2, 0) / (1000 * np.sum(hw ** 2)); f = np.fft.rfftfreq(10000, 1e-3)
ax.loglog(f[1:], np.sqrt(psd[1:]) * 1e6, lw=0.6, color='#11845A')
ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('noise ASD [µV/√Hz]')
ax.set_title(f'{len(ws)} quiet windows of partial 000, RMS {ws.std(1).mean() * 1e3:.3f} mV', loc='left', fontsize=10)
for a_ in fig.axes: a_.grid(alpha=0.25)
out = os.path.join(root, f'overview_run{a.run}.png'); fig.savefig(out, dpi=120, bbox_inches='tight')
os.makedirs(os.path.join(PROJ, 'results', 'plots', 'OFcompare'), exist_ok=True)
fig.savefig(os.path.join(PROJ, 'results', 'plots', 'OFcompare', f'overview_run{a.run}.png'), dpi=120, bbox_inches='tight')
print(out)
