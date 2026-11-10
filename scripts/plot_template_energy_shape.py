"""Pulse shape vs energy for the eval-set clean templates (one detector, seed 1, all energies).

Each template is normalised to its own maximum. Panels: overlay (full window and zoom on the rise),
difference to the 2615 keV template, and shape numbers vs energy: rise time 10-90 %, peak time,
decay time 90-30 %, area / peak, Pearson r and relative residual RMS to the 2615 keV template.

Usage: python scripts/plot_template_energy_shape.py [--level 7p5mV]
Output: results/plots/template_shape/template_shape_vs_energy.png, _numbers.png and .json
"""
import os, sys, glob, json, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/mnt/d/storage/research/DDPM4bolometer/eval_data_OFcompare'
FS, PRE, REF = 1000.0, 1500, 2615


def crossing(y, level, start, rising):
    """First sample index (linearly interpolated) after start where y crosses level."""
    seg = y[start:]
    k = np.flatnonzero(seg >= level if rising else seg <= level)[0]
    i = start + k
    return i - 1 + (level - y[i - 1]) / (y[i] - y[i - 1])


def shape_numbers(y):
    ip = int(np.argmax(y))
    t10 = crossing(y, 0.1, PRE, True); t90 = crossing(y, 0.9, PRE, True)
    d90 = crossing(y, 0.9, ip, False); d30 = crossing(y, 0.3, ip, False)
    return dict(rise_10_90_ms=(t90 - t10) / FS * 1e3, peak_time_ms=(ip - PRE) / FS * 1e3,
                decay_90_30_ms=(d30 - d90) / FS * 1e3, area_over_peak_ms=float(y[PRE:].sum() / FS * 1e3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--level', default='7p5mV'); a = ap.parse_args()
    T = {}
    for fn in glob.glob(f'{D}/{a.level}/eval_efficiency_*keV.h5'):
        with h5py.File(fn) as f:
            T[int(round(f.attrs['energy_kev']))] = f['clean_template'][:].astype(float)
    E = sorted(T); amp = {e: T[e].max() for e in E}; N = {e: T[e] / amp[e] for e in E}
    ref = N[REF]
    rows = {}
    for e in E:
        r = np.corrcoef(N[e][PRE:], ref[PRE:])[0, 1]
        d = N[e] - ref
        rows[e] = dict(peak_V=float(amp[e]), peak_per_keV_uV=float(amp[e] / e * 1e6), **shape_numbers(N[e]),
                       pearson_r=float(r), one_minus_r=float(1 - r),
                       max_abs_diff=float(np.abs(d).max()), rms_diff=float(np.sqrt((d[PRE:] ** 2).mean())))
    t = (np.arange(len(ref)) - PRE) / FS
    cmap = plt.get_cmap('viridis'); col = {e: cmap(i / (len(E) - 1)) for i, e in enumerate(E)}

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for e in E:
        lw = 1.6 if e in (1, REF) else 0.9
        ax[0, 0].plot(t, N[e], color=col[e], lw=lw, label=f'{e} keV')
        ax[0, 1].plot(t * 1e3, N[e], color=col[e], lw=lw)
        ax[1, 0].plot(t, (N[e] - ref) * 100, color=col[e], lw=lw)
        ax[1, 1].plot(t * 1e3, (N[e] - ref) * 100, color=col[e], lw=lw)
    ax[0, 0].set(xlim=(-0.5, 8.5), xlabel='Time from onset (s)', ylabel='Normalised amplitude', title='All energies, normalised to peak')
    ax[0, 1].set(xlim=(-20, 300), xlabel='Time from onset (ms)', ylabel='Normalised amplitude', title='Zoom: rise and peak')
    ax[1, 0].set(xlim=(-0.5, 8.5), xlabel='Time from onset (s)', ylabel='Difference to 2615 keV (% of peak)', title='Normalised template minus 2615 keV template')
    ax[1, 1].set(xlim=(-20, 300), xlabel='Time from onset (ms)', ylabel='Difference to 2615 keV (% of peak)', title='Zoom: difference on the rise')
    for x in ax.flat: x.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2, title='Energy')
    fig.suptitle(f'Eval-set clean templates ({a.level} set, one detector), shape vs energy', fontsize=13)
    fig.tight_layout()
    out = f'{PROJ}/results/plots/template_shape/template_shape_vs_energy'
    fig.savefig(out + '.png', dpi=150); plt.close(fig)

    keys = [('rise_10_90_ms', 'Rise time 10-90 % (ms)'), ('peak_time_ms', 'Peak time after onset (ms)'),
            ('decay_90_30_ms', 'Decay time 90-30 % (ms)'), ('area_over_peak_ms', 'Area / peak (ms)'),
            ('one_minus_r', '1 - Pearson r vs 2615 keV'), ('peak_per_keV_uV', 'Peak per keV (µV/keV)')]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    Ea = np.array(E, float)
    for x, (k, lab) in zip(ax.flat, keys):
        v = np.array([rows[e][k] for e in E])
        m = v > 0 if k == 'one_minus_r' else np.ones(len(v), bool)   # the reference itself has 1 - r = 0
        x.plot(Ea[m], v[m], 'o-', color='#2f6f9f', ms=5)
        x.set_xscale('log'); x.set_xlabel('Energy (keV)'); x.set_ylabel(lab); x.grid(alpha=0.3, which='both')
        if k == 'one_minus_r':
            x.set_yscale('log'); x.axvline(REF, color='k', ls=':', lw=0.8)
            x.text(REF, x.get_ylim()[0] * 1.5, ' reference', fontsize=8)
    fig.suptitle(f'Shape numbers of the normalised templates vs energy ({a.level} set)', fontsize=13)
    fig.tight_layout(); fig.savefig(out + '_numbers.png', dpi=150); plt.close(fig)
    json.dump(dict(level=a.level, reference_keV=REF, rows=rows), open(out + '.json', 'w'), indent=1)
    print(f"{'E':>5} {'peak uV/keV':>11} {'rise':>7} {'tpeak':>6} {'decay':>7} {'area/pk':>8} {'1-r':>9} {'max|d|%':>8} {'rms d%':>7}")
    for e in E:
        r = rows[e]
        print(f"{e:5d} {r['peak_per_keV_uV']:11.3f} {r['rise_10_90_ms']:7.2f} {r['peak_time_ms']:6.0f} {r['decay_90_30_ms']:7.1f} "
              f"{r['area_over_peak_ms']:8.1f} {r['one_minus_r']:9.2e} {r['max_abs_diff']*100:8.3f} {r['rms_diff']*100:7.3f}")
    print('Saved ->', out + '.png', out + '_numbers.png')


if __name__ == '__main__':
    main()
