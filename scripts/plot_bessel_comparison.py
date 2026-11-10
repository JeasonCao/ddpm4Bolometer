"""Raw (noisy) vs raw + Bessel vs DDPM zero-init + Bessel vs clean template, time and frequency domain.

Bessel = the simulator readout chain (scripts/apply_bessel_to_ddpm.py): raw + Bessel from
eval_data_bessel/bessel_sim_noisy, DDPM + Bessel from eval_data_bessel/bessel_sim.
Per noise level and energy, one figure with 2 x 2 panels:
  time domain, full window and zoom on the rise (one event)
  PSD averaged over N events (Welch, 1 kHz)
  PSD of the residual, averaged over N events. The raw window is compared with the clean template;
  the two filtered ones with the clean template passed through the same chain, because the second
  filter changes the pulse shape itself (delay ~7 ms, slower rise) and that deterministic change
  would otherwise dominate the residual.

Usage: python scripts/plot_bessel_comparison.py --level 7p5mV --energies 15 2615
Output: results/plots/bessel/bessel_comparison_{level}_{E}keV.png
"""
import os, sys, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np, h5py
from scipy.signal import welch, resample_poly
from src.basics.filters import apply_bessel_decimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/mnt/d/storage/research/DDPM4bolometer'
FS, PRE = 1000.0, 1500
LAB = {'1p382mV': '1.382 mV', '7p5mV': '7.5 mV'}
SER = [('raw', 'Raw (no denoising)', '#9a9a9a', 0.6), ('rawB', 'Raw + Bessel', '#2a9d8f', 0.9),
       ('ddB', 'DDPM + Bessel', '#c2603c', 1.1)]


def chain(x):
    """Same chain as scripts/apply_bessel_to_ddpm.py."""
    return apply_bessel_decimate(resample_poly(x, 10, 1), 10000.0, 1000.0, 120.0, 6)[:x.size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='7p5mV'); ap.add_argument('--energies', type=int, nargs='+', default=[15, 2615])
    ap.add_argument('--event', type=int, default=0); ap.add_argument('--n_psd', type=int, default=500)
    a = ap.parse_args()
    for E in a.energies:
        fn = f'eval_efficiency_{E}keV.h5'; n = a.n_psd
        with h5py.File(f'{D}/eval_data_bessel/bessel_sim/efficiency_{a.level}/{fn}') as f:
            W = dict(raw=f['waveforms_noisy'][:n].astype(float), ddB=f['waveforms_denoised_zeroinit_bessel_sim'][:n].astype(float))
            clean = f['clean_template'][:].astype(float)
        with h5py.File(f'{D}/eval_data_bessel/bessel_sim_noisy/efficiency_{a.level}/{fn}') as f:
            W['rawB'] = f['waveforms_noisy_bessel_sim'][:n].astype(float)
        cleanB = chain(clean)
        ref = dict(raw=clean, rawB=cleanB, ddB=cleanB)
        t = (np.arange(clean.size) - PRE) / FS; i = a.event
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        for x, (lo, hi), unit in ((ax[0, 0], (-1, 8.5), 1), (ax[0, 1], (-0.03, 0.4), 1e3)):
            for k, lab, c, lw in SER:
                x.plot(t * unit, W[k][i] * 1e3, color=c, lw=lw, label=lab)
            x.plot(t * unit, clean * 1e3, color='k', lw=0.9, ls='--', label='Clean template')
            x.set_xlim(lo * unit, hi * unit); x.set_ylabel('Amplitude (mV)'); x.grid(alpha=0.3)
        ax[0, 0].set_xlabel('Time from onset (s)'); ax[0, 1].set_xlabel('Time from onset (ms)')
        ax[0, 0].set_title(f'Event {i}, full window'); ax[0, 1].set_title(f'Event {i}, zoom on the rise')
        ax[0, 0].legend(fontsize=9)

        def psd(w):
            f, p = welch(w, fs=FS, nperseg=2000, axis=-1); return f[1:], np.atleast_2d(p).mean(0)[1:]
        for k, lab, c, lw in SER:
            ax[1, 0].loglog(*psd(W[k]), color=c, lw=1.0, label=lab)
            ax[1, 1].loglog(*psd(W[k] - ref[k]), color=c, lw=1.0,
                            label={'raw': 'Raw − clean', 'rawB': 'Raw + Bessel − (clean + Bessel)', 'ddB': 'DDPM + Bessel − (clean + Bessel)'}[k])
        ax[1, 0].loglog(*psd(clean), color='k', lw=0.9, ls='--', label='Clean template')
        for x in ax[1]:
            x.axvline(69.35, color='k', lw=0.6, ls=':'); x.set_xlabel('Frequency (Hz)'); x.set_ylabel('PSD (V$^2$/Hz)')
            x.grid(alpha=0.3, which='both'); x.legend(fontsize=9)
        ax[1, 0].set_title(f'Mean PSD of the waveforms ({n} events); dotted: Bessel −3 dB, 69.35 Hz')
        ax[1, 1].set_title(f'Mean PSD of the residual ({n} events)')
        fig.suptitle(f'{LAB[a.level]} noise, {E} keV: raw vs raw + Bessel vs DDPM + Bessel', fontsize=13)
        fig.tight_layout()
        out = f'{PROJ}/results/plots/bessel/bessel_comparison_{a.level}_{E}keV.png'
        os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
        rms = {k: np.sqrt(((W[k] - ref[k]) ** 2).mean()) * 1e6 for k in W}
        print(f"{a.level} {E} keV: residual RMS (uV) raw {rms['raw']:.1f}  raw+B {rms['rawB']:.1f}  DDPM+B {rms['ddB']:.2f}")


if __name__ == '__main__':
    main()
