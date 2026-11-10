"""
2026-10-08: three samplers at 2615 keV -- spectrum, resolution, PSD.

  cold      DDIM eta=0, x_T ~ N(0,I)        <- the production method until now
  zeroinit  DDIM eta=0, x_T = 0             <- the current production method
  nonoise   DDPM posterior mean (no stochastic term), x_T ~ N(0,I)

All three use the same 50 network evaluations per event and the same N=5000
events. Amplitudes come from the real tri-exp fit, calibrated with one shared
slope (k = 782.06 keV/V, the 5-point zero-intercept fit of the 1.382 mV
zero-init set) so peak positions as well as widths are comparable.
"""
import os, sys, glob
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py
from scipy.signal import welch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_gauss_fit import fit_gaussian, gaussian

K = 782.06
E_TRUE = 2615.0
F_SAMPLE = 1000.0
N_PSD = 200

EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/eval_efficiency_2615keV.h5'
NN   = '/mnt/d/storage/research/DDPM4bolometer/nonoise_out/efficiency_1p382mV/eval_efficiency_2615keV.h5'

C_NOISY, C_COLD, C_ZERO, C_NN = '#8a8780', '#2a78d6', '#eb6834', '#1baf7a'
METHODS = [
    ('cold',     'efficiency_1p382mV_cold', C_COLD, 'DDIM η=0, $x_T$ = white noise'),
    ('zeroinit', '1p382mV_zeroinit',        C_ZERO, 'DDIM η=0, $x_T$ = 0'),
    ('nonoise',  '1p382mV_nonoise',         C_NN,   '--no_noise, $x_T$ = white noise'),
]


def load_amps():
    out = {}
    with h5py.File(f'results/fit_results/1p382mV_zeroinit/fit_efficiency_2615keV.h5') as f:
        out['noisy'] = f['amp_noisy'][:].astype(float) * K
    for key, d, _, _ in METHODS:
        with h5py.File(f'results/fit_results/{d}/fit_efficiency_2615keV.h5') as f:
            out[key] = f['amp_denoised'][:].astype(float) * K
    return out


def mean_psd(w):
    f, p = welch(w, fs=F_SAMPLE, nperseg=min(4096, w.shape[1]), axis=-1)
    return f, p.mean(axis=0)


def main():
    amps = load_amps()
    fits = {k: fit_gaussian(v) for k, v in amps.items()}
    for k, g in fits.items():
        print(f'  {k:9s} mu={g["mu"]:9.2f} keV  FWHM={2.355*g["sigma"]:.4f} keV')

    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.26)

    order = [('noisy', C_NOISY, 'No denoising')] + \
            [(k, c, lab) for k, _, c, lab in METHODS]

    # (a) spectrum, full range
    ax = fig.add_subplot(gs[0, 0])
    # Range over every series, not just the noisy one -- the narrow peaks sit
    # outside the noisy distribution's central percentiles and were being clipped.
    allv = np.concatenate([amps[k] for k, _, _ in order])
    lo, hi = np.percentile(allv, [0.3, 99.7])
    pad = 0.06 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 110)
    for k, c, lab in order:
        ax.hist(amps[k], bins=bins, histtype='step', lw=1.7, color=c,
                label=f'{lab}  —  {2.355*fits[k]["sigma"]:.3f} keV')
    ax.set_xlim(bins[0], bins[-1])
    ax.annotate('all four peaks sit ~12 keV below the true 2615 keV —\n'
                'shared calibration non-linearity, common to every method',
                (0.02, 0.60), xycoords='axes fraction', fontsize=7.8,
                color='0.35', va='top')
    ax.set_title('(a)  Reconstructed energy spectrum, N = 5000', fontsize=11, loc='left')
    ax.set_xlabel('Reconstructed energy (keV)'); ax.set_ylabel('Counts')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    # (b) spectrum, zoomed on the two narrow peaks
    ax = fig.add_subplot(gs[0, 1])
    ctr = fits['zeroinit']['mu']
    bz = np.linspace(ctr - 1.6, ctr + 1.6, 110)
    for k, c, lab in order:
        if k == 'noisy':
            continue
        ax.hist(amps[k], bins=bz, histtype='step', lw=1.7, color=c, label=lab)
        g = fits[k]
        xs = np.linspace(bz[0], bz[-1], 400)
        n_in = ((amps[k] >= bz[0]) & (amps[k] < bz[-1])).sum()
        w = bz[1] - bz[0]
        ax.plot(xs, n_in * w * np.exp(-0.5*((xs-g['mu'])/g['sigma'])**2)
                / (g['sigma']*np.sqrt(2*np.pi)), color=c, lw=1, ls=':', alpha=0.8)
    ax.set_title('(b)  Same spectrum, zoomed  (dotted = Gaussian fit)',
                 fontsize=11, loc='left')
    ax.set_xlabel('Reconstructed energy (keV)'); ax.set_ylabel('Counts')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    # (c) resolution
    ax = fig.add_subplot(gs[1, 0])
    labs = [lab for _, _, lab in order]
    vals = [2.355*fits[k]['sigma'] for k, _, _ in order]
    cols = [c for _, c, _ in order]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.56)
    for yi, v in zip(y, vals):
        ax.text(v*1.04, yi, f'{v:.4f} keV', va='center', fontsize=9.5,
                color='#1d2328')
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlim(0, max(vals)*1.28)
    ax.set_xlabel('Energy resolution FWHM (keV)')
    ax.set_title('(c)  Resolution  (tri-exp fit, N = 5000)', fontsize=11, loc='left')
    ax.grid(alpha=0.25, axis='x')

    # (d) PSD
    ax = fig.add_subplot(gs[1, 1])
    with h5py.File(EVAL) as f:
        idx = np.arange(N_PSD)
        wv = {'noisy': f['waveforms_noisy'][idx].astype(np.float64),
              'cold': f['waveforms_denoised_cold'][idx].astype(np.float64),
              'zeroinit': f['waveforms_denoised_zeroinit'][idx].astype(np.float64)}
        tmpl = f['clean_template'][:].astype(np.float64)
    with h5py.File(NN) as f:
        wv['nonoise'] = f['waveforms_denoised_nonoise'][idx].astype(np.float64)

    fc, pc = welch(tmpl, fs=F_SAMPLE, nperseg=4096)
    ax.loglog(fc, pc, color='k', lw=1.3, ls='--', alpha=0.7, label='True (noiseless)')
    for k, c, lab in order:
        fq, p = mean_psd(wv[k])
        ax.loglog(fq, p, color=c, lw=1.5, label=lab)
    ax.axvline(100, color='0.6', ls=':', lw=1)
    ax.set_title('(d)  Power spectral density  (mean of 200 events)',
                 fontsize=11, loc='left')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel(r'PSD (V$^2$/Hz)')
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    ax.grid(alpha=0.25, which='both')

    fig.suptitle('2615 keV, 1.382 mV RMS baseline noise — three samplers, '
                 'same 50 network evaluations per event', fontsize=12.5, y=0.975)
    out = os.path.join(PROJ, 'results', 'plots', 'paper_zeroinit',
                       'three_samplers_2615keV.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out}')

    print('\nHF floor (mean PSD above 100 Hz, V^2/Hz):')
    for k, _, lab in order:
        fq, p = mean_psd(wv[k])
        print(f'  {k:9s} {p[fq>=100].mean():.4e}')
    print('THREE_SAMPLER_FIG_DONE')


if __name__ == '__main__':
    main()
