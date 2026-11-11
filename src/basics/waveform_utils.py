"""Shared waveform plotting utilities."""

import numpy as np
from src.basics.fit import fit_pulse

F_SAMPLE = 1000.0


def plot_waveform_pair(ax_n, ax_d, wf_noisy, wf_denoised, snr_db,
                       fs=F_SAMPLE, event_label='', clean_template=None):
    """Draw one before/after row into ax_n (noisy) and ax_d (denoised).

    Each panel: raw waveform + tri-exp fit (A, chi²/ndf annotated).
    clean_template: noiseless ground-truth pulse; overlaid when provided so you
    can see how close denoising got to the true shape.
    """
    t = np.arange(len(wf_noisy)) / fs

    for ax, wf, color in [(ax_n, wf_noisy, 'steelblue'), (ax_d, wf_denoised, 'tomato')]:
        ax.plot(t, wf, color=color, lw=0.8, alpha=0.65, label='Waveform')

        if clean_template is not None:
            t_tmpl = np.arange(len(clean_template)) / fs
            ax.plot(t_tmpl, clean_template, 'k--', lw=1.0, alpha=0.45,
                    label='True pulse (no noise)')

        fr = fit_pulse(wf, n_pulses=1, fs=fs)
        if fr.success and fr.model_fn is not None:
            model = fr.model_fn(t, *fr.params)
            amp  = fr.peak_amps[0] if fr.peak_amps else float('nan')
            chi2 = fr.chi2_per_ndf
            ax.plot(t, model, color='mediumpurple', lw=1.5,
                    label=f'Fit  A={amp:.4f} V  χ²/ndf={chi2:.2f}')

        annot = f'{event_label}  SNR={snr_db:.1f} dB' if event_label else f'SNR={snr_db:.1f} dB'
        ax.text(0.02, 0.95, annot, transform=ax.transAxes, fontsize=8, va='top')
        ax.set_ylabel('Amplitude (V)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
