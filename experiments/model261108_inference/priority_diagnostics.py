"""Read-only diagnostics of the completed priority points."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'code'), str(HERE / 'plot_deps'),
               r'D:\storage\research\DDPM4bolometer\_development\train_clean_triexp\deps']
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read_json, save_json
from src.basics.fit import fit_pulse

DATA = Path(r'D:\storage\research\DDPM4bolometer')
NOTES = Path(r'D:\Obsidian\note\课题\DDPM4Bolometer')
OUT = NOTES / 'results/plots/model261108_nonoise'


def analyze(energy):
    name = f'eval_efficiency_{energy}keV.h5'
    relative = Path('efficiency_1p382mV') / name
    sources = {'Old model': (DATA / 'nonoise_out' / relative, 'waveforms_denoised_nonoise',
                            NOTES / f'results/fit_results/1p382mV_nonoise/fit_efficiency_{energy}keV.h5'),
               'New model': (DATA / 'model261108_inference' / relative, 'waveforms_denoised',
                            NOTES / f'results/fit_results/model261108_nonoise/efficiency_1p382mV/fit_efficiency_{energy}keV.h5')}
    with h5py.File(DATA / 'eval_data_holdout' / relative) as raw:
        clean = raw['clean_template'][:]
    summary = read_json(OUT / f'comparison_{energy}keV.json')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    metrics = {}
    clean_fit = fit_pulse(clean, n_pulses=1, fs=1000.)
    clean_fit_curve = clean_fit.model_fn(np.arange(len(clean)) / 1000., *clean_fit.params)
    clean_fit_rms = float(np.sqrt(np.mean((clean_fit_curve - clean)**2)))
    colors = {'Old model': '#2374ce', 'New model': '#e86830'}
    sample_amplitudes = {}
    for label, (wave_path, field, fit_path) in sources.items():
        with h5py.File(wave_path) as data, h5py.File(fit_path) as fit:
            wave = data[field][:200].astype(float)
            chi2 = fit['chi2_denoised'][:]
            sigma = np.std(wave[:, :500], axis=1)
            fit_rms = np.sqrt(chi2[:200] * (10000 - 7) / 10000) * sigma
            residual = wave - clean
            sample_amplitudes[label] = fit['amp_denoised'][:] * summary['calibration_keV_per_V']
            metrics[label] = {'fit_success_count': int(fit['success_denoised'][:].sum()),
                              'quality_gate_count': int(fit['converged_denoised'][:].sum()),
                              'chi2_median_all': float(np.median(chi2)),
                              'baseline_std_median_V_first200': float(np.median(sigma)),
                              'fit_residual_rms_median_V_first200': float(np.median(fit_rms)),
                              'clean_residual_rms_median_V_first200': float(np.median(np.sqrt(np.mean(residual**2, axis=1))))}
            t = np.arange(10000) / 1000.
            axes[0, 0].plot(t, residual[0] * 1e6, color=colors[label], label=label, lw=.8)
            axes[0, 1].plot(t, residual.mean(0) * 1e6, color=colors[label], label=label, lw=1)
            axes[1, 1].hist(chi2[np.isfinite(chi2)], bins=np.geomspace(.05, 20000, 120),
                            histtype='step', color=colors[label], label=label)
    all_values = np.concatenate(list(sample_amplitudes.values()))
    low, high = np.percentile(all_values, [.2, 99.8])
    bins = np.linspace(low, high, 101)
    x = (bins[1:] + bins[:-1]) / 2
    for label, amps in sample_amplitudes.items():
        axes[1, 0].hist(amps, bins=bins, histtype='step', color=colors[label], label=label)
        g = summary['gaussian_fits'][label]
        density = np.exp(-.5 * ((x-g['mu'])/g['sigma'])**2) / (g['sigma']*np.sqrt(2*np.pi))
        axes[1, 0].plot(x, density * np.isfinite(amps).sum() * np.diff(bins)[0], ':', color=colors[label])
    axes[1, 1].axvline(10, color='black', ls='--', lw=1, label='Existing gate χ²/ndf = 10')
    axes[1, 1].set_xscale('log')
    if energy == 2615:
        axes[1, 1].set_xlim(.5, 100)
    titles = ['Residual to clean, event 0', 'Mean residual to clean, first 200 events',
              'Energy spectrum zoom; Gaussian shape overlay', 'Tri-exponential fit quality, 5000 events']
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[0]:
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Residual (µV)')
    axes[1, 0].set_xlabel('Energy (keV), shared historical calibration')
    axes[1, 0].set_ylabel('Counts')
    axes[1, 1].set_xlabel('χ²/ndf, σ estimated from denoised baseline')
    axes[1, 1].set_ylabel('Counts')
    fig.suptitle(f'{energy} keV, 1.382 mV — both models use Gaussian start + no_noise\n'
                 'Energy spectra use all successful fits; existing quality gate is reported separately', fontsize=12)
    fig.savefig(OUT / f'details_{energy}keV.png', dpi=150)
    plt.close(fig)
    metrics['clean_template'] = {'tri_exp_fit_success': bool(clean_fit.success),
                                 'tri_exp_fit_residual_rms_V': clean_fit_rms}
    save_json(OUT / f'diagnostics_{energy}keV.json', metrics)
    print(energy, metrics)


for energy in (2615, 15):
    analyze(energy)
