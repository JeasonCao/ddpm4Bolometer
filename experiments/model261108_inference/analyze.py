"""Priority-point fits and plots using the existing project fit implementations."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'code'), str(HERE / 'plot_deps'),
               r'D:\storage\research\DDPM4bolometer\_development\train_clean_triexp\deps']
import h5py
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_fitting import fit_dataset
from scripts.run_gauss_fit import fit_gaussian, process_file
from common import digest, read_json, save_json

DATA = Path(r'D:\storage\research\DDPM4bolometer')
NOTES = Path(r'D:\Obsidian\note\课题\DDPM4Bolometer')
RUN = DATA / 'model261108_inference'
K = 782.06  # Shared historical slope used by plot_three_samplers_2615.py.


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def analyze(energy, preview=False):
    name = f'eval_efficiency_{energy}keV.h5'
    relative = Path('efficiency_1p382mV') / name
    source = DATA / 'eval_data_holdout' / relative
    new = RUN / relative
    old = DATA / 'nonoise_out' / relative
    fit_dir = NOTES / 'results/fit_results/model261108_nonoise/efficiency_1p382mV'
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_path = fit_dir / f'fit_efficiency_{energy}keV.h5'
    old_fit = NOTES / f'results/fit_results/1p382mV_nonoise/fit_efficiency_{energy}keV.h5'
    if preview:
        new = old
        fit_path = old_fit
    else:
        with h5py.File(new) as data:
            assert data.attrs['source_sha256'] == digest(source)
            assert np.array_equal(data['source_row'][:], np.arange(5000))
        if not fit_path.exists():
            # External links avoid duplicating noisy waveforms for the existing fitter.
            linked = RUN / f'fit_input_{energy}.h5'
            with h5py.File(source) as raw, h5py.File(linked, 'w') as staged:
                staged.attrs.update(dict(raw.attrs))
                for key in raw:
                    staged[key] = h5py.ExternalLink(str(source), '/' + key)
                staged['waveforms_denoised'] = h5py.ExternalLink(str(new), '/waveforms_denoised')
            temporary = fit_path.with_suffix('.h5.partial')
            fit_dataset(str(linked), str(temporary), workers=4, chi2_max=10., amp_min=1e-5,
                        amp_max=9.9, amp_dev=.20)
            process_file(str(temporary))
            with h5py.File(temporary, 'a') as fit:
                fit.attrs['source_sha256'] = digest(source)
                fit.attrs['denoised_sha256'] = digest(new)
            temporary.replace(fit_path)
        with h5py.File(fit_path) as fit:
            assert fit.attrs['denoised_sha256'] == digest(new)
            assert len(fit['amp_denoised']) == 5000
    with h5py.File(source) as raw, h5py.File(old) as prior, h5py.File(new) as fresh:
        clean = raw['clean_template'][:]
        waves = {'No denoising': raw['waveforms_noisy'][:200],
                 'Old model': prior['waveforms_denoised_nonoise'][:200],
                 'New model': fresh['waveforms_denoised_nonoise' if preview else 'waveforms_denoised'][:200]}
    with h5py.File(old_fit) as prior, h5py.File(fit_path) as fresh:
        amplitudes = {'No denoising': prior['amp_noisy'][:] * K,
                      'Old model': prior['amp_denoised'][:] * K,
                      'New model': fresh['amp_denoised'][:] * K}
        success = {label: int(fresh[f'success_{label}'][:].sum()) for label in ('noisy', 'denoised')}
    fits = {key: fit_gaussian(value) for key, value in amplitudes.items()}
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    colors = {'No denoising': '#8a8780', 'Old model': '#2374ce', 'New model': '#e86830'}
    all_amplitudes = np.concatenate(list(amplitudes.values()))
    low, high = np.percentile(all_amplitudes[np.isfinite(all_amplitudes)], [.3, 99.7])
    bins = np.linspace(low - .06 * (high - low), high + .06 * (high - low), 110)
    t = np.arange(len(clean)) / 1000.
    axes[0, 0].plot(t, clean, 'k--', label='Clean', lw=1)
    for label, wave in waves.items():
        color = colors[label]
        axes[0, 0].plot(t, wave[0], color=color, lw=.8, label=label)
        axes[0, 1].plot(t, (wave[0] - clean) * 1000, color=color, lw=.7, label=label)
        f, psd = welch(wave, fs=1000, nperseg=4096, axis=-1)
        _, residual_psd = welch(wave - clean, fs=1000, nperseg=4096, axis=-1)
        axes[1, 0].loglog(f[1:], psd.mean(0)[1:], color=color, label=label)
        axes[1, 1].loglog(f[1:], residual_psd.mean(0)[1:], color=color, label=label)
        values = amplitudes[label]
        values = values[np.isfinite(values)]
        axes[2, 0].hist(values, bins=bins, histtype='step', color=color, label=label)
    f, psd = welch(clean, fs=1000, nperseg=4096)
    axes[1, 0].loglog(f[1:], psd[1:], 'k--', label='Clean')
    labels = list(fits)
    widths = [2.355 * fits[k]['sigma'] for k in labels]
    axes[2, 1].barh(labels, widths, color=[colors[k] for k in labels])
    for i, width in enumerate(widths):
        axes[2, 1].text(width, i, f'  {width:.4f} keV', va='center')
    titles = ['Waveform, event 0', 'Residual to clean, event 0',
              'Mean waveform PSD, first 200 events', 'Mean residual PSD, first 200 events',
              'Tri-exponential reconstructed energy', 'Gaussian FWHM, finite fit amplitudes']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.grid(alpha=.2)
    for ax in axes.flat:
        handles, names = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 1].set_ylabel('Residual (mV)')
    for ax in axes[1]:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V²/Hz)')
    axes[2, 0].set_xlabel('Energy (keV), shared historical calibration')
    axes[2, 0].set_ylabel('Counts')
    axes[2, 1].set_xlabel('FWHM (keV)')
    axes[2, 1].set_xlim(0, max(widths) * 1.35)
    prefix = 'PIPELINE PREVIEW — old data in both slots — ' if preview else ''
    fig.suptitle(f'{prefix}{energy} keV, 1.382 mV; shared slope {K} keV/V', fontsize=13)
    destination = HERE / 'validation' if preview else NOTES / 'results/plots/model261108_nonoise'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f'comparison_{energy}keV.png', dpi=150)
    plt.close(fig)
    historical = read_json(NOTES / 'results/plots/OFcompare/full_run999998_resolution_vs_energy.json')
    report = {'energy_kev': energy, 'calibration_keV_per_V': K, 'fit_success': success,
              'gaussian_fits': fits, 'source_file': str(source), 'fit_file': str(fit_path),
              'historical_OF': historical['peaks'].get(str(energy), {}).get('OF'),
              'historical_OF_note': 'Run 999998 continuous-stream data; different events and calibration. Reference only.',
              'preview': preview}
    save_json(destination / f'comparison_{energy}keV.json', finite_json(report))
    if not preview:
        save_json(RUN / f'analysis_{energy}_complete.json', {'energy_kev': energy,
                  'gaussian_converged': fits['New model']['converged'], 'fit_success': success,
                  'plot': str(destination / f'comparison_{energy}keV.png')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy', type=int, choices=[15, 2615], required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    analyze(args.energy, args.preview)
