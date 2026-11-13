"""Calibrate and compare raw, old zero-init, new no_noise and historical OF.

Reuses canonical Gaussian fitting, the established five-point zero-intercept
calibration formula, and the historical OF trigger-matching/selection routine.
All primary amplitudes use successful finite fits (no tri-exp chi2 gate).
Conventional gated statistics and the historical OF convention are retained
as diagnostics. No input file or shared scientific default is modified.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_gauss_fit import fit_gaussian
from src.basics.plot_utils import CALIB_ENERGIES_KEV
from experiments.of_comparison import plot_full_of_vs_ddpm as historical_of

METHODS = ('raw', 'old_zeroinit', 'new_nonoise', 'OF')
LABEL = {'raw': 'No denoising + tri-exp', 'old_zeroinit': 'Old model: zero-init + tri-exp',
         'new_nonoise': 'New model: no_noise + tri-exp', 'OF': 'Historical Octopus OF (different events)'}
COLOR = {'raw': '#888780', 'old_zeroinit': '#2274c9', 'new_nonoise': '#e46c35', 'OF': '#139b73'}
LEVELS = {'1p382mV': (999998, '1.382 mV'), '7p5mV': (999999, '7.5 mV')}
FWHM = 2.355


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def summarize(amps, n_total, success_count, gate=None):
    values = np.asarray(amps, dtype=float)
    values = values[np.isfinite(values)]
    result = {'gaussian': fit_gaussian(values), 'n_total': n_total,
              'n_success': success_count, 'n_used': len(values),
              'std_amplitude': float(np.std(values)),
              'mean_amplitude': float(np.mean(values))}
    if gate is not None:
        result['n_quality_gate'] = len(gate)
        result['gated_gaussian'] = fit_gaussian(np.asarray(gate, dtype=float))
    # Same robust-core definition used by the historical full OF comparison,
    # retained for diagnosing sensitivity to Gaussian fitting range/tails.
    if len(values) >= 30 and np.std(values) > 0:
        sigma, error, _ = historical_of.width(values)
        result['robust_core_sigma_amplitude'] = float(sigma)
        result['robust_core_sigma_err_amplitude'] = float(error)
    return result


def calibration(rows, method):
    points = []
    for energy in CALIB_ENERGIES_KEV:
        g = rows[int(energy)][method]['gaussian']
        assert g['converged'] and np.isfinite(g['mu_err']) and g['mu_err'] > 0, (energy, method, g)
        points.append((float(energy), g['mu'], g['mu_err']))
    energy, amp, err = np.array(points).T
    w = 1 / err**2
    k = float(np.sum(w * energy * amp) / np.sum(w * amp**2))
    # Conditional statistical error only; detector nonlinearity is reported
    # explicitly via residuals, not hidden in an inflated statistical error.
    k_stat = k / np.sqrt(np.sum(w * amp**2))
    return {'k_keV_per_amplitude': k, 'intercept_keV': 0., 'k_stat_err': float(k_stat),
            'points': [{'energy_keV': e, 'mu': a, 'mu_err': s,
                        'predicted_keV': k*a, 'residual_keV': k*a-e}
                       for e,a,s in points]}


def collect(args):
    historical_of.D = str(args.data_root)
    report = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'selection': 'Tri-exp: solver success and finite amplitude, no chi2 or amplitude-deviation cut.',
              'OF_selection': 'Existing good signal triggers matched within 300 samples; delay-at-limit events excluded.',
              'calibration': 'Per noise level and method: E=k*A, five lines 239,511,727,1461,2615 keV; weights=1/mu_err^2.',
              'gaussian_definition': 'Canonical scripts.run_gauss_fit.fit_gaussian for all four methods; FWHM=2.355*sigma.',
              'error_bars': 'Gaussian width statistical errors conditional on the fitted calibration slope.',
              'plot_warning': 'Hollow markers indicate Gaussian SPECTRUM chi2/ndf > 10; diagnostic display only, no additional event cut.',
              'limitations': ['Historical OF uses separate stream events and daily noise states; holdout noise is drawn per window.',
                             'Old model uses DDIM eta=0, x_T=0; new model uses Gaussian x_T and DDPM no_noise.',
                             'New vs old changes architecture/conditioning, training loss (L2 vs L1), and sampler together.',
                             'Gaussian core width is not the full standard deviation or reconstruction efficiency.',
                             'Calibration is evaluated on the same peak samples; residual nonlinearity is retained.'],
              'code_sha256': digest(__file__), 'groups': {}, 'input_files': []}
    for level, (run, title) in LEVELS.items():
        old_dir = args.notes_root / f'results/fit_results/{level}_zeroinit'
        new_dir = args.notes_root / f'results/fit_results/model261108_nonoise/efficiency_{level}'
        of, _ = historical_of.of_amplitudes(run)
        legacy = json.loads((args.notes_root / f'results/plots/OFcompare/full_run{run}_resolution_vs_energy.json').read_text())
        energies = sorted(of)
        assert len(energies) == 17
        for path in (args.data_root/f'OFcompare_local/run{run}/of_ideal.csv',
                     args.data_root/f'OFcompare_truth/run{run}/truth/events.csv'):
            report['input_files'].append({'path': str(path), 'sha256': digest(path)})
        rows = {}
        for energy in energies:
            name = f'fit_efficiency_{energy}keV.h5'
            paths = {'old': old_dir/name, 'new': new_dir/name}
            rows[energy] = {}
            for which, path in paths.items():
                report['input_files'].append({'path': str(path), 'sha256': digest(path)})
                with h5py.File(path, 'r') as f:
                    assert len(f['amp_denoised']) == 5000
                    # The raw reference always comes from the same historical
                    # zero-init fit file, including the two earlier priority points.
                    fields = [('old_zeroinit', 'denoised'), ('raw', 'noisy')] if which == 'old' else [('new_nonoise', 'denoised')]
                    for method, field in fields:
                        a = f[f'amp_{field}'][:].astype(float)
                        success = f[f'success_{field}'][:].astype(bool)
                        gate = f[f'converged_{field}'][:].astype(bool)
                        rows[energy][method] = summarize(a[success], len(a), int(success.sum()), a[gate & np.isfinite(a)])
                        rows[energy][method]['chi2_median'] = float(np.nanmedian(f[f'chi2_{field}'][:]))
            src = of[energy]
            assert src['n_true'] == 5000
            rows[energy]['OF'] = summarize(src['amp'], src['n_true'], len(src['amp']))
            rows[energy]['OF']['n_delay_at_limit'] = src['n_at_limit']
            historical = legacy['peaks'][str(energy)]['OF']
            rows[energy]['OF']['historical_per_peak_fwhm_keV'] = FWHM * historical['sigma_keV']
        calibrations = {method: calibration(rows, method) for method in METHODS}
        for energy, row in rows.items():
            for method, value in row.items():
                k = calibrations[method]['k_keV_per_amplitude']
                g = value['gaussian']
                value.update(mean_energy_keV=k*g['mu'], bias_keV=k*g['mu']-energy,
                             std_energy_keV=k*value['std_amplitude'],
                             fwhm_keV=FWHM*k*g['sigma'] if g['converged'] else np.nan,
                             fwhm_err_keV=FWHM*k*g['sigma_err'] if g['converged'] else np.nan,
                             robust_core_fwhm_keV=FWHM*k*value.get('robust_core_sigma_amplitude', np.nan))
                if 'gated_gaussian' in value:
                    gg = value['gated_gaussian']
                    value['gated_fwhm_same_calibration_keV'] = FWHM*k*gg['sigma'] if gg['converged'] else np.nan
        report['groups'][level] = {'noise_label': title, 'historical_OF_run': run,
                                   'calibrations': calibrations, 'peaks': rows}
    return report


def values(group, method, key):
    energies = np.array(sorted(group['peaks']), dtype=float)
    return energies, np.array([group['peaks'][int(e)][method][key] for e in energies])


def save_figure(fig, destination):
    fig.savefig(destination.with_suffix('.png'), dpi=180)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)


def plot_resolution(report, output, high_only=False):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), sharex='col',
                             gridspec_kw={'height_ratios': [2.6, 1]}, layout='constrained')
    for col, (level, group) in enumerate(report['groups'].items()):
        ax, ratio = axes[:, col]
        for method in METHODS:
            e, y = values(group, method, 'fwhm_keV')
            _, err = values(group, method, 'fwhm_err_keV')
            mask = e >= 239 if high_only else np.ones(len(e), dtype=bool)
            ax.errorbar(e[mask], y[mask], yerr=err[mask], color=COLOR[method],
                        marker={'raw':'^','old_zeroinit':'s','new_nonoise':'o','OF':'D'}[method],
                        ls='--' if method == 'OF' else '-', lw=1.5, ms=4, capsize=2, label=LABEL[method])
            warning=np.array([group['peaks'][int(x)][method]['gaussian']['chi2ndf'] > 10 for x in e])
            if np.any(mask & warning):
                ax.plot(e[mask & warning],y[mask & warning],linestyle='none',
                        marker={'raw':'^','old_zeroinit':'s','new_nonoise':'o','OF':'D'}[method],
                        ms=5,mfc='white',mec=COLOR[method],mew=1.2,zorder=5)
        _, new = values(group, 'new_nonoise', 'fwhm_keV')
        for denominator, label, style in [('old_zeroinit','New / old zero-init','-'),('OF','New / historical OF','--')]:
            _, base = values(group, denominator, 'fwhm_keV')
            ratio.plot(e[mask], (new/base)[mask], style, color=COLOR[denominator], marker='o', ms=3, label=label)
        ax.set_title(group['noise_label'] + ' RMS input noise', fontsize=12)
        ax.set_yscale('log')
        if not high_only:
            ax.set_xscale('log')
        ratio.set_yscale('log')
        ratio.axhline(1., color='#555555', lw=.9, ls=':')
        ax.set_ylabel('Gaussian FWHM (keV)')
        ratio.set_ylabel('FWHM ratio')
        ratio.set_xlabel('True energy (keV)')
        ratio.legend(fontsize=8, loc='best')
        ax.legend(fontsize=8, loc='upper left' if high_only else 'best')
        for a in (ax, ratio):
            a.grid(alpha=.22, which='both')
    fig.suptitle('Energy resolution: raw, old zero-init, new model, and historical OF', fontsize=14)
    fig.supxlabel('Five-line calibration per method; successful tri-exp fits without chi-square gating.\n'
                  'Historical OF uses different events/noise evolution; dashed curves are reference comparisons.\n'
                  'Hollow markers: Gaussian spectrum fit χ²/ndf > 10. These widths need spectral-shape checks.', fontsize=8.5)
    save_figure(fig, output / ('resolution_high_energy' if high_only else 'resolution_vs_energy'))


def plot_calibration(report, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6), layout='constrained')
    for col, (level, group) in enumerate(report['groups'].items()):
        for method in METHODS:
            calibration_result = group['calibrations'][method]
            pts = calibration_result['points']
            energy = np.array([p['energy_keV'] for p in pts])
            amplitude = np.array([p['mu'] for p in pts])
            k = calibration_result['k_keV_per_amplitude']
            axes[0,col].plot(energy, amplitude, 'o', ms=4, color=COLOR[method],
                             label=LABEL[method] + f'; k={k:.3f}')
            axes[0,col].plot([0,3034], np.array([0,3034])/k, color=COLOR[method], alpha=.7,
                             ls='--' if method=='OF' else '-')
            e,bias = values(group, method, 'bias_keV')
            axes[1,col].plot(e, 100*bias/e, marker='o', ms=3, color=COLOR[method],
                             ls='--' if method=='OF' else '-')
            warning=np.array([group['peaks'][int(x)][method]['gaussian']['chi2ndf'] > 10 for x in e])
            if np.any(warning):
                axes[1,col].plot(e[warning],(100*bias/e)[warning],linestyle='none',marker='o',
                                 ms=5,mfc='white',mec=COLOR[method],mew=1.2,zorder=5)
        axes[0,col].set_title(group['noise_label'] + ': five-line zero-intercept calibration')
        axes[0,col].set_xlabel('True calibration energy (keV)')
        axes[0,col].set_ylabel('Gaussian amplitude centroid (V)')
        axes[0,col].legend(fontsize=7.5)
        axes[1,col].axhline(0, color='black', lw=.7)
        axes[1,col].set_xscale('log')
        axes[1,col].set_yscale('symlog', linthresh=1)
        axes[1,col].set_xlabel('True energy (keV)')
        axes[1,col].set_ylabel('Reconstructed centroid bias (%)')
        for ax in axes[:,col]:
            ax.grid(alpha=.22, which='both')
    fig.suptitle('Calibration and residual nonlinearity', fontsize=14)
    fig.supxlabel('239, 511, 727, 1461, 2615 keV; weights = inverse centroid variance.\n'
                  '3034 keV and low-energy points are excluded from calibration. OF is a different-event reference.\n'
                  'Hollow residual points have poor Gaussian fits; their fitted centroids are shape-dependent.', fontsize=8.5)
    save_figure(fig, output/'calibration')


def plot_quality(report, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8), layout='constrained')
    for col, group in enumerate(report['groups'].values()):
        for method in METHODS:
            e, n = values(group, method, 'n_used')
            axes[0,col].plot(e, 100*n/5000, '-', color=COLOR[method], marker='o', ms=3, label=LABEL[method])
            if method != 'OF':
                _, gate = values(group, method, 'n_quality_gate')
                axes[0,col].plot(e, 100*gate/5000, ':', color=COLOR[method], lw=1.5)
            _, fwhm = values(group, method, 'fwhm_keV')
            _, std = values(group, method, 'std_energy_keV')
            axes[1,col].plot(e, std/(fwhm/FWHM), color=COLOR[method], marker='o', ms=3,
                             ls='--' if method=='OF' else '-')
        axes[0,col].set_title(group['noise_label'] + ': solid = used; dotted = original chi-square gate')
        axes[0,col].set_ylabel('Fraction of 5000 events (%)')
        axes[0,col].set_ylim(-3,103)
        axes[0,col].legend(fontsize=7.5)
        axes[1,col].set_ylabel('Full standard deviation / Gaussian sigma')
        axes[1,col].axhline(1,color='black',ls=':',lw=.8)
        axes[1,col].set_yscale('log')
        for ax in axes[:,col]:
            ax.set_xscale('log'); ax.grid(alpha=.22,which='both'); ax.set_xlabel('True energy (keV)')
    fig.suptitle('Selection and non-Gaussian tails: diagnostics, not efficiency measurements', fontsize=13)
    save_figure(fig, output/'fit_quality')


def export(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(json_safe(report), indent=2), encoding='utf-8')
    columns=['noise','energy_keV','method','fwhm_keV','fwhm_err_keV','mean_energy_keV','bias_keV',
             'std_energy_keV','robust_core_fwhm_keV','n_total','n_success','n_used','n_quality_gate']
    with (output/'resolution_table.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for level,group in report['groups'].items():
            for energy,row in group['peaks'].items():
                for method,value in row.items():
                    writer.writerow({'noise':level,'energy_keV':energy,'method':method,
                                     **{k:value.get(k,'') for k in columns[3:]}})
    lines=['# Model261108 full-range comparison','',
           'Main curves use successful finite tri-exponential fits without the chi-square gate; original gate statistics are retained.',
           'Four methods use the same canonical Gaussian fitter and independent five-line zero-intercept calibration.',
           'OF reuses historical amplitudes (no new OF processing). Its stream events/noise evolution differ from the holdout dataset.',
           'Old model: zero-init DDIM. New model: Gaussian start + DDPM no_noise. This is a combined model/loss/sampler comparison.',
           'Gaussian fit uncertainties do not include calibration nonlinearity. See calibration residual and quality figures.','',
           '| Noise | Energy (keV) | Raw FWHM | Old zero-init | New no_noise | Historical OF | New / old |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for level,group in report['groups'].items():
        for e in (15,2615,3034):
            row=group['peaks'][e]
            vals=[row[m]['fwhm_keV'] for m in METHODS]
            lines.append(f'| {level} | {e} | '+ ' | '.join(f'{v:.5f}' for v in vals)+f' | {vals[2]/vals[1]:.3f} |')
    lines += ['', '## Non-Gaussian low-energy warning', '',
              'New-model 1.382 mV / 2 keV and 7.5 mV / 10 keV spectra give broad-fit FWHM 0.575 and 2.981 keV, versus robust-core widths 0.095 and 0.494 keV. Single-Gaussian resolution is not reliable at these points; the primary values are retained and flagged, not silently replaced.',
              'At 7.5 mV / 15-25 keV the new model has pronounced low-energy tails. A narrower central peak alone does not establish better full-distribution reconstruction.',
              'Hollow markers denote Gaussian spectrum chi2/ndf > 10 as a display warning only. This is separate from the stored per-waveform tri-exponential quality gate; no event selection is changed.',
              '', '## Provenance and numerical conventions', '',
              'Raw/old fits are reused read-only from the historical zero-init fit directories; new fits use the current canonical fitter.',
              'A local sample reproduced low-noise cached amplitudes within tolerance; one high-noise 1-keV raw fit differed by about 0.1% when recomputed. Cached values are preserved, not claimed bitwise identical across environments.',
              'OF matching and delay-limit exclusion follow experiments/of_comparison/plot_full_of_vs_ddpm.py.',
              'The JSON also retains robust-core widths and historical per-peak OF widths to expose convention dependence.',
              'The primary global calibration replaces the historical OF per-peak scale; this is a new comparison figure, not an overwrite of historical results.',
              'Input file SHA-256 hashes, Gaussian statistics, gates and calibration residuals are in comparison.json.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')



def plot_outlier_spectra(report,args):
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),layout='constrained',gridspec_kw={'height_ratios':[2.2,1]})
    for col,(level,energy) in enumerate([('1p382mV',2),('7p5mV',10)]):
        group=report['groups'][level]
        residuals={}
        for method,folder in [('old_zeroinit',f'{level}_zeroinit'),('new_nonoise',f'model261108_nonoise/efficiency_{level}')]:
            with h5py.File(args.notes_root/f'results/fit_results/{folder}/fit_efficiency_{energy}keV.h5','r') as f:
                amplitude=f['amp_denoised'][:].astype(float)
                success=f['success_denoised'][:].astype(bool)
                residuals[method]=amplitude[success & np.isfinite(amplitude)]*group['calibrations'][method]['k_keV_per_amplitude']-energy
        allvalues=np.concatenate(list(residuals.values()))
        bins=np.linspace(allvalues.min()-.01,allvalues.max()+.01,160)
        for method,x in residuals.items():
            axes[0,col].hist(x,bins=bins,density=True,histtype='step',color=COLOR[method],label=LABEL[method],lw=1.2)
            axes[1,col].hist(x,bins=bins,histtype='step',color=COLOR[method],lw=1.1)
        v=group['peaks'][energy]['new_nonoise']
        axes[0,col].set_title(f'{group["noise_label"]}, {energy} keV: strongly non-Gaussian')
        axes[0,col].text(.03,.63,
             f'New model\nFull-range Gaussian FWHM = {v["fwhm_keV"]:.3f} keV\n'
             f'Robust-core Gaussian FWHM = {v["robust_core_fwhm_keV"]:.3f} keV\n'
             f'Full-range Gaussian χ²/ndf = {v["gaussian"]["chi2ndf"]:.1f}',
             transform=axes[0,col].transAxes,fontsize=9,
             bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        axes[0,col].legend(fontsize=8,loc='upper left')
        axes[0,col].set_ylabel('Probability density (1/keV)')
        axes[1,col].set_ylabel('Counts (log)');axes[1,col].set_yscale('log')
        for ax in axes[:,col]:
            ax.axvline(0,color='black',ls=':',lw=.8)
            ax.set_xlabel('Reconstructed energy − true energy (keV)');ax.grid(alpha=.2)
    fig.suptitle('Low-energy outliers: a single Gaussian width is inadequate',fontsize=14)
    fig.supxlabel('Same successful-fit amplitudes; no additional event cut.\n'
                  'The two Gaussian windows give different widths. Robust-core values are diagnostics, not replacements.',fontsize=9)
    save_figure(fig,args.output/'low_energy_outliers')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--notes-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-fit-manifest',type=Path)
    parser.add_argument('--wait-timeout-minutes',type=float,default=180.)
    args=parser.parse_args()
    if args.wait_for_fit_manifest:
        started=time.monotonic()
        while True:
            if args.wait_for_fit_manifest.exists():
                state=json.loads(args.wait_for_fit_manifest.read_text())
                if state.get('complete'):
                    assert len(state['files'])==34
                    assert all(digest(r['path'])==r['sha256'] for r in state['files'])
                    break
            status_path=args.wait_for_fit_manifest.parent/'fit_status.json'
            if status_path.exists():
                status=json.loads(status_path.read_text())
                if status.get('stage')=='failed':
                    raise RuntimeError(status)
            if time.monotonic()-started > 60*args.wait_timeout_minutes:
                raise TimeoutError('Full fitting did not complete within the configured wait.')
            time.sleep(30)
    report=collect(args)
    export(report,args.output)
    plot_resolution(report,args.output)
    plot_resolution(report,args.output,high_only=True)
    plot_calibration(report,args.output)
    plot_quality(report,args.output)
    plot_outlier_spectra(report,args)
    for level,group in report['groups'].items():
        print(level,'calibration',{m:round(c['k_keV_per_amplitude'],5) for m,c in group['calibrations'].items()})
        for energy,row in group['peaks'].items():
            print(energy,{m:round(v['fwhm_keV'],5) for m,v in row.items()},flush=True)
    print('COMPARISON_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
