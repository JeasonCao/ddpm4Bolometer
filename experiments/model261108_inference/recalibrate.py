"""Restore the agreed calibration policy without rerunning inference or fits.

Full-range figures: five high-energy lines, zero intercept.
Before/after pairs: share the calibration of the denoised estimator.
A four-method overlay uses the new-model calibration for its raw reference.
Low-energy reconstruction acceptance: all available lines <= 511 keV.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.model261108_inference import full_comparison as fc

def fit_slope(group, method, energies):
    pts = [(e, group['peaks'][e][method]['gaussian']) for e in energies]
    assert all(g['converged'] and g['mu_err'] > 0 for _, g in pts)
    energy = np.array([e for e, _ in pts], float)
    amp = np.array([g['mu'] for _, g in pts])
    err = np.array([g['mu_err'] for _, g in pts])
    w = 1 / err**2
    k = float(np.sum(w*energy*amp)/np.sum(w*amp**2))
    return {'k_keV_per_amplitude': k, 'intercept_keV': 0.,
            'energies_keV': list(energies),
            'points': [{'energy_keV': e, 'mu': a, 'mu_err': s,
                        'predicted_keV': k*a, 'residual_keV': k*a-e,
                        'leverage_fraction': float(ww*a*a/np.sum(w*amp**2))}
                       for e,a,s,ww in zip(energy,amp,err,w)]}

def update_raw(group):
    # Preserve the independent raw calibration as an explicit historical diagnostic.
    group['raw_independent_calibration_previous'] = copy.deepcopy(group['calibrations']['raw'])
    group['calibrations']['raw'] = copy.deepcopy(group['calibrations']['new_nonoise'])
    group['calibrations']['raw']['derived_from'] = 'new_nonoise; shared before/after calibration'
    group['calibrations']['raw']['points'] = copy.deepcopy(group['raw_independent_calibration_previous']['points'])
    k = group['calibrations']['raw']['k_keV_per_amplitude']
    for p in group['calibrations']['raw']['points']:
        p['predicted_keV'] = k*p['mu']
        p['residual_keV'] = k*p['mu']-p['energy_keV']
    for row in group['peaks'].values():
        v = row['raw']; g = v['gaussian']
        v['mean_energy_keV'] = k*g['mu']
        v['std_energy_keV'] = k*v['std_amplitude']
        v['fwhm_keV'] = fc.FWHM*k*g['sigma']
        v['fwhm_err_keV'] = fc.FWHM*k*g['sigma_err']
        v['robust_core_fwhm_keV'] = fc.FWHM*k*v.get('robust_core_sigma_amplitude',np.nan)
        gg = v['gated_gaussian']
        v['gated_fwhm_same_calibration_keV'] = fc.FWHM*k*gg['sigma'] if gg['converged'] else np.nan
    for energy,row in group['peaks'].items():
        row['raw']['bias_keV'] = row['raw']['mean_energy_keV']-energy
    group['raw_reference_by_method'] = {}
    for method in ('old_zeroinit','new_nonoise'):
        kk = group['calibrations'][method]['k_keV_per_amplitude']
        group['raw_reference_by_method'][method] = {
            str(e): {'fwhm_keV': fc.FWHM*kk*row['raw']['gaussian']['sigma'],
                     'mean_energy_keV': kk*row['raw']['gaussian']['mu']}
            for e,row in group['peaks'].items()}

def paired_resolution(report, output):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex=True,sharey='row')
    for r,(level,group) in enumerate(report['groups'].items()):
        for c,method in enumerate(('old_zeroinit','new_nonoise')):
            ax=axes[r,c]
            e,y=fc.values(group,method,'fwhm_keV')
            k=group['calibrations'][method]['k_keV_per_amplitude']
            raw=np.array([group['raw_reference_by_method'][method][str(int(x))]['fwhm_keV'] for x in e])
            ax.plot(e,raw,'^-',color=fc.COLOR['raw'],ms=4,label='Before: raw')
            ax.plot(e,y,'o-',color=fc.COLOR[method],ms=4,label=fc.LABEL[method])
            ax.set(xscale='log',yscale='log',xlabel='True energy (keV)',ylabel='Gaussian FWHM (keV)',
                   title=f"{group['noise_label']}; shared k = {k:.4f} keV/V")
            ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    fig.suptitle('Before/after pairs: each pair shares its denoised calibration')
    fig.supxlabel('Five high-energy lines. Successful finite fits; no chi-square gate. Non-Gaussian low-energy widths remain diagnostic.',fontsize=9)
    fc.save_figure(fig,output/'paired_resolution')

def low_efficiency(report, notes, output):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    result={}
    for r,(level,group) in enumerate(report['groups'].items()):
        energies=sorted(e for e in group['peaks'] if e<=511)
        result[level]={}
        for c,method in enumerate(('old_zeroinit','new_nonoise')):
            calib=fit_slope(group,method,energies)
            k=calib['k_keV_per_amplitude']
            values=[]
            for e in energies:
                old=notes/f'results/fit_results/{level}_zeroinit/fit_efficiency_{e}keV.h5'
                new=notes/f'results/fit_results/model261108_nonoise/efficiency_{level}/fit_efficiency_{e}keV.h5'
                rec={'energy_keV':e}
                for kind,path,field in [('raw',old,'noisy'),('denoised',old if method=='old_zeroinit' else new,'denoised')]:
                    with h5py.File(path) as f:
                        a=f['amp_'+field][:].astype(float)
                        good=f['success_'+field][:].astype(bool)&np.isfinite(a)
                        passed=good&(np.abs(k*a-e)/e<.05)
                        fraction=float(passed.mean())
                        rec[kind]={'n_total':len(a),'n_pass':int(passed.sum()),'fraction':fraction,
                                   'binomial_se':float(np.sqrt(fraction*(1-fraction)/len(a)))}
                values.append(rec)
            result[level][method]={'calibration':calib,'rows':values}
            ax=axes[r,c]
            for kind,color,label in [('raw','#888780','Before: raw'),('denoised',fc.COLOR[method],fc.LABEL[method])]:
                ax.errorbar(energies,[100*v[kind]['fraction'] for v in values],
                            yerr=[100*v[kind]['binomial_se'] for v in values],
                            color=color,marker='o',ms=3,lw=1.2,label=label)
            ax.set(xscale='log',ylim=(-2,102),xlabel='True energy (keV)',ylabel='Within 5% of true energy (%)',
                   title=f"{group['noise_label']}; shared low-energy k = {k:.3f}")
            ax.grid(alpha=.2,which='both');ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Energy reconstruction acceptance: calibration on energies <= 511 keV')
    fig.supxlabel('All 5,000 windows in the denominator; successful finite amplitudes and |E_reco/E_true - 1| < 5%.\n'
                  'No chi-square cut; no DT-trigger condition. Gaussian calibration centroids at non-Gaussian peaks require caution.',fontsize=9)
    fc.save_figure(fig,output/'reconstruction_efficiency_low')
    return result

def reconstruction(report, output):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex='col')
    for c,group in enumerate(report['groups'].values()):
        for method in fc.METHODS:
            e,mu=fc.values(group,method,'mean_energy_keV')
            axes[0,c].plot(e,mu,'o-',color=fc.COLOR[method],ms=3,label=fc.LABEL[method])
            axes[1,c].plot(e,100*(mu-e)/e,'o-',color=fc.COLOR[method],ms=3)
            bad=np.array([group['peaks'][int(x)][method]['gaussian']['chi2ndf']>10 for x in e])
            for ax,y in [(axes[0,c],mu),(axes[1,c],100*(mu-e)/e)]:
                ax.plot(e[bad],y[bad],'o',ms=5,mfc='white',mec=fc.COLOR[method],linestyle='none')
        axes[0,c].plot([1,3034],[1,3034],'k:',lw=1)
        axes[0,c].set(xscale='log',yscale='log',ylabel='Reconstructed centroid (keV)',title=group['noise_label'])
        axes[0,c].legend(fontsize=7)
        axes[1,c].axhline(0,color='black',lw=.8)
        axes[1,c].set(xscale='log',yscale='symlog',xlabel='True energy (keV)',ylabel='Centroid bias (%)')
        for ax in axes[:,c]:ax.grid(alpha=.2,which='both')
    fig.suptitle('Reconstructed energy and residual nonlinearity: high-energy calibration')
    fig.supxlabel('Raw shares the new-model scale. Old/new models and historical OF have their own five-line calibrations.\n'
                  'Hollow points: poor Gaussian spectrum fit; fitted centroids are shape-dependent.',fontsize=9)
    fc.save_figure(fig,output/'reconstruction_quality')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notes-root',type=Path,required=True)
    p.add_argument('--source-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.source_report.read_text())
    for g in report['groups'].values():
        g['peaks']={int(k):v for k,v in g['peaks'].items()}
    for entry in report['input_files']:
        assert fc.digest(entry['path'])==entry['sha256'],entry['path']
    report['previous_report_sha256']=fc.digest(a.source_report)
    report['previous_code_sha256']=report['code_sha256']
    report['code_sha256']=fc.digest(__file__)
    report['calibration']='Full-range: five high-energy lines; raw shares new-model k. Paired before/after plots share each denoised k. Efficiency: <=511-keV calibration.'
    report['raw_reference']='new_nonoise'
    for group in report['groups'].values():
        update_raw(group)
        for method in ('old_zeroinit','new_nonoise','OF'):
            assert np.isclose(fit_slope(group,method,[239,511,727,1461,2615])['k_keV_per_amplitude'],
                              group['calibrations'][method]['k_keV_per_amplitude'],rtol=1e-12)
    fc.LABEL['raw']='No denoising (new-model scale)'
    fc.export(report,a.output)
    fc.plot_resolution(report,a.output)
    fc.plot_resolution(report,a.output,high_only=True)
    fc.plot_calibration(report,a.output)
    reconstruction(report,a.output)
    paired_resolution(report,a.output)
    eff=low_efficiency(report,a.notes_root,a.output)
    (a.output/'efficiency_low.json').write_text(json.dumps(fc.json_safe(eff),indent=2),encoding='utf-8')
    fc.plot_quality(report,a.output)
    fc.plot_outlier_spectra(report,a)
    text=['# Revised calibration, 2026-11-13','',
          'Full-range resolution and reconstructed energy use 239/511/727/1461/2615 keV, weighted through the origin; 3034 keV is excluded.',
          'The four-method overlay uses the NEW model calibration for its single raw reference. Each old/new before-after pair also appears separately with its own shared calibration.',
          'Low-energy reconstruction acceptance uses every available energy <=511 keV, with one denoised-derived k per before/after pair.',
          'All events remain available. The original chi-square gate is reported but is not applied to the comparison. A 5% energy window is used only for the explicitly labelled acceptance figure.',
          'No new DT trigger calculation, OF filtering, neural inference or batch waveform fitting was performed.',
          'Poor low-energy Gaussian fits remain flagged; splitting calibration ranges cannot turn a non-Gaussian spectrum into a Gaussian.',
          'The OF data use different stream events; the old/new comparison changes architecture, loss and sampler. Historical results are preserved.',
          'comparison.json preserves the previous independent raw calibration and both pair-specific raw reference values.',
          'efficiency_low.json records the exact low-energy calibration points, statistical weights and all acceptance counts.',
          'The old output directory is untouched.']
    (a.output/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    print('RECALIBRATION_COMPLETE',a.output,flush=True)
    for level,g in report['groups'].items():
        print(level,'raw change',g['raw_independent_calibration_previous']['k_keV_per_amplitude'],
              '->',g['calibrations']['raw']['k_keV_per_amplitude'],
              'low k',{m:v['calibration']['k_keV_per_amplitude'] for m,v in eff[level].items()},flush=True)

if __name__=='__main__':main()
