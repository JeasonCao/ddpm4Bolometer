"""Summarize saved low-energy diagnostics and magnify high-energy fit residuals.

Read-only inputs. Produces descriptive figures and an auditable fixed-deficit
summary; the energy-defined plot groups never become event-quality cuts.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.model261108_inference import waveform_tail_diagnostics as td
from experiments.model261108_inference.full_comparison import digest, json_safe, save_figure

def high_zoom(report, output):
    examples=[r for r in report['high_energy_examples'] if r['chi2_quantile']==.5]
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    metrics=[]
    for c,rec in enumerate(examples):
        raw_path,out_path,fit_path=[Path(x['path']) for x in rec['sources']]
        for source in rec['sources']:
            assert digest(source['path'])==source['sha256']
        f,_,_=td.load_fit(fit_path)
        with h5py.File(raw_path) as h: clean=h['clean_template'][:].astype(float)
        with h5py.File(out_path) as h: y=h['waveforms_denoised'][rec['row']].astype(float)
        t=np.arange(len(y))/1000
        fit=td.reconstruct(f,[rec['row']],t)[0]
        cf=td.fit_pulse(clean,n_pulses=1,fs=1000.)
        assert cf.success
        clean_fit=cf.model_fn(t,*cf.params)
        for r,(start,end,title) in enumerate([(1.492,1.525,'Rising edge'),(1.60,1.70,'Peak region')]):
            mask=(t>=start)&(t<=end)
            for a,color,style,label in [(y,'#df7539','-','New model output'),
                                       (fit,'black','--','Saved tri-exp fit'),
                                       (clean,'#139b73',':','Clean truth')]:
                axes[r,c].plot(t[mask],a[mask],color=color,ls=style,lw=1.3,label=label)
            axes[r,c].set(xlim=(start,end),ylabel='Voltage (V)',title=title)
        for r,(start,end) in enumerate([(1.48,1.75),(1.52,2.5)],start=2):
            mask=(t>=start)&(t<=end)
            axes[r,c].plot(t[mask],(y-fit)[mask]*1000,color='#8e4da0',label='Output - saved fit')
            axes[r,c].plot(t[mask],(clean-clean_fit)[mask]*1000,color='#139b73',ls='--',label='Clean - its tri-exp fit')
            sigma=rec['baseline_sigma_mV']
            axes[r,c].axhspan(-sigma,sigma,color='#aaaaaa',alpha=.3,label='Output baseline ±1σ')
            axes[r,c].set(xlim=(start,end),ylabel='Residual (mV)',
                          title='Residual including onset' if r==2 else 'Residual after onset (magnified)')
        axes[0,c].set_title(f"{rec['noise']}, row {rec['row']}, χ²/ndf={rec['chi2_ndf']:.1f}\nRising edge")
        for ax in axes[:,c]:
            ax.grid(alpha=.2);ax.set_xlabel('Time (s)');ax.legend(fontsize=7)
        rclean=clean-clean_fit
        onset=(t>=1.49)&(t<=1.52)
        metrics.append({**rec,'clean_residual_sse_fraction_1p49_1p52s':float(np.sum(rclean[onset]**2)/np.sum(rclean**2)),
                        'clean_residual_peak_mV':float(np.max(np.abs(rclean))*1000),
                        'clean_residual_peak_time_s':float(t[np.argmax(np.abs(rclean))])})
    fig.suptitle('2615 keV: local fit mismatch is also present in the noiseless template')
    fig.supxlabel('Saved fits are not changed. The clean-template fit is a diagnostic control; the original quality gate is retained.',fontsize=9)
    save_figure(fig,output/'high_energy_fit_zoom')
    return metrics

def overview(report,output):
    keys=['1p382mV_2keV','7p5mV_10keV']
    fig,axes=plt.subplots(2,3,figsize=(15,8.8),layout='constrained')
    for r,key in enumerate(keys):
        rec=report['cases'][key]
        with np.load(output/key/'per_event_features.npz') as h: a={k:h[k] for k in h.files}
        x=a['energy_keV'];groups=a['group']
        mix=rec['mixtures'][1];grid=np.linspace(x.min(),x.max(),500)
        axes[r,0].hist(x,bins=120,density=True,histtype='step',color='black',label='All 5,000 events')
        for i,(mu,sd,w) in enumerate(zip(mix['means'],mix['sigmas'],mix['weights'])):
            axes[r,0].plot(grid,w*np.exp(-.5*((grid-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)),
                           color=td.COLORS[2 if i==0 else 1],ls='--',label='Broad component' if i==0 else 'Narrow component')
        axes[r,0].axvline(rec['group_definition']['tail_below'],color=td.COLORS[2],ls=':')
        axes[r,0].set(xlabel='Reconstructed energy (keV)',ylabel='Density (1/keV)',title=key.replace('_',' / '))
        axes[r,0].legend(fontsize=7)
        td.scatter(axes[r,1],x,a['fit_t_rise']*1000,groups,'Reconstructed energy (keV)','Fitted rise time (ms)')
        rho=spearmanr(x,a['fit_t_rise']).statistic
        axes[r,1].set_title(f'Lower energy accompanies slower rise; ρ={rho:.3f}',fontsize=10)
        axes[r,1].axhline(200,color='black',ls=':',lw=.8)
        axes[r,1].text(.02,.98,'200 ms = existing fit upper bound',transform=axes[r,1].transAxes,va='top',fontsize=7)
        v=a['input_noise_power_1_20_Hz_fraction']
        td.scatter(axes[r,2],v,x,groups,'Input-noise power fraction, 1–20 Hz','Reconstructed energy (keV)')
        rho=spearmanr(x,v).statistic
        axes[r,2].set_title(f'Association with low-frequency noise; ρ={rho:.3f}',fontsize=10)
        axes[r,2].legend(fontsize=7,markerscale=2)
        for ax in axes[r,:]:ax.grid(alpha=.18)
    fig.suptitle('Low-energy tails: spectrum, reconstructed shape, and input noise')
    fig.supxlabel('Blue: diagnostic peak core; orange: below μ_peak−3σ_peak; gray: transition. All events retained.\n'
                  'Two Gaussian curves describe shape only; three components fit better by BIC. Correlation does not establish cause.',fontsize=9)
    save_figure(fig,output/'low_energy_overview')

def fixed_deficits(report,output):
    summary={}
    for key,rec in report['cases'].items():
        with np.load(output/key/'per_event_features.npz') as h: a={k:h[k] for k in h.files}
        with h5py.File(rec['sources'][2]['path']) as h: true=float(h.attrs['true_amp_V'])
        groups=a['group'];tail=groups==2
        assert np.array_equal(a['source_row'],np.arange(5000))
        assert all(np.all(np.isfinite(a[k])) for k in a if k not in ('group','source_row'))
        assert sum(rec['counts'].values())==5000
        vals={'n':5000,'true_peak_V':true,
              'fit_peak_below_90pct_truth':int((a['fit_amp']<.9*true).sum()),
              'direct_peak_below_90pct_truth':int((a['output_peak_ratio']<.9).sum()),
              'tail_rise_at_upper_bound':int((tail&(a['fit_t_rise']>=.2*(1-1e-5))).sum()),
              'tail_count_energy_defined':int(tail.sum()),
              'fit_direct_peak_spearman':rec['fit_vs_direct_peak_spearman']}
        for k in ['fit_t_rise','fit_t_fall1','fit_t_fall2','output_peak_ratio',
                  'output_projection_ratio','output_area_ratio','fit_minus_direct_peak_ratio',
                  'input_noise_power_1_20_Hz_fraction','input_noise_projection_ratio','input_scale_V']:
            vals[k]={g:float(np.median(a[k][groups==i])) for i,g in [(1,'core'),(2,'tail')]}
        summary[key]=vals
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,level in zip(axes,('1p382mV','7p5mV')):
        rows=[(int(k.split('_')[1][:-3]),v) for k,v in summary.items() if k.startswith(level+'_')]
        rows.sort()
        for field,label,color in [('fit_peak_below_90pct_truth','Tri-exp fitted peak','#df7539'),
                                   ('direct_peak_below_90pct_truth','Direct maximum - baseline','#139b73')]:
            ax.plot([e for e,_ in rows],[100*v[field]/v['n'] for _,v in rows],'o-',color=color,label=label)
        ax.set(xlabel='True energy (keV)',ylabel='Peak deficit >10% (%)',title=level,ylim=(-.5,31))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fixed truth-based deficit: independent of mixture group boundaries')
    fig.supxlabel('All 5,000 events in each denominator; peak <90% of clean peak. Diagnostic only, no event removal and no recalibration.',fontsize=9)
    save_figure(fig,output/'fixed_peak_deficit')
    return summary

def check_sampled_noise_alignment(report):
    # Identity check only, not a new independent validation cohort.
    checks=[]
    for key1,key2,ratio in [('1p382mV_2keV','1p382mV_5keV',1.),
                            ('7p5mV_10keV','7p5mV_50keV',1.),
                            ('1p382mV_2keV','7p5mV_10keV',7.5/1.382)]:
        rows=np.unique(np.linspace(0,4999,32,dtype=int))
        with h5py.File(report['cases'][key1]['sources'][0]['path']) as f, h5py.File(report['cases'][key2]['sources'][0]['path']) as g:
            a=f['waveforms_noisy'][rows].astype(float)-f['clean_template'][:].astype(float)
            b=g['waveforms_noisy'][rows].astype(float)-g['clean_template'][:].astype(float)
        checks.append({'case_a':key1,'case_b':key2,'sampled_rows':rows.tolist(),'expected_scale_ratio':ratio,
                       'rms_difference_relative_to_b':float(np.sqrt(np.mean((a*ratio-b)**2)/np.mean(b*b))),
                       'flattened_noise_correlation':float(np.corrcoef(a.ravel(),b.ravel())[0,1])})
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Completed waveform_tail_diagnostics directory')
    args=p.parse_args()
    path=args.output/'diagnostics.json'
    report=json.loads(path.read_text())
    assert report['complete'] and len(report['cases'])==8
    summary={'code_sha256':digest(__file__),'diagnostics_sha256':digest(path),
             'feature_file_hashes':{key:digest(args.output/key/'per_event_features.npz') for key in report['cases']},
             'high_energy':high_zoom(report,args.output),
             'fixed_deficits':fixed_deficits(report,args.output),
             'sampled_noise_alignment':check_sampled_noise_alignment(report)}
    overview(report,args.output)
    (args.output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2),encoding='utf-8')
    print('SUMMARY_COMPLETE',flush=True)
    print(json.dumps(summary['sampled_noise_alignment'],indent=2),flush=True)

if __name__=='__main__':main()
