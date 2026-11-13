"""Report paired single-pulse comparisons, preserving the old 4980-event selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_readout_fit.model import Forward
from experiments.cuore_readout_fit.run_study import load_event,dump
from experiments.cuore_fit_ranges.stable_fitter import evaluate as bare
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
def identity(r):return r['source_file'],r['row']

def main(a):
    root=Path(a.input);out=Path(a.plots);out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((root/'baseline.json').read_text());shared=read(root/'shared.jsonl')
    free=read(root/'free.jsonl')
    summary={}
    for group,rows in [('calibration_train',shared),('heldout_test',shared),('free_test_diagnostic',free)]:
        subset=[r for r in rows if (r['source_file']=='cuoreTest.h5')==('test' in group)]
        if group=='calibration_train':
            allowed=json.loads((root/'manifest.json').read_text())['training_calibration']
            subset=[r for r in subset if r['row'] in allowed]
        tags=sorted(set(r['tag'] for r in subset))
        selected={identity(r) for r in subset}
        subset += [r for r in baseline if identity(r) in selected]
        stats={}
        for tag in ['bare']+tags:
            rr=[r for r in subset if r['tag']==tag]
            stats[tag]=dict(n=len(rr),success=sum(r['success'] for r in rr),
                median_rms_percent=float(np.median([r['nrmse']*100 for r in rr])),
                median_peak_region_rms_percent=float(np.median([r['peak_region_rms']*100 for r in rr])),
                median_abs_peak_error_percent=float(np.median([abs(r['peak_time_value_error'])*100 for r in rr])),
                median_condition=float(np.median([r['condition'] for r in rr])),
                collapsed=sum(r['delta']<.05 for r in rr),
                any_boundary=sum(bool(r.get('boundary',[])) for r in rr),
                median_tau_s=np.median([r['tau'] for r in rr],axis=0).tolist())
            if rr and rr[0].get('k') is not None:
                stats[tag]['k_quantiles']=np.quantile([r['k'] for r in rr],[0,.5,1]).tolist()
        summary[group]=stats
    # Paired validation plot: no hyperparameter selection uses these events.
    validation=[r for r in shared if r['source_file']=='cuoreTest.h5']
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    colors={'bare':'#d17a20','bessel_shared':'#2a9d8f','rc_shared':'#2166ac','nonlinear_shared':'#8750a1'}
    labels={'bare':'Bare 3-exp','bessel_shared':'+ Bessel','rc_shared':'+ RC + Bessel','nonlinear_shared':'+ nonlinear RC + Bessel'}
    for ax,metric,title in zip(axs,['nrmse','peak_region_rms','peak_time_value_error'],
                             ['Full-window RMS','Leading edge + peak RMS','Absolute error at observed maximum']):
        for tag in colors:
            rr=sorted([r for r in (baseline if tag=='bare' else validation) if r['source_file']=='cuoreTest.h5' and r['tag']==tag],
                      key=lambda r:r['amp_adc'])
            ax.plot([r['amp_adc'] for r in rr],[100*abs(r[metric]) for r in rr],'.-',ms=4,lw=.7,color=colors[tag],label=labels[tag])
        ax.set_xlabel('Observed amplitude [ADC]');ax.set_ylabel('% of observed amplitude');ax.set_title(title);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('24 held-out single pulses | response chosen using training pulses only',fontsize=13)
    fig.tight_layout();save(fig,out,'heldout_comparison')
    # Repeat the exact previous examples; no cherry-picking by improvement.
    rows=[385,3629,2864,3519]
    fig,axs=plt.subplots(3,4,figsize=(17,10),sharex='col')
    for col,row in enumerate(rows):
        y,ident=load_event(a.data,'cuoreTraining.h5',row);t=np.arange(len(y))/1000.;tp=t[np.argmax(y)]
        rr=[next(r for r in baseline if identity(r)==('cuoreTraining.h5',row))]
        rr += [next(r for r in shared if identity(r)==('cuoreTraining.h5',row) and r['tag']==tag)
               for tag in ['bessel_shared','rc_shared']]
        axs[0,col].plot(t,y,color='.25',lw=.7,label='Data')
        axs[1,col].plot(t,y,color='.25',lw=.8)
        for r in rr:
            pred=bare(t,np.array(r['p']))[0] if r['kind']=='bare' else Forward(len(y),r['kind'],r['fixed_fc'],r['fixed_rc'],r['oversample']).evaluate(r['p'])[0]
            for ax in axs[:2,col]:ax.plot(t,pred,color=colors[r['tag']],lw=1.2,label=labels[r['tag']])
            axs[2,col].plot(t,100*(y-pred),color=colors[r['tag']],lw=.8,label=labels[r['tag']])
        for ax in axs[:,col]:ax.grid(alpha=.2)
        axs[0,col].set_title(f'Training row {row} | event {ident["event_id"]}')
        axs[0,col].set_xlim(tp-.45,tp+1.2)
        axs[1,col].set_ylim(.85,1.03)
        axs[2,col].axhline(0,color='.5',lw=.6);axs[2,col].set_xlabel('Time [s]')
    axs[0,0].set_ylabel('Normalized ADC')
    axs[1,0].set_ylabel('Peak detail')
    axs[2,0].set_ylabel('Data - fit [% of amplitude]')
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Same four examples as the previous report | shared effective response',fontsize=15)
    fig.tight_layout();save(fig,out,'same_four_examples')
    settings=json.loads((root/'shared_settings.json').read_text())
    grid_rows=sum([read(root/name) for name in ['grid.jsonl','refine0.jsonl','refine1.jsonl']],[])
    sensitivity={}
    for tolerance in [.01,.05]:
        near=[dict(r) for r in settings['grid_rc'] if not r['failures'] and r['mse']<=settings['rc']['mse']*(1+tolerance)]
        for r in near:
            r['median_tau_s']=np.median([x['tau'] for x in grid_rows if x['tag']==r['tag']],axis=0).tolist()
        medians=[r['median_tau_s'] for r in near]
        sensitivity[str(tolerance)]=dict(nsettings=len(near),settings=near,median_tau_min=np.min(medians,axis=0).tolist(),median_tau_max=np.max(medians,axis=0).tolist())
    dump(root/'shared_response_sensitivity.json',sensitivity)
    manifest=json.loads((root/'manifest.json').read_text())
    initialization={}
    for kind in ['bessel','rc']:
        grid_by_row={r['row']:r for r in grid_rows if r['tag']==settings[kind]['tag']}
        calibration=[r for r in shared if r['tag']==kind+'_shared' and r['source_file']=='cuoreTraining.h5' and r['row'] in manifest['training_calibration']]
        initialization[kind]=max(grid_by_row[r['row']]['nrmse']/r['nrmse']-1 for r in calibration)
    summary['selected_setting_one_vs_three_start_max_relative_rms_excess']=initialization
    fig,ax=plt.subplots(figsize=(7,4.5))
    for fc in [4.,6.,8.,12.,20.]:
        rr=sorted([r for r in settings['grid_rc'] if r['fc']==fc and '_ref' not in r['tag']],key=lambda r:r['rc'])
        ax.plot([r['rc']*1000 for r in rr],[100*np.sqrt(r['mse']) for r in rr],'o-',label=f'{fc:g} Hz')
    ax.set_xlabel('Shared effective RC [ms]');ax.set_ylabel('Calibration aggregate RMS [%]')
    ax.set_title('16 training pulses | Bessel cutoff sensitivity');ax.grid(alpha=.2);ax.legend(title='Bessel -3 dB')
    fig.tight_layout();save(fig,out,'shared_response_profile')
    if (root/'full_single.jsonl').exists():
        full=read(root/'full_single.jsonl')
        old={ (r['file'],r['row']):r for r in read(Path(a.old)/'fits.jsonl') if r['label']==0}
        if len(full)==5000:
            rr=sorted(full,key=identity)
            oldr=[old[identity(r)] for r in rr]
            paired=np.array([[o['nrmse'],r['nrmse']] for o,r in zip(oldr,rr)])
            summary['full_single']=dict(n=len(rr),success=sum(r['success'] for r in rr),
                rms_percent_quantiles_old=np.quantile(paired[:,0]*100,[0,.1,.5,.9,.99,1]).tolist(),
                rms_percent_quantiles_new=np.quantile(paired[:,1]*100,[0,.1,.5,.9,.99,1]).tolist(),
                median_paired_rms_reduction_percent=float(np.median((1-paired[:,1]/paired[:,0])*100)),
                improved=int(np.sum(paired[:,1]<paired[:,0])),
                worse_over_one_percent_relative=int(np.sum(paired[:,1]>paired[:,0]*1.01)),
                collapsed=sum(r['delta']<.05 for r in rr),boundary=sum(bool(r['boundary']) for r in rr),
                median_tau_s=np.median([r['tau'] for r in rr],axis=0).tolist(),
                median_condition=float(np.median([r['condition'] for r in rr])),
                rms_to_baseline_noise_median=float(np.median([r['nrmse']/o['baseline_noise_fraction'] for r,o in zip(rr,oldr)])),
                multistart_tau_span_quantiles=np.quantile([r['near_optimum_tau_relative_span'] for r in rr],[.5,.9,.99,1]).tolist(),
                multistart_both_converged=sum(all(v['success'] for v in r['starts']) for r in rr),
                median_peak_region_rms_percent=float(np.median([r['peak_region_rms']*100 for r in rr])),
                median_abs_peak_error_percent=float(np.median([abs(r['peak_time_value_error'])*100 for r in rr])))
            chosen=[r for r in rr if old[identity(r)]['collapsed'][0] and old[identity(r)]['success']]
            assert len(chosen)==4980
            fig,axs=plt.subplots(1,3,figsize=(14,4))
            for j,ax in enumerate(axs):
                ax.scatter([r['amp_adc'] for r in chosen],[r['tau'][j]*1000 for r in chosen],s=3,alpha=.3,color='#2166ac',rasterized=True)
                ax.set_xlabel('Observed amplitude [ADC]');ax.set_ylabel('Effective time constant [ms]')
                ax.set_title(['Rise','Fast decay','Slow decay'][j]);ax.grid(alpha=.2)
            fig.suptitle('Same 4980 previously degenerate singles | shared RC + Bessel | not calibrated thermal constants')
            fig.tight_layout();save(fig,out,'tau_vs_amplitude_same_4980')
            fig,axs=plt.subplots(1,2,figsize=(11,4.5))
            axs[0].scatter(paired[:,0]*100,paired[:,1]*100,s=3,alpha=.25,rasterized=True)
            lim=max(paired.max()*100,1);axs[0].plot([0,lim],[0,lim],'--',color='.5')
            axs[0].set_xlabel('Bare 3-exp RMS [%]');axs[0].set_ylabel('Shared RC + Bessel RMS [%]')
            for j,label,color in [(0,'Bare 3-exp','#d17a20'),(1,'Shared RC + Bessel','#2166ac')]:
                axs[1].hist(paired[:,j]*100,bins=70,histtype='step',label=label,color=color)
            axs[1].set_xlabel('RMS / observed amplitude [%]');axs[1].set_ylabel('Single pulses');axs[1].legend()
            fig.suptitle('Paired comparison: all 5000 single pulses');fig.tight_layout()
            save(fig,out,'full_single_residuals')
    dump(root/'report.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['input','plots','data','old']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
