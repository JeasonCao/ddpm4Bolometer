"""Plot only the 4,980 unresolved single events and representative time-domain fits.
No refitting; original and previous result files remain unchanged.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_fitter import evaluate

def main():
    p=argparse.ArgumentParser()
    for k in ('data','fits','output'): p.add_argument('--'+k,required=True)
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in (Path(a.fits)/'fits.jsonl').read_text().splitlines()]
    ss=sorted([r for r in rows if r['label']==0 and r['success'] and r['collapsed'][0]],key=lambda r:r['nrmse'])
    assert len(ss)==4980
    quantiles=[0,.1,.5,.9,.99,1]
    stats=dict(count=len(ss),selection='label=0, success=True, collapsed[0]=True; no additional quality cuts',
        rms_fraction_quantiles=dict(zip(map(str,quantiles),np.quantile([r['nrmse'] for r in ss],quantiles).tolist())),
        residual_to_baseline_noise_quantiles=dict(zip(map(str,quantiles),np.quantile([r['residual_to_baseline_noise'] for r in ss],quantiles).tolist())),
        input_fit_sha256=hashlib.sha256((Path(a.fits)/'fits.jsonl').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),examples=[])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for j,ax in enumerate(axs):
        ax.scatter([r['amplitude_adc']/1000 for r in ss],[r['tau'][0][j] for r in ss],
                   s=6,alpha=.35,color='#2469a0',rasterized=True)
        ax.set(xlabel='Window amplitude ($10^3$ ADC)',ylabel='Time constant (s)',
               title=[r'Rise: $\tau_r$',r'Fast decay: $\tau_{d1}$',r'Slow decay: $\tau_{d2}$'][j])
        ax.grid(alpha=.2)
    fig.suptitle('CUORE | 4,980 single-pulse windows only\nRise and fast-decay constants unresolved; 20 separated cases excluded.',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(out/f'tau_vs_amplitude_single_4980.{ext}',dpi=200)
    plt.close(fig)

    fig,axs=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for col,(q,tag) in enumerate([(0.1,'10th percentile'),(.5,'Median'),(.9,'90th percentile'),(1.,'Largest RMS')]):
        r=ss[int(round(q*(len(ss)-1)))]
        path=Path(a.data)/r['file']
        with h5py.File(path,'r') as f:
            assert int(f['eventId'][r['row']])==r['event_id']
            y=(f['Waveform'][r['row']]-r['offset_adc'])/r['amplitude_adc']
        t=np.arange(len(y))/1000.; pred=evaluate(t,np.array(r['p']))[0];res=y-pred
        recomputed=float(np.sqrt(np.mean(res**2)))
        assert np.isclose(recomputed,r['nrmse'],rtol=1e-9,atol=1e-12)
        peak_t=t[np.argmax(y)]
        zoom=(max(0,peak_t-.3),min(10,peak_t+.65))
        region=(t>=zoom[0])&(t<=zoom[1])
        meta=dict(file=r['file'],row=r['row'],event_id=r['event_id'],rank_percentile=q,
                  amplitude_adc=r['amplitude_adc'],tau_s=r['tau'][0],
                  full_window_rms_fraction=r['nrmse'],peak_zoom_rms_fraction=float(np.sqrt(np.mean(res[region]**2))),
                  max_abs_residual_fraction=float(np.max(np.abs(res))),baseline_noise_fraction=r['baseline_noise_fraction'],
                  peak_fit_value=float(pred[np.argmax(y)]),zoom_seconds=zoom)
        stats['examples'].append(meta)
        for ax in axs[:2,col]:
            ax.plot(t,y,color='#1666aa',lw=1,label='Raw waveform')
            ax.plot(t,pred,color='#e1812c',lw=1.35,ls='--',label='Fit')
            ax.set(ylabel='Amplitude / window amplitude')
            ax.grid(alpha=.18)
        axs[0,col].set(xlim=(0,10),xlabel='Time (s)',
            title=f'{tag} | RMS / amplitude = {r["nrmse"]:.3%}\n{r["file"].replace("cuore","")}, row {r["row"]}, ID {r["event_id"]}')
        axs[1,col].set(xlim=zoom,xlabel='Time (s)',title='Rise and peak enlarged')
        axs[2,col].plot(t,100*res,color='#80529b',lw=.8)
        axs[2,col].axhline(0,color='black',lw=.7)
        axs[2,col].set(xlim=(0,10),xlabel='Time (s)',ylabel='(Data - fit) / amplitude (%)',
                      title=f'Full-window residual | max {100*np.max(np.abs(res)):.2f}%')
        axs[2,col].grid(alpha=.18)
    axs[0,0].legend(fontsize=9,loc='upper right')
    fig.suptitle('Representative fits from the 4,980 unresolved single events\nChosen by residual-RMS rank; no refitting. Whole-window RMS includes the pre-trigger baseline.',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(out/f'time_domain_examples_4980.{ext}',dpi=180)
    plt.close(fig)
    (out/'selection_and_quality.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
