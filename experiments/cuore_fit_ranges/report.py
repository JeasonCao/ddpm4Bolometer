"""Plots and identity/hash checks for independent CUORE range study."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_fitter import evaluate

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main():
 p=argparse.ArgumentParser()
 for k in ('data','pilot','full','old','plots'):p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.plots);out.mkdir(parents=True,exist_ok=True)
 rows=read(Path(a.full)/'fits.jsonl');old=read(Path(a.old)/'fits.jsonl')
 oldmap={(r['file'],r['row']):r for r in old}
 pilot=read(Path(a.pilot)/'fits.jsonl')
 manifest=json.loads((Path(a.full)/'manifest.json').read_text())
 verify={}
 for name,n in [('cuoreTraining.h5',9000),('cuoreTest.h5',1000)]:
  path=Path(a.data)/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sources'][name]
  with h5py.File(path,'r') as src,h5py.File(Path(a.full)/name.replace('.h5','_fit.h5'),'r') as f:
   assert np.array_equal(f['row'][:],np.arange(n))
   assert np.array_equal(f['event_id'][:],src['eventId'][:])
   assert np.array_equal(f['amplitude_adc'][:],src['normalizationScale'][:].ravel())
   assert np.array_equal(f['offset_adc'][:],src['normalizationOffset'][:].ravel())
   assert 'Waveform' not in f
   verify[name]=dict(rows=n,source_hash_unchanged=True,identity_exact_match=True)
 summary={}
 for cls in (0,1):
  ss=[r for r in rows if r['label']==cls];good=[r for r in ss if r['success']]
  paired=[r for r in good if oldmap[(r['file'],r['row'])]['success']]
  summary[str(cls)]=dict(total=len(ss),success=len(good),
    collapsed_window=sum(any(r['collapsed']) for r in good),
    collapsed_dominant=sum(r['collapsed'][r['dominant']] for r in good),
    boundary=sum(any(r['boundary']) for r in good),diagnostic_pass=sum(r['diagnostic_pass'] for r in good),
    pulse_counts=dict(Counter(r['n_pulses'] for r in good)),
    median_old_nrmse=float(np.median([oldmap[(r['file'],r['row'])]['nrmse'] for r in paired])),
    median_new_nrmse=float(np.median([r['nrmse'] for r in paired])),
    median_relative_rms_improvement=float(np.median([1-r['nrmse']/oldmap[(r['file'],r['row'])]['nrmse'] for r in paired])),
    worse_by_over_1percent=sum(r['nrmse']>oldmap[(r['file'],r['row'])]['nrmse']*1.01 for r in paired))
 verify['summary']=summary
 verify['failure_reasons']=dict(Counter(r.get('message','') for r in rows if not r['success']))
 (out/'comparison_and_verification.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 for tag,sel in [('all_windows',rows),('single_pulses',[r for r in rows if not r['label']])]:
  fig,axs=plt.subplots(1,3,figsize=(15,5.6),layout='constrained')
  groups=[(1,True,'Pileup, unresolved rise/fast','#cd873c','x'),(1,False,'Pileup, separated','#e65c00','.'),
          (0,True,'Single, unresolved rise/fast','#687785','x'),(0,False,'Single, separated','#0066bb','o')]
  for j,ax in enumerate(axs):
   for cls,collapsed,label,c,marker in groups:
    rr=[r for r in sel if r['success'] and r['label']==cls and r['collapsed'][r['dominant']]==collapsed]
    if rr:ax.scatter([r['amplitude_adc']/1000 for r in rr],[r['tau'][r['dominant']][j] for r in rr],
             s=8 if collapsed else 15,alpha=.45 if collapsed else .8,c=c,marker=marker,linewidths=.5,
             label=f'{label} ({len(rr):,})',rasterized=True)
   ax.set(xlabel='Window amplitude ($10^3$ ADC)',ylabel='Time constant (s)',title=[r'Rise: $\tau_r$',r'Fast decay: $\tau_{d1}$',r'Slow decay: $\tau_{d2}$'][j])
   if tag=='all_windows':ax.set_yscale('log')
   ax.grid(alpha=.18)
  leg=axs[0].legend(fontsize=8,loc='best')
  for h in leg.legend_handles:h.set_alpha(1)
  fig.suptitle(f'CUORE | stable tri-exponential refit | {tag.replace("_"," ")}\nRise/fast overlap is explicitly flagged; separated does not mean physically validated.\nPileup: largest fitted component per window; full window amplitude on x-axis.',fontsize=12)
  for ext in ('png','pdf'):fig.savefig(out/f'tau_vs_amplitude_{tag}.{ext}',dpi=200)
  plt.close(fig)
 # Old/new residual comparison, paired rows only, same residual definition.
 fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
 for cls,ax in enumerate(axs):
  ss=[r for r in rows if r['label']==cls and r['success'] and oldmap[(r['file'],r['row'])]['success']]
  x=np.array([oldmap[(r['file'],r['row'])]['nrmse']*100 for r in ss]);y=np.array([r['nrmse']*100 for r in ss])
  ax.scatter(x,y,s=6,alpha=.35);mi=min(x.min(),y.min())*.8;ma=max(x.max(),y.max())*1.2
  ax.plot([mi,ma],[mi,ma],color='black',lw=1,ls='--')
  ax.set(xscale='log',yscale='log',xlabel='Old residual RMS / amplitude (%)',ylabel='New residual RMS / amplitude (%)',title=['Single pulses','Pileup: joint fit, up to 3 pulses'][cls]);ax.grid(alpha=.2)
 fig.suptitle('Fit residual comparison | pileup changes are not a range-only comparison')
 for ext in ('png','pdf'):fig.savefig(out/f'residual_comparison.{ext}',dpi=180)
 plt.close(fig)
 # Range sensitivity on the 144-window pilot.
 fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 for cls,c,label in [(0,'#0066bb','Single'),(1,'#d48331','Pileup')]:
  for split,ls in [('Training','-'),('Test','--')]:
   med=[];collapse=[]
   for cfg in ('compact','moderate','wide'):
    ss=[r for r in pilot if r['config']==cfg and r['label']==cls and split in r['file'] and r['success']]
    med.append(np.median([r['nrmse']*100 for r in ss]))
    collapse.append(100*np.mean([any(r['collapsed']) for r in ss]))
   axs[0].plot(range(3),med,marker='o',color=c,ls=ls,label=f'{label}, {split}')
   axs[1].plot(range(3),collapse,marker='o',color=c,ls=ls,label=f'{label}, {split}')
 for ax in axs:ax.set_xticks(range(3),['Compact','Moderate','Wide']);ax.grid(alpha=.2)
 axs[0].set(ylabel='Median residual RMS / amplitude (%)',title='Residual insensitive to wider search')
 axs[1].set(ylabel='Windows with unresolved rise/fast (%)',title='Degeneracy persists across ranges',ylim=(90,101))
 axs[0].legend(fontsize=9);fig.suptitle('144-window pilot, 3 starts per range; same stable model in all ranges')
 for ext in ('png','pdf'):fig.savefig(out/f'range_comparison.{ext}',dpi=180)
 plt.close(fig)
 prof=json.loads((Path(a.pilot)/'numerical_checks_and_profiles.json').read_text())['profiles']
 fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
 for r in prof:
  pts=r['points'];rss=np.array([v['nrmse']**2 for v in pts]);delta=np.array([v['delta'] for v in pts])
  ax.plot(delta,(rss/rss.min()-1)*100,marker='.',label=f'{r["amplitude_adc"]/1000:.1f}k ADC, row {r["row"]}')
 ax.set(xlabel=r'Fixed $\delta=\log(\tau_{d1}/\tau_r)$; 0 means coincident times',ylabel='Residual sum-of-squares increase (%)',
        title='All other parameters refitted at each separation\nDescriptive profiles, not confidence intervals')
 ax.grid(alpha=.2);ax.legend()
 for ext in ('png','pdf'):fig.savefig(out/f'separation_profiles.{ext}',dpi=180)
 plt.close(fig)
 # Representative raw waveforms and new fit.
 examples=[]
 for cls in (0,1):
  ss=sorted([r for r in rows if r['label']==cls and r['success']],key=lambda r:r['nrmse'])
  examples.extend(ss[int(q*(len(ss)-1))] for q in (.1,.5,.9))
 fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
 for ax,r in zip(axs.flat,examples):
  with h5py.File(Path(a.data)/r['file'],'r') as f:y=(f['Waveform'][r['row']]-r['offset_adc'])/r['amplitude_adc']
  t=np.arange(len(y))/1000
  ax.plot(t,y,lw=.8,label='Data');ax.plot(t,evaluate(t,np.array(r['p']))[0],lw=1,label='New fit')
  ax.set(xlabel='Time (s)',ylabel='Normalized amplitude',title=f'Class {r["label"]}, {r["file"]}, row {r["row"]}\n{r["n_pulses"]} pulse(s), residual {r["nrmse"]:.2%}')
 axs[0,0].legend();fig.savefig(out/'fit_examples.png',dpi=180);plt.close(fig)
 print(json.dumps(verify,indent=2))
if __name__=='__main__':main()
