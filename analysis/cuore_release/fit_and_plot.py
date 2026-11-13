"""Fit CUORE public release 20721645 and plot amplitude versus three time constants.
Inputs read-only; all rows retained; pileup plotted using largest fitted component.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, concurrent.futures, hashlib, importlib.util, json, shutil, sys, time
from pathlib import Path
import numpy as np
import h5py

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init(snapshot):
    global fit
    spec=importlib.util.spec_from_file_location('fitter_snapshot',snapshot)
    fit=importlib.util.module_from_spec(spec); sys.modules[spec.name]=fit; spec.loader.exec_module(fit)
    original=fit.curve_fit
    def capped(*a,**kw):
        kw['maxfev']=2000
        return original(*a,**kw)
    fit.curve_fit=capped

def one(job):
    path,row,label=job; start=time.perf_counter()
    r=dict(file=Path(path).name,row=row,label=label,success=False)
    try:
        with h5py.File(path,'r') as f:
            raw=f['Waveform'][row]; off=float(f['normalizationOffset'][row].item())
            amp=float(f['normalizationScale'][row].item())
            r.update(event_id=int(f['eventId'][row]),amplitude_adc=amp,offset_adc=off)
        y=(raw-off)/amp*.01  # Numerical scale ONLY, not volts; no clipping.
        detected=len(fit._detect_peaks(y-fit._baseline(y),1000.,2))
        fr=fit.fit_pulse(y,n_pulses=1+label,fs=1000.)
        r.update(success=bool(fr.success),message=fr.message,detected_peaks=detected)
        if fr.success:
            pred=fr.model_fn(np.arange(len(y))/1000.,*fr.params)
            rms=float(np.sqrt(np.mean((pred-y)**2))/.01)
            _,lo,hi=fit._setup_single(y,1000.)
            p=fr.params if label==0 else np.r_[fr.params[0],fr.params[1:7]]
            bound=any(min((p[j]-lo[j])/(hi[j]-lo[j]),(hi[j]-p[j])/(hi[j]-lo[j]))<.001 for j in (3,4,5,6))
            if label:
                p=fr.params[7:13]; tr1=fr.params[3]
                lows=[max(.01,tr1*.8),.01,max(.01,tr1*1.5),.01]
                highs=[min(.2,tr1*1.2),.99,10.,10.]
                bound |= any(min((p[j+2]-l)/(h-l),(h-p[j+2])/(h-l))<.001 for j,l,h in zip(range(4),lows,highs))
            ordered=all(a<b<c for a,b,c in zip(fr.tau_r,fr.tau_d,fr.tau_d2))
            sep=all(c/b>1.05 for b,c in zip(fr.tau_d,fr.tau_d2))
            good=rms<=.02 and not bound and ordered and sep
            if label: good &= detected==2 and min(fr.peak_amps)>.0002
            r.update(params=fr.params.tolist(),tau_r=fr.tau_r,tau_fast=fr.tau_d,tau_slow=fr.tau_d2,
                     alpha=fr.alpha,peak_amp_adc=(np.array(fr.peak_amps)*amp/.01).tolist(),
                     onset_s=fr.onset_times,dominant=int(np.argmax(fr.peak_amps)),nrmse=rms,
                     chi2_ndf=fr.chi2_per_ndf,near_bound=bool(bound),diagnostic_pass=bool(good),
                     ordered=bool(ordered),separated=bool(sep))
    except Exception as e: r.update(success=False,message=repr(e))
    r['elapsed_s']=time.perf_counter()-start
    return r

def export(rows,a,manifest):
    out,plots=Path(a.output),Path(a.plots)
    for name in ('cuoreTraining.h5','cuoreTest.h5'):
        rr=sorted([r for r in rows if r['file']==name],key=lambda r:r['row'])
        if not rr: continue
        with h5py.File(out/name.replace('.h5','_fit.h5'),'w') as f:
            f.attrs['provenance_json']=json.dumps(manifest)
            f.attrs['source_file']=str(Path(a.data)/name)
            f.attrs['units']='tau/onset: s; amplitude: ADC; raw fit params: numerical normalized units'
            f.attrs['label_source']='Labels' if 'Training' in name else 'source rowOrder and cleanPulseCount attributes'
            f.attrs['normalization']='(Waveform-offset)/scale*0.01, no clipping'
            f.attrs['params_layout']='single B,A,t0,tr,alpha,td1,td2; pileup B,A1,t01,tr1,alpha1,td11,td21,A2,t02,tr2,alpha2,td12,td22; NaN padding'
            for k in ('row','event_id','label','success','amplitude_adc','offset_adc','detected_peaks','dominant','nrmse','chi2_ndf','near_bound','diagnostic_pass','ordered','separated','elapsed_s'):
                f.create_dataset(k,data=[r.get(k,np.nan) for r in rr])
            for k,w in [('params',13),('tau_r',2),('tau_fast',2),('tau_slow',2),('alpha',2),('peak_amp_adc',2),('onset_s',2)]:
                arr=np.full((len(rr),w),np.nan)
                for i,r in enumerate(rr):
                    v=r.get(k,[]); arr[i,:len(v)]=v
                f.create_dataset(k,data=arr)
            f.create_dataset('message',data=[r.get('message','') for r in rr],dtype=h5py.string_dtype())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    keys=['tau_r','tau_fast','tau_slow']; titles=['Rise time constant','Fast decay time constant','Slow decay time constant']
    for tag,selected in [('all_windows',rows),('single_pulses',[r for r in rows if r['label']==0])]:
        fig,axes=plt.subplots(1,3,figsize=(15,5.3),layout='constrained')
        groups=[(1,False,'Pileup: flagged','#e8b787','x'),(1,True,'Pileup: diagnostic pass','#de7c22','.'),(0,False,'Single: flagged','#bbbbbb','x'),(0,True,'Single: diagnostic pass','#1764a0','.')]
        for ax,k,title in zip(axes,keys,titles):
            for cls,good,label,color,marker in groups:
                ss=[r for r in selected if r['success'] and r['label']==cls and r['diagnostic_pass']==good]
                if ss: ax.scatter([r['amplitude_adc'] for r in ss],[r[k][r['dominant']] for r in ss],s=8,alpha=.4,color=color,marker=marker,linewidths=.4,label=f'{label} (n={len(ss):,})',rasterized=True)
            ax.set(xlabel='Window amplitude: maximum - baseline (ADC)',ylabel='Time constant (s)',title=title,yscale='log'); ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        ok=sum(r['success'] for r in selected)
        fig.suptitle(f'CUORE public release | {tag.replace("_"," ")} | fitted {ok:,}/{len(selected):,}\nOne point per window; pileup uses largest fitted component. Logarithmic time axes.',fontsize=13)
        for ext in ('png','pdf'): fig.savefig(plots/f'tau_vs_amplitude_{tag}.{ext}',dpi=200)
        plt.close(fig)
    init(str(out/'fitter_snapshot.py'))
    examples=[]
    for cls in (0,1):
        ss=sorted([r for r in rows if r['label']==cls and r['success']],key=lambda r:r['nrmse'])
        if ss: examples.extend(ss[int(q*(len(ss)-1))] for q in (.1,.5,.9))
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,r in zip(axes.flat,examples):
        with h5py.File(Path(a.data)/r['file'],'r') as f: y=f['Waveform'][r['row']]
        t=np.arange(len(y))/1000; model=fit.biexp_single if r['label']==0 else fit.biexp_double_free
        pred=model(t,*r['params'])/.01
        ax.plot(t,(y-r['offset_adc'])/r['amplitude_adc'],lw=.8,label='Data'); ax.plot(t,pred,lw=1,label='Fit')
        ax.set(title=f'Class {r["label"]} | {r["file"]}, row {r["row"]}\nRMS / amplitude = {r["nrmse"]:.2%}',xlabel='Time (s)',ylabel='Normalized amplitude')
    axes[0,0].legend(); fig.savefig(plots/'fit_examples.png',dpi=160); plt.close(fig)
    summary=dict(rows=len(rows),success=sum(r['success'] for r in rows),groups={str(c):dict(rows=sum(r['label']==c for r in rows),success=sum(r['label']==c and r['success'] for r in rows),diagnostic_pass=sum(r['label']==c and r.get('diagnostic_pass',False) for r in rows)) for c in (0,1)},elapsed_fit_s_sum=sum(r['elapsed_s'] for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ('data','fitter','output','plots'): p.add_argument('--'+k,required=True)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--pilot',type=int,default=0)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); Path(a.plots).mkdir(parents=True,exist_ok=True)
    snapshot=out/'fitter_snapshot.py'
    if not snapshot.exists(): shutil.copy2(a.fitter,snapshot)
    manifest=dict(source_url='https://zenodo.org/records/20721645',fs_hz=1000,sampling_reference='https://arxiv.org/html/2607.02548v1',fitter_sha256=sha(snapshot),script_sha256=sha(__file__),normalization_peak=.01,maxfev=2000,workers=a.workers,pilot=a.pilot,
      diagnostic_rules='RMS/window amplitude <=2%; shape params not within 0.1% of bound range; tr<td1<td2; td2/td1>1.05; pileup two detected peaks and each fitted peak>2% window amplitude. Exploratory screen, not uncertainty validation.',
      limitations='Sequential two-pulse model for pileup, which can contain >2 pulses. P2 rise constrained +/-20% of P1. Main plot uses largest fitted component versus full-window raw amplitude, not component energy. Single setup actually raises decay lower bounds to 0.06708 s and 0.45 s.',sources={})
    jobs=[]
    for name in ('cuoreTraining.h5','cuoreTest.h5'):
        path=Path(a.data)/name; manifest['sources'][name]=dict(path=str(path),sha256=sha(path))
        with h5py.File(path,'r') as f:
            n=len(f['Waveform']); assert f['Waveform'].shape[1]==10000
            if 'Labels' in f: labels=f['Labels'][:]
            else:
                assert f.attrs['rowOrder']=='clean rows followed by pileup rows'
                count=int(f.attrs['cleanPulseCount']); labels=np.r_[np.zeros(count,dtype=int),np.ones(n-count,dtype=int)]
            for cls in (0,1):
                ids=np.flatnonzero(labels==cls)
                if a.pilot: ids=ids[np.linspace(0,len(ids)-1,a.pilot,dtype=int)]
                jobs.extend((str(path),int(i),cls) for i in ids)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    checkpoint=out/'fits.jsonl'
    rows=[json.loads(s) for s in checkpoint.read_text(encoding='utf-8').splitlines()] if checkpoint.exists() else []
    done={(r['file'],r['row']) for r in rows}; jobs=[j for j in jobs if (Path(j[0]).name,j[1]) not in done]
    start=time.perf_counter(); print(f'Pending {len(jobs)}; cached {len(rows)}',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(snapshot),)) as pool, checkpoint.open('a',encoding='utf-8') as stream:
        futures=[pool.submit(one,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            r=future.result(); rows.append(r); stream.write(json.dumps(r)+'\n'); stream.flush()
            if len(rows)%100==0 or a.pilot: print(f'{len(rows)} rows; elapsed {time.perf_counter()-start:.1f}s; latest {r["elapsed_s"]:.2f}s',flush=True)
    export(rows,a,manifest); print(f'Complete; wall {time.perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__': main()
