"""CLI: range pilot, full refit and reports for CUORE release."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, concurrent.futures, hashlib, json, time, shutil
from pathlib import Path
import numpy as np
import h5py
from stable_fitter import CONFIGS,fit,solve,evaluate

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_job(path,row):
    with h5py.File(path,'r') as f:
        amp=float(f['normalizationScale'][row].item()); offset=float(f['normalizationOffset'][row].item())
        y=(f['Waveform'][row]-offset)/amp
        return y,amp,offset,int(f['eventId'][row])

def worker(job):
    path,row,label,config,stage=job
    start=time.perf_counter()
    r=dict(file=Path(path).name,row=row,label=label,config=config,success=False)
    try:
        y,amp,offset,event_id=load_job(path,row)
        r.update(amplitude_adc=amp,offset_adc=offset,event_id=event_id)
        if stage=='pilot':
            results=fit(y,label,config,stride=1,nseeds=3,max_nfev=500)
        else:
            # Coarse candidates initialize full-sample solves; outputs always fit all 10000 samples.
            coarse=fit(y,label,config,stride=5,nseeds=3,max_nfev=350)
            results=[]
            for c in coarse[:2]:
                z=solve(np.arange(len(y))/1000.,y,c['p'],CONFIGS[config],c['windows'],max_nfev=600)
                for key in ('detected_peaks','missing_peak','windows','seed','n_pulses'): z[key]=c[key]
                results.append(z)
        successful=[z for z in results if z['success']]
        best=min(successful or results,key=lambda z:z['nrmse'])
        r.update(best)
        # Keep all starts and unsuccessful iterates in pilot; compact full-run diagnostics.
        near=[z for z in results if z['nrmse']<=best['nrmse']*1.005]
        tau=np.array([z['tau'] for z in near])
        r['near_optimum_tau_relative_span']=float(np.max(np.ptp(tau,axis=0)/np.maximum(np.array(best['tau']),1e-9))) if len(near)>1 else 0.
        r['start_nrmse']=[z['nrmse'] for z in results]
        r['start_success']=[z['success'] for z in results]
        if stage=='pilot': r['all_starts']=results
        r['peak_amp_adc']=(np.array(r['peak_amp'])*amp).tolist()
        r['baseline_noise_fraction']=float(np.std(y[:500]))
        r['residual_to_baseline_noise']=r['nrmse']/max(r['baseline_noise_fraction'],1e-12)
        r['diagnostic_pass']=bool(r['success'] and r['nrmse']<=.02 and not any(r['collapsed'])
            and not any(r['boundary']) and not r['missing_peak'] and r['condition']<1e6
            and r['near_optimum_tau_relative_span']<.1)
    except Exception as e: r.update(success=False,message=repr(e))
    r['total_elapsed_s']=time.perf_counter()-start
    return r

def select_jobs(a):
    jobs=[]
    for name in ('cuoreTraining.h5','cuoreTest.h5'):
        path=Path(a.data)/name
        with h5py.File(path,'r') as f:
            n=len(f['Waveform']); amp=f['normalizationScale'][:].ravel()
            labels=f['Labels'][:] if 'Labels' in f else np.r_[np.zeros(int(f.attrs['cleanPulseCount']),int),np.ones(int(f.attrs['pileupPulseCount']),int)]
            for cls in (0,1):
                ids=np.flatnonzero(labels==cls)
                if a.stage=='pilot':
                    count=(64 if cls==0 else 32) if 'Training' in name else (32 if cls==0 else 16)
                    order=ids[np.argsort(amp[ids])]
                    ids=order[np.linspace(0,len(order)-1,count,dtype=int)]
                for i in ids:
                    for cfg in (list(CONFIGS) if a.stage=='pilot' else [a.config]):
                        jobs.append((str(path),int(i),cls,cfg,a.stage))
    return jobs

def export(rows,a,manifest):
    out=Path(a.output)
    if a.stage=='full':
        for name in ('cuoreTraining.h5','cuoreTest.h5'):
            rr=sorted([r for r in rows if r['file']==name],key=lambda r:r['row'])
            with h5py.File(out/name.replace('.h5','_fit.h5'),'w') as f:
                f.attrs['manifest_json']=json.dumps(manifest)
                f.attrs['source_file']=str(Path(a.data)/name)
                f.attrs['normalization']='(Waveform-normalizationOffset)/normalizationScale; no clipping'
                f.attrs['parameter_layout']='B + repeated(log C,log D,t0,log tr,delta,log slow_gap); NaN padded; seconds; amplitude normalized'
                f.attrs['formula']='C*(exp(-t/tf)-exp(-t/tr))/delta + D*(exp(-t/ts)-exp(-t/tr)); tf=tr*exp(delta), ts=tf+exp(loggap); delta=0 uses (t/tr)*exp(-t/tr)'
                f.attrs['collapsed_definition']='delta < 0.05: rise and fast decay not resolved by this diagnostic'
                f.attrs['label_source']='Labels' if 'Training' in name else 'rowOrder, cleanPulseCount, pileupPulseCount attributes'
                for key in ('row','event_id','label','success','amplitude_adc','offset_adc','n_pulses','dominant','nrmse','condition','missing_peak','detected_peaks','diagnostic_pass','near_optimum_tau_relative_span','baseline_noise_fraction','residual_to_baseline_noise','total_elapsed_s'):
                    f.create_dataset(key,data=[r.get(key,np.nan) for r in rr])
                for key,width in [('p',19),('alpha',3),('peak_amp_adc',3),('collapsed',3),('boundary',3)]:
                    values=np.full((len(rr),width),np.nan)
                    for i,r in enumerate(rr):
                        v=r.get(key,[]); values[i,:len(v)]=v
                    f.create_dataset(key,data=values)
                taus=np.full((len(rr),3,3),np.nan)
                for i,r in enumerate(rr):
                    v=r.get('tau',[]); taus[i,:len(v),:]=v
                f.create_dataset('tau_s',data=taus)
                f.create_dataset('message',data=[r.get('message','') for r in rr],dtype=h5py.string_dtype())
    summary={}
    for cfg in sorted(set(r['config'] for r in rows)):
        summary[cfg]={}
        for split in ('Training','Test'):
            summary[cfg][split]={}
            for cls in (0,1):
                ss=[r for r in rows if r['config']==cfg and split in r['file'] and r['label']==cls]
                good=[r for r in ss if r['success']]
                summary[cfg][split][str(cls)]=dict(n=len(ss),success=len(good),
                    collapsed=sum(any(r['collapsed']) for r in good),boundary=sum(any(r['boundary']) for r in good),
                    diagnostic_pass=sum(r['diagnostic_pass'] for r in good),
                    median_nrmse=float(np.median([r['nrmse'] for r in good])) if good else None,
                    median_condition=float(np.median([r['condition'] for r in good])) if good else None,
                    median_start_tau_span=float(np.median([r['near_optimum_tau_relative_span'] for r in good])) if good else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser()
    for key in ('data','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--stage',choices=['pilot','full'],required=True);p.add_argument('--config',default='moderate',choices=list(CONFIGS))
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(stage=a.stage,config=a.config,ranges=CONFIGS,fs_hz=1000,
        sources={n:sha(Path(a.data)/n) for n in ('cuoreTraining.h5','cuoreTest.h5')},
        code={n:sha(Path(__file__).parent/n) for n in ('stable_fitter.py','run_study.py')},
        notes='Independent equivalent ordered tri-exponential family with exact repeated-pole limit; full samples for final solve. Joint independent shape fit for up to 3 detected pileup pulses. No P2 +/-20% rise constraint. No physical-identifiability guarantee.',
        diagnostic='RMS<=2%, no near-bound coefficients/time limits, no delta<0.05, no missing pileup peak, scaled-Jacobian condition<1e6, near-best start tau span<10%. Exploratory only.')
    mp=out/'manifest.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest,'Cannot mix changed inputs or methods with checkpoint.'
    else: mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for n in manifest['code']: shutil.copy2(Path(__file__).parent/n,out/(n+'.snapshot'))
    checkpoint=out/'fits.jsonl'
    rows=[json.loads(s) for s in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    done={(r['file'],r['row'],r['config']) for r in rows}
    jobs=[j for j in select_jobs(a) if (Path(j[0]).name,j[1],j[3]) not in done]
    start=time.perf_counter();print(f'Pending {len(jobs)}',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool,checkpoint.open('a',encoding='utf-8') as stream:
        futures=[pool.submit(worker,j) for j in jobs]
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();rows.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
            if len(rows)%50==0: print(f'{len(rows)} fits, wall {time.perf_counter()-start:.1f}s',flush=True)
    export(rows,a,manifest);print(f'COMPLETE {time.perf_counter()-start:.1f}s',flush=True)
if __name__=='__main__':main()
