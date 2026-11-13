"""Small-sample calibration, held-out checks and independent single-pulse refits."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,hashlib,argparse,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_readout_fit.model import fit,Forward
from experiments.cuore_fit_ranges.stable_fitter import evaluate as bare

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_event(root,source,row):
    with h5py.File(Path(root)/source,'r') as h:
        amp=float(h['normalizationScale'][row,0]);offset=float(h['normalizationOffset'][row,0])
        y=(h['Waveform'][row]-offset)/amp
        identity=dict(source_file=source,row=int(row),event_id=int(np.asarray(h['eventId'][row]).item()),
                      amp_adc=amp,offset_adc=offset)
    return y,identity

def worker(job):
    root,source,row,kind,fc,rc,nstarts,tag=job
    y,identity=load_event(root,source,row)
    start=time.perf_counter()
    attempts=fit(y,kind,fc,rc,nstarts=nstarts)
    r=attempts[0]
    close=[a for a in attempts if a['success'] and a['nrmse']<=r['nrmse']*1.001]
    span=float(np.max(np.ptp([a['tau'] for a in close],axis=0)/r['tau'])) if len(close)>1 else 0.
    return dict(**identity,**r,tag=tag,total_elapsed=time.perf_counter()-start,near_optimum_tau_relative_span=span,near_optimum_start_count=len(close),
                starts=[dict(success=a['success'],nrmse=a['nrmse'],nfev=a['nfev']) for a in attempts])

def execute(jobs,out,workers):
    existing=[]
    if out.exists():
        existing=[json.loads(s) for s in out.read_text(encoding='utf-8').splitlines() if s.strip()]
    key=lambda r:(r['source_file'],r['row'],r['tag'])
    done={key(r) for r in existing}
    pending=[j for j in jobs if (j[1],j[2],j[-1]) not in done]
    with out.open('a',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,j):j for j in pending}
        for n,future in enumerate(as_completed(futures),1):
            r=future.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();existing.append(r)
            if n%50==0 or n==len(pending):print(out.name,n,'/',len(pending),flush=True)
    return existing

def selection(root,n):
    result={}
    for name,count in [('cuoreTraining.h5',4500),('cuoreTest.h5',500)]:
        with h5py.File(root/name,'r') as h:
            a=h['normalizationScale'][:count,0]
        order=np.argsort(a)
        result[name]=[int(order[int(round(q*(count-1)))]) for q in np.linspace(.02,.98,n)]
    return result

def baseline(old,keys,root):
    wanted=set(keys);rr=[]
    for s in (old/'fits.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(s);source=r.get('source_file',r.get('file'))
        if (source,r['row']) not in wanted:continue
        y,ident=load_event(root,source,r['row']);t=np.arange(len(y))/1000.
        pred=bare(t,np.asarray(r['p']))[0];res=y-pred;tp=t[np.argmax(y)]
        rr.append(dict(**ident,kind='bare',tag='bare',success=r['success'],p=r['p'],
            tau=r['tau'][0],delta=float(r['p'][5]),alpha=r['alpha'][0],condition=r['condition'],
            nrmse=float(np.sqrt(np.mean(res**2))),
            peak_region_rms=float(np.sqrt(np.mean(res[(t>tp-.3)&(t<tp+.65)]**2))),
            peak_time_value_error=float(pred[np.argmax(y)]-1),max_abs_residual=float(np.max(abs(res)))))
    assert len(rr)==len(wanted),(len(rr),len(wanted))
    return rr

def select_grid(records):
    groups={}
    for r in records:groups.setdefault(r['tag'],[]).append(r)
    summary=[]
    for tag,rr in groups.items():
        summary.append(dict(tag=tag,fc=rr[0]['fixed_fc'],rc=rr[0]['fixed_rc'],
            mse=float(np.mean([r['nrmse']**2 for r in rr])),n=len(rr),
            failures=sum(not r['success'] for r in rr)))
    eligible=[r for r in summary if not r['failures']]
    if not eligible:raise RuntimeError('No grid setting converged on every calibration sample')
    return min(eligible,key=lambda r:r['mse']),summary

def pilot(args):
    root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cal=selection(root,16)['cuoreTraining.h5'];val=selection(root,24)['cuoreTest.h5']
    examples=[385,3629,2864,3519]
    keys=[('cuoreTraining.h5',r) for r in sorted(set(cal+examples))]+[('cuoreTest.h5',r) for r in val]
    manifest=dict(training_calibration=cal,test_validation=val,examples=examples,fs_hz=1000,
        rules='Amplitude ranks selected before fitting; shared response tuned ONLY on 16 training singles.',
        normalization='(raw-normalizationOffset)/normalizationScale, no clipping',
        sources={f:dict(sha256=sha(root/f),bytes=(root/f).stat().st_size) for f in ['cuoreTraining.h5','cuoreTest.h5']})
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError('Existing manifest differs; use a fresh output directory')
    dump(out/'manifest.json',manifest)
    dump(out/'baseline.json',baseline(Path(args.old),keys,root))
    jobs=[(str(root),s,r,k,None,None,2,k+'_free') for s,r in keys for k in ['bessel','rc','nonlinear']]
    execute(jobs,out/'free.jsonl',args.workers)
    jobs=[]
    for fc in [2.,2.5,3.,3.5,4.,5.,6.,8.,12.]:
        jobs += [(str(root),'cuoreTraining.h5',r,'bessel',fc,None,1,f'bessel_{fc}') for r in cal]
    for fc in [4.,6.,8.,12.,20.]:
        for rc in [.02,.04,.06,.08,.10,.14]:
            jobs += [(str(root),'cuoreTraining.h5',r,'rc',fc,rc,1,f'rc_{fc}_{rc}') for r in cal]
    rr=execute(jobs,out/'grid.jsonl',args.workers)
    best_b,bs=select_grid([r for r in rr if r['kind']=='bessel'])
    best_r,rs=select_grid([r for r in rr if r['kind']=='rc'])
    # Local grid refinement remains calibration-only. Record every tried setting.
    all_rows=rr
    for iteration in range(2):
        jobs=[]
        for kind,best in [('bessel',best_b),('rc',best_r)]:
            fcs=[best['fc']*v for v in [.85,1.,1.15]]
            rcs=[None] if kind=='bessel' else [best['rc']*v for v in [.85,1.,1.15]]
            for fc in fcs:
                for rc in rcs:
                    tag=f'{kind}_ref{iteration}_{fc:.7g}_{rc}'
                    jobs += [(str(root),'cuoreTraining.h5',r,kind,fc,rc,1,tag) for r in cal]
        all_rows += execute(jobs,out/f'refine{iteration}.jsonl',args.workers)
        best_b,bs=select_grid([r for r in all_rows if r['kind']=='bessel'])
        best_r,rs=select_grid([r for r in all_rows if r['kind']=='rc'])
    settings=dict(bessel=best_b,rc=best_r,grid_bessel=bs,grid_rc=rs)
    dump(out/'shared_settings.json',settings)
    jobs=[]
    for s,r in keys:
        jobs += [(str(root),s,r,'bessel',best_b['fc'],None,3,'bessel_shared'),
                 (str(root),s,r,'rc',best_r['fc'],best_r['rc'],3,'rc_shared'),
                 (str(root),s,r,'nonlinear',best_r['fc'],best_r['rc'],3,'nonlinear_shared')]
    rr=execute(jobs,out/'shared.jsonl',args.workers)
    print('PILOT_COMPLETE',json.dumps({k:settings[k] for k in ['bessel','rc']}),flush=True)

def full(args):
    root=Path(args.data);out=Path(args.output)
    settings=json.loads((out/'shared_settings.json').read_text())
    best=settings['rc']
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for source,info in manifest['sources'].items():
        if sha(root/source)!=info['sha256']: raise RuntimeError('Input changed since pilot')
    config=dict(model='rc',fc=best['fc'],rc=best['rc'],nstarts=2,oversample=2,fs_hz=1000)
    if (out/'full_config.json').exists() and json.loads((out/'full_config.json').read_text())!=config:
        raise RuntimeError('Full-run configuration changed; use a fresh output directory')
    dump(out/'full_config.json',config)
    started=time.perf_counter()
    jobs=[(str(root),s,r,'rc',best['fc'],best['rc'],2,'rc_shared')
          for s,n in [('cuoreTraining.h5',4500),('cuoreTest.h5',500)] for r in range(n)]
    rows=execute(jobs,out/'full_single.jsonl',args.workers)
    assert len(rows)==5000
    for source in ['cuoreTraining.h5','cuoreTest.h5']:
        rr=sorted([r for r in rows if r['source_file']==source],key=lambda r:r['row'])
        path=out/(Path(source).stem+'_readout_fit.h5')
        with h5py.File(path,'w') as h,h5py.File(root/source,'r') as src:
            h.attrs['source_file']=source;h.attrs['source_sha256']=sha(root/source)
            h.attrs['model']='effective linear RC + causal sixth-order Bessel; NOT calibrated thermal inversion'
            h.attrs['fs_hz']=1000.;h.attrs['bessel_norm']='mag';h.attrs['fc_hz']=best['fc'];h.attrs['rc_s']=best['rc']
            h.attrs['parameter_order']='baseline,log_C,log_D,t0_s,log_tau_r,log_tau_fast_over_r,log_tau_slow_minus_fast'
            h.attrs['waveforms_stored']=False;h.attrs['oversample']=2
            for field in ['row','event_id','amp_adc','offset_adc','success','nrmse','peak_region_rms',
                          'max_abs_residual','peak_time_value_error','tau','p','delta','alpha','condition','total_elapsed',
                          'near_optimum_tau_relative_span','near_optimum_start_count']:
                h.create_dataset(field,data=np.asarray([r[field] for r in rr]))
            h.create_dataset('boundary',data=np.asarray([bool(r['boundary']) for r in rr]))
            for field in ['normalizationOffset','normalizationMaximum','normalizationScale']:
                h.create_dataset(field,data=src[field][:len(rr)])
            for field,value in src.attrs.items():h.attrs['source_'+field]=value
    dump(out/'full_complete.json',dict(n=5000,success=sum(r['success'] for r in rows),
        settings=best,invocation_elapsed_s=time.perf_counter()-started,h5_sha256={p.name:sha(p) for p in out.glob('*_readout_fit.h5')}))
    print('FULL_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','full'])
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--old',required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();{'pilot':pilot,'full':full}[args.stage](args)
