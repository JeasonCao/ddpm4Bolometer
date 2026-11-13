"""Amplitude-conditioned CUORE pulse-shape distributions and noise controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,argparse,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import periodogram
from scipy.stats import spearmanr,theilslopes
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_readout_fit.model import Forward,fit
from experiments.cuore_readout_fit.run_study import sha,load_event
CENTERS=[6500.,10000.,18000.,28500.]
HALF_WIDTH=.025
KEYS=['tau_r_ms','tau_fast_ms','tau_slow_ms','alpha','rise_10_90_ms','decay_90_30_ms']

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quant(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),q16=float(np.quantile(x,.16)),q84=float(np.quantile(x,.84)),
                q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),std=float(np.std(x,ddof=1)) if len(x)>1 else None)
def crossing(y,peak,level,rising):
    a,b=(y[:peak],y[1:peak+1]) if rising else (y[peak:-1],y[peak+1:])
    inds=np.flatnonzero((a<=level)&(b>level) if rising else (a>=level)&(b<level))
    if len(inds)==0:raise ValueError('Missing fractional-height crossing')
    i=int(inds[-1] if rising else inds[0])+(0 if rising else peak)
    return (i+(level-y[i])/(y[i+1]-y[i]))/1000.
def shape_features(y,sigma=3):
    z=gaussian_filter1d(y,sigma)
    peak=int(np.argmax(z));height=float(z[peak]);z=z/height
    t10=crossing(z,peak,.1,True);t50=crossing(z,peak,.5,True);t90=crossing(z,peak,.9,True)
    d90=crossing(z,peak,.9,False);d30=crossing(z,peak,.3,False)
    return dict(rise_10_90_ms=(t90-t10)*1000,decay_90_30_ms=(d30-d90)*1000,
                t50_s=t50,smoothed_peak=height,peak_time_s=peak/1000.)

def extract(a):
    root=Path(a.data);fits=Path(a.fits);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(l) for l in (fits/'full_single.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (Path(a.old)/'fits.jsonl').read_text().splitlines()]
    previous={(r['file'],r['row']):bool(r['collapsed'][0] and r['success']) for r in old if r['label']==0}
    manifest=dict(sources={f:sha(root/f) for f in ['cuoreTraining.h5','cuoreTest.h5']},
                  fit_h5={f:sha(fits/f) for f in ['cuoreTraining_readout_fit.h5','cuoreTest_readout_fit.h5']},
                  fit_json_sha256=sha(fits/'full_single.jsonl'),fs_hz=1000,energy_available=False,
                  center_adc=CENTERS,half_width_fraction=HALF_WIDTH,primary_statistics='all 5000 labeled single pulses',
                  scatter_selection='previous 4980-point selection, no separate markers for other 20',
                  smoothing_sigma_samples=[2,3,5],alignment='smoothed rising 50% crossing; no time stretching',
                  noise_model='stationary Gaussian spectral synthesis from pooled pretrigger periodograms; RMS sampled per band',
                  seed=261113,noise_repeats=a.repeats)
    for source in manifest['sources']:
        with h5py.File(fits/(Path(source).stem+'_readout_fit.h5')) as saved:
            assert saved.attrs['source_sha256']==manifest['sources'][source]
    dump(out/'manifest.json',manifest)
    t=np.arange(10000)/1000.;aligned_t=np.arange(10000)/1000.-3.
    selected={i:[] for i in range(4)};selected_psd={i:[] for i in range(4)}
    rows=[];params=[];normalization=[];band_indices={i:[] for i in range(4)}
    fw=Forward(10000,'rc',5.1,.068,2)
    start=time.perf_counter()
    handles={s:h5py.File(root/s,'r') for s in ['cuoreTraining.h5','cuoreTest.h5']}
    try:
        for n,r in enumerate(sorted(records,key=lambda r:(r['source_file'],r['row']))):
            src=handles[r['source_file']];raw=src['Waveform'][r['row']]
            y=(raw-r['offset_adc'])/r['amp_adc'];base=raw[:2500]-np.mean(raw[:2500])
            f=shape_features(y);f2=shape_features(y,2);f5=shape_features(y,5)
            p=np.asarray(r['p']);pred=fw.evaluate(p)[0];pf=shape_features(pred-p[0])
            band=next((j for j,c in enumerate(CENTERS) if abs(r['amp_adc']/c-1)<=HALF_WIDTH),-1)
            row=dict(index=n,source_file=r['source_file'],source_row=r['row'],event_id=r['event_id'],
                amplitude_adc=r['amp_adc'],offset_adc=r['offset_adc'],legacy_4980=previous[r['source_file'],r['row']],
                band=band,baseline_rms_adc=float(np.std(base,ddof=1)),baseline_noise_fraction=float(np.std(base,ddof=1)/r['amp_adc']),
                tau_r_ms=r['tau'][0]*1000,tau_fast_ms=r['tau'][1]*1000,tau_slow_ms=r['tau'][2]*1000,
                alpha=r['alpha'],rise_10_90_ms=f['rise_10_90_ms'],decay_90_30_ms=f['decay_90_30_ms'],
                rise_sigma2_ms=f2['rise_10_90_ms'],rise_sigma5_ms=f5['rise_10_90_ms'],
                decay_sigma2_ms=f2['decay_90_30_ms'],decay_sigma5_ms=f5['decay_90_30_ms'],
                fit_rise_10_90_ms=pf['rise_10_90_ms'],fit_decay_90_30_ms=pf['decay_90_30_ms'],
                t50_s=f['t50_s'],smoothed_peak_adc=f['smoothed_peak']*r['amp_adc'],
                nrmse=r['nrmse'],collapsed=r['delta']<.05,condition=r['condition'])
            rows.append(row);params.append(p);normalization.append(r['amp_adc'])
            if band>=0:
                aligned=np.interp(aligned_t+f['t50_s'],t,y/f['smoothed_peak'],left=0.,right=float(y[-1]/f['smoothed_peak']))
                selected[band].append(aligned.astype('f4'));band_indices[band].append(n)
                freq,psd=periodogram(base,fs=1000.,detrend='constant',scaling='density')
                selected_psd[band].append(psd)
            if (n+1)%1000==0:print('features',n+1,'/5000',flush=True)
    finally:
        for h in handles.values():h.close()
    assert len(rows)==5000 and sum(r['legacy_4980'] for r in rows)==4980
    csvwrite(out/'event_features.csv',rows)
    with h5py.File(out/'shape_distributions.h5','w') as h:
        h.attrs['energy_available']=False;h.attrs['amplitude_units']='ADC';h.attrs['source_manifest']='manifest.json'
        g=h.create_group('events')
        for key in rows[0]:
            values=[r[key] for r in rows]
            g.create_dataset(key,data=values,dtype=h5py.string_dtype('utf-8') if isinstance(values[0],str) else None)
        g.create_dataset('fit_p',data=params)
        h.create_dataset('aligned_time_s',data=aligned_t)
        h.create_dataset('quantile_levels',data=[.025,.16,.5,.84,.975])
        for j,c in enumerate(CENTERS):
            b=h.create_group(f'bands/{j}');b.attrs['center_adc']=c;b.attrs['half_width_fraction']=HALF_WIDTH
            b.create_dataset('event_indices',data=band_indices[j])
            b.create_dataset('raw_aligned_waveform_quantiles',data=np.quantile(selected[j],[.025,.16,.5,.84,.975],axis=0),compression='gzip')
            b.create_dataset('noise_frequency_hz',data=freq)
            b.create_dataset('noise_psd_adc2_per_hz',data=np.mean(selected_psd[j],axis=0))
            local=[rows[k] for k in band_indices[j]]
            v=np.array([[r[k] for k in ['tau_r_ms','tau_fast_ms','tau_slow_ms','alpha']] for r in local])
            z=np.log(v[:,:3]);z=np.column_stack([z,v[:,3]])
            med=np.median(z,axis=0);mad=np.median(abs(z-med),axis=0)
            pos=int(np.argmin(np.sum(((z-med)/np.maximum(mad,1e-8))**2,axis=1)))
            representative=band_indices[j][pos];b.attrs['representative_index']=representative
            p=np.array(params[representative]);p[0]=0.
            signal=fw.evaluate(p)[0];signal/=np.max(signal)
            b.create_dataset('fixed_model_template',data=signal,compression='gzip')
            empirical=np.median(np.asarray(selected[j]),axis=0)
            empirical=gaussian_filter1d(empirical,3);empirical-=np.mean(empirical[:2500]);empirical/=np.max(empirical)
            b.create_dataset('fixed_empirical_template',data=empirical,compression='gzip')
    dump(out/'extraction_complete.json',dict(n=5000,bands={str(j):len(v) for j,v in selected.items()},
        elapsed_s=time.perf_counter()-start))
    print('EXTRACTION_COMPLETE',flush=True)

def noise_worker(job):
    file,band,kind,rep,seed=job;rng=np.random.default_rng(seed)
    with h5py.File(file,'r') as h:
        b=h[f'bands/{band}'];inds=b['event_indices'][:]
        center=float(b.attrs['center_adc']);signal=b[f'fixed_{kind}_template'][:]
        freq=b['noise_frequency_hz'][:];psd=b['noise_psd_adc2_per_hz'][:]
        donor=int(rng.choice(inds));sigma=float(h['events/baseline_rms_adc'][donor])
        rid=int(b.attrs['representative_index']);initial=h['events/fit_p'][rid]
    frequencies=np.fft.rfftfreq(10000,.001)
    s=np.interp(frequencies,freq[1:],psd[1:]);s[0]=0 # flat continuation below the lowest resolved frequency
    re=rng.normal(size=len(s));im=rng.normal(size=len(s))
    fourier=(re+1j*im)*np.sqrt(s*1000*10000/4)
    fourier[0]=0;fourier[-1]=re[-1]*np.sqrt(s[-1]*1000*10000)
    noise=np.fft.irfft(fourier,n=10000);noise*=sigma/np.std(noise)
    raw=center*signal+noise;offset=np.mean(raw[:2500]);scale=np.max(raw)-offset
    y=(raw-offset)/scale
    # A generic start plus a near-central start; both optimize against noisy full windows.
    initial=np.array(initial);initial[0]=0.
    rr=fit(y,'rc',5.1,.068,nstarts=1,initial=initial)
    r=rr[0];feat=shape_features(y)
    return dict(band=band,kind=kind,replicate=rep,seed=seed,noise_donor_index=donor,
        noise_rms_adc=sigma,synthetic_baseline_rms_adc=float(np.std(noise[:2500])),
        amplitude_adc=float(scale),success=r['success'],nrmse=r['nrmse'],
        tau_r_ms=r['tau'][0]*1000,tau_fast_ms=r['tau'][1]*1000,tau_slow_ms=r['tau'][2]*1000,
        alpha=r['alpha'],rise_10_90_ms=feat['rise_10_90_ms'],decay_90_30_ms=feat['decay_90_30_ms'],
        collapsed=r['delta']<.05,elapsed_s=r['elapsed'])

def sensitivity_worker(job):
    data,source,row,fc,rc,index,band=job
    y,_=load_event(data,source,row);r=fit(y,'rc',fc,rc,nstarts=2)[0]
    return dict(index=index,band=band,source_file=source,source_row=row,fc=fc,rc=rc,
        success=r['success'],nrmse=r['nrmse'],tau_r_ms=r['tau'][0]*1000,
        tau_fast_ms=r['tau'][1]*1000,tau_slow_ms=r['tau'][2]*1000,alpha=r['alpha'])

def execute(jobs,fn,path,workers):
    if path.exists():raise RuntimeError('Control output exists; do not accidentally duplicate a run')
    with path.open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(fn,j) for j in jobs]
        for n,fut in enumerate(as_completed(pending),1):
            r=fut.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
            if n%100==0 or n==len(jobs):print(path.name,n,'/',len(jobs),flush=True)

def controls(a):
    out=Path(a.output);file=out/'shape_distributions.h5'
    jobs=[(str(file),band,kind,rep,261113+band*10000+kindid*1000+rep)
          for band in range(4) for kindid,kind in enumerate(['model','empirical']) for rep in range(a.repeats)]
    execute(jobs,noise_worker,out/'noise_controls.jsonl',a.workers)
    rng=np.random.default_rng(261113);jobs=[]
    with h5py.File(file,'r') as h:
        for band in range(4):
            indices=rng.choice(h[f'bands/{band}/event_indices'][:],size=8,replace=False)
            for index in indices:
                source=h['events/source_file'][index].decode();row=int(h['events/source_row'][index])
                for fc,rc in [(5.1,.068),(5.1,.08),(6.,.068),(6.,.08)]:
                    jobs.append((a.data,source,row,fc,rc,int(index),band))
    execute(jobs,sensitivity_worker,out/'response_sensitivity.jsonl',a.workers)
    print('CONTROLS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','controls'])
    for name in ['data','fits','old','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--repeats',type=int,default=96);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();{'extract':extract,'controls':controls}[a.stage](a)
