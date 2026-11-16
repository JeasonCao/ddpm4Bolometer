"""Compare measured CUORE baseline noise with simulated evaluation noise.

Energy-equivalent point RMS is distinct from fixed-template baseline resolution.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,sys,hashlib,time
from pathlib import Path
import numpy as np,h5py
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_shape_distribution.run_study import shape_features
from scripts.window_of import estimate
from analysis.eval_noise_shape.run_analysis import ENERGIES,LEVELS,sha
KEYS=['rise_10_90_ms','decay_90_30_ms'];COLORS=['#277db3','#d78437','#4a9858','#8863b2']
def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def qstats(v):
 v=np.asarray(v);v=v[np.isfinite(v)];q=np.quantile(v,[.025,.16,.5,.84,.975]);return dict(n=len(v),mean=float(v.mean()),std=float(v.std(ddof=1)),q025=float(q[0]),q16=float(q[1]),median=float(q[2]),q84=float(q[3]),q975=float(q[4]),half68=float((q[3]-q[1])/2))
def save(fig,p,name):
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(p/(name+'.'+ext),dpi=170,bbox_inches='tight')
 plt.close(fig)
def main(a):
 started=time.perf_counter();data,notes,out,plots=map(Path,[a.data,a.notes,a.output,a.plots]);out.mkdir(exist_ok=True,parents=True);plots.mkdir(exist_ok=True,parents=True)
 plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 prior=notes/'results/analysis/cuore_energy_shape_20261114';rows=read(prior/'calibrated_event_features.csv');cal=json.loads((prior/'calibration.json').read_text())['groups']
 with h5py.File(data/'eval_data_holdout/efficiency_1p382mV/eval_efficiency_1keV.h5','r') as h:
  clean=h['clean_template'][:];k0=1/(clean.max()-clean[:1400].mean())
 bn=np.empty((5000,2500));source=[]
 for name in ['cuoreTraining.h5','cuoreTest.h5']:
  path=data/'CUORE_Data_Release_for_ML_Applications'/name;source.append(dict(path=str(path),sha256=sha(path)))
  ids=[i for i,r in enumerate(rows) if r['source_file']==name]
  with h5py.File(path,'r') as h:
   for start in range(0,len(ids),128):
    ii=np.array(ids[start:start+128]);rr=np.array([int(rows[i]['source_row']) for i in ii]);block=h['Waveform'][rr,:2500].astype(float)
    bn[ii]=block-block.mean(axis=1,keepdims=True)
    np.testing.assert_array_equal(h['eventId'][rr].ravel(),[int(rows[i]['event_id']) for i in ii])
 groups=np.array([r['baseline_group'] for r in rows]);legacy=np.array([r['legacy_4980']=='True' for r in rows]);E=np.array([float(r['energy_reference_kev']) for r in rows]);ac=np.array([float(r['corrected_amplitude_adc']) for r in rows])
 gain=np.zeros(5000);keVperADC=np.zeros(5000);localkeVperADC=np.zeros(5000)
 for g in ['A','B']:
  m=groups==g;p=cal[g];base=np.array([float(rows[i]['offset_adc']) for i in np.flatnonzero(m)])
  gain[m]=1+p['gain_slope_per_1000_baseline']*(base-p['baseline_reference_adc'])/1000
  keVperADC[m]=p['energy_c1']/10000/gain[m]
  localkeVperADC[m]=(p['energy_c1']+2*p['energy_c2']*ac[m]/10000)/10000/gain[m]
 rms=bn.std(axis=1,ddof=1);np.testing.assert_allclose(rms,[float(r['baseline_rms_adc']) for r in rows],rtol=1e-10)
 eq=rms*keVperADC/k0*1000;er=rms*keVperADC;baseline_rows=[];noise_summary={}
 for i,r in enumerate(rows):baseline_rows.append(dict(source_file=r['source_file'],source_row=r['source_row'],event_id=r['event_id'],group=groups[i],energy_reference_kev=E[i],rms_adc=rms[i],point_rms_kev_zero_slope=er[i],equivalent_sim_rms_mV=eq[i],equivalent_sim_mV_local_slope=rms[i]*localkeVperADC[i]/k0*1000))
 table(out/'cuore_baseline_per_event.csv',baseline_rows)
 for g in ['A','B']:
  m=groups==g;noise_summary[g]=dict(rms_adc=qstats(rms[m]),point_rms_keV=qstats(er[m]),equivalent_sim_rms_mV=qstats(eq[m]),local_slope_equivalent_mV=qstats((rms*localkeVperADC/k0*1000)[m]))
 dump(out/'rms_summary.json',dict(sim_low_energy_slope_keV_per_V=k0,groups=noise_summary,equivalence='Match energy-equivalent raw sample RMS using extrapolated zero-amplitude slope; not hardware ADC-to-mV conversion or OF resolution.'))
 # All 5000 pretrigger windows; no amplitude/quality clipping.
 noise={'CUORE A':bn[groups=='A']*keVperADC[groups=='A',None],'CUORE B':bn[groups=='B']*keVperADC[groups=='B',None]}
 simpath=data/'eval_data_holdout/efficiency_1p382mV/eval_efficiency_1keV.h5'
 with h5py.File(simpath,'r') as h:
  residual=h['waveforms_noisy'][:,:2500]-h['clean_template'][:2500];residual-=residual.mean(axis=1,keepdims=True)
 noise['模拟 1.382 mV']=residual*k0;noise['模拟 7.5 mV']=noise['模拟 1.382 mV']*(7.5/1.382)
 source.append(dict(path=str(simpath),sha256=sha(simpath)))
 short_rms={name:qstats(nn.std(axis=1,ddof=1)/k0*1000) for name,nn in noise.items()}
 dump(out/'same_window_rms_summary.json',short_rms)
 spectra={};spectral_rows=[];psd_stats=[]
 for name,nn in noise.items():
  f,p=periodogram(nn,fs=1000,window='hann',detrend=False,axis=1,scaling='density');pp=p.mean(axis=0);spectra[name]=(f,pp)
  for lo,hi in [(0,5),(5,20),(20,100),(100,500)]:
   frac=float(pp[(f>lo)&(f<=hi)].sum()/pp[f>0].sum());psd_stats.append(dict(dataset=name,band_low_hz=lo,band_high_hz=hi,power_fraction=frac))
  spectral_rows.extend(dict(dataset=name,frequency_hz=float(ff),psd_keV2_per_Hz=float(v)) for ff,v in zip(f,pp))
 table(out/'noise_psd.csv',spectral_rows);table(out/'noise_band_fractions.csv',psd_stats)
 # Fixed-time, short-window, Hann-tapered OF-like baseline estimate. Cross-fit noise PSD.
 bank={};ww=np.load(prior/'aligned_peak_waveforms.npz');tt=ww['time_s'];t=np.arange(2500)/1000
 for g in ['A','B']:bank['CUORE '+g]=np.interp(t-.75,tt,np.median(ww[g+'_2615'],axis=0),left=0,right=0)
 with h5py.File(data/'eval_data_holdout/efficiency_1p382mV/eval_efficiency_2615keV.h5','r') as h:
  cc=h['clean_template'][:];cc-=cc[:1400].mean();cc/=cc.max();sf=shape_features(cc);bank['模拟']=np.interp(t-.75+sf['t50_s'],np.arange(10000)/1000,cc,left=0,right=0)
 ofrows=[];samples={}
 for name,nn in noise.items():
  # Subtract window mean for AP and noise; exclude DC after taper as well.
  for apname,ap in bank.items():
   w=np.hanning(2500);template=(ap-ap.mean())*w;X=np.fft.rfft((nn-nn.mean(axis=1,keepdims=True))*w,axis=1);freq=np.fft.rfftfreq(2500,.001)
   for cutoff in [10,20,40]:
    amps=np.empty(len(nn))
    for fold in [0,1]:
     test=np.arange(len(nn))%2==fold;ps=np.mean(np.abs(X[~test])**2,axis=0);ps[freq>cutoff]=np.max(ps)*1e30
     amps[test]=estimate(X[test],template,ps,np.array([0.]),mode='standard',floor=0)[0]
     # Linear response check: adding one template unit must add exactly one estimator unit.
     if fold==0:
      z=X[test][:3];a0=estimate(z,template,ps,np.array([0.]),floor=0)[0];a1=estimate(z+np.fft.rfft(template),template,ps,np.array([0.]),floor=0)[0];np.testing.assert_allclose(a1-a0,1,rtol=1e-10,atol=1e-10)
    q=qstats(amps);ofrows.append(dict(dataset=name,template=apname,cutoff_hz=cutoff,sigma_keV=q['std'],half68_keV=q['half68'],FWHM_equiv_2p355sigma_keV=2.354820045*q['std'],n=q['n'],mean_keV=q['mean']))
    if cutoff==20 and apname==('模拟' if name.startswith('模拟') else name):samples[name]=amps
 table(out/'short_window_baseline_estimates.csv',ofrows);np.savez_compressed(out/'baseline_estimator_samples.npz',**samples)
 print('RMS',json.dumps(noise_summary),flush=True)
 print('OF primary',[r for r in ofrows if r['cutoff_hz']==20 and r['template']==('模拟' if r['dataset'].startswith('模拟') else r['dataset'])],flush=True)
 # Apply exactly the direct-shape definition used for CUORE, to 1000 paired rows/case.
 ids=np.sort(np.random.default_rng(261117).choice(5000,1000,replace=False));simstats=[];allfeatures=[]
 if a.reuse_direct:
  allfeatures=read(out/'sim_direct_time_per_event.csv');simstats=read(out/'sim_direct_time_statistics.csv')
  assert len(allfeatures)==34000 and len(simstats)==68
  for r in simstats:
   for k in list(r):
    if k not in ['noise_level','feature']:r[k]=int(r[k]) if k in ['energy_kev','failed','n'] else float(r[k])
  for level in LEVELS:
   for e in ENERGIES:
    selected=[int(r['source_row']) for r in allfeatures if r['noise_level']==level and int(r['energy_kev'])==e]
    np.testing.assert_array_equal(selected,ids)
 else:
  for level in LEVELS:
   for e in ENERGIES:
    path=data/f'eval_data_holdout/efficiency_{level}/eval_efficiency_{e}keV.h5';vals=[];failed=0
    with h5py.File(path,'r') as h:
     for start in range(0,len(ids),100):
      batchids=ids[start:start+100];block=h['waveforms_noisy'][batchids].astype(float)
      # Match the available pretrigger duration: simulator baseline 0–1.4s, CUORE 0–2.5s.
      block-=block[:,:1400].mean(axis=1,keepdims=True)
      for row,v in zip(batchids,block):
       try:
        sf=shape_features(v);vals.append([sf[k] for k in KEYS]);allfeatures.append(dict(noise_level=level,energy_kev=e,source_row=int(row),valid=True,**{k:sf[k] for k in KEYS}))
       except (ValueError,FloatingPointError):
        failed+=1;allfeatures.append(dict(noise_level=level,energy_kev=e,source_row=int(row),valid=False,**{k:'' for k in KEYS}))
    vv=np.array(vals)
    for j,key in enumerate(KEYS):simstats.append(dict(noise_level=level,energy_kev=e,feature=key,failed=failed,**qstats(vv[:,j])))
    print(f'direct {level} {e}keV: {len(vals)}/1000',flush=True)
 table(out/'sim_direct_time_per_event.csv',allfeatures);table(out/'sim_direct_time_statistics.csv',simstats)
 # Monoenergetic peak-window comparisons, not mixed-energy broad bins.
 peaks=read(prior/'calibration_peaks.csv');models=json.loads((prior/'trend_models.json').read_text())['models'];comparison=[]
 for g in ['A','B']:
  for e in [511,727,2615]:
   pk=next(p for p in peaks if p['group']==g and round(float(p['reference_energy_kev']))==e)
   mask=(groups==g)&legacy&(np.abs(ac-float(pk['mu_adc']))<=2*float(pk['sigma_adc']))
   for key in KEYS:
    vals=np.array([float(rows[i][key]) for i in np.flatnonzero(mask)]);co=np.array(models[g][key]['mean_coefficients']);x=(E[mask]-1500)/1000
    detrend=vals-sum(co[j]*x**j for j in range(len(co)));obs=qstats(detrend)
    for level in LEVELS:
     sim=next(s for s in simstats if s['noise_level']==level and s['energy_kev']==e and s['feature']==key)
     comparison.append(dict(group=g,energy_kev=e,feature=key,sim_noise=level,cuore_n=len(vals),cuore_mean_ms=float(vals.mean()),sim_mean_ms=sim['mean'],cuore_half68_ms=obs['half68'],sim_half68_ms=sim['half68'],width_ratio_cuore_over_sim=obs['half68']/sim['half68'],relative_width_ratio=(obs['half68']/vals.mean())/(sim['half68']/sim['mean']),cuore_std_ms=obs['std'],sim_std_ms=sim['std'],peak_background_fraction=float(pk['estimated_background_fraction'])))
 table(out/'direct_shape_comparison.csv',comparison)
 # Fitted-tau comparison is secondary because the estimators differ.
 cs=read(prior/'peak_residual_statistics.csv');ss=read(notes/'results/analysis/eval_noise_shape_20261116/scatter_statistics.csv');taurows=[]
 for g in ['A','B']:
  for e in [511,727,2615]:
   for ck,sk in [('tau_r_ms','t_rise'),('tau_fast_ms','t_fall1'),('tau_slow_ms','t_fall2')]:
    cr=next(r for r in cs if r['group']==g and round(float(r['energy_kev']))==e and r['feature']==ck)
    for level in LEVELS:
     sr=next(r for r in ss if r['noise_level']==level and int(r['energy_kev'])==e and r['feature']==sk)
     taurows.append(dict(group=g,energy_kev=e,feature=ck,sim_noise=level,cuore_half68_ms=float(cr['residual_half68']),sim_half68_ms=float(sr['half68_ms']),ratio=float(cr['residual_half68'])/float(sr['half68_ms']),warning='different waveform and fitting models; descriptive only'))
 table(out/'tau_comparison_descriptive_only.csv',taurows)
 fig,axs=plt.subplots(1,2,figsize=(13,4.8))
 for g,col in zip(['A','B'],COLORS):
  v=eq[groups==g];axs[0].hist(v,bins=np.linspace(1,8,100),histtype='step',color=col,label=f'CUORE {g} (全样本统计)');sv=np.sort(v);axs[1].plot(sv,np.arange(1,len(sv)+1)/len(sv),color=col,label=f'CUORE {g}')
 for val,col in zip([1.382,7.5],COLORS[2:]):
  for ax in axs:ax.axvline(val,color=col,ls='--',label=f'模拟 {val} mV')
 axs[0].set(xlabel='按能量响应换算的模拟等效 RMS / mV',ylabel='事件数',yscale='log');axs[1].set(xlabel='模拟等效 RMS / mV',ylabel='累计比例',xlim=(1,8));axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
 fig.suptitle('CUORE基线RMS分布：不是ADC硬件电压，也不是OF分辨率');save(fig,plots,'01_equivalent_noise_rms')
 fig,axs=plt.subplots(1,2,figsize=(13,4.8))
 for (name,(f,p)),col in zip(spectra.items(),COLORS):
  axs[0].loglog(f[1:],np.sqrt(p[1:]),color=col,label=name);axs[1].semilogx(f[1:],np.cumsum(p[1:])/p[1:].sum(),color=col,label=name)
 axs[0].set(xlabel='频率 / Hz',ylabel='能量等效噪声 ASD / keV/√Hz');axs[1].set(xlabel='频率 / Hz',ylabel='累计噪声功率比例')
 for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle('相同2.5秒窗口、Hann谱估计：总RMS相近不代表信号频带内噪声相近');save(fig,plots,'02_noise_spectra')
 fig,axs=plt.subplots(1,2,figsize=(13,4.8))
 for j,key in enumerate(KEYS):
  for g,col in zip(['A','B'],COLORS):
   rr=[r for r in comparison if r['group']==g and r['feature']==key and r['sim_noise']==LEVELS[0]];axs[j].plot([r['energy_kev'] for r in rr],[r['cuore_half68_ms'] for r in rr],'o-',color=col,label=f'CUORE {g} 能峰附近')
  for level,col in zip(LEVELS,COLORS[2:]):
   rr=[r for r in simstats if r['noise_level']==level and r['feature']==key];axs[j].plot([r['energy_kev'] for r in rr],[r['half68'] for r in rr],'o--',ms=3,color=col,label=level.replace('p','.'))
  axs[j].set(xscale='log',yscale='log',xlabel='能量 / keV',ylabel=('直接上升10–90%' if j==0 else '直接下降90–30%')+'：68%半宽 / ms',xlim=(400,3200));axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
 fig.suptitle('同一定义的直接时间散布：CUORE每峰全部选中事件；模拟每点固定抽样1000例');save(fig,plots,'03_direct_shape_spread_comparison')
 fig,axs=plt.subplots(1,3,figsize=(15,4.7))
 for ax,key,label in zip(axs,['tau_r_ms','tau_fast_ms','tau_slow_ms'],['τr','τfast','τslow']):
  for g,col in zip(['A','B'],COLORS):
   rr=[r for r in taurows if r['group']==g and r['feature']==key and r['sim_noise']==LEVELS[0]];ax.plot([r['energy_kev'] for r in rr],[r['cuore_half68_ms'] for r in rr],'o-',color=col,label=f'CUORE {g}')
  for level,col in zip(LEVELS,COLORS[2:]):
   rr=[r for r in taurows if r['group']=='A' and r['feature']==key and r['sim_noise']==level];ax.plot([r['energy_kev'] for r in rr],[r['sim_half68_ms'] for r in rr],'o--',color=col,label=level.replace('p','.'))
  ax.set(xscale='log',yscale='log',xlabel='能量 / keV',ylabel=label+'：68%半宽 / ms');ax.grid(alpha=.2)
 axs[0].legend(fontsize=8);fig.suptitle('拟合τ散布仅作描述：CUORE为RC+Bessel有效拟合，模拟为标准三指数，不能直接扣方差');save(fig,plots,'04_tau_spread_descriptive')
 dump(out/'manifest.json',dict(sources=source,script_sha256=sha(__file__),calibration_sha256=sha(prior/'calibration.json'),feature_source_sha256=sha(prior/'calibrated_event_features.csv'),direct_shape_definition_sha256=sha(Path(__file__).resolve().parents[2]/'experiments/cuore_shape_distribution/run_study.py'),baseline_window_samples=2500,sim_direct_sample_rows=ids.tolist(),elapsed_s=time.perf_counter()-started,limitations=['Reference energy scale extrapolated to zero amplitude, not official calibration.','2.5s tapered fixed-time baseline estimate is not official full-window CUORE OF/trigger resolution; sigma and Gaussian-equivalent FWHM are descriptive.','Pretrigger baselines of selected pulse events may not represent an unbiased random-trigger noise ensemble.','AP comes from 2615keV; zero-energy pulse shape not supplied.','CUORE shape statistics include intrinsic/working-point/model/noise effects; fixed simulator has noise+fit only.','Simulator baseline mean uses1.4s (onset1.5s), CUORE2.5s.','No direct CUORE information below500keV.']))
 dump(out/'verification.json',dict(passed=True,baseline_rms_recomputed_all5000=True,source_ids_checked=True,of_unit_response_verified=True,sim_direct_cases=34,sim_direct_rows=len(allfeatures)))
 print('COMPARISON',[r for r in comparison if r['energy_kev']==2615],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['data','notes','output','plots']:p.add_argument('--'+k,required=True)
 p.add_argument('--reuse-direct',action='store_true')
 main(p.parse_args())
