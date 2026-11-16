"""Quantify raw-noisy fit scatter at the fixed-template evaluation energies."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.basics.fit import fit_pulse,_TAU_R_MIN,_TAU_R_MAX,_TAU_D_MIN,_TAU_D_MAX,_TAU_R_GUESS,_TAU_D1_GUESS,_TAU_ORDER_FACTOR
ENERGIES=[1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034]
LEVELS=['1p382mV','7p5mV'];RMS=[1.382,7.5];COLORS=['#267db3','#d78332']
KEYS=['t_rise','t_fall1','t_fall2'];LABELS=['上升常数 τr','快下降常数 τfast','慢下降常数 τslow']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,p,name):
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(p/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
def main(a):
 start=time.perf_counter();data,notes,out,plots=map(Path,[a.data,a.notes,a.output,a.plots]);out.mkdir(exist_ok=True,parents=True);plots.mkdir(exist_ok=True,parents=True)
 plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 rng=np.random.default_rng(261116);sample=np.sort(rng.choice(5000,32,replace=False));source=[];stats=[];quality=[];cleanrows=[];cases={};wave={};templates={};refs={};noise_ref=None;reuse_max=0.;max_rms_error=0.
 bounds=np.array([[_TAU_R_MIN,_TAU_R_MAX],[max(_TAU_D_MIN,_TAU_R_GUESS*_TAU_ORDER_FACTOR),_TAU_D_MAX],[max(_TAU_D_MIN,_TAU_D1_GUESS*_TAU_ORDER_FACTOR),_TAU_D_MAX]])*1000
 with h5py.File(out/'per_event_scatter.h5','w') as saved:
  saved.attrs['selection']='success_noisy AND finite all three taus; no chi2, amplitude efficiency or boundary cuts'
  saved.attrs['units']='time constants in ms; rows preserve original file order'
  for level,target in zip(LEVELS,RMS):
   for e in ENERGIES:
    raw=data/f'eval_data_holdout/efficiency_{level}/eval_efficiency_{e}keV.h5'
    fit=notes/f'results/fit_results/model261108_nonoise/efficiency_{level}/fit_efficiency_{e}keV.h5'
    rawhash=sha(raw);fithash=sha(fit)
    with h5py.File(raw,'r') as rh,h5py.File(fit,'r') as fh:
     assert str(fh.attrs['source_sha256'])==rawhash
     assert float(rh.attrs['energy_kev'])==e and int(rh.attrs['n_events'])==5000 and float(rh.attrs['f_sample'])==1000
     clean=rh['clean_template'][:];noisy=rh['waveforms_noisy'][sample];resid=noisy-clean
     assert clean.shape==(10000,) and fh['t_rise_noisy'].shape==(5000,)
     assert np.all(fh['energy_true'][:]==e)
     assert np.allclose(fh['noise_rms'][:],rh['noise_rms'][:],rtol=0,atol=0)
     for name in rh['noise_params']:assert np.allclose(rh['noise_params'][name][:],fh['noise_params'][name][:],rtol=0,atol=0,equal_nan=True)
     err=float(np.max(np.abs(np.std(resid,axis=1)*1000-target)));max_rms_error=max(max_rms_error,err)
     assert err<1e-5
     scaled=resid/(target/1000)
     if noise_ref is None:noise_ref=scaled
     reuse_max=max(reuse_max,float(np.max(np.abs(scaled-noise_ref))))
     if e not in templates:
      templates[e]=clean
      fr=fit_pulse(clean,n_pulses=1,fs=1000.)
      assert fr.success
      ref=np.array([fr.tau_r[0],fr.tau_d[0],fr.tau_d2[0]])*1000;refs[e]=ref
      pred=fr.model_fn(np.arange(10000)/1000,*fr.params)
      cleanrows.append(dict(energy_kev=e,template_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),tau_r_ms=float(ref[0]),tau_fast_ms=float(ref[1]),tau_slow_ms=float(ref[2]),alpha=float(fr.alpha[0]),fit_rms_V=float(np.sqrt(np.mean((clean-pred)**2))),source=str(raw)))
     else:assert np.array_equal(clean,templates[e]),'Two noise levels do not share the same clean template'
     yy=np.column_stack([fh[k+'_noisy'][:].astype(float)*1000 for k in KEYS]);success=fh['success_noisy'][:];finite=np.isfinite(yy).all(axis=1);valid=success&finite
     conv=fh['converged_noisy'][:];eff=fh['efficient_noisy'][:]
     touch=np.isclose(yy,bounds[:,0],rtol=0,atol=1e-4*(bounds[:,1]-bounds[:,0]))|np.isclose(yy,bounds[:,1],rtol=0,atol=1e-4*(bounds[:,1]-bounds[:,0]))
     # After sorting, an original slow-component lower-bound solution can appear as fast.
     touch[:,1] |= np.isclose(yy[:,1],bounds[2,0],rtol=0,atol=1e-4*(bounds[1,1]-bounds[1,0]))
     alpha=fh['alpha_noisy'][:];base=fh['baseline_noisy'][:]
     alpha_bound=np.isclose(alpha,.01,atol=1e-4)|np.isclose(alpha,.99,atol=1e-4)
     baseline_bound=np.isclose(np.abs(base),.001,atol=1e-7)
     gr=saved.create_group(f'{level}/{e}keV');gr.attrs['source_file']=str(raw);gr.attrs['fit_file']=str(fit);gr.attrs['source_sha256']=rawhash;gr.attrs['fit_sha256']=fithash
     for name,v in [('source_row',np.arange(5000)),('tau_ms',yy),('delta_from_clean_fit_ms',yy-refs[e]),('valid',valid),('success_noisy',success),('converged_noisy',conv),('efficient_noisy',eff),('tau_near_bound',touch),('alpha_near_bound',alpha_bound),('baseline_near_bound',baseline_bound)]:gr.create_dataset(name,data=v,compression='gzip')
     cases[(level,e)]=(yy,valid,touch)
     qrow=dict(noise_level=level,energy_kev=e,total=5000,valid=int(valid.sum()),solver_failed=int((~success).sum()),nonfinite=int((~finite).sum()),historical_converged=int(conv.sum()),historical_efficient=int(eff.sum()),any_tau_bound=int((valid&touch.any(axis=1)).sum()),alpha_bound=int((valid&alpha_bound).sum()),baseline_bound=int((valid&baseline_bound).sum()))
     quality.append(qrow)
     for j,key in enumerate(KEYS):
      v=yy[valid,j];q=np.quantile(v,[.025,.16,.5,.84,.975]);ref=refs[e][j];boundary=touch[valid,j]
      stats.append(dict(noise_level=level,noise_rms_mV=target,energy_kev=e,feature=key,n=len(v),clean_fit_ms=float(ref),mean_ms=float(np.mean(v)),median_ms=float(q[2]),mean_bias_ms=float(v.mean()-ref),median_bias_ms=float(q[2]-ref),std_ms=float(np.std(v,ddof=1)),half68_ms=float((q[3]-q[1])/2),q025_ms=float(q[0]),q16_ms=float(q[1]),q84_ms=float(q[3]),q975_ms=float(q[4]),relative_std_pct=float(np.std(v,ddof=1)/ref*100),relative_half68_pct=float((q[3]-q[1])/2/ref*100),near_bound_count=int(boundary.sum()),near_bound_fraction=float(boundary.mean())))
     if e in [5,50,2615]:
      peak=float(np.max(clean)-np.mean(clean[:1000]));wave[(level,e)]=(clean/peak,noisy/peak)
    source.append(dict(raw=str(raw),raw_sha256=rawhash,fit=str(fit),fit_sha256=fithash))
    print(f'{level} {e} keV: valid={valid.sum()}, near_tau_bound={qrow["any_tau_bound"]}',flush=True)
 csvout(out/'scatter_statistics.csv',stats);csvout(out/'fit_quality.csv',quality);csvout(out/'clean_template_fits.csv',cleanrows)
 # Per-event scatter: horizontal jitter is display only, never an energy uncertainty.
 fig,axs=plt.subplots(1,3,figsize=(16,5))
 for j,ax in enumerate(axs):
  for l,col,shift in zip(LEVELS,COLORS,[-.018,.018]):
   for e in ENERGIES:
    yy,valid,touch=cases[(l,e)];xx=e*np.exp(rng.uniform(-.012,.012,valid.sum())+shift);ax.scatter(xx,yy[valid,j],s=1,alpha=.08,color=col,rasterized=True)
   rr=[r for r in stats if r['noise_level']==l and r['feature']==KEYS[j]]
   ax.plot(ENERGIES,[r['median_ms'] for r in rr],color=col,label=l.replace('p','.'));ax.fill_between(ENERGIES,[r['q16_ms'] for r in rr],[r['q84_ms'] for r in rr],color=col,alpha=.12)
  ax.plot(ENERGIES,[refs[e][j] for e in ENERGIES],'k--',lw=1.4,label='无噪声模板拟合基准')
  ax.set(xscale='log',yscale='log',xlabel='真实能量 / keV',ylabel=LABELS[j]+' / ms');ax.grid(alpha=.2,which='both')
 axs[0].legend(fontsize=8);fig.suptitle('原始验证集：17能量×2噪声×5000事件；全部有效拟合点，保留边界解\n实线/色带：中位数及16–84%；横向微抖动仅为显示，不代表能量不确定度',fontsize=12)
 save(fig,plots,'01_all_event_scatter')
 fig,axs=plt.subplots(2,3,figsize=(15,9))
 for j in range(3):
  for l,col in zip(LEVELS,COLORS):
   rr=[r for r in stats if r['noise_level']==l and r['feature']==KEYS[j]]
   axs[0,j].plot(ENERGIES,[r['std_ms'] for r in rr],'o-',color=col,ms=3,label=l.replace('p','.')+' SD')
   axs[0,j].plot(ENERGIES,[r['half68_ms'] for r in rr],'--',color=col,label=l.replace('p','.')+' 68%半宽')
   axs[1,j].plot(ENERGIES,[r['relative_half68_pct'] for r in rr],'o-',color=col,ms=3,label=l.replace('p','.'))
  axs[0,j].set(ylabel=LABELS[j]+'：观测散布 / ms',yscale='log');axs[1,j].set(ylabel=LABELS[j]+'：68%半宽 / clean拟合值 (%)',yscale='log')
  for ax in axs[:,j]:ax.set(xscale='log',xlabel='真实能量 / keV');ax.grid(alpha=.2,which='both')
 axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8);fig.suptitle('固定模板条件下的噪声＋拟合散布：低能边界/非高斯区不能用单一高斯宽度概括')
 save(fig,plots,'02_noise_fit_widths')
 fig,axs=plt.subplots(1,3,figsize=(16,4.7))
 for j,ax in enumerate(axs):
  for l,col in zip(LEVELS,COLORS):
   rr=[r for r in stats if r['noise_level']==l and r['feature']==KEYS[j]]
   ax.plot(ENERGIES,[r['median_bias_ms']/r['clean_fit_ms']*100 for r in rr],'o-',color=col,ms=3,label=l.replace('p','.'))
  ax.axhline(0,color='#777',lw=.8);ax.set(xscale='log',xlabel='真实能量 / keV',ylabel=LABELS[j]+'：中位数偏移 / clean拟合值 (%)');ax.grid(alpha=.2)
 axs[0].legend();fig.suptitle('加噪后的中心偏移与事件散布是两件事；基准为无噪声模板的同拟合器结果')
 save(fig,plots,'03_bias_vs_energy')
 fig,axs=plt.subplots(1,3,figsize=(16,4.7))
 for l,col in zip(LEVELS,COLORS):
  rr=[r for r in quality if r['noise_level']==l]
  for ax,key in zip(axs,['solver_failed','any_tau_bound','historical_converged']):ax.plot(ENERGIES,[100*r[key]/5000 for r in rr],'o-',ms=3,color=col,label=l.replace('p','.'))
 for ax,label in zip(axs,['求解失败 / 总事件 (%)','任一τ接近实际边界候选 / 总事件 (%)','历史χ²与幅度门限通过 / 总事件 (%)']):ax.set(xscale='log',xlabel='真实能量 / keV',ylabel=label);ax.grid(alpha=.2)
 axs[0].legend();fig.suptitle('质量诊断：本轮散布图未采用历史χ²、效率或去边界筛选；失败计数单列')
 save(fig,plots,'04_fit_quality')
 fig,axs=plt.subplots(2,3,figsize=(15,8))
 for i,l in enumerate(LEVELS):
  for j,ax in enumerate(axs[i]):
   for e,col in zip([5,50,2615],['#3694bd','#d28a35','#8058b1']):
    yy,valid,_=cases[(l,e)];v=(yy[valid,j]-refs[e][j])/refs[e][j]*100
    # Each distribution integrates to one; full ranges, no Gaussian assumption.
    ax.hist(v,bins=80,density=True,histtype='step',color=col,label=f'{e} keV')
   ax.set(yscale='log',xlabel=LABELS[j]+' 相对clean拟合值的偏差 (%)',ylabel=l.replace('p','.')+' 概率密度');ax.grid(alpha=.2)
 axs[0,0].legend();fig.suptitle('同能量拟合分布：5、50、2615 keV（全范围，概率密度对数轴）')
 save(fig,plots,'05_selected_distributions')
 fig,axs=plt.subplots(2,3,figsize=(15,8));t=np.arange(10000)/1000
 for i,l in enumerate(LEVELS):
  for j,e in enumerate([5,50,2615]):
   clean,ww=wave[(l,e)];lo,med,hi=np.quantile(ww,[.16,.5,.84],axis=0);ax=axs[i,j]
   ax.fill_between(t,lo,hi,color=COLORS[i],alpha=.25,label='32个固定抽样窗口16–84%');ax.plot(t,med,color=COLORS[i],lw=.8,label='抽样中位数');ax.plot(t,clean,'k--',lw=1,label='固定clean模板')
   ax.set(xlim=(1,5),xlabel='时间 / s',ylabel='幅度 / clean峰高',title=f'{l.replace("p",".")} · {e} keV');ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=7);fig.suptitle('实际原始加噪波形的散布：未做DDPM，所有窗使用clean峰高归一化；32例仅用于时域展示')
 save(fig,plots,'06_waveform_envelopes')
 # High-energy noise scaling check and full provenance.
 ratios=[]
 for e in ENERGIES:
  for k in KEYS:
   r1=next(r for r in stats if r['energy_kev']==e and r['feature']==k and r['noise_level']==LEVELS[0]);r2=next(r for r in stats if r['energy_kev']==e and r['feature']==k and r['noise_level']==LEVELS[1])
   ratios.append(dict(energy_kev=e,feature=k,std_ratio_7p5_over_1p382=r2['std_ms']/r1['std_ms'],half68_ratio_7p5_over_1p382=r2['half68_ms']/r1['half68_ms'],noise_rms_ratio=7.5/1.382))
 csvout(out/'noise_scaling_ratios.csv',ratios)
 # Check normalized clean shapes actually vary with energy; same within energy only.
 cref=templates[2615]/np.max(templates[2615]);shape_diff={str(e):float(np.max(np.abs(templates[e]/np.max(templates[e])-cref))) for e in ENERGIES}
 verification=dict(passed=True,events=170000,cases=34,same_clean_between_levels_all17=True,raw_hashes_match_fit_metadata=True,all_noise_metadata_match=True,sampled_rows=sample.tolist(),sampled_noise_rms_max_error_mV=max_rms_error,standardized_noise_sample_max_difference_across_cases=reuse_max,noise_reuse_is_sample_check_only=True,clean_normalized_max_difference_vs2615=shape_diff)
 dump(out/'verification.json',verification)
 dump(out/'manifest.json',dict(sources=source,script_sha256=sha(__file__),fit_core_sha256=sha(Path(__file__).resolve().parents[2]/'src/basics/fit.py'),generator_sha256=sha(Path(__file__).resolve().parents[2]/'scripts/generate_eval_dataset.py'),energies=ENERGIES,levels=LEVELS,parameter_bounds_ms=bounds.tolist(),fast_additional_boundary_candidate_ms=float(bounds[2,0]),boundary_note='Actual setup bounds, including possible 450 ms pre-sort slow bound mapped to fast; candidate proximity, not proof of active constraints.',boundary_tolerance_fraction_of_range=1e-4,selection='solver success and finite three taus; NO chi2/amplitude/efficiency/boundary cuts',clean_reference='17 new noiseless-template fits with canonical fitter; not physical ODE tau truth',source_fit_fields='*_noisy only; no denoised arrays used',elapsed_s=time.perf_counter()-start,limitations=['Conditional on simulated templates, noise and the existing estimator; not transferable CUORE variance subtraction.','Cross-energy and cross-level noise is paired/reused, not independent replicates.','Boundary-limited low-SNR scatter is not an unconstrained Gaussian sigma.','Noisy parameter scatter includes interaction with model mismatch; clean fit offset is kept separate.']))
 print('DONE',json.dumps(verification),flush=True)
 for r in stats:
  if r['energy_kev'] in [50,2615]:print(json.dumps(r),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['data','notes','output','plots']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
