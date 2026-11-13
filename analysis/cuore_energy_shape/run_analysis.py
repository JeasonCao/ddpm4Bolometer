"""Read-only, exploratory gain calibration and conditional CUORE shape statistics."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np,h5py
from scipy.optimize import least_squares
from scipy.stats import norm,normaltest,skew,kurtosis,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
KEYS=['tau_r_ms','tau_fast_ms','tau_slow_ms','alpha','rise_10_90_ms','decay_90_30_ms']
LABELS=['上升常数 τr / ms','快下降常数 τfast / ms','慢下降常数 τslow / ms','快分量权重 α','直接上升时间 10–90% / ms','直接下降时间 90–30% / ms']
COL={'A':'#287bb5','B':'#d27c32'}
EE=np.array([510.999,583.191,727.330,911.204,968.971,2614.511])
SEEDS={'A':[5685,6485,8065,10065,10695,28095],'B':[5815,6645,8260,10305,10945,28725]}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(4*1024**2),b''): h.update(c)
    return h.hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,p,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(p/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
def poly(x,d):return np.column_stack([x**i for i in range(d+1)])
def fitmu(x,y,d,w):return np.linalg.lstsq(poly(x,d)*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
def main(args):
    root,inp,out,plots=map(Path,[args.data,args.features,args.output,args.plots])
    out.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cuore_spectrum'))
    from plot_spectrum import peakfit
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=list(csv.DictReader(inp.open(encoding='utf-8-sig')))
    a=np.array([float(r['amplitude_adc']) for r in rows]);b=np.array([float(r['offset_adc']) for r in rows])
    y=np.array([[float(r[k]) for k in KEYS] for r in rows]);legacy=np.array([r['legacy_4980']=='True' for r in rows])
    assert len(rows)==5000 and legacy.sum()==4980 and np.isfinite(y).all()
    assert len({(r['source_file'],int(r['source_row'])) for r in rows})==5000
    hashes={n:sha(root/n) for n in ['cuoreTraining.h5','cuoreTest.h5']}
    assert hashes==json.loads((inp.parent/'manifest.json').read_text(encoding='utf-8-sig'))['sources']
    energy=np.full(5000,np.nan);ac=np.zeros(5000);groups=np.where(b>-70000,'A','B');cal={};peaks=[];masks={};rng=np.random.default_rng(261114)
    for g in ['A','B']:
        m=groups==g;ref=float(np.median(b[m]));hi=m&(a>27500);x=(b[hi]-ref)/1000
        sol=least_squares(lambda p:(a[hi]-p[0]-p[1]*x)/40,[np.median(a[hi]),0],loss='soft_l1')
        p=sol.x;gain=1+(p[1]/p[0])*(b[m]-ref)/1000
        assert sol.success and np.all(gain>0)
        ac[m]=a[m]/gain;local=[]
        for e,seed in zip(EE,SEEDS[g]):
            f=peakfit(ac[m],seed,140 if e<1000 else 220)
            assert f['fit_success'] and not f['mu_bound']
            local.append(dict(group=g,reference_energy_kev=float(e),**f))
        anchors=[local[1],local[-1]];xx=np.array([r['mu_adc']/10000 for r in anchors]);c=np.linalg.solve(np.column_stack([xx,xx**2]),[583.191,2614.511])
        energy[m]=c[0]*ac[m]/10000+c[1]*(ac[m]/10000)**2
        assert np.min(c[0]+2*c[1]*ac[m]/10000)>0
        for f in local:
            xx=f['mu_adc']/10000;f['mapped_energy_kev']=float(c[0]*xx+c[1]*xx**2);f['residual_kev']=f['mapped_energy_kev']-f['reference_energy_kev']
            f['sigma_kev']=float((c[0]+2*c[1]*xx)*f['sigma_adc']/10000);f['anchor']=f in anchors
            mm=m&(np.abs(ac-f['mu_adc'])<=2*f['sigma_adc']);f['selected_all']=int(mm.sum());f['selected_legacy']=int((mm&legacy).sum())
            sig=f['signal_area']*(norm.cdf(2)-norm.cdf(-2));bg=f['background_per_adc']*4*f['sigma_adc'];f['estimated_background_fraction']=float(bg/(sig+bg))
            masks[(g,int(round(f['reference_energy_kev'])))]=mm
        check=[f for f in local if not f['anchor']]
        assert max(abs(f['residual_kev']) for f in check)<10,'Independent peaks fail calibration check'
        cal[g]=dict(n=int(m.sum()),baseline_reference_adc=ref,high_peak_n=int(hi.sum()),high_peak_intercept_adc=float(p[0]),high_peak_slope_per_1000_baseline=float(p[1]),gain_slope_per_1000_baseline=float(p[1]/p[0]),energy_c1=float(c[0]),energy_c2=float(c[1]),max_check_residual_kev=max(abs(f['residual_kev']) for f in check))
        peaks.extend(local)
    dump(out/'calibration.json',dict(formula='Acorr=A/[1+g*(baseline-baseline_reference)/1000]; E=c1*(Acorr/10000)+c2*(Acorr/10000)^2',groups=cal));table(out/'calibration_peaks.csv',peaks)
    print('Calibration:',json.dumps(cal),flush=True)
    for i,r in enumerate(rows):r.update(baseline_group=groups[i],corrected_amplitude_adc=ac[i],energy_reference_kev=energy[i])
    table(out/'calibrated_event_features.csv',rows)
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for g in ['A','B']:
        m=groups==g;axs[0].hist(energy[m],bins=np.arange(450,2701,4),histtype='step',color=COL[g],label=f'组{g}：{m.sum()}例')
        pp=[v for v in peaks if v['group']==g and not v['anchor']]
        axs[1].plot([v['reference_energy_kev'] for v in pp],[v['residual_kev'] for v in pp],'o-',color=COL[g],label=f'组{g}：未参与定标的峰')
    for e in EE:axs[0].axvline(e,color='#777',ls=':',lw=.7)
    axs[0].set(yscale='log',ylabel='事件数 / 4 keV',title='分别作基线增益修正与两峰定标后的参考能谱');axs[0].legend()
    axs[1].axhline(0,color='#777',lw=.8);axs[1].set(xlabel='参考能量 / keV',ylabel='峰位偏差 / keV');axs[1].legend();save(fig,plots,'01_calibration_check')
    edges=np.linspace(500,2700,9);bins=np.clip(np.digitize(energy,edges)-1,0,7)
    models={};bstats=[];pstats=[];res=np.full_like(y,np.nan);curves=[];sensitivity=[];corr={}
    for g in ['A','B']:
        m=(groups==g)&legacy&(energy>=500)&(energy<=2650);x=(energy[m]-1500)/1000;yy=y[m];bb=bins[m]
        w=1/np.bincount(bb,minlength=8)[bb];folds=rng.permutation(np.arange(len(x)))%5;models[g]={}
        for j,key in enumerate(KEYS):
            scores={}
            for d in [1,2]:
                err=np.zeros(len(x))
                for fold in range(5):
                    train=folds!=fold;test=~train;c=fitmu(x[train],yy[train,j],d,w[train]);err[test]=(yy[test,j]-poly(x[test],d)@c)**2
                scores[d]=float(np.sqrt(np.average(err,weights=w)))
            d=2 if scores[2]<.98*scores[1] else 1;c=fitmu(x,yy[:,j],d,w);rr=yy[:,j]-poly(x,d)@c;res[m,j]=rr;local=[]
            for k in range(8):
                sel=bb==k
                if sel.sum()<30:continue
                v=yy[sel,j];r=rr[sel];ee=energy[m][sel];q=np.quantile(r,[.16,.5,.84]);boot=rng.choice(r,size=(300,len(r)),replace=True);width=np.std(boot,axis=1,ddof=1)
                z=dict(group=g,feature=key,energy_mean_kev=float(ee.mean()),energy_low=float(edges[k]),energy_high=float(edges[k+1]),n=int(sel.sum()),mean=float(v.mean()),median=float(np.median(v)),mean_se=float(v.std(ddof=1)/np.sqrt(len(v))),residual_std=float(r.std(ddof=1)),residual_half68=float((q[2]-q[0])/2),std_ci_low=float(np.quantile(width,.025)),std_ci_high=float(np.quantile(width,.975)))
                local.append(z);bstats.append(z)
            sx=np.array([(v['energy_mean_kev']-1500)/1000 for v in local]);sv=np.array([v['residual_std'] for v in local]);sc=np.linalg.lstsq(poly(sx,1),np.log(sv),rcond=None)[0]
            hc=np.linalg.lstsq(poly(sx,1),np.log([v['residual_half68'] for v in local]),rcond=None)[0]
            models[g][key]=dict(degree=d,mean_coefficients=c.tolist(),cv_rmse_linear=scores[1],cv_rmse_quadratic=scores[2],log_std_coefficients=sc.tolist(),log_half68_coefficients=hc.tolist(),n=int(m.sum()),units='dimensionless' if key=='alpha' else 'ms')
            am=(groups==g)&(energy>=500)&(energy<=2650);ax=(energy[am]-1500)/1000;ab=bins[am];aw=1/np.bincount(ab,minlength=8)[ab];ca=fitmu(ax,y[am,j],d,aw)
            for e in [583.191,911.204,2614.511]:
                z=np.array([(e-1500)/1000]);sensitivity.append(dict(group=g,feature=key,energy_kev=e,mean_legacy=float((poly(z,d)@c)[0]),mean_all5000=float((poly(z,d)@ca)[0])))
            grid=np.linspace(511,2615,150);gx=(grid-1500)/1000
            for e,mu,sd in zip(grid,poly(gx,d)@c,np.exp(poly(gx,1)@sc)):curves.append(dict(group=g,feature=key,energy_kev=float(e),conditional_mean=float(mu),observed_residual_std=float(sd)))
            for p in [v for v in peaks if v['group']==g]:
                pm=masks[(g,int(round(p['reference_energy_kev'])))]&legacy;px=(energy[pm]-1500)/1000;r=y[pm,j]-poly(px,d)@c
                if len(r)<20:continue
                q=np.quantile(r,[.16,.5,.84]);sd=float(r.std(ddof=1));center=float(r.mean())
                pstats.append(dict(group=g,feature=key,energy_kev=p['reference_energy_kev'],n=len(r),residual_mean=center,residual_std=sd,residual_median=float(q[1]),residual_half68=float((q[2]-q[0])/2),skewness=float(skew(r,bias=False)),excess_kurtosis=float(kurtosis(r,bias=False)),normality_p=float(normaltest(r).pvalue),tail_outside_3sd=float(np.mean(np.abs(r-center)>3*sd)),background_fraction=p['estimated_background_fraction'],residual_baseline_spearman=float(spearmanr(r,b[pm]).statistic)))
        scale=np.column_stack([np.exp(poly(x,1)@np.array(models[g][k]['log_std_coefficients'])) for k in KEYS]);z=res[m]/scale
        corr[g]=dict(features=KEYS,pearson=np.corrcoef(z.T).tolist(),spearman=spearmanr(z).statistic.tolist())
        table(out/f'empirical_residual_vectors_{g}.csv',[dict(source_file=rows[i]['source_file'],source_row=rows[i]['source_row'],energy_kev=float(energy[i]),**{k:float(v) for k,v in zip(KEYS,zr)}) for i,zr in zip(np.flatnonzero(m),z)])
    ps=np.array([v['normality_p'] for v in pstats]);order=np.argsort(ps);qv=np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
    for idx,q in zip(order,qv):pstats[idx]['normality_q_bh']=float(min(q,1))
    table(out/'energy_bin_statistics.csv',bstats);table(out/'peak_residual_statistics.csv',pstats);table(out/'trend_curves.csv',curves);table(out/'selection_sensitivity.csv',sensitivity)
    dump(out/'trend_models.json',dict(x_definition='x=(E_keV-1500)/1000; mean=sum(c[i]*x**i); SD=exp(s0+s1*x)',mean_weighting='equal total weight in each occupied 275-keV bin',selection='legacy4980, 500<=E<=2650',degree_rule='quadratic only if fivefold weighted held-out RMSE improves >2%',models=models));dump(out/'residual_correlations.json',corr)
    grid=np.linspace(511,2615,200);gx=(grid-1500)/1000
    for g in ['A','B']:
        m=(groups==g)&legacy;fig,axs=plt.subplots(2,3,figsize=(15,8))
        for j,(ax,key,label) in enumerate(zip(axs.flat,KEYS,LABELS)):
            mod=models[g][key];mu=poly(gx,mod['degree'])@mod['mean_coefficients'];sd=np.exp(poly(gx,1)@mod['log_std_coefficients'])
            ax.scatter(energy[m],y[m,j],s=3,alpha=.12,color=COL[g],rasterized=True);ax.fill_between(grid,mu-sd,mu+sd,color=COL[g],alpha=.17,label='趋势 ± 观测残差SD（非置信区间）')
            ax.plot(grid,mu,color='#222',lw=1.8,label='线性中心趋势' if mod['degree']==1 else '二次中心趋势');ss=[v for v in bstats if v['group']==g and v['feature']==key]
            ax.errorbar([v['energy_mean_kev'] for v in ss],[v['mean'] for v in ss],yerr=[v['mean_se'] for v in ss],fmt='o',ms=4,color='#b32e35',label='分箱均值 ± 标准误');ax.set(xlabel='参考能量 / keV',ylabel=label,xlim=(480,2700));ax.grid(alpha=.2)
            if j==0:ax.legend(fontsize=7)
        fig.suptitle(f'基线组{g}：能量趋势与观测散布（沿用原4980例选择；全部点保留）');save(fig,plots,f'02_shape_trends_{g}')
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,key,label in zip(axs.flat,KEYS,LABELS):
        for g in ['A','B']:
            ss=[v for v in bstats if v['group']==g and v['feature']==key]
            ax.errorbar([v['energy_mean_kev'] for v in ss],[v['residual_std'] for v in ss],yerr=[[max(0,v['residual_std']-v['std_ci_low']) for v in ss],[max(0,v['std_ci_high']-v['residual_std']) for v in ss]],fmt='o',ms=4,color=COL[g],label=f'{g}组 SD / 95%自助区间')
            ax.plot(grid,np.exp(poly(gx,1)@models[g][key]['log_std_coefficients']),color=COL[g]);ax.plot([v['energy_mean_kev'] for v in ss],[v['residual_half68'] for v in ss],':',color=COL[g],label=f'{g}组 68%半宽')
        ax.set(xlabel='参考能量 / keV',ylabel=label+' 的散布');ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=7);fig.suptitle('去趋势后的散布：实线为正值SD模型；虚线检验长尾影响');save(fig,plots,'03_width_vs_energy')
    for g in ['A','B']:
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for j,(ax,key,label) in enumerate(zip(axs.flat,KEYS,LABELS)):
            for e,col in [(583,'#3390bb'),(911,'#c88530'),(2615,'#7f5aa3')]:
                m=masks[(g,e)]&legacy;mod=models[g][key];r=y[m,j]-poly((energy[m]-1500)/1000,mod['degree'])@mod['mean_coefficients'];z=(r-r.mean())/r.std(ddof=1)
                ax.hist(z,bins=np.linspace(-5,7,41),weights=np.ones(len(z))/(len(z)*.3),histtype='step',color=col,label=f'{e} keV，N={len(z)}')
            zz=np.linspace(-5,7,200);ax.plot(zz,norm.pdf(zz),'k--',label='标准高斯');ax.set(xlabel='各峰残差减去该峰均值，再除以标准差',ylabel='概率密度',title=label);ax.grid(alpha=.2)
        axs.flat[0].legend(fontsize=8);fig.suptitle(f'组{g}：同能峰附近分布与高斯对照（全样本统计；图展示−5至7σ）');save(fig,plots,f'04_gaussian_check_{g}')
    time=np.arange(10000)/1000;gridt=np.arange(-.25,3.001,.002);waves={}
    with h5py.File(root/'cuoreTraining.h5','r') as tr,h5py.File(root/'cuoreTest.h5','r') as te:
        handles={'cuoreTraining.h5':tr,'cuoreTest.h5':te}
        for g in ['A','B']:
            for e in [583,911,2615]:
                aligned=[]
                for i in np.flatnonzero(masks[(g,e)]&legacy):
                    r=rows[i];h=handles[r['source_file']];row=int(r['source_row']);assert int(np.asarray(h['eventId'][row]).item())==int(r['event_id'])
                    raw=h['Waveform'][row];v=(raw-b[i])/float(r['smoothed_peak_adc']);aligned.append(np.interp(gridt+float(r['t50_s']),time,v))
                waves[f'{g}_{e}']=np.array(aligned)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for row,g in enumerate(['A','B']):
        for e,col in [(583,'#3390bb'),(911,'#c88530'),(2615,'#7f5aa3')]:
            ww=waves[f'{g}_{e}'];lo,med,hi=np.quantile(ww,[.16,.5,.84],axis=0)
            for ax in axs[row]:ax.plot(gridt,med,color=col,label=f'{e} keV，N={len(ww)}');ax.fill_between(gridt,lo,hi,color=col,alpha=.12)
        axs[row,0].set(xlim=(-.2,3),ylim=(-.03,1.05),ylabel=f'组{g}：峰高归一化幅值');axs[row,1].set(xlim=(-.1,.12),ylim=(.05,.95));axs[row,2].set(xlim=(.3,1.4),ylim=(.1,.9))
        for ax,title in zip(axs[row],['完整波形','上升沿放大','下降沿放大']):ax.set(xlabel='相对上升沿50%时刻 / s',title=title);ax.grid(alpha=.2)
        axs[row,0].legend(fontsize=8)
    fig.suptitle('定标后不同能峰的实际波形：中位数及16–84%区间（对齐时间与峰高，不拉伸时间）');save(fig,plots,'05_waveforms_by_energy');np.savez_compressed(out/'aligned_peak_waveforms.npz',time_s=gridt,**waves)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,g in zip(axs,['A','B']):
        cc=np.array(corr[g]['spearman']);im=ax.imshow(cc,vmin=-1,vmax=1,cmap='coolwarm')
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{cc[i,j]:.2f}',ha='center',va='center',fontsize=9)
        labs=['τr','τfast','τslow','α','rise','decay'];ax.set_xticks(range(6),labs);ax.set_yticks(range(6),labs);ax.set_title(f'组{g} 标准化残差秩相关')
    fig.colorbar(im,ax=list(axs),shrink=.8);fig.suptitle('生成波形应保留联合分布，不能独立抽取各参数')
    for ext in ['png','pdf']:fig.savefig(plots/f'06_residual_correlations.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    assert hashes=={n:sha(root/n) for n in hashes}
    manifest=dict(source_sha256=hashes,features_sha256=sha(inp),spectrum_code_sha256=sha(Path(__file__).resolve().parents[1]/'cuore_spectrum/plot_spectrum.py'),script_sha256=sha(__file__),seed=261114,calibration_all=5000,shape_legacy=4980,calibration_official=False,normality_tests=len(pstats),normality_rejections_bh05=sum(v['normality_q_bh']<.05 for v in pstats),limitations=['Empirical groups, no official energy/run/time labels.','Amplitude-derived energy and shape share measurement errors.','969-keV blend and continuum contamination affect peak selections.','Observed scatter includes noise, fit degeneracy and working-point variation; not latent physical variance. Adding this scatter to clean pulses and then adding noise can double-count noise.','No validated extrapolation below 511 or above 2615 keV.','Coefficients/residual vectors are exploratory, not an approved generator.'])
    dump(out/'manifest.json',manifest);dump(out/'verification.json',dict(passed=True,source_hashes_unchanged=True,identities_unique=True,energy_finite=bool(np.isfinite(energy).all()),calibration_checks_passed=True))
    print(json.dumps(manifest,ensure_ascii=False),flush=True);print('Trends:',json.dumps(models),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['data','features','output','plots']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
