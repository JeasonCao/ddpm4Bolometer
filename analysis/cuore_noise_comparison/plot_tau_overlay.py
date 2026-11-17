"""Overlay existing CUORE effective taus and pre-DDPM evaluation fitted taus."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_training(folder, selection):
    files=sorted(folder.glob('fit_clean_*.h5'))
    assert len(files)==10
    with np.load(selection) as old:
        shards=old['source_shard']; indices=old['window_idx']
    assert len(shards)==61225 and len(set(zip(shards.tolist(),indices.tolist())))==61225
    energies=[]; taus=[]; identities=[]; sources=[]; fingerprint=None
    for shard,path in enumerate(files):
        with h5py.File(path,'r') as f:
            assert bool(f.attrs['complete']) and np.all(f['done'][:])
            if fingerprint is None: fingerprint=str(f.attrs['fingerprint'])
            assert str(f.attrs['fingerprint'])==fingerprint
            yy=np.column_stack([f[k][:,0] for k in ['t_rise','t_fall1','t_fall2']])*1000
            onset=np.abs(f['onset_error'][:,0]); relative=f['relative_rms'][:]
            good=(f['n_pulses_fit'][:]==1)&f['success'][:].astype(bool)&np.isfinite(relative)&np.isfinite(yy).all(axis=1)&(yy>0).all(axis=1)
            flagged=f['tau_boundary'][:].astype(bool)|f['decay_degenerate'][:].astype(bool)|(relative>.01)|(onset>.05)|~np.isfinite(onset)
            mask=good&~flagged
            wi=f['window_idx'][:]
            np.testing.assert_array_equal(wi[mask],indices[shards==shard])
            e=f['energies_1'][:][mask]
            assert np.isfinite(e).all() and (e>0).all()
            energies.append(e); taus.append(yy[mask]); identities.extend((shard,int(i)) for i in wi[mask])
        sources.append(dict(path=str(path),sha256=sha(path)))
    te=np.concatenate(energies); ty=np.concatenate(taus)
    np.testing.assert_array_equal(np.array(identities),np.column_stack([shards,indices]))
    return te,ty,identities,dict(selection=str(selection),selection_sha256=sha(selection),fit_sources=sources,fingerprint=fingerprint,selected=len(te),energy_range_kev=[float(te.min()),float(te.max())],tau_ranges_ms=[[float(v.min()),float(v.max())] for v in ty.T],selection_check='Recomputed single/success/finite/residual/onset/boundary criteria; exact equality to prior selected source_shard/window_idx')

def main():
    p=argparse.ArgumentParser()
    for key in ['cuore','simulation','output','plots']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--train-fits', type=Path)
    p.add_argument('--train-selection', type=Path)
    a=p.parse_args()
    if bool(a.train_fits)!=bool(a.train_selection): p.error('Both training arguments are required together')
    a.output.mkdir(parents=True,exist_ok=True); a.plots.mkdir(parents=True,exist_ok=True)
    with a.cuore.open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['legacy_4980']=='True']
    assert len(rows)==4980
    ce=np.array([float(r['energy_reference_kev']) for r in rows])
    cy=np.array([[float(r[k]) for k in ['tau_r_ms','tau_fast_ms','tau_slow_ms']] for r in rows])
    assert np.isfinite(ce).all() and np.isfinite(cy).all() and (cy>0).all()
    training=None
    if a.train_fits:
        te,ty,identities,training=read_training(a.train_fits,a.train_selection)
        with (a.output/'training_selected_parameters.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['source_shard','window_idx','energy_true_kev','tau_r_ms','tau_fast_ms','tau_slow_ms'])
            for ident,e,yy in zip(identities,te,ty): w.writerow([*ident,float(e),*map(float,yy)])
    cases={}; summaries=[]; checks=[]
    levels=['1p382mV','7p5mV']; colors=['#267db3','#e28b26']; purple='#8751a6'
    expected=[1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034]
    with h5py.File(a.simulation,'r') as h:
        for level in levels:
            energies=sorted(int(k[:-3]) for k in h[level])
            assert energies==expected
            cases[level]=[]
            for e in energies:
                g=h[f'{level}/{e}keV']; yy=g['tau_ms'][:]; mask=g['valid'][:].astype(bool)
                np.testing.assert_array_equal(g['source_row'][:],np.arange(5000))
                assert yy.shape==(5000,3)
                assert np.isfinite(yy[mask]).all() and (yy[mask]>0).all()
                cases[level].append((e,yy[mask]))
                checks.append(dict(level=level,energy_kev=e,total=5000,plotted=int(mask.sum())))
                for j,key in enumerate(['tau_r','tau_fast','tau_slow']):
                    q=np.quantile(yy[mask,j],[0,.025,.16,.5,.84,.975,1])
                    summaries.append(dict(dataset=level,energy_kev=e,feature=key,n=int(mask.sum()),**dict(zip(['min_ms','q025_ms','q16_ms','median_ms','q84_ms','q975_ms','max_ms'],map(float,q)))))
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    titles=['上升时间常数 τr','快下降时间常数 τfast','慢下降时间常数 τslow']
    labels=['模拟验证集 · 1.382 mV','模拟验证集 · 7.5 mV','CUORE · 4,980 单事件（A/B 合并）']
    handles=[Line2D([],[],marker='o',linestyle='None',markersize=6,color=c,label=l) for c,l in zip(colors+[purple],labels)]
    if training:
        handles.append(Line2D([],[],marker='o',linestyle='None',markersize=6,color='#369448',label='训练集 clean · 61,225 筛选单事件'))
    manifest_plots=[]
    for high in [False,True]:
        fig,axs=plt.subplots(1,3,figsize=(17,5.8))
        cutoff=450 if high else 0
        cm=ce>=cutoff
        if training: tm=(te>=cutoff)&((te<=3150) if high else True)
        for j,ax in enumerate(axs):
            if training: ax.scatter(te[tm],ty[tm,j],s=2.5,alpha=.14 if high else .09,color='#369448',rasterized=True,linewidths=0,zorder=1)
            for level,col in reversed(list(zip(levels,colors))):
                points=[(e,v) for e,v in cases[level] if e>=cutoff]
                for e,yy in points:
                    # Exact energy coordinates: no jitter or density resampling.
                    ax.scatter(np.full(len(yy),e),yy[:,j],s=2.2,alpha=.10,color=col,rasterized=True,linewidths=0)
                ee=np.array([e for e,v in points]); qq=np.array([np.quantile(v[:,j],[.16,.5,.84]) for e,v in points])
                ax.plot(ee,qq[:,1],color=col,lw=.8 if level==levels[0] else 2,marker='o',ms=2 if level==levels[0] else 4,zorder=5 if level==levels[0] else 4)
                ax.vlines(ee,qq[:,0],qq[:,2],color=col,lw=1.2 if level==levels[0] else 3,zorder=5 if level==levels[0] else 4)
            ax.scatter(ce[cm],cy[cm,j],s=3.5,alpha=.27,color=purple,rasterized=True,linewidths=0,zorder=3)
            ax.set(title=titles[j],xlabel='能量 / keV（CUORE：参考定标；模拟：真值）',ylabel='拟合时间常数 / ms')
            if high:
                ax.set_xlim(450,3150)
                ax.set_xticks([500,1000,1500,2000,2500,3000])
            else:
                ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.85,5700 if training else 3500)
                ax.set_xticks([1,10,100,500,1000,3000],labels=['1','10','100','500','1000','3000'])
            if training and not high: ax.set_xticks([1,10,100,500,1000,3000,5000],labels=['1','10','100','500','1k','3k','5k'])
            ax.grid(alpha=.18,which='both')
        title='高能区：450–3,150 keV（线性坐标；保留该区间全部有效点）' if high else '全能区：17 个模拟能量点与 CUORE 叠加（横纵轴均为对数）'
        if training: title='训练 clean ＋ 两档加噪验证集 ＋ CUORE：'+('高能区 450–3,150 keV' if high else '全能区（横纵轴均为对数）')
        fig.suptitle(title,fontsize=15,y=.985)
        if training: handles[-1].set_label(f'训练 clean · 图内 {int(tm.sum()):,} 个筛选单事件')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2 if training else 3,frameon=False,fontsize=10)
        note='训练：无噪声、ODE 参数变化，沿用单事件质量筛选；验证：固定模板加噪、DDPM 前；折线/竖线为验证集的中位数/中央 68%。' if training else '模拟：原始加噪波形，DDPM 前；点为逐窗拟合，折线/粗竖线为中位数/中央 68%。无额外去边界筛选。'
        fig.text(.5,.065,note,ha='center',fontsize=9.5)
        fig.text(.5,.025,'模型不同：CUORE 为共享 RC+Bessel 有效响应拟合；模拟为标准三指数。参数叠加仅供描述性比较，不等同于真实热常数。',ha='center',fontsize=10,color='#555555')
        fig.subplots_adjust(left=.055,right=.99,bottom=.19,top=.77 if training else .80,wspace=.25)
        name='02_high_energy_overlay' if high else '01_full_energy_overlay'
        for ext in ['png','pdf']:
            fig.savefig(a.plots/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
        manifest_plots.append(dict(training_points=int(tm.sum()) if training else None,name=name,cuore_points=int(cm.sum()),simulation_points={l:sum(len(v) for e,v in cases[l] if e>=cutoff) for l in levels},energy_min=cutoff,display_only=True))
    with (a.output/'simulation_quantiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    manifest=dict(training=training,inputs=[dict(path=str(x),sha256=sha(x)) for x in [a.cuore,a.simulation]],cuore_selection='legacy_4980 only; A/B pooled; previous 20 excluded',simulation_selection='Existing valid mask: solver success and finite taus, retain boundaries; *_noisy only',cuore_energy_range_kev=[float(ce.min()),float(ce.max())],cuore_below500=int((ce<500).sum()),units='ms and keV',checks=checks,plots=manifest_plots,source_sha256=sha(__file__),limitations=['Different waveform fit models: descriptive effective-parameter comparison only.','Low-energy noise and parameter degeneracy/bounds widen fitted distributions; no Gaussian assumption.','Plot density not comparable as event counts and energy sampling differ.','Simulation median lines connect discrete energies for display, not a fitted energy law.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(training=training,cuore_range=manifest['cuore_energy_range_kev'],plots=manifest_plots,passed=True),ensure_ascii=False))

if __name__=='__main__':
    main()
