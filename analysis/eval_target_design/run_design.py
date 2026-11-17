"""Steps 1-2 only: empirical center/observed-width draft; no waveform generation."""
import argparse,csv,hashlib,json,html
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,d,name):
    for ext in ['png','pdf']:fig.savefig(d/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
KEYS=['tau_r_ms','tau_fast_ms','tau_slow_ms']
LABELS=['上升常数 τr','快下降常数 τfast','慢下降常数 τslow']
ENERGIES=np.array([1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034])

def main():
    p=argparse.ArgumentParser()
    for key in ['notes','output','plots']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.plots.mkdir(parents=True,exist_ok=True)
    prior=a.notes/'results/analysis/cuore_energy_shape_20261114'
    sources=[prior/'calibrated_event_features.csv',prior/'trend_models.json',prior/'energy_bin_statistics.csv',a.notes/'results/analysis/cuore_eval_train_tau_overlay_20261117/training_selected_parameters.csv',a.notes/'results/analysis/eval_noise_shape_20261116/scatter_statistics.csv']
    rr=read(sources[0]);models=json.loads(sources[1].read_text(encoding='utf-8-sig'))['models'];bins=read(sources[2]);tr=read(sources[3]);old=read(sources[4])
    ce=np.array([float(r['energy_reference_kev']) for r in rr]);cy=np.array([[float(r[k]) for k in KEYS] for r in rr]);cg=np.array([r['baseline_group'] for r in rr]);legacy=np.array([r['legacy_4980']=='True' for r in rr])
    te=np.array([float(r['energy_true_kev']) for r in tr]);ty=np.array([[float(r[k]) for k in KEYS] for r in tr]);assert len(te)==61225
    assert np.isfinite(ty).all() and (ty>0).all()
    coeff={}; slopes={}; widths={}; fitrows=[]
    # Independently reproduce original equal-energy-bin weighted linear fits.
    for g in ['A','B']:
        mask=legacy&(cg==g)&(ce>=500)&(ce<=2650);x=(ce[mask]-1500)/1000;ib=np.clip(np.digitize(ce[mask],np.linspace(500,2700,9))-1,0,7);w=1/np.bincount(ib,minlength=8)[ib];X=np.column_stack([np.ones(len(x)),x])
        cc=np.linalg.lstsq(X*np.sqrt(w[:,None]),cy[mask]*np.sqrt(w[:,None]),rcond=None)[0]
        coeff[g]=cc;slopes[g]=cc[1];widths[g]={}
        for j,key in enumerate(KEYS):
            m=models[g][key];assert m['degree']==1
            np.testing.assert_allclose(cc[:,j],m['mean_coefficients'],rtol=1e-10,atol=1e-9)
            b=[r for r in bins if r['group']==g and r['feature']==key];bx=np.array([(float(r['energy_mean_kev'])-1500)/1000 for r in b]);design=np.column_stack([np.ones(len(bx)),bx])
            sd=np.linalg.lstsq(design,np.log([float(r['residual_std']) for r in b]),rcond=None)[0]
            hh=np.linalg.lstsq(design,np.log([float(r['residual_half68']) for r in b]),rcond=None)[0]
            np.testing.assert_allclose(sd,m['log_std_coefficients'],rtol=1e-10,atol=1e-9);np.testing.assert_allclose(hh,m['log_half68_coefficients'],rtol=1e-10,atol=1e-9)
            widths[g][key]=(sd,hh)
            fitrows.append(dict(group=g,feature=key,n=int(mask.sum()),mean_at_1500_ms=cc[0,j],slope_ms_per_MeV=cc[1,j],log_sd_intercept_at1500=sd[0],log_sd_slope_per_MeV=sd[1],log_h68_intercept_at1500=hh[0],log_h68_slope_per_MeV=hh[1]))
    def width(g,e,kind):
        effective=np.clip(np.asarray(e),511,2615);xx=(effective-1500)/1000
        return np.array([np.exp(widths[g][k][kind][0]+widths[g][k][kind][1]*xx) for k in KEYS]).T
    def local(e):
        m=np.abs(te-e)<=max(100,.1*e)
        assert m.sum()>=30
        return m
    # Choose a real nearby clean waveform vector closest to local log-space median.
    ids=np.flatnonzero(local(2615));log=np.log(ty[ids]);med=np.median(log,axis=0);cov=np.cov(log.T);delta=log-med;d=np.einsum('ij,jk,ik->i',delta,np.linalg.inv(cov),delta);anchor_idx=ids[np.argmin(d)];anchor=ty[anchor_idx]
    def center(g,e,k=1):return anchor+np.expand_dims((np.maximum(np.asarray(e),511)-2615)/1000,-1)*slopes[g]*k
    # A is the primary, narrower observed-width scenario. Shared slope factor preserves slope ratios.
    grid=np.unique(np.r_[np.linspace(511,3034,100),ENERGIES[ENERGIES>=511]])
    bounds=np.array([np.quantile(ty[local(e)],[.025,.16,.84,.975],axis=0) for e in grid])
    cache=[]
    for e in grid:
        lv=np.log(ty[local(e)]);mean=lv.mean(axis=0);chol=np.linalg.cholesky(np.cov(lv.T));z=np.linalg.solve(chol,(lv-mean).T).T;tree=cKDTree(z)
        distance_limit=float(np.quantile(tree.query(z,k=11)[0][:,-1],.95))
        cache.append((mean,chol,tree,distance_limit))
    scan=[]
    for k in np.linspace(1,0,101):
        mu=center('A',grid,k);sd=width('A',grid,0)
        marginal=((mu>=bounds[:,1])&(mu<=bounds[:,2])&(mu-2*sd>=bounds[:,0])&(mu+2*sd<=bounds[:,3])).all()
        ratios=[]
        for row,(mean,chol,tree,lim) in zip(mu,cache):
            zt=np.linalg.solve(chol,np.log(row)-mean);ratios.append(float(tree.query(zt,k=10)[0][-1]/lim))
        worst=int(np.argmax(ratios));joint=max(ratios)<=1
        scan.append(dict(slope_factor=float(k),marginal_pass=bool(marginal),center_joint_pass=bool(joint),worst_neighbor_distance_ratio=max(ratios),worst_energy_kev=float(grid[worst]),all_pass=bool(marginal and joint)))
    passing=[r['slope_factor'] for r in scan if r['all_pass']]
    if not passing:raise RuntimeError('No slope factor passes draft guard; revise anchor/guard explicitly')
    maximum_factor=max(passing)
    factor=float(np.floor((maximum_factor+1e-9)*10)/10) # round down to a simple draft value
    selected=next(r for r in scan if abs(r['slope_factor']-factor)<1e-9)
    assert selected['all_pass']
    table(a.output/'slope_attenuation_scan.csv',scan)
    targets=[];supports=[];curve=[]
    for e in ENERGIES:
        mu=center('A',e,factor);alt=center('B',e,factor);sa=width('A',e,0);ha=width('A',e,1);sb=width('B',e,0);hb=width('B',e,1)
        status='low_energy_frozen_reference_not_validated' if e<511 else ('high_energy_extrapolated_center_frozen_width' if e>2615 else 'within_CUORE_energy_span')
        row=dict(energy_kev=int(e),status=status,width_reference_energy_kev=int(np.clip(e,511,2615)),clean_width_status='PENDING_new_center_noise_and_fit_calibration')
        for j,key in enumerate(KEYS):
            label=key.removesuffix('_ms');row.update({label+'_center_ms':float(mu[j]),label+'_A_observed_sd_ms':float(sa[j]),label+'_A_observed_h68_ms':float(ha[j]),label+'_B_observed_sd_ms':float(sb[j]),label+'_B_observed_h68_ms':float(hb[j]),label+'_B_center_alternative_ms':float(alt[j]),label+'_clean_sd_ms':None})
        targets.append(row)
        if e>=511:
            m=local(e);v=ty[m];q=np.quantile(v,[.025,.16,.5,.84,.975],axis=0)
            # 3-parameter density diagnostic, not a full waveform/inverse-ODE guarantee.
            lv=np.log(v);mean=lv.mean(axis=0);chol=np.linalg.cholesky(np.cov(lv.T));z=np.linalg.solve(chol,(lv-mean).T).T;zt=np.linalg.solve(chol,np.log(mu)-mean)
            tree=cKDTree(z);baseline=tree.query(z,k=11)[0][:,-1];distance=float(tree.query(zt,k=10)[0][-1]);rank=float(np.mean(baseline<=distance))
            for j,key in enumerate(KEYS):
                supports.append(dict(energy_kev=int(e),feature=key,n_local=int(m.sum()),local_energy_min=float(te[m].min()),local_energy_max=float(te[m].max()),train_q025=q[0,j],train_q16=q[1,j],train_median=q[2,j],train_q84=q[3,j],train_q975=q[4,j],target_mean=mu[j],A_mean_minus_2sd=mu[j]-2*sa[j],A_mean_plus_2sd=mu[j]+2*sa[j],A_center_inside_16_84=bool(q[1,j]<=mu[j]<=q[3,j]),A_2sd_inside_2p5_97p5=bool(q[0,j]<=mu[j]-2*sa[j] and mu[j]+2*sa[j]<=q[4,j]),B_same_A_center_2sd_inside_2p5_97p5=bool(q[0,j]<=mu[j]-2*sb[j] and mu[j]+2*sb[j]<=q[4,j]),center_3tau_knn_sparsity_percentile=rank*100))
    for e in np.unique(np.r_[np.geomspace(1,511,120),np.linspace(511,3034,180)]):
        for j,key in enumerate(KEYS):curve.append(dict(energy_kev=float(e),feature=key,target_center_ms=float(center('A',e,factor)[j]),A_observed_sd_ms=float(width('A',e,0)[j]),A_observed_h68_ms=float(width('A',e,1)[j]),B_observed_sd_ms=float(width('B',e,0)[j]),B_observed_h68_ms=float(width('B',e,1)[j])))
    table(a.output/'cuore_center_width_coefficients.csv',fitrows);table(a.output/'targets_17_energy_points.csv',targets);table(a.output/'training_support_checks.csv',supports);table(a.output/'target_curves.csv',curve)
    # Old-template noise widths are warnings only; do not subtract from new target widths.
    noisecheck=[]
    for r in old:
        e=int(r['energy_kev']);j=['t_rise','t_fall1','t_fall2'].index(r['feature']);sd=float(r['std_ms']);target=float(width('A',e,0)[j])
        noisecheck.append(dict(noise_level=r['noise_level'],energy_kev=e,feature=KEYS[j],old_template_noise_fit_sd_ms=sd,draft_A_observed_sd_ms=target,old_sd_over_draft=sd/target,old_noise_exceeds_draft=sd>target,usable_for_variance_subtraction=False))
    table(a.output/'old_template_noise_warning_only.csv',noisecheck)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'A':'#267db3','B':'#e28b26'}
    eg=np.linspace(500,3034,250)
    fig,axs=plt.subplots(1,3,figsize=(16,5.3))
    for j,ax in enumerate(axs):
        for g in ['A','B']:
            m=legacy&(cg==g);ax.scatter(ce[m],cy[m,j],s=3,alpha=.10,color=colors[g],rasterized=True,linewidths=0)
            val=coeff[g][0,j]+coeff[g][1,j]*(eg-1500)/1000;inside=eg<=2615
            ax.plot(eg[inside],val[inside],color=colors[g],lw=2,label=f'{g}：斜率 {slopes[g][j]:+.2f} ms/MeV')
            ax.plot(eg[~inside],val[~inside],color=colors[g],ls='--',lw=1.4)
        ax.set(title=LABELS[j],xlabel='CUORE 参考能量 / keV',ylabel='有效拟合时间常数 / ms',xlim=(450,3100));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('CUORE A/B 分开拟合中心趋势：按能量区间等权；虚线为外推',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,a.plots,'01_cuore_linear_centers')
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for j,key in enumerate(KEYS):
        for i,kind in enumerate([0,1]):
            ax=axs[i,j]
            for g in ['A','B']:
                b=[r for r in bins if r['group']==g and r['feature']==key];field='residual_std' if kind==0 else 'residual_half68'
                ax.scatter([float(r['energy_mean_kev']) for r in b],[float(r[field]) for r in b],color=colors[g],s=25)
                ax.plot(eg,width(g,eg,kind)[:,j],color=colors[g],label=g)
            ax.axvline(2615,color='#777',ls=':',lw=1);ax.set(title=LABELS[j] if i==0 else '',xlabel='能量 / keV',ylabel=('残差标准差 SD' if kind==0 else '残差中央 68% 半宽')+' / ms',yscale='log',xlim=(450,3100));ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('观测展宽：SD 与中央 68% 半宽分别建模；2615 keV 以上暂冻结宽度',fontsize=14);fig.tight_layout(rect=(0,0,1,.94));save(fig,a.plots,'02_cuore_observed_widths')
    for high in [False,True]:
        fig,axs=plt.subplots(1,3,figsize=(17,5.7));ee=np.linspace(450,3034,260) if high else np.geomspace(1,5407,400);ee=ee[ee<=3034]
        mu=center('A',ee,factor);hh=width('A',ee,1);cut=(te>=450)&(te<=3150) if high else np.ones(len(te),bool)
        for j,ax in enumerate(axs):
            ax.scatter(te[cut],ty[cut,j],color='#369448',s=2,alpha=.12 if high else .06,rasterized=True,linewidths=0,label='训练 clean：筛选单事件')
            m=legacy;ax.scatter(ce[m],cy[m,j],color='#8751a6',s=3,alpha=.16,rasterized=True,linewidths=0,label='CUORE 有效拟合（未平移）')
            ax.fill_between(ee,mu[:,j]-hh[:,j],mu[:,j]+hh[:,j],color='#d24445',alpha=.28,label='主草案中心 ± A 观测68%半宽')
            ax.plot(ee,mu[:,j],color='#bd242a',lw=2,label='主草案中心（A斜率）')
            ax.plot(ee,center('B',ee,factor)[:,j],color='#3b60a1',ls='--',lw=1.2,label='B斜率备选中心')
            ax.scatter(ENERGIES,center('A',ENERGIES,factor)[:,j],color='#bd242a',s=15,zorder=5)
            ax.set(title=LABELS[j],xlabel='能量 / keV',ylabel='拟合时间常数 / ms');ax.grid(alpha=.2)
            if high:ax.set_xlim(450,3150)
            else:
                ax.set(xscale='log',yscale='log',xlim=(.85,5700));ax.set_xticks([1,10,100,500,1000,3000,5000],labels=['1','10','100','500','1k','3k','5k'])
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=9,frameon=False)
        fig.suptitle('17点目标草案：'+('高能区' if high else '全能区；低于511 keV冻结为未验证参考'),fontsize=14,y=.995)
        fig.text(.5,.035,'红色宽度带是待达到的观测展宽参考，不是已确定的 clean 展宽；不同拟合模型仅作描述性对照。',ha='center',fontsize=10)
        fig.subplots_adjust(left=.055,right=.99,top=.78,bottom=.16,wspace=.25);save(fig,a.plots,'04_target_high_energy' if high else '03_target_full_energy')
    # Compact high-energy view of target bands within local training quantiles.
    fig,axs=plt.subplots(1,3,figsize=(15,4.6));he=ENERGIES[ENERGIES>=511]
    for j,key in enumerate(KEYS):
        ax=axs[j];ss=[r for r in supports if r['feature']==key];ax.fill_between(he,[r['train_q025'] for r in ss],[r['train_q975'] for r in ss],color='#369448',alpha=.10,label='局部训练 2.5–97.5%');ax.fill_between(he,[r['train_q16'] for r in ss],[r['train_q84'] for r in ss],color='#369448',alpha=.20,label='局部训练 16–84%')
        ax.errorbar(he,center('A',he,factor)[:,j],yerr=2*width('A',he,0)[:,j],color='#bd242a',fmt='o-',ms=4,capsize=3,label='目标中心 ± 2 SD（观测参考）')
        ax.set(title=LABELS[j],xlabel='能量 / keV',ylabel='时间常数 / ms');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('边际范围检查：±2 SD 仅作设计检查，不宣称高斯95%覆盖或联合可实现性',fontsize=12);fig.tight_layout(rect=(0,0,1,.93));save(fig,a.plots,'05_training_margin_check')
    summary=dict(stage='DRAFT steps 1-2 only; no ODE/clean/noise generation or inference',reference_energy_kev=2615,anchor_source=tr[anchor_idx],anchor_local_count=len(ids),anchor_selection='Actual vector nearest local log-space coordinatewise median in Mahalanobis distance, within ±10% of 2615 keV; source event energy is not exactly 2615',slope_policy='Absolute ms/MeV, preserve A relative slope ratios with common attenuation; no fractional-width rescaling',slope_factor_A=factor,max_passing_slope_factor_grid=maximum_factor,selected_slope_guard=selected,full_slope_guard=scan[0],A_slopes_ms_per_MeV=slopes['A'].tolist(),B_slopes_ms_per_MeV=slopes['B'].tolist(),primary_group='A narrower scenario, provisional; B kept separate as alternate',center_policy='Borrow anchor vector at reference 2615; linear above511, freeze below511; actual ODE center not yet determined',width_policy='Reuse separate log-linear SD/H68 fits; absolute ms; clip reference E to[511,2615]; low-energy values reference-only, not promised observable widths',clean_width='Not estimated or variance-subtracted; requires matching center waveform and selected noise/estimator',guard='A center in local training16–84%; center±2 observed SD inside2.5–97.5%, 100-point high-energy grid plus five nodes; local ±max(100keV,10%E); whitened log-3tau target 10-neighbor distance <= local leave-one-out 10-neighbor distance95%; chosen maximum factor rounded down to0.1',guard_passed=True,knn_warning='3-tau center sparsity only, no alpha/no waveform/no covariance-band guarantee',inputs=[dict(path=str(s),sha256=sha(s)) for s in sources],script_sha256=sha(__file__),checks=dict(reproduced_all_6_center_fits=True,reproduced_all_12_width_fits=True,n_training=61225,n_targets=len(targets),all_high_energy_A_guards=all(r['A_2sd_inside_2p5_97p5'] and r['A_center_inside_16_84'] for r in supports)))
    dump(a.output/'design_summary.json',summary)
    # Offline report, table is a proposal, not a production configuration.
    def fmt(v):return f'{v:.2f}'
    tab='<table><tr><th>E / keV</th><th>中心 τr / τfast / τslow（ms）</th><th>A观测SD（ms）</th><th>A观测68%半宽（ms）</th><th>状态</th></tr>'
    for r in targets:
        prefix=['tau_r','tau_fast','tau_slow'];cells=[str(r['energy_kev']),' / '.join(fmt(r[k+'_center_ms']) for k in prefix),' / '.join(fmt(r[k+'_A_observed_sd_ms']) for k in prefix),' / '.join(fmt(r[k+'_A_observed_h68_ms']) for k in prefix),'低能参考，待定' if r['energy_kev']<511 else ('中心外推，宽度冻结' if r['energy_kev']>2615 else 'CUORE范围内趋势参考')];tab+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>'
    tab+='</table>'
    body=f'''<!doctype html><meta charset="utf-8"><title>17点推理集目标草案</title><style>body{{font:16px sans-serif;max-width:1250px;margin:30px auto;line-height:1.6}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px}}img{{width:100%}}.note{{background:#fff2d9;padding:15px}}</style><h1>新推理集：中心与观测展宽草案（仅步骤1、2）</h1><p class="note">不是生成配置。未生成波形、未反求ODE参数、未运行DDPM；clean展宽留空，未扣除噪声。主草案选较窄的A，B独立保留，不混合成高斯。</p><p>参考中心（2615 keV）：{anchor.tolist()} ms。借用训练分片{tr[anchor_idx]['source_shard']}第{tr[anchor_idx]['window_idx']}窗（真实能量{float(tr[anchor_idx]['energy_true_kev']):.3f} keV）的实际三参数组合；并不证明其ODE参数在2615keV恰好产生此向量。完整斜率的联合中心检查未通过，网格最大通过比例为{maximum_factor:.2f}，向下取整为草案共用斜率系数k={factor:.2f}，A原斜率{slopes['A'].tolist()} ms/MeV。</p><p>中心：μ(E)=参考中心+k×A斜率×(max(E,511)−2615)/1000。SD和68%半宽分别复用CUORE原有log线性拟合，绝对单位ms；没有按中心比例放大。低于511keV冻结作为未验证参考，3034中心线性外推，宽度冻结到2615；避免向未观测区无限扩大或缩小。两档噪声应共用clean，参考噪声条件尚未决定。</p>{tab}<h2>中心与训练分布</h2><img src="../../plots/eval_target_draft_20261117/04_target_high_energy.png"><img src="../../plots/eval_target_draft_20261117/03_target_full_energy.png"><h2>CUORE分组趋势与展宽</h2><img src="../../plots/eval_target_draft_20261117/01_cuore_linear_centers.png"><img src="../../plots/eval_target_draft_20261117/02_cuore_observed_widths.png"><h2>范围检查</h2><img src="../../plots/eval_target_draft_20261117/05_training_margin_check.png"><p>高能中心及±2观测SD通过本次边际设计检查；中心还通过了本次3参数近邻检查，但展宽的联合可实现性、alpha和实际波形尚未验证。training_support_checks.csv另列3参数中心局部近邻稀疏度，不包含alpha、实际波形和分布尾部；B宽度可能越出核心区。训练仅限此前质量筛选61225个单事件，不含pileup与标记样本。</p><p>旧模板的噪声/拟合SD仅在old_template_noise_warning_only.csv用于预警，不作为新中心的噪声预算，不作方差扣除。低能观测目标尤其可能无法实现。两边拟合模型不同，CUORE观测展宽包含基线状态、噪声与参数退化，A/B均不假定高斯。本草案不是复现真实热常数。</p><p>后续：确认主/备方案及低能规则；再确定ODE中心、匹配噪声条件与拟合贡献，校准clean联合展宽。所有待做工作均未执行。</p>'''
    (a.output/'target_draft_report.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(anchor=anchor.tolist(),factor=factor,maximum_factor=maximum_factor,effective_slopes=(factor*slopes['A']).tolist(),selected_guard=selected,high_knn_ranks=[r['center_3tau_knn_sparsity_percentile'] for r in supports if r['feature']==KEYS[0]],passed=True),ensure_ascii=False))

if __name__=='__main__':main()
