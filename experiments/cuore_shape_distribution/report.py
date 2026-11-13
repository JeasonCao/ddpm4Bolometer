"""Tables, scientific figures and a portable Chinese report for CUORE shape analysis."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,argparse,base64,html
from pathlib import Path
import numpy as np,h5py
from scipy.stats import spearmanr,theilslopes,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_shape_distribution.run_study import CENTERS,HALF_WIDTH,KEYS,dump,csvwrite,quant
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False})
LABELS=['有效上升常数 τr（ms）','有效快下降常数 τf（ms）','有效慢下降常数 τs（ms）',
        '混合权重 α','波形上升 10–90%（ms）','波形下降 90–30%（ms）']
COLORS=['#3478b5','#d58b24','#409477','#a766a7']
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def save(fig,path,name):
    fig.savefig(path/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(path/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def halfwidth(v):return float((np.quantile(v,.84)-np.quantile(v,.16))/2)
def width_ci(v,seed):
    rng=np.random.default_rng(seed);v=np.asarray(v);idx=rng.integers(0,len(v),(400,len(v)))
    q=np.quantile(v[idx],[.16,.84],axis=1);w=(q[1]-q[0])/2
    return np.quantile(w,[.025,.975]).tolist()
def main(a):
    root=Path(a.input);plots=Path(a.plots);plots.mkdir(parents=True,exist_ok=True)
    with h5py.File(root/'shape_distributions.h5') as h:
        data={key:h['events'][key][:] for key in h['events']}
        aligned_t=h['aligned_time_s'][:]
        waves={j:h[f'bands/{j}/raw_aligned_waveform_quantiles'][:] for j in range(4)}
    amp=data['amplitude_adc'];mask=data['legacy_4980'].astype(bool)
    assert len(amp)==5000 and mask.sum()==4980
    bandrows=[];binned=[];correlations={};summary={}
    for j,c in enumerate(CENTERS):
        ind=data['band']==j;rows={}
        for k in KEYS:
            for name,sel in [('all_single',ind),('legacy_4980',ind&mask)]:
                q=quant(data[k][sel])
                bandrows.append(dict(band=j,center_adc=c,low_adc=c*(1-HALF_WIDTH),high_adc=c*(1+HALF_WIDTH),
                    selection=name,feature=k,**q,half68_width=(q['q84']-q['q16'])/2))
            slope,intercept,_,_=theilslopes(data[k][ind],amp[ind]/c-1.)
            residual=data[k][ind]-(slope*(amp[ind]/c-1.)+intercept)
            rows[k]=dict(**quant(data[k][ind]),detrended_half68=halfwidth(residual),
                         fractional_amplitude_slope=float(slope))
        rows['n']=int(ind.sum());rows['baseline_rms_adc']=quant(data['baseline_rms_adc'][ind])
        rows['selection_sensitivity']={}
        for k in KEYS:
            rows['selection_sensitivity'][k]=dict(all_median=float(np.median(data[k][ind])),
                legacy_median=float(np.median(data[k][ind&mask])),
                all_half68=halfwidth(data[k][ind]),legacy_half68=halfwidth(data[k][ind&mask]))
        rows['width_sensitivity']={}
        for width in [.005,.01,.025]:
            sel=abs(amp/c-1)<=width
            rows['width_sensitivity'][str(width)]=dict(n=int(sel.sum()),
                features={k:quant(data[k][sel]) for k in KEYS})
        summary[str(j)]=rows
        z=np.column_stack([np.log(data[k][ind]) for k in KEYS[:3]]+[np.log(np.clip(data['alpha'][ind],1e-8,1-1e-8)/(1-np.clip(data['alpha'][ind],1e-8,1-1e-8)))])
        detrended=np.empty_like(z)
        for k in range(4):
            slope,intercept,_,_=theilslopes(z[:,k],amp[ind]/c-1.)
            detrended[:,k]=z[:,k]-(slope*(amp[ind]/c-1.)+intercept)
        correlations[str(j)]=dict(features=['log_tau_r','log_tau_fast','log_tau_slow','logit_alpha'],
            detrended_spearman=spearmanr(detrended).statistic.tolist(),
            observed_log_parameter_covariance=np.cov(z,rowvar=False).tolist(),
            detrended_log_parameter_covariance=np.cov(detrended,rowvar=False).tolist())
    for low in np.arange(5000,30000,1000):
        ind=(amp>=low)&(amp<low+1000)
        if not np.any(ind):continue
        for k in KEYS:
            binned.append(dict(low_adc=float(low),high_adc=float(low+1000),center_adc=float(np.median(amp[ind])),feature=k,**quant(data[k][ind])))
    csvwrite(root/'narrow_band_statistics.csv',bandrows);csvwrite(root/'amplitude_bin_statistics.csv',binned)
    dump(root/'parameter_correlations.json',correlations)
    summary['smoothing_sensitivity']={k:dict(median_abs_2vs5_ms=float(np.median(abs(data[f'{k}_sigma2_ms']-data[f'{k}_sigma5_ms']))),
                p99_abs_2vs5_ms=float(np.quantile(abs(data[f'{k}_sigma2_ms']-data[f'{k}_sigma5_ms']),.99))) for k in ['rise','decay']}
    summary['global_spearman']={k:float(spearmanr(amp,data[k]).statistic) for k in KEYS}
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,k,label in zip(axs.flat,KEYS,LABELS):
        ax.scatter(amp[mask],data[k][mask],s=3,alpha=.15,color='#5186ac',rasterized=True)
        rows=[r for r in binned if r['feature']==k and r['n']>=20]
        x=[r['center_adc'] for r in rows]
        ax.fill_between(x,[r['q16'] for r in rows],[r['q84'] for r in rows],color='#e59535',alpha=.25,label='区间内16–84%分布')
        ax.plot(x,[r['median'] for r in rows],color='#bd681c',lw=2,label='区间中位数（全部单事件）')
        ax.set_xlabel('幅值（ADC；未标定为能量）');ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('随幅值的形状变化：点为原4980例，统计曲线使用全部5000例',fontsize=15)
    fig.tight_layout();save(fig,plots,'01_amplitude_trends')
    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for j,c in enumerate(CENTERS):
        w=waves[j][2];label=f'{c:,.0f} ADC ±2.5%'
        axs[0].plot(aligned_t,w,color=COLORS[j],label=label,lw=1.7)
        axs[1].plot(aligned_t,w,color=COLORS[j],lw=1.7)
        axs[2].plot(aligned_t,100*(w-waves[0][2]),color=COLORS[j],lw=1.7)
    axs[0].set_xlim(-.15,2);axs[1].set_xlim(-.1,.2);axs[2].set_xlim(-.15,2)
    axs[0].set_ylabel('归一化波形中位数');axs[1].set_ylabel('上升沿与峰顶');axs[2].set_ylabel('相对最低幅值组的差（峰高百分比）')
    axs[0].legend(fontsize=9)
    for ax in axs:ax.set_xlabel('相对上升50%时刻（s）');ax.grid(alpha=.2)
    fig.suptitle('直接比较实测波形：只对齐与幅值归一化，不伸缩时间轴',fontsize=15)
    fig.tight_layout();save(fig,plots,'02_typical_waveforms')
    fig,axs=plt.subplots(2,4,figsize=(17,8))
    for j,c in enumerate(CENTERS):
        q=waves[j];n=summary[str(j)]['n']
        axs[0,j].fill_between(aligned_t,q[0],q[4],color=COLORS[j],alpha=.12,label='2.5–97.5%')
        axs[0,j].fill_between(aligned_t,q[1],q[3],color=COLORS[j],alpha=.35,label='16–84%')
        axs[0,j].plot(aligned_t,q[2],color=COLORS[j],lw=1.5,label='中位数')
        axs[1,j].fill_between(aligned_t,100*(q[0]-q[2]),100*(q[4]-q[2]),color=COLORS[j],alpha=.12)
        axs[1,j].fill_between(aligned_t,100*(q[1]-q[2]),100*(q[3]-q[2]),color=COLORS[j],alpha=.35)
        axs[1,j].axhline(0,color=COLORS[j],lw=1)
        axs[0,j].set_title(f'{c:,.0f} ADC ±2.5% | N={n}')
        for ax in axs[:,j]:ax.set_xlim(-.15,1.5);ax.grid(alpha=.2);ax.set_xlabel('相对上升50%时刻（s）')
    axs[0,0].set_ylabel('实测归一化波形');axs[1,0].set_ylabel('相对本组中位数（峰高百分比）')
    axs[0,0].legend(fontsize=9)
    for j in range(4):axs[1,j].set_ylim(-1.5,1.5)
    fig.suptitle('窄幅值区间内的波形分布：阴影是事件间散布，不是均值误差',fontsize=15)
    fig.tight_layout();save(fig,plots,'03_within_band_waveforms')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,k,label in zip(axs.flat,KEYS,LABELS):
        values=[data[k][data['band']==j] for j in range(4)]
        parts=ax.violinplot(values,positions=np.arange(4),showextrema=False,quantiles=[[.16,.5,.84]]*4)
        for body,col in zip(parts['bodies'],COLORS):body.set_facecolor(col);body.set_alpha(.6)
        parts['cquantiles'].set_color('#222222')
        ax.set_xticks(np.arange(4),['6,500','10,000','18,000','28,500'])
        ax.set_xlabel('区间中心幅值（ADC；每组±2.5%）');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('窄幅值区间内的参数分布：横线为16%、50%、84%分位数',fontsize=15)
    fig.tight_layout();save(fig,plots,'04_within_band_distributions')
    controls=read(root/'noise_controls.jsonl')
    assert len(controls)==4*2*json.loads((root/'manifest.json').read_text())['noise_repeats']
    csvwrite(root/'noise_controls.csv',controls)
    noise_stats=[];fig,axs=plt.subplots(2,3,figsize=(15,8))
    for metric,(ax,k,label) in enumerate(zip(axs.flat,KEYS,LABELS)):
        for kindid,kind in enumerate(['observed','observed_detrended','model','empirical']):
            widths=[];lo=[];hi=[]
            for j in range(4):
                rr=[r for r in controls if r['band']==j and r['kind']==kind and r['success']]
                ind=data['band']==j
                if kind in ('observed','observed_detrended'):
                    vals=data[k][ind].copy()
                    if kind=='observed_detrended':vals-=summary[str(j)][k]['fractional_amplitude_slope']*(amp[ind]/CENTERS[j]-1)
                else:vals=np.array([r[k] for r in rr])
                width=halfwidth(vals);ci=width_ci(vals,7000+metric*100+kindid*10+j)
                noise_stats.append(dict(band=j,center_adc=CENTERS[j],kind=kind,feature=k,n=len(vals),
                    median=float(np.median(vals)),half68_width=width,width_ci_low=ci[0],width_ci_high=ci[1]))
                widths.append(width);lo.append(max(0,width-ci[0]));hi.append(max(0,ci[1]-width))
            ax.bar(np.arange(4)+(kindid-1.5)*.19,widths,.18,yerr=[lo,hi],capsize=2,
                color=['#647b91','#93b1cc','#d39a43','#589e80'][kindid],label=['实测事件间散布','扣除区间内幅值趋势','固定拟合形状+噪声','固定实测中位形状+噪声'][kindid])
        ax.set_xticks(np.arange(4),['6,500','10,000','18,000','28,500'])
        ax.set_xlabel('幅值（ADC）');ax.set_ylabel(label+'：中央68%半宽');ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('噪声能产生多少“形状散布”？误差线为半宽的bootstrap 95%区间',fontsize=14)
    fig.tight_layout();save(fig,plots,'05_noise_scatter_control')
    csvwrite(root/'noise_width_comparison.csv',noise_stats)
    summary['noise_control']=dict(n=len(controls),success=sum(r['success'] for r in controls),
        collapsed=sum(r['collapsed'] for r in controls),warning='Stationary Gaussian pretrigger PSD approximation; not intrinsic-variance subtraction.',
        baseline_rms_ratio_quantiles=np.quantile([r['synthetic_baseline_rms_adc']/r['noise_rms_adc'] for r in controls],[.16,.5,.84]).tolist())
    summary['noise_width_ratio_observed_over_empirical']={}
    for j in range(4):
        summary['noise_width_ratio_observed_over_empirical'][str(j)]={}
        for k in KEYS:
            obs=next(r for r in noise_stats if r['band']==j and r['feature']==k and r['kind']=='observed')
            control=next(r for r in noise_stats if r['band']==j and r['feature']==k and r['kind']=='empirical')
            summary['noise_width_ratio_observed_over_empirical'][str(j)][k]=obs['half68_width']/control['half68_width']
    summary['detrended_noise_width_ratio_observed_over_empirical']={}
    for j in range(4):
        summary['detrended_noise_width_ratio_observed_over_empirical'][str(j)]={}
        for k in KEYS:
            obs=next(r for r in noise_stats if r['band']==j and r['feature']==k and r['kind']=='observed_detrended')
            control=next(r for r in noise_stats if r['band']==j and r['feature']==k and r['kind']=='empirical')
            summary['detrended_noise_width_ratio_observed_over_empirical'][str(j)][k]=obs['half68_width']/control['half68_width']
    fig,axs=plt.subplots(1,4,figsize=(16,4.3))
    for j,ax in enumerate(axs):
        mat=np.array(correlations[str(j)]['detrended_spearman']);im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(4),['τr','τf','τs','α']);ax.set_yticks(range(4),['τr','τf','τs','α']);ax.set_title(f'{CENTERS[j]:,.0f} ADC')
        for row in range(4):
            for col in range(4):ax.text(col,row,f'{mat[row,col]:.2f}',ha='center',va='center',color='white' if abs(mat[row,col])>.65 else 'black')
    fig.colorbar(im,ax=axs.tolist(),shrink=.7,label='Spearman r（去除区间内幅值线性趋势）')
    fig.suptitle('时间常数与权重并不独立：相关中也包含噪声和拟合耦合',fontsize=14)
    save(fig,plots,'06_parameter_correlations')
    sens=read(root/'response_sensitivity.jsonl');csvwrite(root/'response_sensitivity.csv',sens)
    assert len(sens)==128
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    settings=sorted(set((r['fc'],r['rc']) for r in sens));sensitivity={}
    for sid,(fc,rc) in enumerate(settings):
        rr=[r for r in sens if (r['fc'],r['rc'])==(fc,rc)]
        sensitivity[f'{fc}_{rc}']={}
        for ax,k in zip(axs,KEYS[:3]):
            med=[float(np.median([r[k] for r in rr if r['band']==j])) for j in range(4)]
            ax.plot(CENTERS,med,'o-',label=f'{fc:g} Hz / {rc*1000:g} ms',lw=1.5)
            sensitivity[f'{fc}_{rc}'][k]=dict(band_medians=med,high_vs_low_percent=100*(med[-1]/med[0]-1))
    for ax,label in zip(axs,LABELS[:3]):ax.set_xlabel('幅值（ADC）');ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend(fontsize=9)
    fig.suptitle('读出设置敏感性：同32个事件，4组近最优共享响应设置',fontsize=14)
    fig.tight_layout();save(fig,plots,'07_readout_sensitivity')
    summary['response_sensitivity']=sensitivity;summary['response_sensitivity_success']=sum(r['success'] for r in sens)
    summary['amplitude_minmax_adc']=[float(amp.min()),float(amp.max())]
    summary['source_n']=5000;summary['scatter_n']=4980
    summary['offset_association']={'amplitude_offset_spearman':float(spearmanr(amp,data['offset_adc']).statistic)}
    z=np.column_stack([np.ones(len(amp)),rankdata(data['offset_adc'])])
    x=rankdata(amp);rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0]
    for k in ['rise_10_90_ms','decay_90_30_ms']:
        y=rankdata(data[k]);ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
        summary['offset_association'][k+'_partial_rank_r']=float(np.corrcoef(rx,ry)[0,1])
    fig,axs=plt.subplots(2,3,figsize=(14,7.8))
    for ax,k,label in zip(axs.flat,KEYS,LABELS):
        rr=[r for r in binned if r['feature']==k and r['n']>=20]
        xx=[r['center_adc'] for r in rr]
        ax.fill_between(xx,[r['q16'] for r in rr],[r['q84'] for r in rr],alpha=.22,color='#3478b5',label='事件间16–84%范围')
        ax.plot(xx,[r['median'] for r in rr],color='#275f8c',lw=2,label='中位数')
        for j,c in enumerate(CENTERS):
            qq=summary[str(j)][k]
            ax.errorbar(c,qq['median'],yerr=[[qq['median']-qq['q16']],[qq['q84']-qq['median']]],fmt='o',color=COLORS[j],capsize=3,ms=4)
        ax.set_xlabel('幅值（ADC，未标定为能量）');ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=9)
    fig.suptitle('总体趋势与事件间分布：全部5000例；彩色点为四个窄幅值区间',fontsize=14)
    fig.tight_layout();save(fig,plots,'08_distribution_summary')
    dump(root/'summary.json',summary)
    generate_html(root,plots,summary)
    print(json.dumps({k:summary[k] for k in ['global_spearman','smoothing_sensitivity','noise_control','noise_width_ratio_observed_over_empirical']},indent=2))
    for j,c in enumerate(CENTERS):
        print('BAND',c,'N',summary[str(j)]['n'],[(k,summary[str(j)][k]['median'],summary[str(j)][k]['q16'],summary[str(j)][k]['q84']) for k in KEYS],flush=True)

def generate_html(root,plots,s):
    cells=[]
    for j,c in enumerate(CENTERS):
        r=s[str(j)]
        cells.append('<tr><td>'+f'{c:,.0f} ±2.5%'+'</td><td>'+str(r['n'])+'</td>'+''.join(
            '<td>'+f'{r[k]["median"]:.2f}<br><small>[{r[k]["q16"]:.2f}, {r[k]["q84"]:.2f}]</small></td>' for k in KEYS[:3]+KEYS[4:])+'</tr>')
    images=[]
    for path in sorted(plots.glob('0*.png')):
        images.append('<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(path.stem)+'</figcaption></figure>')
    conclusions='<h2>本次数据支持的主要结论</h2><ul>'
    low=s['0'];high=s['3']
    for k,label in zip(['rise_10_90_ms','decay_90_30_ms'],['直接上升10–90%时间','直接下降90–30%时间']):
        change=100*(high[k]['median']/low[k]['median']-1)
        conclusions+=f'<li>幅值从6500到28500 ADC，{label}中位数从{low[k]["median"]:.2f}到{high[k]["median"]:.2f}ms，变化{change:+.2f}%。</li>'
    ratios=s['noise_width_ratio_observed_over_empirical']
    rise_ratios=[ratios[str(j)]['rise_10_90_ms'] for j in range(4)]
    decay_ratios=[ratios[str(j)]['decay_90_30_ms'] for j in range(4)]
    tau_ratios=[ratios[str(j)][k] for j in range(4) for k in KEYS[:3]]
    conclusions+=f'<li>直接上升时间的观测分布半宽约为固定实测模板加噪声对照的{min(rise_ratios):.2f}–{max(rise_ratios):.2f}倍；目前不能据此要求很大的真实上升沿随机变化。</li>'
    conclusions+=f'<li>直接下降时间对应比值约{min(decay_ratios):.2f}–{max(decay_ratios):.2f}；三个拟合时间常数约{min(tau_ratios):.2f}–{max(tau_ratios):.2f}。存在超出本次噪声模型的散布，但仍可能混有模型误差、慢漂移或其它未建模效应，不能全部视作本征热学变化。图5同时提供扣除窄区间内幅值趋势后的分布宽度。</li>'
    detrended=s['detrended_noise_width_ratio_observed_over_empirical']
    dr=[detrended[str(j)]['rise_10_90_ms'] for j in range(4)]
    dd=[detrended[str(j)]['decay_90_30_ms'] for j in range(4)]
    dt=[detrended[str(j)][k] for j in range(4) for k in KEYS[:3]]
    conclusions+=f'<li><b>更接近“同幅值”的比较：</b>扣除区间内幅值趋势后，上升时间散布/噪声对照为{min(dr):.2f}–{max(dr):.2f}，下降为{min(dd):.2f}–{max(dd):.2f}，有效时间常数为{min(dt):.2f}–{max(dt):.2f}。高幅值组的原始下降时间散布有明显的区间宽度贡献。</li>'
    conclusions+='<li>有效参数有强相关：生成时应保持联合结构。四组近最优读出设置保留了低/高幅值间的变化方向，但绝对tau会变化；慢分量的小样本曲线并非严格单调。</li>'
    conclusions+='<li>将直接上升/下降时间与幅值的秩相关在控制基线偏置后重算，趋势仍保留；这不是纯能量因果效应的证明。其它敏感性数值见summary.json。</li></ul>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CUORE 波形与幅值分布</title>
<style>body{max-width:1250px;margin:35px auto;padding:0 24px;font-family:"Microsoft YaHei",sans-serif;color:#253244;line-height:1.7;background:#f7f9fc}h1,h2{color:#1c466d}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6dfeb;padding:10px;text-align:center}small{color:#667}img{width:100%;background:white}figure{margin:28px 0}figcaption{font-size:12px;color:#667}.notice{background:#e9f1fa;padding:18px;border-left:5px solid #3478b5}</style>
<h1>CUORE：随幅值的形状变化与区间内分布</h1>
<p>2026-11-13 · 5000个标注为单事件的窗口 · 统计包含全部单事件；散点沿用先前4980点选择。</p>
<div class="notice"><b>目前没有能量标定。</b>所有横轴均为ADC幅值，“窄幅值区间”不等于严格的同能量。
公开数据来自同一dataset-calorimeter的标定运行，且选取重建能量&gt;500 keV的触发事件；不能据此约束低于500 keV或不同探测器的形状。
<a href="https://arxiv.org/html/2607.02548v1">数据发布说明（附录A）</a></div>
<h2>四个代表区间</h2><p>表中为中位数，[16%,84%]是事件间分布，不是中位数误差。前三列为共享RC+Bessel模型的有效时间常数，后两列直接由轻度平滑后的波形测量。</p>
<table><thead><tr><th>幅值（ADC）</th><th>N</th><th>τr（ms）</th><th>τf（ms）</th><th>τs（ms）</th><th>上升10–90%（ms）</th><th>下降90–30%（ms）</th></tr></thead><tbody>'''+''.join(cells)+'''</tbody></table>
<h2>如何用于后续验证集</h2>
<ol><li>以幅值条件下的典型实测波形、直接上升/下降时间与联合参数分布作为匹配目标。</li>
<li>保持时间常数与混合权重的关联；不要把三个时间常数独立抽样。</li>
<li>窄区间内观测散布包含噪声、剩余幅值差异和模型误差。噪声对照用于估计其量级，不直接做方差相减，不将全部散布作为clean形状散布再加一次噪声。</li>
<li>有效时间常数不是ODE热学参数。生成验证集应比较最终读出波形及这些观测指标，不能直接把τ填入热学参数。</li>
<li>本次借用了训练与测试文件做描述性分析；若以后据此调生成器，应另留独立事件作最终验证。</li></ol>
<h2>方法与限制</h2><p>直接指标使用3ms高斯平滑，并检查2/5ms敏感性；波形仅按上升50%时刻对齐及平滑峰高归一化，没有时间伸缩。
拟合使用固定共享5.1Hz（-3dB）、RC=68ms。图7使用四组近最优共享设置检查结论稳定性。拟合时间常数仍是模型条件下的有效参数。</p>
<p>每个幅值区间，对固定拟合模板与固定实测中位模板各做96次加噪再拟合。噪声由同区间2.5秒基线的平均功率谱构造，RMS从实测分布抽取，假设平稳高斯过程；最低可分辨频率0.4Hz以下按平坦谱延伸，不能覆盖未测到的慢漂移、非平稳或信号相关噪声。模板不变，因此对照产生的散布并非真实形状变化。</p>
<h2>数据文件</h2><ul>
<li>event_features.csv：5000行事件特征、原始文件/行号/ID、有效参数、直接波形指标和标记。</li>
<li>narrow_band_statistics.csv / amplitude_bin_statistics.csv：窄区间及全幅值分箱统计。</li>
<li>shape_distributions.h5：事件表、拟合参数、对齐波形分位数、基线噪声谱和固定模板。</li>
<li>noise_controls.csv / noise_width_comparison.csv：全部噪声对照与分布半宽比较。</li>
<li>parameter_correlations.json / response_sensitivity.csv：联合分布和读出设置敏感性。</li>
<li>summary.json / manifest.json：汇总与来源哈希。完整原始波形不复制，依据源文件+行号回查。</li></ul>'''+''.join(images)+'</html>'
    text=text.replace('<h2>四个代表区间</h2>',conclusions+'<h2>四个代表区间</h2>')
    repeats=json.loads((root/'manifest.json').read_text())['noise_repeats']
    text=text.replace('各做96次',f'各做{repeats}次')
    (root/'CUORE_shape_report.html').write_text(text,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--plots',required=True)
    main(p.parse_args())
