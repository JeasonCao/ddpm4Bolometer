"""Export compact tables, verification and a local Chinese report from saved results."""
import argparse,csv,json,html,shutil,hashlib
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_analysis import KEYS,LABELS,COL,poly,table,dump,save,sha

def main(a):
    out,plots,root=map(Path,[a.output,a.plots,a.data])
    read=lambda n:list(csv.DictReader((out/n).open(encoding='utf-8-sig')))
    events=read('calibrated_event_features.csv');peaks=read('calibration_peaks.csv');stats=read('peak_residual_statistics.csv')
    models=json.loads((out/'trend_models.json').read_text())['models']
    cal=json.loads((out/'calibration.json').read_text())['groups']
    manifest=json.loads((out/'manifest.json').read_text())
    assert len(events)==5000
    for name in ['cuoreTraining.h5','cuoreTest.h5']:
        with h5py.File(root/name,'r') as h:
            for r in [v for v in events if v['source_file']==name]:
                i=int(r['source_row']);assert int(h['eventId'][i].item())==int(r['event_id'])
                assert np.isclose(float(h['normalizationScale'][i,0]),float(r['amplitude_adc']),rtol=0,atol=1e-8)
                assert np.isclose(float(h['normalizationOffset'][i,0]),float(r['offset_adc']),rtol=0,atol=1e-8)
    predictions=[]
    for g in ['A','B']:
        for key in KEYS:
            m=models[g][key]
            assert m['degree']==1
            for e in [583.191,911.204,2614.511]:
                x=np.array([(e-1500)/1000])
                predictions.append(dict(group=g,feature=key,energy_kev=e,mean=float((poly(x,1)@m['mean_coefficients'])[0]),
                    observed_std_model=float(np.exp((poly(x,1)@m['log_std_coefficients'])[0])),half68_model=float(np.exp((poly(x,1)@m['log_half68_coefficients'])[0]))))
    table(out/'predictions_at_peaks.csv',predictions)
    coeff=[]
    for g in ['A','B']:
        for key in KEYS:
            m=models[g][key];c=m['mean_coefficients'];s=m['log_std_coefficients'];h=m['log_half68_coefficients']
            coeff.append(dict(group=g,feature=key,mean_intercept_at_E0=c[0]-1.5*c[1],mean_slope_per_MeV=c[1],std_exp_intercept=s[0]-1.5*s[1],std_exp_slope_per_MeV=s[1],half68_exp_intercept=h[0]-1.5*h[1],half68_exp_slope_per_MeV=h[1]))
    table(out/'simple_coefficients.csv',coeff)
    # Compact central-distribution view; full-range figures retain every plotted outlier.
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(14,8));grid=np.linspace(511,2615,200);x=(grid-1500)/1000
    for j in range(3):
        key=KEYS[j];values=[]
        for g in ['A','B']:
            rr=[r for r in events if r['baseline_group']==g and r['legacy_4980']=='True'];ee=np.array([float(r['energy_reference_kev']) for r in rr]);yy=np.array([float(r[key]) for r in rr]);values.extend(yy)
            m=models[g][key];mu=poly(x,1)@m['mean_coefficients'];sd=np.exp(poly(x,1)@m['log_std_coefficients']);hw=np.exp(poly(x,1)@m['log_half68_coefficients'])
            axs[0,j].scatter(ee,yy,s=3,alpha=.10,color=COL[g],rasterized=True);axs[0,j].plot(grid,mu,color=COL[g],lw=2,label=f'组{g}均值')
            axs[1,j].plot(grid,sd,color=COL[g],lw=2,label=f'组{g} 观测SD');axs[1,j].plot(grid,hw,'--',color=COL[g],label=f'组{g} 68%半宽')
        low,high=np.quantile(values,[.005,.99]);axs[0,j].set_ylim(low,high)
        outside=sum(v<low or v>high for v in values);axs[0,j].text(.03,.95,f'纵轴放大：{outside}点在图外',transform=axs[0,j].transAxes,va='top',fontsize=8)
        axs[0,j].set(ylabel=LABELS[j],xlabel='参考能量 / keV');axs[1,j].set(ylabel=LABELS[j]+' 的散布',xlabel='参考能量 / keV')
        for ax in axs[:,j]:ax.grid(alpha=.2);ax.set_xlim(480,2700)
    axs[0,0].legend(loc='lower left',fontsize=8);axs[1,0].legend(fontsize=8)
    fig.suptitle('时间常数的连续描述：上排为线性中心趋势，下排为观测散布模型（仅在511–2615 keV内使用）')
    save(fig,plots,'07_compact_tau_summary')
    rejected=sum(float(r['normality_q_bh'])<.05 for r in stats)
    trs=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['group'],r['feature'],f"{r['mean_intercept_at_E0']:.5g} + ({r['mean_slope_per_MeV']:.5g}) E",f"exp({r['std_exp_intercept']:.5g} + ({r['std_exp_slope_per_MeV']:.5g}) E)"] )+'</tr>' for r in coeff)
    imgs=[('07_compact_tau_summary','时间常数的中心与宽度：紧凑版'),('01_calibration_check','参考定标检查'),('05_waveforms_by_energy','实际波形的变化'),('02_shape_trends_A','A组全部六指标与全部点'),('02_shape_trends_B','B组全部六指标与全部点'),('03_width_vs_energy','观测散布、区间与长尾'),('04_gaussian_check_A','A组同能峰附近高斯性'),('04_gaussian_check_B','B组同能峰附近高斯性'),('06_residual_correlations','参数联合分布')]
    figures=''.join(f'<h2>{title}</h2><img src="{html.escape(ospath)}" alt="{title}">' for name,title in imgs for ospath in [str(Path('../../plots')/plots.name/(name+'.png')).replace('\\','/')])
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CUORE能量与波形分布</title><style>body{{font:17px/1.7 system-ui;max-width:1300px;margin:30px auto;padding:0 25px;color:#243044}}img{{width:100%;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border:1px solid #ddd}}.box{{background:#eef4fa;padding:18px}}h2{{margin-top:40px}}</style>
    <h1>CUORE：能量—波形中心—散布（2026-11-14）</h1><div class="box">中心趋势可以先用直线；散布随能量减小。但不能统一假设独立高斯：{rejected}/{len(stats)}个峰区/指标检验在BH校正后拒绝高斯，τ与α残差高度相关。所有宽度为观测散布，尚未分离噪声和拟合不确定度。</div>
    <h2>定标与选择</h2><p>5000单事件全量给出参考能量，A/B组按基线−70000 ADC分开。各组用高幅值区(a&gt;27500)的Huber线性幅值—基线关系作乘法增益修正，再以583.191/2614.511 keV两峰定标为零截距二次式。独立511/727/911/969 keV峰最大偏差A={cal['A']['max_check_residual_kev']:.2f} keV，B={cal['B']['max_check_residual_kev']:.2f} keV。B组727峰较弱且有较多连续本底，969峰可能混峰。不是官方能量/分辨率测量，也没有完全消除工作点变化。</p>
    <p>图与形状分析沿用此前4980例选择；连续拟合进一步限500–2650 keV，实际A=2712、B=2264。另存全部5000例中心趋势敏感性对照。各峰采用拟合峰中心±2σ选事件，带有连续本底，不能当成绝对纯净、完全同能量的样本。</p>
    <h2>简单模型怎么用</h2><p>θ(E)=μ(E)+ε(E)。μ=a+bE；观测标准差先用σ=exp(c+dE)，同时提供68%半宽模型。E以MeV计，时间以ms计。下表是候选描述式，不是批准投入生成器的物理真值。中心趋势按8个275-keV箱等权拟合；五折验证中二次均未比线性改善2%以上。宽度模型是分箱残差宽度的描述拟合，未做独立生成验证，不能声称唯一最优形式。未拒绝高斯不等于证明高斯。</p>
    <table><tr><th>组</th><th>指标</th><th>均值</th><th>观测SD</th></tr>{trs}</table>
    <h2>生成验证集前的约束</h2><ol><li>分组保存均值/宽度，不能把不同工作点直接混为一个高斯。</li><li>若采用有效三指数模型，τr、τfast、τslow、α要联合抽样并保留相关性及物理约束；经验残差向量已保存。经验向量也含观测误差，只能作为候选统计材料。</li><li>不能把全部观测σ作为clean波形真散布再加同等噪声，这会重复计算噪声。应以已有噪声对照或前向模拟检验分离本征散布。</li><li>这些τ对应既有共享RC+Bessel有效读出模型，不是直接测量的ODE热参数；若继续使用ODE生成，应通过前向波形指标匹配，而非把τ当作ODE参数。</li><li>只有约511–2615 keV的实测支持，不能直接外推到几keV或3034 keV。增益与定标系数误差未纳入宽度区间。</li></ol>
    <p>数据：<a href="calibrated_event_features.csv">逐事件定标表</a> · <a href="simple_coefficients.csv">简单系数</a> · <a href="peak_residual_statistics.csv">峰区分布检验</a> · <a href="energy_bin_statistics.csv">分箱宽度</a> · <a href="trend_models.json">模型设置</a>。</p>{figures}</html>'''
    (out/'CUORE_energy_shape_report.html').write_text(text,encoding='utf-8')
    dump(out/'independent_verification.json',dict(passed=True,all5000_ids_amplitudes_baselines_match_h5=True,all_reference_energies_finite=all(np.isfinite(float(r['energy_reference_kev'])) for r in events),all_linear_selected=True,normality_rejections_bh05=rejected,model_range_kev=[511,2615]))
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['run_analysis.py','summarize.py','README.md']:
        p=Path(__file__).parent/name
        if p.exists():shutil.copy2(p,snap/name)
    print('Independent verification and report complete.')
    for r in predictions:
        if r['group']=='A' and r['feature'] in KEYS[:3]:print(r)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['output','plots','data']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
