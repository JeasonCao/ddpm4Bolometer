"""Audit saved summaries, check window-length sensitivity, and assemble a report."""
import argparse,json,csv,sys,html
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_analysis import read,dump,table,qstats,save,estimate,COLORS

def main(a):
 out,plots,notes,data=map(Path,[a.output,a.plots,a.notes,a.data]);prior=notes/'results/analysis/cuore_energy_shape_20261114'
 stats=read(out/'sim_direct_time_statistics.csv');ev=read(out/'sim_direct_time_per_event.csv');assert len(ev)==34000
 for r in stats:
  vals=[float(v[r['feature']]) for v in ev if v['noise_level']==r['noise_level'] and v['energy_kev']==r['energy_kev'] and v['valid']=='True']
  q=qstats(vals);np.testing.assert_allclose([q['half68'],q['std'],q['median']],[float(r['half68']),float(r['std']),float(r['median'])],rtol=1e-12)
 of=read(out/'short_window_baseline_estimates.csv');samples=np.load(out/'baseline_estimator_samples.npz')
 for name in samples.files:
  row=next(r for r in of if r['dataset']==name and r['template']==('模拟' if name.startswith('模拟') else name) and r['cutoff_hz']=='20')
  np.testing.assert_allclose(np.std(samples[name],ddof=1),float(row['sigma_keV']),rtol=1e-12)
 # Window-length sensitivity for the two CUORE groups; no new events or fits.
 events=read(prior/'calibrated_event_features.csv');cal=json.loads((prior/'calibration.json').read_text(encoding='utf-8-sig'))['groups'];waves=np.load(prior/'aligned_peak_waveforms.npz');nn=np.empty((5000,2500))
 for name in ['cuoreTraining.h5','cuoreTest.h5']:
  idx=np.array([i for i,r in enumerate(events) if r['source_file']==name]);rows=np.array([int(events[i]['source_row']) for i in idx])
  with h5py.File(data/'CUORE_Data_Release_for_ML_Applications'/name,'r') as h:nn[idx]=h['Waveform'][rows,:2500]
 sensitivity=[]
 for g in ['A','B']:
  idx=np.array([i for i,r in enumerate(events) if r['baseline_group']==g]);p=cal[g];base=np.array([float(events[i]['offset_adc']) for i in idx]);gain=1+p['gain_slope_per_1000_baseline']*(base-p['baseline_reference_adc'])/1000
  noise=nn[idx]*p['energy_c1']/10000/gain[:,None];unitpulse=np.median(waves[g+'_2615'],axis=0)
  for n in [1500,2000,2500]:
   nt=noise[:,:n];nt=nt-nt.mean(axis=1,keepdims=True);t=np.arange(n)/1000;ap=np.interp(t-.3*n/1000,waves['time_s'],unitpulse,left=0,right=0);w=np.hanning(n);ap=(ap-ap.mean())*w;X=np.fft.rfft(nt*w,axis=1);freq=np.fft.rfftfreq(n,.001);amp=np.empty(len(nt))
   for fold in [0,1]:
    test=np.arange(len(nt))%2==fold;ps=np.mean(np.abs(X[~test])**2,axis=0);ps[freq>20]=ps.max()*1e30;amp[test]=estimate(X[test],ap,ps,np.array([0.]),floor=0)[0]
   sensitivity.append(dict(group=g,window_s=n/1000,sigma_keV=float(amp.std(ddof=1)),FWHM_equiv_keV=float(amp.std(ddof=1)*2.354820045),half68_keV=qstats(amp)['half68']))
 table(out/'window_length_sensitivity.csv',sensitivity)
 print('Window sensitivity',sensitivity)
 # Correct zoom: do not autoscale y using off-screen low-energy simulated points.
 comparison=read(out/'direct_shape_comparison.csv');plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(13,4.8))
 for j,key in enumerate(['rise_10_90_ms','decay_90_30_ms']):
  for g,col in zip(['A','B'],COLORS):
   rr=[r for r in comparison if r['group']==g and r['feature']==key and r['sim_noise']=='1p382mV'];axs[j].plot([float(r['energy_kev']) for r in rr],[float(r['cuore_half68_ms']) for r in rr],'o-',color=col,label=f'CUORE {g}')
  for level,col in zip(['1p382mV','7p5mV'],COLORS[2:]):
   rr=[r for r in stats if r['noise_level']==level and r['feature']==key and float(r['energy_kev'])>=400];axs[j].plot([float(r['energy_kev']) for r in rr],[float(r['half68']) for r in rr],'o--',color=col,label=level.replace('p','.'))
  axs[j].set(xscale='log',yscale='log',xlim=(450,3200),xlabel='能量 / keV',ylabel=('直接上升10–90%' if j==0 else '直接下降90–30%')+'：68%半宽 / ms');axs[j].legend(fontsize=8);axs[j].grid(alpha=.2)
 fig.suptitle('共同能区的直接时间散布：CUORE按能峰选取并去趋势；模拟每点1000例')
 save(fig,plots,'03_direct_shape_spread_comparison')
 rms=json.loads((out/'rms_summary.json').read_text(encoding='utf-8-sig'))['groups'];short=json.loads((out/'same_window_rms_summary.json').read_text(encoding='utf-8-sig'))
 rr=''.join(f"<tr><td>{g}</td><td>{rms[g]['equivalent_sim_rms_mV']['median']:.2f}</td><td>{rms[g]['equivalent_sim_rms_mV']['q16']:.2f}–{rms[g]['equivalent_sim_rms_mV']['q84']:.2f}</td><td>{rms[g]['equivalent_sim_rms_mV']['q025']:.2f}–{rms[g]['equivalent_sim_rms_mV']['q975']:.2f}</td></tr>" for g in ['A','B'])
 primary=[r for r in of if r['cutoff_hz']=='20' and r['template']==('模拟' if r['dataset'].startswith('模拟') else r['dataset'])]
 br=''.join(f"<tr><td>{r['dataset']}</td><td>{float(r['sigma_keV']):.3f}</td><td>{float(r['FWHM_equiv_2p355sigma_keV']):.3f}</td></tr>" for r in primary)
 imgs=''.join(f'<h2>{title}</h2><img src="../../plots/{plots.name}/{name}.png">' for name,title in [('01_equivalent_noise_rms','RMS分布'),('02_noise_spectra','噪声频谱'),('03_direct_shape_spread_comparison','同定义的直接时间散布'),('04_tau_spread_descriptive','拟合参数：仅作描述')])
 doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CUORE与验证集：噪声和形状散布</title><style>body{{font:17px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#243044}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 20px;border:1px solid #ccc}}.note{{padding:18px;background:#eef4fa}}</style><h1>CUORE与17点验证集：噪声及波形散布</h1><p>2026-11-17。原始波形只读；未运行DDPM或重拟合事件。</p><div class="note">总RMS：CUORE约相当于模拟器的2mV。直接波形时间的中央散布处于现有两档之间，属于同一量级。可是噪声谱、中心波形和拟合参数分布并不相同；不能据此称验证集已复现实测分布。</div>
 <h2>RMS和分布</h2><p>5000单事件各取前2500点，减本窗均值，再算RMS。使用分组参考定标在零幅度处的斜率，映射到模拟器低能增益774.348 keV/V；这里的mV是模拟等效，不是CUORE硬件实际电压。零能量处定标是外推。采用事件局部斜率时中位数为A1.96、B2.20mV，可看作该选择的敏感性。</p><table><tr><th>组</th><th>中位数/mV</th><th>16–84%</th><th>2.5–97.5%</th></tr>{rr}</table><p>模拟名义RMS为完整10秒窗口设定值；在同样2.5秒片段上，中位数分别为{short['模拟 1.382 mV']['median']:.3f}/{short['模拟 7.5 mV']['median']:.3f}mV，也存在有限窗口的RMS分布。CUORE A较窄，B右尾较长。</p>
 <h2>基线能量分辨率：明确估计器</h2><p>发布资料没有给出此样本专属的官方基线分辨率。这里使用2.5秒预触发基线，分两半交叉估计功率谱并评价，配合各组2615keV中位模板、固定时刻、Hann加窗和20Hz频带进行标准PSD加权估幅。没有扫描最大值。下表的FWHM是2.355×SD的高斯等效值，并非对非高斯分布实测半高全宽，也不是官方全窗OF/触发分辨率。</p><table><tr><th>数据</th><th>σ/keV</th><th>等效FWHM/keV</th></tr>{br}</table><p>10/20/40Hz截止结果稳定；另存1.5/2.0/2.5秒窗口敏感性，窗口长度会改变答案。校准和AP系统误差未纳入。脉冲预触发段来自已经筛选的事件，不等于独立无偏随机触发噪声样本。不可把采样点RMS直接乘2.355叫OF分辨率。</p>
 <h2>分布是否同量级</h2><p>同一定义：3ms高斯平滑后的直接10–90%上升时间、90–30%下降时间。模拟每个17点/噪声档固定抽取1000例，CUORE使用511/727/2615峰附近的既有窗口选择，扣除窄能量趋势。低能模拟缺失交叉点的事件保留失败标志；共同能区均成功。CUORE某些峰有连续本底，B727尤明显，不能将选中窗口当绝对纯净单能样本。</p><p>2615keV中央68%半宽：上升CUORE A/B为0.223/0.286ms，模拟1.382/7.5mV为0.093/0.506ms；下降CUORE为1.259/2.615ms，模拟为1.103/5.786ms。CUORE中央散布落在两档之间。分布中心不同：CUORE上升约149.4–149.7ms、模拟约75.3–75.6ms；CUORE下降约577–583ms、模拟约771–772ms。B组上升有长尾，SD结论不等于68%宽度结论。</p><p>三个有效τ的散布差距更大，但两边拟合器不同：CUORE共享RC+Bessel有效模型 vs 模拟标准三指数。因此这些τ比值不能直接当成真实形状多样性的倍率，更不能直接扣方差。</p>
 <h2>对验证集的意义</h2><p>当前两档可作为噪声强弱压力测试；约2mV可作为匹配CUORE总RMS的候选档，但仅改RMS不够。需要同时匹配噪声PSD/低频成分、中心波形和观测形状分布。新的clean内在变化幅度仍需匹配模板/噪声/估计器的前向对照，避免把噪声或拟合退化重复加入。公开数据没有500keV以下实测支持，无法验证17点中的低能区域。</p><p>来源：<a href="https://arxiv.org/html/2607.02548v1">数据发布说明</a>；<a href="https://arxiv.org/html/2510.25720v1">CUORE分析方法</a>。数值表均保存在本目录。</p>{imgs}</html>'''
 (out/'CUORE_noise_comparison_report.html').write_text(doc,encoding='utf-8')
 dump(out/'independent_verification.json',dict(passed=True,direct_rows=34000,summary_rows_recomputed=len(stats),baseline_estimator_sample_sd_recomputed=True,window_sensitivity_cases=len(sensitivity)))
 print('Report and independent verification complete.')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['output','plots','notes','data']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
