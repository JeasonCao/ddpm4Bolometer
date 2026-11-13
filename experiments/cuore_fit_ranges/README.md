# CUORE 三指数拟合范围研究（独立实验）

## 问题与结论边界

用户授权：另写拟合程序，比较参数范围，再重新拟合公开 CUORE 全数据集。
此目录属于探索性实验，不更改项目 src/basics/fit.py，也不替换已有模拟数据、
DDPM 或能量分辨率分析的默认拟合方法。源 HDF5 只读，旧结果保留。

初步和正式小样本证据都表明：简单放宽 alpha 上界会使原参数 A 增大、
tau_r 与 tau_d1 更接近，波形残差几乎不变。不能将“不再碰0.99边界”
当作恢复了三个可独立测量的物理时间常数。全量结果的数值成功与参数
可辨识性分开报告。

## 输入和单位

Zenodo 20721645：cuoreTraining.h5 9000窗口，cuoreTest.h5 1000窗口。
共5000单事件、5000pileup，采样率1kHz，每窗10,000点。
训练标签来自 Labels，测试类别来自显式 rowOrder/cleanPulseCount/
pileupPulseCount 属性，仅用于描述分析，不用于声称盲测分类性能。
采样率依据 https://arxiv.org/html/2607.02548v1。

没有逐事件能量字段。横轴使用 normalizationScale，即最大值减前2500点
基线，单位ADC。拟合输入 (Waveform-offset)/scale，不裁剪；不是ADC转伏特。
最终幅值恢复为ADC，时间常数与t0单位秒。拟合不复制波形。

## 程序

- fitter.py：原始 A/alpha 参数坐标中的独立解析Jacobian拟合器；比较旧形状
  范围、只放宽alpha、中等/宽范围及负前因子探索分支。legacy名字只表示
  旧形状边界参考，不是旧算法逐行复制（初值、onset窗口、优化器及基线范围不同）。
- probe.py：跨幅值初步范围/初值探查，保存所有初值结果。
- stable_fitter.py：稳定参数化、解析Jacobian、峰检测、单/多脉冲联合拟合。
- run_study.py：正式pilot、全量执行、检查点、来源哈希和无波形HDF5输出。
- check_initialization.py：对照直接全点多初值与加速初始化后的全点求解。
- check_and_profile.py：导数校验、已知参数波形恢复、固定两常数间距的残差剖面。
- report.py：输入不变及行身份核验、新旧残差、范围比较、新散点和拟合示例图。

所有入口用 --data/--output 等参数指定路径。无自动commit/push。

## 稳定参数化的准确含义

原模型：B+A[-exp(-dt/tr)+alpha exp(-dt/tf)+(1-alpha)exp(-dt/ts)]。

新计算形式：
  B + C [exp(-dt/tf)-exp(-dt/tr)]/delta
    + D [exp(-dt/ts)-exp(-dt/tr)]
  tf = tr exp(delta), ts = tf + gap, dt >= 0
  C,D > 0, delta >= 0, gap > 0。

delta>0时与原式完全等价：
  A=C/delta+D，alpha=C/(C+D*delta)。
同时明确采用 tr<=tf<ts 的有序、正系数模型子集。它不是原程序所有无序
参数分支的全集。delta=0时使用精确重根极限：
  C (dt/tr) exp(-dt/tr) + D [exp(-dt/ts)-exp(-dt/tr)]。
该极限只有两个不同的指数时间尺度，不能再把重合的tr/tf说成独立测得。

这种参数化避免用很大的A配合alpha接近1来抵消两个指数项。
delta<0.05（tf/tr<1.0513）标为“rise/fast未分开”；这是诊断标记，
不删数据，也不是统计置信区间。

## 三套正式搜索范围

| 设置 | tr (s) | tf/tr | ts-tf (s) |
|---|---|---|---|
| compact | 0.01–0.25 | 1–20 | 0.01–10 |
| moderate | 0.002–0.6 | 1–100 | 0.002–15 |
| wide | 0.001–2 | 1–1000 | 0.001–30 |

注意：后两列是比值和差值，不是tf、ts的独立上下限。允许超过10s窗口
的搜索候选只是为了暴露长尾不确定性，绝不证明长时间常数可测。
所有设置共同：基线±0.1归一化幅值，C在1e-6–100，D在1e-6–10。
开始时刻限制在观测峰前0.9s至峰前1ms，并按脉冲时间顺序限定窗口。

选择moderate用于全量数值探索：单事件三套结果几乎一致；pilot训练
pileup中moderate减少了一些额外边界命中，继续扩大至wide未带来实质
残差收益。这不是经过物理验证的推荐先验，也没有解决三常数退化问题。

## 正式pilot和实现核验

144个按幅值分层的窗口：训练64单+32pileup，测试32单+16pileup。
每窗每范围3初值，全部原始采样点求解，共432组范围结果、1296次初值求解。
测试文件样本用于独立描述性交叉核对，不作盲测性能声明。

三套范围下96个单事件都趋向重根；单事件残差中位数训练约0.8852%、
测试约0.8841%，不随范围实质改变。未据此宣称波形模型已精确适配实测数据。

解析Jacobian与数值差分最大相对偏差约1.6e-8。
两组非退化已知参数波形的时间常数恢复相对误差<=1.3e-12；
与原函数的数值等价误差<4e-16。这是实现校验，不是实测物理验证。
24窗加速初始化核验中，最终RMS相对增加最大0.03281%，多数近机器精度。
三条固定delta残差剖面只描述目标函数形状，不提供统计置信区间。

## 全量策略及与旧结果的区别

3个初值在每5个点取1点的网格上寻找候选，选2个候选，再用全部10,000
原始采样点求解。最终结果从成功求解中取残差最小者；全部失败仍保存
最佳迭代及失败状态。源数据不降采样保存，最终拟合和残差都使用全点。

pileup使用高斯平滑后的显著峰检测，最多容纳3个可见脉冲；无法检出第二峰
会保留缺峰标记和初始化候选，不能视作可靠分解。模型同时拟合所有脉冲，
各分量形状独立，不使用旧顺序减法或P2上升常数±20%约束。超过3脉冲、
重叠不可分辨、窗口截断和基线漂移仍可能使模型不足。

所以pileup的改善包含联合拟合和可见第三脉冲，不能归结为仅放宽参数范围。

探索性diagnostic_pass还要求：RMS/幅值<=2%；没有额外参数边界命中；
各分量delta>=0.05；不缺峰；列归一化Jacobian条件数<1e6；
近最优初值之间tau相对跨度<10%。这一标记并非官方标准或可靠性证明。
对于多于3个峰的事件，即使某些检查通过，也不能声称已完整分解。

## 输出与对应关系

results/fit_results/cuore_range_study_20261112/pilot/
results/fit_results/cuore_range_study_20261112/full_moderate/
results/plots/cuore_range_study_20261112/

新HDF5逐行对应源文件（row从0开始），保留event_id、label、offset、
整窗幅值、全部分量参数、tau_s[窗口,分量,(rise,fast,slow)]、峰幅值、
退化/边界标记、残差、条件数、耗时及失败消息。没有波形。
19列p为baseline加最多3组(logC,logD,t0,logtr,delta,loggap)，不足部分NaN。
所有结果包含代码/输入SHA-256，pilot/full各保存执行代码快照。
图中pileup取最大拟合分量，一窗一点，横轴仍为整窗幅值。


## 全量结果（2026-11-12）

- 全量计算677.2s（约11.3分钟，8 CPU workers），10000窗口均保留。
  单事件5000/5000数值成功；pileup4990/5000数值成功，10个达到迭代上限。
- 4980/5000单事件仍为rise/fast未分开。20个初筛通过事件与旧版完全相同。
  单事件额外范围边界命中从旧版4980降为0，但退化并未消失，不能把这个
  边界通过率变化当成物理可靠性改善。
- 单事件三tau中位数：旧[0.191766,0.200202,1.059526]s；
  新[0.196011,0.196033,1.060248]s。前两项合并而慢项变化很小。
- 单事件残差RMS/幅值中位数0.878064% -> 0.877867%，基本不变。
- 新旧共同数值成功的pileup为4965窗，残差中位数1.121478% -> 0.933880%；
  逐窗相对残差改善中位数11.22%。仍有129个共同事件残差相对增加>1%，
  所以不能宣称全面改善。
- 新版成功pileup中3734窗用2脉冲、1256窗用3脉冲；4978窗至少一个分量
  rise/fast未分开；1332窗还有额外边界问题，没有通过全部初筛的pileup。
  部分长常数只被推向了新范围的上界，不能解读为测得几十秒物理常数。
- 所有10000行的源row/event_id/offset/amplitude精确匹配，输入SHA-256未变。
  图已目视核验；没有截掉异常成功点。全量/单事件散点、范围比较、残差比较、
  分离参数剖面均有PNG/PDF，另有拟合示例图。

结论：本次完成了独立程序、范围测试和全量重拟合，但未找到一套仅靠
调整范围就能可靠提取三个独立时间常数的设置。当前搜索范围只作数值
探索，不升级为正式物理分析默认。后续需另行研究模型是否适合这些实测
波形以及可辨识性；本轮没有改成其他物理模型或修改主线拟合。

## 仅4980单事件的散点和时域检查（2026-11-12）

用户要求不再画20个分离事件，只显示4980个rise/fast未分开的单事件。
plot_single_4980.py仅读取已完成拟合，不重拟合；新图独立保存在
results/plots/cuore_range_study_20261112/single_4980/，旧图保留。

4980个事件整窗RMS/幅值的P10/P50/P90/最大为0.7796%/0.8776%/0.9510%/1.5716%。
按残差排序抽取P10、P50、P90、最大四例，Training源row分别385、3629、2864、3519，
event ID分别614、5693、4482、5522。每例显示全窗口、上升/峰区放大和全窗口残差；
来源/参数/选样规则记录在selection_and_quality.json，可准确回到源波形。

四例峰区RMS/幅值约2.39%、2.63%、2.62%、4.84%；最大局部绝对偏差约12.61%、
13.49%、12.75%、20.02%。原始峰值时刻的拟合幅值分别为0.9706、0.9670、0.9665、
0.9483（按原窗口幅值归一）。整体轮廓和尾部接近，但上升沿及峰顶存在系统偏差。
整窗RMS包含大量基线，会稀释脉冲关键部位误差；参数退化与波形局部失配需分开解释。

## 2026-11-13 interpretation correction after checking the original reference

Carrettoni & Vignati (2011), Eq. 4.1 models temperature. Their voltage fits use
Eqs. 4.15–4.17, including thermistor, bias circuit and Bessel response.
Source: https://www.roma1.infn.it/people/vignati/curriculum/JINST_6_2011_P08007.pdf
This experiment fits ADC waveforms directly with the simplified exponential
shape. Its mismatch/degeneracy does not establish failure of the paper's full
model. Prioritize checking that signal chain before attributing failure to an
insufficient number of exponentials. No new experiment was run for this check.
