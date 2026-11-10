# 离散窗口 OF

独立 CPU 脚本，仅依赖 NumPy、h5py。事件、AP、噪声分别由 HDF5 文件和字段指定；不调用 Octopus 或 DDPM。输入只读，已有输出拒绝覆盖。

## 输入与输出

- `--input / --input-key`：二维事件数组 `(事件数, 采样点数)`。
- `--template-file / --ap-key`：一维现成 AP（默认使用方式）；或 `--ap-pulses-file / --ap-key` 指定二维脉冲集合求平均。脉冲必须预先对齐、筛选，程序不自动识别堆积或对齐。`--ap-average raw` 为等权原幅度平均；`peak-normalized` 为逐条扣基线、归一化后等权平均，适合混合能量。最终 AP 都归一化到正峰高 1，适用正极性脉冲。
- `--noise-files ... / --noise-key`：一份或多份二维噪声窗口，共同字段名，每个窗口等权。是否足以描述事件的噪声由使用者决定；程序不会将事件减模板自动当作纯噪声。
- 三者采样率、长度和电压单位须一致。必须提供 `--sample-rate`（Hz）；识别到采样率属性时核验一致，无属性时此参数是用户对采样率的声明。
- 默认三者各减去前 1400 点平均值；可用 `--baseline-samples` 调整。无额外窗函数。
- 输出 HDF5 中 `standard/`、`legacy/` 含 `amplitude`、`delay_samples`、`delay_seconds`、`valid`、`at_delay_boundary`。幅度单位同事件输入；`event_index` 对应原始行号，不自动复制来源的事件 ID。非法事件保留行号，幅度/时移为 NaN。
- 同时保存 AP、两种原始平均谱、频率、基线、样本数及 `config_json`。`complete=True` 才表示全部完成；失败可能留下 `complete=False` 的文件，不自动覆盖续跑。

## 模式和约定

- `--weighting both` 默认一次算两种：standard 为 `mean(abs(rFFT(noise))**2)` 分母，采用实信号单边频谱权重（内部频点乘 2、端点乘 1）；legacy 为 `mean(abs(rFFT(noise)))` 分母及旧实现等权频点求和。
- 保存的谱是未按 Hz 归一化的原始 FFT 平均谱，不应直接标注为 V²/Hz。
- 默认 `--dc exclude`；可显式 `include`。谱下限默认是正谱中位数乘 `1e-12`，避免除零；`--spectral-floor 0` 禁用下限但遇零谱会报错。
- 默认固定时移 0。`--delay-range 10 --delay-step 0.5` 搜索 ±10 点。正时移表示事件晚于 AP；这与旧 `of_amplitudes()` 返回的时移符号相反。时移采用周期 DFT 平移，应使脉冲远离窗口边界。
- 搜索允许带符号幅度，选择使加权残差平方最小的时移；极大负幅度也可能被选中。纯噪声/低能事件的扫描估幅可能有选择偏差，边界事件只标记，不剔除。
- 标准模式对偶数窗口的 Nyquist 端点使用实值分数平移，并按每个时移的模板范数计算；支持奇数、偶数窗口。
- legacy 与旧公式对照时显式设置 `--dc include --spectral-floor 0`，使用相同预处理、AP和时移网格。旧脚本对部分真模板未扣基线；必须先确认模板基线一致。兼容模式不声称复刻整个 Octopus 触发/预处理流程。

## 命令模板

下面的文件名和字段名是示例，占位内容须替换为实际输入：

```bash
PY=/home/wsl_0vbb/mambaforge3/envs/DDPM/bin/python
$PY scripts/window_of.py \
  --input events.h5 --input-key waveforms_noisy \
  --template-file templates.h5 --ap-key clean_template \
  --noise-files noise.h5 --noise-key noise \
  --sample-rate 1000 --weighting both \
  --output of_results.h5
```

改为 `--ap-pulses-file pulses.h5 --ap-key pulses --ap-average peak-normalized` 即从脉冲集合构建 AP。`--max-events 32` 可限量验证事件；噪声谱仍使用指定的全部噪声窗口。大噪声池首次检查宜指定一个分片。

不执行能量定标、高斯拟合或自动质量筛选，也不把非平稳残差谱视为最优性保证。建议 AP/噪声估计集与评价事件独立，并记录来源。
