"""
Generate a self-contained, animated HTML explainer of the DT trigger
algorithm, built on real numeric data from one event (run through the
actual src/basics/trigger.dt_trigger_blind code). Single panel, one clean
high-SNR example, time-based (not frame-based) stepping so it can be slowed
down to whatever pace is actually watchable.

Usage
-----
python scripts/test/generate_dt_trigger_animation.py \
    --eval_file /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p2mV/eval_efficiency_50keV.h5 \
    --event_idx 0 --kind denoised \
    --output results/plots/trigger/dt_trigger_animated_demo.html
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import json
import h5py
import numpy as np

from src.basics.trigger import dt_trigger_blind, rolling_baseline_threshold

WINDOW_MS  = 300.0
DEAD_MS    = 300.0
NSIGMA     = 4.0
N_ABOVE    = 5
MATCH_PRE_MS  = 100.0
MATCH_POST_MS = 200.0
DISP_LO_MS, DISP_HI_MS = -80.0, 150.0


def extract_series(x, fs, onset_idx, window_n, dead_n):
    dv = np.diff(x)
    mu, rms, threshold = rolling_baseline_threshold(dv, window_n, nsigma=NSIGMA)
    idxs, _, _, _ = dt_trigger_blind(x, fs, window_n, n_above=N_ABOVE, nsigma=NSIGMA,
                                     dead_samples=dead_n)

    disp_lo = onset_idx + int(round(DISP_LO_MS * 1e-3 * fs))
    disp_hi = onset_idx + int(round(DISP_HI_MS * 1e-3 * fs))

    t_ms = ((np.arange(disp_lo, disp_hi) - onset_idx) / fs * 1e3).tolist()
    wave_mV = (x[disp_lo:disp_hi] * 1e3).tolist()

    # dv/threshold for original index j use dv[j-1] / threshold[j-window_n-1]
    dv_orig_idx = np.arange(disp_lo, disp_hi - 1)
    dv_mV  = (dv[dv_orig_idx - 1] * 1e3).tolist()
    thr_mV = (threshold[dv_orig_idx - window_n - 1] * 1e3).tolist()
    dv_t_ms = (((dv_orig_idx) - onset_idx) / fs * 1e3).tolist()

    lo_match = onset_idx - int(round(MATCH_PRE_MS * 1e-3 * fs))
    hi_match = onset_idx + int(round(MATCH_POST_MS * 1e-3 * fs))
    matched_idx = [int(i) for i in idxs if lo_match <= i <= hi_match]
    all_trigger_ms = ((idxs - onset_idx) / fs * 1e3).tolist()
    trigger_ms = (matched_idx[0] - onset_idx) / fs * 1e3 if matched_idx else None

    return dict(t_ms=t_ms, wave_mV=wave_mV, dv_t_ms=dv_t_ms, dv_mV=dv_mV, thr_mV=thr_mV,
               trigger_ms=trigger_ms, all_trigger_ms=all_trigger_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--event_idx', type=int, default=0)
    parser.add_argument('--kind', choices=['noisy', 'denoised'], default='denoised')
    parser.add_argument('--output', default=os.path.join(
        _RESULTS_DIR, 'plots', 'trigger', 'dt_trigger_animated_demo.html'))
    args = parser.parse_args()

    with h5py.File(args.eval_file, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        fs         = float(f.attrs['f_sample'])
        onset_s    = float(f.attrs['onset_s'])
        x = f[f'waveforms_{args.kind}'][args.event_idx]

    onset_idx = int(round(onset_s * fs))
    window_n  = int(round(WINDOW_MS * 1e-3 * fs))
    dead_n    = int(round(DEAD_MS * 1e-3 * fs))

    series = extract_series(x, fs, onset_idx, window_n, dead_n)
    data = {
        'energy_kev': energy_kev, 'kind': args.kind,
        'window_ms': WINDOW_MS, 'dead_ms': DEAD_MS, 'nsigma': NSIGMA, 'n_above': N_ABOVE,
        'match_pre_ms': MATCH_PRE_MS, 'match_post_ms': MATCH_POST_MS,
        'series': series,
    }

    out_html = _build_html(data, args.event_idx)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(out_html)
    print(f'Saved → {args.output}')
    print(f"  triggers found: {series['all_trigger_ms']}")


def _build_html(data, event_idx):
    data_json = json.dumps(data)
    kind_label = '降噪后 (denoised)' if data['kind'] == 'denoised' else '原始 (noisy)'

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>DT Trigger — 动态演示</title>
<style>
{_css()}
</style>
</head>
<body>
<h1>DT（导数阈值）触发器是怎么工作的</h1>
<p class="subtitle">真实事例驱动的动画演示 &mdash; {data['energy_kev']:.0f} keV，{kind_label}，event #{event_idx}，
跑的是真实的 <code>dt_trigger_blind()</code> 代码算出来的数值。</p>

<div class="legend-bar">
  <span class="chip wave">波形</span>
  <span class="chip dv">逐点差分 dV</span>
  <span class="chip thr">滚动阈值（向前看 {data['window_ms']:.0f} ms）</span>
  <span class="chip above">超过阈值的点</span>
  <span class="chip fire">触发点</span>
</div>

<div class="controls">
  <button id="playBtn">&#9654; 播放</button>
  <button id="resetBtn">&#8634; 重置</button>
  <label>每点间隔
    <input type="range" id="speed" min="20" max="400" value="120" step="10">
    <span id="speedLabel">120 ms/点</span>
  </label>
  <span class="counter-label">连续超阈值计数（需要 {data['n_above']}）：</span>
  <span id="counterDisplay" class="counter-display">0 / {data['n_above']}</span>
</div>

<div class="panel">
  <canvas id="canvasMain" width="900" height="430"></canvas>
  <div id="statusMain" class="status not-fired">等待...</div>
</div>

<div class="explain">
  <h2>看动画的时候关注这几点</h2>
  <ol>
    <li><b>上半区是波形本身</b>，下半区是它的逐点差分 dV（斜率）。红色播放头每隔一小段时间走一个采样点。</li>
    <li><b>蓝色阴影区</b>是当前播放头往前数 {data['window_ms']:.0f} ms 的"回看窗口"——黑色虚线阈值就是从这一段算出来的（均值 + {data['nsigma']:.0f}&times;标准差），算法从来看不到播放头右边、未来的信息。</li>
    <li>dV 的点冒到阈值线上面就变橙色，右上角计数器 +1；中间断了一个点（掉回阈值以下）计数器立刻清零重来。</li>
    <li>计数器攒到 <b>{data['n_above']} 个连续点</b>，触发立即开火——画面闪一下绿色竖线，状态变成"TRIGGERED"。</li>
    <li>灰色虚线标的是"真实 onset"，淡绿色条带是事后打分用的匹配窗口——这两个东西只是给你看的参考，整套判决逻辑从头到尾都没有用过它们。</li>
  </ol>
</div>

<script>
const DATA = {data_json};
{_js()}
</script>
</body></html>
"""


def _css():
    return """
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
       max-width: 980px; margin: 30px auto; padding: 0 18px; color: #1a1a1a; }
h1 { font-size: 1.7em; margin-bottom: 2px; }
.subtitle { color: #555; margin-top: 0; }
.legend-bar { display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0; font-size: 0.85em; }
.chip { padding: 3px 10px; border-radius: 12px; border: 1px solid #ccc; }
.chip.wave  { border-color: #2980b9; color: #2980b9; }
.chip.dv    { border-color: #8e44ad; color: #8e44ad; }
.chip.thr   { border-color: #444; color: #444; }
.chip.above { border-color: #e67e22; color: #e67e22; background: #fff3e6; }
.chip.fire  { border-color: #27ae60; color: #27ae60; background: #eafaf1; }
.controls { display: flex; align-items: center; gap: 16px; margin: 16px 0; padding: 10px 14px;
            background: #f5f7fa; border-radius: 8px; flex-wrap: wrap; }
.controls button { padding: 6px 16px; font-size: 0.95em; border-radius: 6px; border: 1px solid #aaa;
                   background: white; cursor: pointer; }
.controls button:hover { background: #eee; }
.counter-label { font-size: 0.88em; color: #555; }
.counter-display { font-family: monospace; font-weight: bold; font-size: 1.1em; }
#speedLabel { font-family: monospace; font-size: 0.85em; color: #555; margin-left: 4px; }
.panel { text-align: center; }
canvas { border: 1px solid #ddd; border-radius: 6px; background: #fff; max-width: 100%; }
.status { margin-top: 10px; font-weight: bold; font-size: 1.15em; padding: 8px; border-radius: 6px; }
.status.not-fired { color: #888; background: #f2f2f2; }
.status.fired { color: #1e7e34; background: #d8f5e3; animation: flash 0.6s ease-out; }
@keyframes flash { 0% { background: #7be8a4; } 100% { background: #d8f5e3; } }
.explain { margin-top: 30px; background: #fbfbfb; border: 1px solid #e3e3e3; border-radius: 8px;
           padding: 14px 22px; }
.explain h2 { font-size: 1.1em; }
.explain ol { line-height: 1.8; }
code { font-family: 'SF Mono', Consolas, Monaco, monospace; background: #eee; padding: 1px 5px; border-radius: 3px; }
"""


def _js():
    # All geometry/animation logic; DATA is injected above this script block.
    return r"""
function buildPanel(canvasId, series, windowMs, nAbove, matchPreMs, matchPostMs) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const padL = 56, padR = 16, padT = 22, padMid = 30, padB = 30;
  const waveH = (H - padT - padMid - padB) * 0.42;
  const dvH   = (H - padT - padMid - padB) * 0.58;

  const tAll = series.t_ms;
  const tMin = tAll[0], tMax = tAll[tAll.length - 1];
  const x = t => padL + (t - tMin) / (tMax - tMin) * (W - padL - padR);

  const waveVals = series.wave_mV;
  const dvVals = series.dv_mV;
  const thrVals = series.thr_mV;
  const dvT = series.dv_t_ms;

  const waveMin = Math.min(...waveVals), waveMax = Math.max(...waveVals);
  const waveSpan = Math.max(waveMax - waveMin, 1e-6);
  const yWave = v => padT + waveH - (v - waveMin) / waveSpan * waveH;

  const dvAll = dvVals.concat(thrVals);
  const dvMin = Math.min(...dvAll), dvMax = Math.max(...dvAll);
  const dvSpan = Math.max(dvMax - dvMin, 1e-9);
  const dvTop = padT + waveH + padMid;
  const yDv = v => dvTop + dvH - (v - dvMin) / dvSpan * dvH;

  function axes() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + waveH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, dvTop); ctx.lineTo(padL, dvTop + dvH); ctx.stroke();
    ctx.fillStyle = '#888'; ctx.font = '11px sans-serif';
    ctx.fillText('幅度 (mV)', 4, padT + 10);
    ctx.fillText('dV (mV/sample)', 4, dvTop + 10);
    for (let tt = Math.ceil(tMin / 50) * 50; tt <= tMax; tt += 50) {
      const xx = x(tt);
      ctx.strokeStyle = '#eee';
      ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, dvTop + dvH); ctx.stroke();
      ctx.fillStyle = '#999'; ctx.fillText(tt + 'ms', xx - 12, dvTop + dvH + 16);
    }
    const x0 = x(0);
    ctx.strokeStyle = '#999'; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x0, padT); ctx.lineTo(x0, dvTop + dvH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#999'; ctx.fillText('true onset (参考线)', x0 + 4, padT - 6);
    const mLo = x(-matchPreMs), mHi = x(matchPostMs);
    ctx.fillStyle = 'rgba(46,204,113,0.08)';
    ctx.fillRect(mLo, padT, mHi - mLo, waveH);
  }

  function drawWaveUpTo(idx) {
    ctx.strokeStyle = '#2980b9'; ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i <= idx && i < tAll.length; i++) {
      const xx = x(tAll[i]), yy = yWave(waveVals[i]);
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  }

  function drawDvAndThr(idx) {
    ctx.strokeStyle = '#444'; ctx.setLineDash([5, 3]); ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i <= idx && i < dvT.length; i++) {
      const xx = x(dvT[i]), yy = yDv(thrVals[i]);
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = '#8e44ad'; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= idx && i < dvT.length; i++) {
      const xx = x(dvT[i]), yy = yDv(dvVals[i]);
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    ctx.stroke();

    for (let i = 0; i <= idx && i < dvT.length; i++) {
      const above = dvVals[i] > thrVals[i];
      ctx.fillStyle = above ? '#e67e22' : '#8e44ad';
      ctx.beginPath();
      ctx.arc(x(dvT[i]), yDv(dvVals[i]), above ? 3.4 : 2.0, 0, 7);
      ctx.fill();
    }
  }

  function drawPlayhead(t) {
    const xx = x(t);
    ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, dvTop + dvH); ctx.stroke();
  }

  function drawLookbackWindow(t) {
    const xx0 = x(Math.max(t - windowMs, tMin));
    const xx1 = x(t);
    ctx.fillStyle = 'rgba(52,152,219,0.14)';
    ctx.fillRect(xx0, dvTop, xx1 - xx0, dvH);
  }

  function drawTrigger(tm) {
    const xx = x(tm);
    ctx.strokeStyle = '#27ae60'; ctx.lineWidth = 2.6;
    ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, dvTop + dvH); ctx.stroke();
  }

  return { axes, drawWaveUpTo, drawDvAndThr, drawPlayhead, drawLookbackWindow, drawTrigger,
          dvT, dvVals, thrVals, n: dvT.length };
}

const panel = buildPanel('canvasMain', DATA.series, DATA.window_ms, DATA.n_above,
                         DATA.match_pre_ms, DATA.match_post_ms);

let frame = 0;
let playing = false;
let fired = false;
let timer = null;
const maxFrame = panel.n - 1;

function streakAt(idx) {
  let s = 0;
  for (let i = idx; i >= 0; i--) {
    if (panel.dvVals[i] > panel.thrVals[i]) s++; else break;
  }
  return Math.min(s, DATA.n_above);
}

function render() {
  const idx = Math.min(frame, panel.n - 1);
  const s = streakAt(idx);
  const t = panel.dvT[idx];

  panel.axes();
  panel.drawLookbackWindow(t);
  panel.drawWaveUpTo(frame);
  panel.drawDvAndThr(frame);
  panel.drawPlayhead(t);

  if (!fired && DATA.series.trigger_ms !== null && t >= DATA.series.trigger_ms) {
    fired = true;
  }
  if (fired) panel.drawTrigger(DATA.series.trigger_ms);

  document.getElementById('counterDisplay').textContent = `${s} / ${DATA.n_above}`;

  const st = document.getElementById('statusMain');
  if (fired) {
    st.textContent = `TRIGGERED @ onset +${DATA.series.trigger_ms.toFixed(0)} ms`;
    st.className = 'status fired';
  } else if (frame >= maxFrame) {
    st.textContent = 'NOT TRIGGERED (在这段窗口内)';
    st.className = 'status not-fired';
  } else {
    st.textContent = '扫描中...';
    st.className = 'status not-fired';
  }
}

function tick() {
  if (!playing) return;
  frame += 1;
  render();
  if (frame >= maxFrame) { playing = false; document.getElementById('playBtn').innerHTML = '&#9654; 播放'; return; }
  const delay = parseInt(document.getElementById('speed').value, 10);
  timer = setTimeout(tick, delay);
}

document.getElementById('speed').addEventListener('input', (e) => {
  document.getElementById('speedLabel').textContent = e.target.value + ' ms/点';
});

document.getElementById('playBtn').addEventListener('click', () => {
  playing = !playing;
  document.getElementById('playBtn').innerHTML = playing ? '&#10073;&#10073; 暂停' : '&#9654; 播放';
  if (playing) {
    if (frame >= maxFrame) frame = 0;
    tick();
  } else if (timer) {
    clearTimeout(timer);
  }
});
document.getElementById('resetBtn').addEventListener('click', () => {
  if (timer) clearTimeout(timer);
  frame = 0; playing = false; fired = false;
  document.getElementById('playBtn').innerHTML = '&#9654; 播放';
  render();
});

render();
"""


if __name__ == '__main__':
    main()
