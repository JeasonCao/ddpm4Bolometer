"""Explainer figures, part by part: one convolution, inside one ResBlock, downsampling and
how far one point can see, attention ablation, and a toy 'learn one filter' demo.
Production weights are only READ; runs on CPU so it never competes for the GPU."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, json
PROJ = str(_REPO_ROOT)
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule

OUT = os.path.join(PROJ, "results", "plots", "ddpm_explainer") + "/"
MD = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/'
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
INK, SOFT, GRID = '#18222B', '#5A6670', '#D7DCDA'
C_MEAS, C_DIFF, C_PRED, C_ACC = '#1F5FA8', '#D2622A', '#11845A', '#8A4FB5'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK,
                     'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei'],
                     'axes.unicode_minus': False})
def sty(a, t=None):
    a.grid(alpha=0.25, color=GRID); a.spines[['top', 'right']].set_visible(False)
    if t: a.set_title(t, fontsize=13, loc='left')
torch.set_num_threads(12)
dev = torch.device('cpu')
sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)
net = UNet1D(cond_mode='step').to(dev)
st = torch.load(MD + 'best_model.pt', map_location=dev, weights_only=False)
net.load_state_dict(st['model_state_dict'] if 'model_state_dict' in st else st); net.eval()
R = {}

def event(e, i=0):
    with h5py.File(EVD + f'eval_efficiency_{e}keV.h5') as f:
        noisy = f['waveforms_noisy'][i].astype(np.float64); tmpl = f['clean_template'][:].astype(np.float64)
        den = f['waveforms_denoised_zeroinit'][i].astype(np.float64)
    return noisy, tmpl, den
tt = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
def make_inputs(e, t, seed=3):
    noisy, tmpl, _ = event(e); s = np.abs(noisy).max()
    x0, xo = tmpl / s, noisy / s
    eps = np.random.default_rng(seed).standard_normal(x0.size)
    xt = sch.sqrt_alpha_bar[t-1].item() * x0 + sch.sqrt_one_minus_alpha_bar[t-1].item() * eps
    return x0, xo, xt, eps
tsec = np.arange(10000) / 1000.0

# ================= m1: what one convolution does, sample by sample =================
noisy, tmpl, _ = event(50); s = np.abs(noisy).max(); xo = noisy / s
on = int(np.argmax(tmpl > 0.1 * tmpl.max()))           # onset index
i0 = on - 14; seg = xo[i0:i0 + 30]; n = np.arange(30)
filt = {'平均 [1/3, 1/3, 1/3]': np.array([1, 1, 1]) / 3, '斜率 [−½, 0, ½]': np.array([-.5, 0, .5])}
fig, ax = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={'hspace': 0.42})
k = 17
ax[0].plot(n, seg, 'o-', color=C_MEAS, ms=5, lw=0.8)
ax[0].axvspan(k - 1.5, k + 1.5, color=C_ACC, alpha=0.13)
for m in (-1, 0, 1):
    ax[0].annotate(f'{seg[k+m]:.3f}', (k + m, seg[k + m]), textcoords='offset points',
                   xytext=(0, 9 if m != 0 else -16), ha='center', fontsize=10, color=C_ACC)
sty(ax[0], '输入：50 keV 测量波形在脉冲起点附近的 30 个采样点（1 点 = 1 ms）')
for a, (lab, w) in zip(ax[1:], filt.items()):
    y = np.convolve(seg, w[::-1], mode='same'); y[0] = y[-1] = np.nan
    a.plot(n, y, 'o-', color=C_ACC, ms=5, lw=0.8)
    v = w @ seg[k-1:k+2]
    a.plot([k], [v], 'o', ms=11, mfc='none', mec=INK, mew=1.6)
    a.text(0.02, 0.93, f'圈出的点 = {w[0]:+.2f}×{seg[k-1]:.3f} {w[1]:+.2f}×{seg[k]:.3f} {w[2]:+.2f}×{seg[k+1]:.3f} = {v:.3f}',
           transform=a.transAxes, va='top', fontsize=10.5, color=INK)
    lo_, hi_ = np.nanmin(y), np.nanmax(y); a.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.45 * (hi_ - lo_))
    sty(a, f'输出：用滤波器 {lab} 滑过去')
ax[2].set_xlabel('采样点序号')
fig.savefig(OUT + 'm1_conv_slide.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ================= m2: inside one ResBlock, step by step =================
T_DEMO = 30
x0, xo, xt, eps = make_inputs(15, T_DEMO)
X_t, X_o = tt(xt), tt(xo); TT = torch.full((1,), T_DEMO, dtype=torch.long)
blocks = [('enc0.a', net.enc_blocks[0][0]), ('enc0.b', net.enc_blocks[0][1]), ('enc1.a', net.enc_blocks[1][0]),
          ('enc1.b', net.enc_blocks[1][1]), ('enc2.a', net.enc_blocks[2][0]), ('enc2.b', net.enc_blocks[2][1]),
          ('enc3.a', net.enc_blocks[3][0]), ('enc3.b', net.enc_blocks[3][1]), ('mid.1', net.mid_block1),
          ('mid.2', net.mid_block2)] + [(f'dec{3-j}.{ab}', net.dec_blocks[j][q]) for j in range(4)
                                         for q, ab in ((0, 'a'), (1, 'b'))]
ins = {}
hs = [b.register_forward_pre_hook(lambda m, a, nm=nm: ins.__setitem__(nm, (a[0].detach(), a[1].detach())))
      for nm, b in blocks]
with torch.no_grad(): net(X_t, X_o, TT)
for h in hs: h.remove()
ratio = {}
with torch.no_grad():
    for nm, b in blocks:
        x, te = ins[nm]
        h = b.act(b.norm1(b.conv1(x))) + b.t_proj(te).unsqueeze(-1)
        corr = b.act(b.norm2(b.conv2(h))); sk = b.skip(x)
        ratio[nm] = float(corr.std() / sk.std())
    b = net.enc_blocks[0][1]; x, te = ins['enc0.b']
    c1 = b.conv1(x); n1 = b.norm1(c1); a1 = b.act(n1); tof = b.t_proj(te).unsqueeze(-1); h1 = a1 + tof
    corr = b.act(b.norm2(b.conv2(h1))); outp = corr + x
    # choose a channel whose correction visibly changes the input around the pulse
    score = (corr[0, :, 1300:3000].std(-1) / (x[0, :, 1300:3000].std(-1) + 1e-9))
    ch = int(torch.argsort(score, descending=True)[3])
R['resblock_ratio'] = ratio
sl = slice(1000, 3000); tz = tsec[sl]
g = lambda T: T[0, ch].numpy()[sl]
steps = [(g(x), '① 进入这个块的第 %d 号通道（上一块的输出）' % ch, INK),
         (g(c1), '② 卷积 1：64 条输入各配 3 个抽头、相加（这一步混合了所有通道）', C_MEAS),
         (g(n1), '③ 归一化：把这一组通道整体拉回均值 0、标准差 1', C_MEAS),
         (g(a1), '④ 开关 SiLU：负的部分被压扁，正的部分保留', C_ACC),
         (g(h1), '⑤ 加上步数偏移 %+.2f（只跟 t 有关，整条线一起平移）' % float(tof[0, ch, 0]), C_ACC),
         (g(corr), '⑥ 卷积 2 → 归一化 → 开关：得到"修正量"', C_DIFF),
         (g(outp), '⑦ 输出 = ① + ⑥：原样保留输入，只叠加修正', C_PRED)]
fig, ax = plt.subplots(len(steps), 1, figsize=(12, 13.5), sharex=True, gridspec_kw={'hspace': 0.55})
for a, (y, lab, c) in zip(ax, steps):
    if lab.startswith('⑦'): a.plot(tz, g(x), lw=0.8, color=INK, alpha=0.35)
    a.plot(tz, y, lw=0.8, color=c); a.axhline(0, color=SOFT, lw=0.5)
    sty(a, lab)
ax[-1].set_xlabel('时间 (s)')
fig.savefig(OUT + 'm2_resblock_trace.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ================= m3: how far one point can see, level by level =================
X_o_g = X_o.clone().requires_grad_(True)
probes = [('第一层卷积后', net.conv_in, 1), ('第 0 层（10000 点）', net.enc_attns[0], 1),
          ('第 1 层（5000 点）', net.enc_attns[1], 2), ('第 2 层（2500 点）', net.enc_attns[2], 4),
          ('第 3 层（1250 点）', net.enc_attns[3], 8), ('瓶颈，注意力之前', net.mid_block1, 8),
          ('瓶颈，注意力之后', net.mid_attn, 8)]
rf = []
for lab, mod, stride in probes:
    store = {}
    h = mod.register_forward_hook(lambda m, i, o: store.__setitem__('o', o))
    if X_o_g.grad is not None: X_o_g.grad = None
    net(X_t, X_o_g, TT); h.remove()
    o = store['o']; pos = 5000 // stride
    o[0, :, pos].pow(2).sum().backward()
    gabs = X_o_g.grad[0, 0].abs().numpy().astype(np.float64)
    # GroupNorm gives every point a faint whole-window view (only through the mean and
    # spread), which dominates a cumulative-mass measure; count where the dependence is
    # at least 1% of its peak instead.
    big = np.nonzero(gabs >= 0.01 * gabs.max())[0]; lo, hi = int(big[0]), int(big[-1]) + 1
    rf.append((lab, lo, hi, gabs / gabs.max()))
R['receptive_1pct_of_peak_ms'] = {lab: hi - lo for lab, lo, hi, _ in rf}
fig, ax = plt.subplots(2, 1, figsize=(12, 8.4), gridspec_kw={'height_ratios': [1, 2.1], 'hspace': 0.35})
dec = [(1, '每点 1 ms'), (2, '每点 2 ms'), (4, '每点 4 ms'), (8, '每点 8 ms')]
w0 = slice(on - 60, on + 140)
for j, (d, lab) in enumerate(dec):
    seg = xo[w0]; m = seg[: len(seg) // d * d].reshape(-1, d).mean(1); tx = (np.arange(m.size) * d + d / 2) / 1.0
    ax[0].plot(tx, m - 2.2 * j, 'o-', ms=2.8 if d == 1 else 4, lw=0.7, color=plt.cm.viridis(0.1 + 0.25 * j))
    mid = 100 // d
    ax[0].axvspan((mid - 1) * d, (mid + 2) * d, ymin=0, ymax=1, color=plt.cm.viridis(0.1 + 0.25 * j), alpha=0.0)
    ax[0].plot([(mid - 1) * d, (mid + 2) * d], [-2.2 * j - 0.3] * 2, lw=5,
               color=plt.cm.viridis(0.1 + 0.25 * j), solid_capstyle='butt')
    ax[0].text(2, -2.2 * j + 0.75, f'{lab}：3 个抽头（色块）覆盖 {3*d} ms', fontsize=10.5, va='center')
ax[0].set_yticks([]); ax[0].set_xlabel('时间 (ms，起点附近 200 ms)')
sty(ax[0], '示意：每下采样一次，点数减半，同样 3 个抽头覆盖的时间翻倍')
# Bars are the conv-chain reach computed from kernel sizes and strides (3-tap convs,
# 4-tap stride-2 downsampling); the gradient measurement above agrees wherever GroupNorm's
# faint whole-window coupling does not swamp it (3/7/25 ms at the first levels, 172 vs 176 ms
# at the bottleneck).
reach = [('第一层卷积后', 3), ('第 0 层（10000 点）', 11), ('第 1 层（5000 点）', 30),
         ('第 2 层（2500 点）', 68), ('第 3 层（1250 点）', 144), ('瓶颈，注意力之前', 176),
         ('瓶颈，注意力之后', 10000)]
R['conv_reach_ms'] = dict(reach)
for j, (lab, wms) in enumerate(reach):
    col = C_DIFF if '之后' in lab else plt.cm.viridis(0.08 + 0.13 * j)
    lo = max(0, 5000 - wms // 2); hi = min(10000, 5000 + wms // 2 + 1)
    ax[1].barh(j, max((hi - lo) / 1000, 0.012), left=lo / 1000, height=0.55, color=col, alpha=0.9)
    txt = f'{wms} ms' if wms < 1000 else '整个 10 s 窗口'
    ax[1].text(hi / 1000 + 0.12 if wms < 1000 else 7.6, j, txt, va='center', fontsize=11,
               color=INK if wms < 1000 else 'white')
ax[1].set_yticks(range(len(reach))); ax[1].set_yticklabels([r[0] for r in reach]); ax[1].invert_yaxis()
ax2 = ax[1].twinx(); ax2.plot(tsec, xo, lw=0.4, color=C_MEAS, alpha=0.3); ax2.set_yticks([])
ax[1].set_xlim(0, 10); ax[1].set_xlabel('测量波形上的时间 (s)')
sty(ax[1], '5 s 处的一个点，在各层能直接看见输入的多宽一段（只靠卷积；最后一行加了注意力）')
fig.savefig(OUT + 'm3_downsample_reach.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ================= m4: what attention adds — switch it off =================
def eps_err(e, t, attn_on):
    x0, xo, xt, eps = make_inputs(e, t)
    if not attn_on:
        h = net.mid_attn.register_forward_hook(lambda m, i, o: i[0])
    with torch.no_grad(): eh = net(tt(xt), tt(xo), torch.full((1,), t, dtype=torch.long))[0, 0].numpy()
    if not attn_on: h.remove()
    sa, sb = sch.sqrt_alpha_bar[t-1].item(), sch.sqrt_one_minus_alpha_bar[t-1].item()
    x0h = (xt - sb * eh) / sa
    return float(np.sqrt(np.mean((x0h - x0) ** 2))), x0h, x0
abl = {}
for e in (15, 511, 2615):
    for t in (10, 30, 50):
        on_, _, _ = eps_err(e, t, True); off_, _, _ = eps_err(e, t, False)
        abl[f'{e}keV_t{t}'] = (on_, off_)
R['attention_ablation_x0rms_on_off'] = abl
_, x0h_on, x0r = eps_err(511, 30, True); _, x0h_off, _ = eps_err(511, 30, False)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={'width_ratios': [1.5, 1]})
_, x0h_on, x0r = eps_err(15, 30, True); _, x0h_off, _ = eps_err(15, 30, False)
ax[0].plot(tsec, x0h_off - x0r, lw=0.6, color=C_DIFF, label='关掉注意力')
ax[0].plot(tsec, x0h_on - x0r, lw=0.6, color=C_PRED, label='正常')
ax[0].axhline(0, color=INK, lw=0.6); ax[0].legend(fontsize=10, frameon=False, loc='upper right')
sty(ax[0], '15 keV, t = 30：网络一次猜出的干净脉冲 − 真值')
ax[0].set_xlabel('时间 (s)')
lab = [f'{e} keV' for e in (15, 511, 2615)]
for j, t in enumerate((10, 30, 50)):
    on_ = [abl[f'{e}keV_t{t}'][0] for e in (15, 511, 2615)]; off_ = [abl[f'{e}keV_t{t}'][1] for e in (15, 511, 2615)]
    ax[1].plot(range(3), np.array(off_) / np.array(on_), 'o-', label=f't = {t}', color=plt.cm.viridis(0.15 + 0.35 * j))
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(lab); ax[1].axhline(1, color=SOFT, lw=0.8, ls=':'); ax[1].set_ylim(0, 9)
ax[1].set_ylabel('关掉后误差 ÷ 正常误差'); ax[1].legend(fontsize=10, frameon=False)
sty(ax[1], '关掉注意力，猜测误差放大多少倍')
fig.savefig(OUT + 'm4_attention_off.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# ================= m5: learning, shrunk to one 41-tap filter =================
rng = np.random.default_rng(0)
def pairs(e, idx):
    with h5py.File(EVD + f'eval_efficiency_{e}keV.h5') as f:
        nz = f['waveforms_noisy'][idx].astype(np.float64); tm = f['clean_template'][:].astype(np.float64)
    return nz, np.broadcast_to(tm, nz.shape)
X, Y = [], []
for e in (15, 20, 25, 30, 40):
    a, b = pairs(e, list(range(0, 200))); X.append(a); Y.append(b)
X = np.concatenate(X); Y = np.concatenate(Y); sc = 0.001382 * 3
X /= sc; Y /= sc
Xt = torch.tensor(X, dtype=torch.float32)[:, None]; Yt = torch.tensor(Y, dtype=torch.float32)[:, None]
K = 41
w = torch.zeros(1, 1, K, requires_grad=True)
opt = torch.optim.Adam([w], lr=3e-3)
snaps, losses, want = {}, [], (0, 10, 50, 300, 3000)
xe, ye, de = event(30, 4000); xe, ye, de = xe / sc, ye / sc, de / sc
for step in range(3001):
    if step in want: snaps[step] = w.detach()[0, 0].numpy().copy()
    bi = rng.integers(0, len(Xt), 32)
    pred = torch.nn.functional.conv1d(Xt[bi], w, padding=K // 2)
    loss = (pred - Yt[bi]).pow(2).mean()
    losses.append(loss.item())
    opt.zero_grad(); loss.backward(); opt.step()
def apply(wk, x): return np.convolve(x, wk[::-1], mode='same')
R['toy'] = {'loss_first': losses[0], 'loss_last': float(np.mean(losses[-50:])),
            'meas_err': float(np.sqrt(np.mean((xe - ye) ** 2))),
            'filter_err': float(np.sqrt(np.mean((apply(snaps[3000], xe) - ye) ** 2))),
            'ddpm_err': float(np.sqrt(np.mean((de - ye) ** 2)))}
fig = plt.figure(figsize=(13, 10.5)); gs = fig.add_gridspec(3, len(want), height_ratios=[1, 1.15, 1.25], hspace=0.6, wspace=0.3)
for j, s_ in enumerate(want):
    a = fig.add_subplot(gs[0, j]); a.bar(np.arange(K) - K // 2, snaps[s_], color=C_ACC, width=0.8)
    a.axhline(0, color=SOFT, lw=0.5); sty(a, f'第 {s_} 步'); a.set_xticks([-20, 0, 20])
    if j == 0: a.set_ylabel('41 个抽头')
    b = fig.add_subplot(gs[1, j]); zz = slice(1300, 2600)
    b.plot(tsec[zz], xe[zz], lw=0.4, color=C_MEAS, alpha=0.4); b.plot(tsec[zz], apply(snaps[s_], xe)[zz], lw=0.9, color=C_ACC)
    b.plot(tsec[zz], ye[zz], lw=1.0, color=INK, ls='--'); sty(b); b.set_xticks([1.5, 2.0, 2.5])
    if j == 0: b.set_ylabel('30 keV 事件\n（未参与训练）')
a = fig.add_subplot(gs[2, :3]); a.plot(losses, lw=0.5, color=C_ACC); a.set_yscale('log')
for s_ in want: a.axvline(s_, color=SOFT, lw=0.6, ls=':')
a.set_xlabel('训练步数'); a.set_ylabel('误差（损失）'); sty(a, '每一步：拿 32 道题对答案，把 41 个数各往"误差变小"的方向拧一点')
a = fig.add_subplot(gs[2, 3:]); zz = slice(1450, 1750)
a.plot(tsec[zz], xe[zz], lw=0.5, color=C_MEAS, alpha=0.45, label='测量')
a.plot(tsec[zz], apply(snaps[3000], xe)[zz], lw=1.2, color=C_ACC, label='学到的一个滤波器')
a.plot(tsec[zz], de[zz], lw=1.2, color=C_PRED, label='我们的 DDPM')
a.plot(tsec[zz], ye[zz], lw=1.0, color=INK, ls='--', label='真值')
a.legend(fontsize=9.5, frameon=False, loc='lower right'); sty(a, '放大到上升沿'); a.set_xlabel('时间 (s)')
fig.savefig(OUT + 'm5_toy_learning.png', dpi=130, bbox_inches='tight'); plt.close(fig)

json.dump(R, open(OUT + 'explainer_parts.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False)); print('ALL_DONE')
