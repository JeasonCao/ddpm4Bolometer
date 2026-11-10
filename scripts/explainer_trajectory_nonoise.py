"""Record every reverse step of the production sampler (Gaussian start + --no_noise,
i.e. GaussianDiffusion.sample(stochastic=False)) for one event, and draw the network's
noise guess eps_hat and clean guess x0_hat at each step. Weights are only READ; runs on the GPU."""
import os, sys, json
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
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
C_MEAS, C_DIFF, C_PRED = '#1F5FA8', '#D2622A', '#11845A'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False})
def sty(a, t=None):
    a.grid(alpha=0.25, color=GRID); a.spines[['top', 'right']].set_visible(False)
    if t: a.set_title(t, fontsize=11.5, loc='left')
dev = torch.device('cuda')
sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)
net = UNet1D(cond_mode='step').to(dev)
st = torch.load(MD + 'best_model.pt', map_location=dev, weights_only=False)
net.load_state_dict(st['model_state_dict'] if 'model_state_dict' in st else st); net.eval()
tsec = np.arange(10000) / 1000.0
SHOW = (50, 45, 40, 30, 20, 10, 5, 1)
R = {}

for E in (15, 2615):
    with h5py.File(EVD + f'eval_efficiency_{E}keV.h5') as f:
        noisy = f['waveforms_noisy'][0].astype(np.float64); tmpl = f['clean_template'][:].astype(np.float64)
    s = np.abs(noisy).max(); x0 = tmpl / s; xo = noisy / s
    Xo = torch.tensor(xo, dtype=torch.float32)[None, None].to(dev)
    torch.manual_seed(0); x = torch.randn(1, 1, 10000).to(dev)          # Gaussian start
    rec = {}
    with torch.no_grad():
        for i in reversed(range(50)):                             # same arithmetic as diffusion.sample
            t = i + 1
            e = net(x, Xo, torch.full((1,), t, dtype=torch.long, device=dev))
            sa, sb = sch.sqrt_alpha_bar[i].item(), sch.sqrt_one_minus_alpha_bar[i].item()
            xt = x[0, 0].cpu().numpy().astype(np.float64); eh = e[0, 0].cpu().numpy().astype(np.float64)
            x0h = (xt - sb * eh) / sa
            eps_true = (xt - sa * x0) / sb                        # the noise x_t really holds, given the true pulse
            rec[t] = dict(xt=xt, eh=eh, x0h=x0h, et=eps_true)
            coeff = sch.beta[i] / sch.sqrt_one_minus_alpha_bar[i]
            x = (1.0 / sch.sqrt_alpha[i]) * (x - coeff * e)       # --no_noise: no random term added
    final = x[0, 0].cpu().numpy().astype(np.float64)
    amp = x0.max()
    R[f'{E}keV'] = {
        'x0hat_rms_err_rel_amp': {t: float(np.sqrt(np.mean((rec[t]['x0h'] - x0) ** 2)) / amp) for t in rec},
        'eps_hat_std': {t: float(rec[t]['eh'].std()) for t in rec},
        'eps_hat_vs_true_rms': {t: float(np.sqrt(np.mean((rec[t]['eh'] - rec[t]['et']) ** 2))) for t in rec},
        'final_rms_err_rel_amp': float(np.sqrt(np.mean((final - x0) ** 2)) / amp),
        'meas_rms_err_rel_amp': float(np.sqrt(np.mean((xo - x0) ** 2)) / amp),
        'final_equals_x0hat_t1': float(np.abs(final - rec[1]['x0h']).max())}

    fig, ax = plt.subplots(len(SHOW), 3, figsize=(15.5, 2.05 * len(SHOW) + 0.6), sharex=True,
                           gridspec_kw={'hspace': 0.55, 'wspace': 0.16})
    for r, t in enumerate(SHOW):
        d = rec[t]; sa = sch.sqrt_alpha_bar[t-1].item()
        a = ax[r, 0]; a.plot(tsec, d['xt'], lw=0.35, color=C_DIFF)
        a.plot(tsec, sa * x0, lw=1.0, color=INK, ls='--')
        sty(a, f't = {t}   输入 $x_t$（虚线：其中的信号 {sa:.3f}$\\cdot x_0$）')
        a = ax[r, 1]; a.plot(tsec, d['et'], lw=0.35, color='#9AA3AA')
        a.plot(tsec, d['eh'], lw=0.35, color=C_PRED, alpha=0.85)
        sty(a, '网络猜的高斯噪声 $\\hat\\varepsilon$（绿）  灰：$x_t$ 里实际的噪声')
        a = ax[r, 2]; a.plot(tsec, xo, lw=0.3, color=C_MEAS, alpha=0.35)
        a.plot(tsec, d['x0h'], lw=0.6, color=C_PRED); a.plot(tsec, x0, lw=1.0, color=INK, ls='--')
        err = R[f'{E}keV']['x0hat_rms_err_rel_amp'][t]
        sty(a, f'由 $\\hat\\varepsilon$ 换算的干净波形 $\\hat x_0$   误差 {100*err:.2f}% 脉高')
        lo, hi = x0.min() - 0.35 * amp, amp * 1.35
        if E == 15: lo, hi = xo.min() * 1.1, xo.max() * 1.1
        a.set_ylim(lo, hi)
    for a in ax[-1]: a.set_xlabel('时间 (s)')
    fig.suptitle(f'{E} keV 事件，高斯噪声起步 + no_noise，50 步中的 8 步（右列浅蓝：测量 $\\tilde x$；黑虚线：真值）', x=0.07, ha='left', fontsize=14, y=1.0)
    fig.subplots_adjust(top=0.965)
    fig.savefig(OUT + f'traj_nonoise_{E}keV.png', dpi=120, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for E, c in ((15, C_MEAS), (2615, C_DIFF)):
    d = R[f'{E}keV']; ts = sorted(d['x0hat_rms_err_rel_amp'])
    ax[0].plot(ts, [100 * d['x0hat_rms_err_rel_amp'][t] for t in ts], 'o-', ms=3, color=c, label=f'{E} keV')
    ax[0].axhline(100 * d['meas_rms_err_rel_amp'], color=c, ls=':', lw=1)
    ax[1].plot(ts, [d['eps_hat_vs_true_rms'][t] for t in ts], 'o-', ms=3, color=c, label=f'{E} keV')
ax[0].set_yscale('log'); ax[0].invert_xaxis(); ax[1].invert_xaxis()
ax[0].set_xlabel('步数 t（推理从左往右走）'); ax[0].set_ylabel('$\\hat x_0$ 误差（% 脉高）'); ax[0].legend(frameon=False)
sty(ax[0], '每一步的干净波形猜测有多准（点线：测量本身）')
ax[1].set_xlabel('步数 t'); ax[1].set_ylabel('$\\hat\\varepsilon$ 与实际噪声之差（均方根）'); ax[1].legend(frameon=False)
sty(ax[1], '每一步的高斯噪声猜测有多准')
fig.savefig(OUT + 'traj_nonoise_summary.png', dpi=130, bbox_inches='tight'); plt.close(fig)
json.dump(R, open(OUT + 'traj_nonoise.json', 'w'), indent=1)
for E in (15, 2615):
    d = R[f'{E}keV']
    print(E, 'meas', round(100 * d['meas_rms_err_rel_amp'], 3), 'final', round(100 * d['final_rms_err_rel_amp'], 4),
          'final==x0h(t=1)', d['final_equals_x0hat_t1'])
    print('  x0h err %:', {t: round(100 * d['x0hat_rms_err_rel_amp'][t], 3) for t in SHOW})
    print('  eps std  :', {t: round(d['eps_hat_std'][t], 3) for t in SHOW})
    print('  eps err  :', {t: round(d['eps_hat_vs_true_rms'][t], 4) for t in SHOW})
print('ALL_DONE')
