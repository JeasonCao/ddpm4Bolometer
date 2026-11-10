"""Record every reverse step of the production sampler (Gaussian start + --no_noise,
i.e. GaussianDiffusion.sample(stochastic=False)) for one event, and draw the network's
noise guess eps_hat and clean guess x0_hat at each step. Weights are only READ; runs on the GPU."""
import os, sys, json
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule

OUT = os.path.join(PROJ, "results", "plots", "ddpm_explainer") + "/"
MD = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/'
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
INK, SOFT, GRID = '#18222B', '#5A6670', '#D7DCDA'
C_MEAS, C_DIFF, C_PRED = '#1F5FA8', '#D2622A', '#11845A'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK, 'axes.unicode_minus': False})
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

for E in (2615,):
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
        sty(a, f't = {t}   input $x_t$ (dashed: signal part {sa:.3f}$\\cdot x_0$)')
        a = ax[r, 1]; a.plot(tsec, d['et'], lw=0.35, color='#9AA3AA')
        a.plot(tsec, d['eh'], lw=0.35, color=C_PRED, alpha=0.85)
        sty(a, 'predicted noise $\\hat\\varepsilon$ (green); grey: true noise in $x_t$')
        a = ax[r, 2]; a.plot(tsec, xo, lw=0.3, color=C_MEAS, alpha=0.35)
        a.plot(tsec, d['x0h'], lw=0.6, color=C_PRED); a.plot(tsec, x0, lw=1.0, color=INK, ls='--')
        err = R[f'{E}keV']['x0hat_rms_err_rel_amp'][t]
        sty(a, f'clean estimate $\\hat x_0$ from $\\hat\\varepsilon$   error {100*err:.2f}% of pulse height')
        lo, hi = x0.min() - 0.35 * amp, amp * 1.35
        if E == 15: lo, hi = xo.min() * 1.1, xo.max() * 1.1
        a.set_ylim(lo, hi)
    for a in ax[-1]: a.set_xlabel('time (s)')
    fig.suptitle(f'{E} keV event, Gaussian start + no_noise sampler: 8 of the 50 reverse steps (right column: light blue = measurement $\\tilde x$, black dashed = truth)', x=0.07, ha='left', fontsize=14, y=1.0)
    fig.subplots_adjust(top=0.965)
    fig.savefig(OUT + f'traj_nonoise_{E}keV_en.png', dpi=120, bbox_inches='tight'); plt.close(fig)

print('ALL_DONE')
