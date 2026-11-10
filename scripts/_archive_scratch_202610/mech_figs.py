"""Mechanism figures for the training/inference slides. Real model, real 15 keV eval data."""
import os, sys
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scripts.run_batch_inference import load_model

OUT = "/tmp/claude-1000/-home-wsl-0vbb/4159fbbb-99dd-4ed5-b7cf-493a0436b6d5/scratchpad/deck/img/"
EV = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/eval_efficiency_15keV.h5'
MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
INK, SOFT, GRID = '#18222B', '#5A6670', '#D7DCDA'
C_X0, C_MEAS, C_DIFF, C_PRED = '#18222B', '#1F5FA8', '#D2622A', '#1B9A6C'
plt.rcParams.update({'font.size': 14, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK})
def sty(a, t=None):
    a.grid(alpha=0.25, color=GRID); a.spines[['top', 'right']].set_visible(False)
    if t: a.set_title(t, fontsize=14.5, loc='left')

dev = torch.device('cuda')
D = load_model(MODEL, dev); S = D.schedule
sab = S.sqrt_alpha_bar.cpu().numpy(); s1ab = S.sqrt_one_minus_alpha_bar.cpu().numpy()
with h5py.File(EV) as f:
    noisy = f['waveforms_noisy'][:16].astype(np.float64); tmpl = f['clean_template'][:].astype(np.float64)
sc = np.abs(noisy).max(axis=1, keepdims=True)
X0 = tmpl[None] / sc; XT = noisy / sc                      # normalised clean & measurement, 16 events
ts = np.arange(X0.shape[1]) / 1000.0
rng = np.random.default_rng(3)
x0, xt_obs = X0[0], XT[0]
eps = rng.standard_normal(x0.size)

# ---------- 1: building a training sample ----------
fig, ax = plt.subplots(2, 3, figsize=(14, 5.6), sharey=True)
for a, y, c, t in zip(ax[0], [x0, xt_obs, eps], [C_X0, C_MEAS, C_DIFF],
        ['clean pulse  $x_0$  (simulated)', r'measurement  $\tilde x$ = ($x_0$ + real noise) / s',
         r'injected noise  $\varepsilon \sim \mathcal{N}(0,I)$']):
    a.plot(ts, y, lw=0.6, color=c); sty(a, t)
for a, t in zip(ax[1], (10, 25, 40)):
    y = sab[t-1] * x0 + s1ab[t-1] * eps
    a.plot(ts, y, lw=0.6, color=C_DIFF); a.set_xlabel('time (s)')
    sty(a, f'$x_t$ at t = {t}:   {sab[t-1]:.2f}·$x_0$ + {s1ab[t-1]:.2f}·ε')
ax[0, 0].set_ylim(-3.3, 3.3)
fig.tight_layout(); fig.savefig(OUT + 'tr_sample.png', dpi=130); plt.close(fig)

# ---------- 2: what one network call returns, at several t ----------
x0_t = torch.from_numpy(x0.astype(np.float32))[None, None].to(dev)
xo_t = torch.from_numpy(xt_obs.astype(np.float32))[None, None].to(dev)
ep_t = torch.from_numpy(eps.astype(np.float32))[None, None].to(dev)
SHOW = (5, 20, 35, 50)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
with torch.no_grad():
    for a, t in zip(ax, SHOW):
        xt = S.sqrt_alpha_bar[t-1] * x0_t + S.sqrt_one_minus_alpha_bar[t-1] * ep_t
        e = D._predict_noise(xt, xo_t, torch.full((1,), t, device=dev, dtype=torch.long), None)
        x0h = ((xt - S.sqrt_one_minus_alpha_bar[t-1] * e) / S.sqrt_alpha_bar[t-1])[0, 0].cpu().numpy()
        eh = e[0, 0].cpu().numpy()
        a.plot(ts, xt[0, 0].cpu().numpy(), lw=0.4, color=C_DIFF, alpha=0.35, label='input $x_t$')
        a.plot(ts, x0, lw=2.4, color='#B9C0C5', label='truth $x_0$')
        a.plot(ts, x0h, lw=0.9, color=C_PRED, label=r'guess $\hat x_0$')
        r_eps = np.sqrt(((eh - eps) ** 2).mean()); r_x0 = np.sqrt(((x0h - x0) ** 2).mean())
        sty(a, f't = {t}   (signal {sab[t-1]*100:.1f}%)'); a.set_xlabel('time (s)')
        a.text(0.97, 0.05, f'error in ε̂: {r_eps:.4f}\nerror in x̂₀: {r_x0:.4f}', transform=a.transAxes,
               ha='right', va='bottom', fontsize=11.5, color=SOFT)
        print(f't={t:2d}  rms(eps_hat-eps)={r_eps:.4f}  rms(x0_hat-x0)={r_x0:.5f}')
ax[0].set_ylim(-0.6, 1.4); ax[0].legend(frameon=False, fontsize=11, loc='upper right')
fig.tight_layout(); fig.savefig(OUT + 'tr_what.png', dpi=130); plt.close(fig)

# ---------- 3: geometry of the state ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
a = ax[0]
th = np.linspace(0, np.pi/2, 200); a.plot(np.cos(th), np.sin(th), color=GRID, lw=6, zorder=0)
tt = np.arange(1, 51)
a.plot(sab, s1ab, 'o', ms=4, color=C_DIFF, zorder=2)
a.plot([1], [0], 'o', ms=9, color=C_X0, zorder=3); a.annotate('t = 0\nclean pulse', (1, 0), xytext=(-8, 14), textcoords='offset points', ha='right', fontsize=12.5)
for t in (10, 20, 30, 40):
    a.annotate(f't = {t}', (sab[t-1], s1ab[t-1]), xytext=(9, 5), textcoords='offset points', fontsize=12.5, color=SOFT)
a.plot([sab[49]], [s1ab[49]], 'o', ms=9, color=C_DIFF, zorder=3)
a.annotate('t = 50\npure noise', (sab[49], s1ab[49]), xytext=(14, -38), textcoords='offset points', fontsize=12.5)
k = 24  # one reverse step t=25 -> t=24, drawn as a rotation
a.annotate('', xy=(sab[k-1], s1ab[k-1]), xytext=(sab[k], s1ab[k]),
           arrowprops=dict(arrowstyle='-|>', color=C_MEAS, lw=2.4, mutation_scale=22))
a.annotate('one reverse step\n= a small rotation', (sab[k-1], s1ab[k-1]), xytext=(-150, -40),
           textcoords='offset points', fontsize=12.5, color=C_MEAS)
a.set_xlim(-0.05, 1.12); a.set_ylim(-0.05, 1.12); a.set_aspect('equal')
a.set_xlabel(r'signal coefficient  $\sqrt{\bar\alpha_t}$  (cos φ)'); a.set_ylabel(r'noise coefficient  $\sqrt{1-\bar\alpha_t}$  (sin φ)')
sty(a, r'$x_t = \cos\varphi_t\,x_0 + \sin\varphi_t\,\varepsilon$')
b = ax[1]
phi = np.degrees(np.arctan2(s1ab, sab)); dphi = np.diff(np.concatenate([[0.0], phi]))
b.bar(tt, dphi, color=C_DIFF, width=0.75)
b.set_xlabel('step t'); b.set_ylabel('rotation per step (degrees)')
sty(b, 'angle covered by each step (90° in total)')
fig.tight_layout(); fig.savefig(OUT + 'inf_arc.png', dpi=130); plt.close(fig)
print(f'total angle {phi[-1]:.2f} deg; largest single step {dphi.max():.2f} deg at t={int(np.argmax(dphi))+1}')

# ---------- 4: convergence of the guess vs the state (16 events, white-noise start, DDIM eta=0) ----------
xo = torch.from_numpy(XT.astype(np.float32)).unsqueeze(1).to(dev)
x0true = X0
g = torch.Generator(device=dev); g.manual_seed(11)
x_t = torch.randn(xo.shape, device=dev, generator=g)
err_x0h, err_xt = [], []
with torch.no_grad():
    for tc in range(49, -1, -1):
        e = D._predict_noise(x_t, xo, torch.full((xo.shape[0],), tc + 1, device=dev, dtype=torch.long), None)
        x0h = (x_t - S.sqrt_one_minus_alpha_bar[tc] * e) / S.sqrt_alpha_bar[tc]
        err_xt.append(np.sqrt(((x_t[:, 0].cpu().numpy() - x0true) ** 2).mean()))
        err_x0h.append(np.sqrt(((x0h[:, 0].cpu().numpy() - x0true) ** 2).mean()))
        x_t = (S.sqrt_alpha_bar[tc-1] * x0h + S.sqrt_one_minus_alpha_bar[tc-1] * e) if tc > 0 else x0h
    final = np.sqrt(((x_t[:, 0].cpu().numpy() - x0true) ** 2).mean())
meas_err = np.sqrt(((XT - X0) ** 2).mean())
steps = np.arange(50, 0, -1)
fig, a = plt.subplots(figsize=(12, 5.4))
a.semilogy(steps, err_xt, color=C_DIFF, lw=2.6, label=r'state $x_t$  vs truth')
a.semilogy(steps, err_x0h, color=C_PRED, lw=2.6, label=r"network's guess $\hat x_0$  vs truth")
a.axhline(meas_err, color=C_MEAS, ls='--', lw=1.8, label=r'measurement $\tilde x$  vs truth')
a.invert_xaxis(); a.set_xlabel('step t  (counts down from 50 to 1)'); a.set_ylabel('RMS deviation from the true pulse')
sty(a); a.legend(frameon=False, fontsize=13)
fig.tight_layout(); fig.savefig(OUT + 'inf_converge.png', dpi=130); plt.close(fig)
print(f'measurement err {meas_err:.4f}; guess err at t=50 {err_x0h[0]:.4f}, t=40 {err_x0h[10]:.4f}, t=1 {err_x0h[-1]:.5f}; state err t=50 {err_xt[0]:.3f} t=10 {err_xt[40]:.4f}; final {final:.5f}')
print('MECH_DONE')
