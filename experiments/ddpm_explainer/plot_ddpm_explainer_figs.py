"""Figures for the DDPM explainer, all from the real model and real eval data."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys
PROJ = str(_REPO_ROOT)
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_batch_inference import load_model
from src.ddpm.schedule import DiffusionSchedule

OUT = os.path.join(PROJ, "results", "plots", "ddpm_explainer")
EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/eval_efficiency_15keV.h5'
MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
INK, SOFT, GRID = '#1d2328', '#5a646d', '#dcdad4'
C_CLEAN, C_NOISY, C_DIFF, C_PRED = '#1d2328', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': SOFT, 'axes.labelcolor': INK,
                     'xtick.color': SOFT, 'ytick.color': SOFT, 'axes.titlecolor': INK})

with h5py.File(EVAL) as f:
    noisy = f['waveforms_noisy'][3].astype(np.float64)
    clean = f['clean_template'][:].astype(np.float64)
scale = np.abs(noisy).max()
x0, xt_obs = clean / scale, noisy / scale
t_s = np.arange(len(x0)) / 1000.0
sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5)
sab, s1ab = sch.sqrt_alpha_bar.numpy(), sch.sqrt_one_minus_alpha_bar.numpy()
rng = np.random.default_rng(7)

def style(ax, title=None):
    ax.grid(alpha=0.25, color=GRID); ax.spines[['top', 'right']].set_visible(False)
    if title: ax.set_title(title, fontsize=10.5, loc='left')

# ---------- fig 1: the two kinds of noise ----------
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
xT = rng.standard_normal(len(x0))
for a, y, c, ttl in zip(ax,
        [x0, xt_obs - x0, xt_obs, xT],
        [C_CLEAN, SOFT, C_NOISY, C_DIFF],
        ['clean pulse $x_0$', 'detector noise  $n$', r'measurement  $\tilde x = x_0 + n$',
         r'diffusion start  $x_T \sim \mathcal{N}(0,I)$']):
    a.plot(t_s, y, lw=0.6, color=c); style(a, ttl); a.set_xlabel('time (s)')
ax[0].set_ylabel('normalised amplitude'); ax[0].set_ylim(-3.2, 3.2)
fig.tight_layout(); fig.savefig(f'{OUT}/f1_two_noises.png', dpi=150); plt.close(fig)

# ---------- fig 2: noise schedule ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
t = np.arange(1, 51)
ax[0].plot(t, sch.beta.numpy(), color=C_DIFF, lw=2)
style(ax[0], r'(a)  per-step noise  $\beta_t$  (quadratic, $10^{-4}\to0.5$)')
ax[0].set_xlabel('step t'); ax[0].set_ylabel(r'$\beta_t$')
ax[1].plot(t, sab, color=C_CLEAN, lw=2, label=r'$\sqrt{\bar\alpha_t}$  signal kept')
ax[1].plot(t, s1ab, color=C_DIFF, lw=2, label=r'$\sqrt{1-\bar\alpha_t}$  noise added')
style(ax[1], r'(b)  how much of $x_0$ survives to step t')
ax[1].set_xlabel('step t'); ax[1].legend(frameon=False, fontsize=9)
for tt in (10, 20, 30, 40):
    ax[1].axvline(tt, color=GRID, lw=0.8, zorder=0)
fig.tight_layout(); fig.savefig(f'{OUT}/f2_schedule.png', dpi=150); plt.close(fig)

# ---------- fig 3: forward process ----------
SHOW = [0, 10, 20, 30, 40, 50]
fig, ax = plt.subplots(1, 6, figsize=(14, 2.5), sharey=True)
eps = rng.standard_normal(len(x0))
for a, k in zip(ax, SHOW):
    y = x0 if k == 0 else sab[k-1] * x0 + s1ab[k-1] * eps
    a.plot(t_s, y, lw=0.5, color=C_CLEAN if k == 0 else C_DIFF)
    sig = 1.0 if k == 0 else sab[k-1]
    style(a, f't = {k}' + ('' if k == 0 else f'   signal {sig*100:.0f}%' if sig > 0.01 else f'   signal {sig*100:.1f}%'))
    a.set_xticks([0, 5, 10])
ax[0].set_ylim(-3.5, 3.5); ax[0].set_ylabel('$x_t$')
fig.tight_layout(); fig.savefig(f'{OUT}/f3_forward.png', dpi=150); plt.close(fig)

# ---------- fig 4: reverse process, real model, zero-init DDIM ----------
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff = load_model(MODEL, dev)
x_tilde = torch.from_numpy(xt_obs.astype(np.float32))[None, None].to(dev)
S = diff.schedule
traj_x, traj_x0, traj_eps = {}, {}, {}
with torch.no_grad():
    g = torch.Generator(device=dev); g.manual_seed(11)
    x_t = torch.randn(1, 1, len(x0), device=dev, generator=g)   # textbook start
    for i, tc in enumerate(range(49, -1, -1)):
        step = tc + 1
        cond = torch.full((1,), step, device=dev, dtype=torch.long)
        e = diff._predict_noise(x_t, x_tilde, cond, None)
        x0h = (x_t - S.sqrt_one_minus_alpha_bar[tc] * e) / S.sqrt_alpha_bar[tc]
        traj_x[step], traj_x0[step], traj_eps[step] = (x_t[0,0].cpu().numpy().copy(),
            x0h[0,0].cpu().numpy().copy(), e[0,0].cpu().numpy().copy())
        x_t = (S.sqrt_alpha_bar[tc-1] * x0h + S.sqrt_one_minus_alpha_bar[tc-1] * e) if tc > 0 else x0h
    final = x_t[0,0].cpu().numpy()

SHOWR = [50, 40, 30, 20, 10, 1]
fig, ax = plt.subplots(2, 6, figsize=(14, 4.6), sharey='row')
for j, k in enumerate(SHOWR):
    ax[0, j].plot(t_s, traj_x[k], lw=0.6, color=C_DIFF)
    style(ax[0, j], f'step {k}:  $x_t$')
    ax[1, j].plot(t_s, x0, lw=1.0, color=GRID, label='truth')
    ax[1, j].plot(t_s, traj_x0[k], lw=0.7, color=C_PRED)
    style(ax[1, j], r"network's guess $\hat x_0$")
    for a in ax[:, j]: a.set_xticks([0, 5, 10])
ax[0, 0].set_ylim(-3.2, 3.2); ax[1, 0].set_ylim(-0.4, 1.3)
ax[0, 0].set_ylabel('state carried\nto next step'); ax[1, 0].set_ylabel('guess at\nclean pulse')
fig.tight_layout(); fig.savefig(f'{OUT}/f4_reverse.png', dpi=150); plt.close(fig)

# ---------- fig 5: end result for this event ----------
fig, ax = plt.subplots(1, 1, figsize=(10, 3.2))
ax.plot(t_s, xt_obs * scale * 1e3, lw=0.5, color=C_NOISY, alpha=0.75, label=r'measured  $\tilde x$')
ax.plot(t_s, final * scale * 1e3, lw=1.4, color=C_PRED, label='denoised  (output × scale)')
ax.plot(t_s, x0 * scale * 1e3, lw=1.0, color=C_CLEAN, ls='--', label='true pulse')
style(ax, '15 keV event, 1.382 mV noise:  input → output, back in physical units')
ax.set_xlabel('time (s)'); ax.set_ylabel('mV'); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/f5_result.png', dpi=150); plt.close(fig)

err_obs = np.sqrt(((xt_obs - x0) ** 2).mean()); err_out = np.sqrt(((final - x0) ** 2).mean())
print(f'scale = {scale*1e3:.3f} mV;  RMS error: input {err_obs:.4f}  output {err_out:.5f}  ({err_obs/err_out:.0f}x)')
print('EXPLAIN_FIGS_DONE')
