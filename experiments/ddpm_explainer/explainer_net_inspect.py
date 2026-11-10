"""Inspect the trained U-Net: filters, feature maps, attention, receptive field, one real training step.
Production weights are only READ; nothing is saved back."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, copy, json
PROJ = str(_REPO_ROOT)
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion

OUT = os.path.join(PROJ, "results", "plots", "ddpm_explainer") + "/"
MD = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/'
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
INK, SOFT, GRID = '#18222B', '#5A6670', '#D7DCDA'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK})
def sty(a, t=None):
    a.grid(alpha=0.25, color=GRID); a.spines[['top', 'right']].set_visible(False)
    if t: a.set_title(t, fontsize=13.5, loc='left')
dev = torch.device('cuda')
sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)
def load(path):
    m = UNet1D(cond_mode='step').to(dev)
    st = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(st, dict) and 'model_state_dict' in st: st = st['model_state_dict']
    m.load_state_dict(st); m.eval(); return m
trained = load(MD + 'best_model.pt')
torch.manual_seed(0); rnd = UNet1D(cond_mode='step').to(dev).eval()

with h5py.File(EVD + 'eval_efficiency_15keV.h5') as f:
    noisy = f['waveforms_noisy'][0].astype(np.float64); tmpl = f['clean_template'][:].astype(np.float64)
s = np.abs(noisy).max(); x0 = tmpl / s; xt_obs = noisy / s
L = x0.size; tsec = np.arange(L) / 1000.0
eps = np.random.default_rng(3).standard_normal(L)
T_DEMO = 30
xt = sch.sqrt_alpha_bar[T_DEMO-1].item() * x0 + sch.sqrt_one_minus_alpha_bar[T_DEMO-1].item() * eps
tt = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
X_t, X_o = tt(xt), tt(xt_obs); TT = torch.full((1,), T_DEMO, device=dev, dtype=torch.long)

# ---- 1: first-layer filter frequency responses ----
def fresp(w):  # w: (64, 3) taps -> |H(f)| on 0..500 Hz
    f = np.linspace(0, 500, 251); z = np.exp(-2j*np.pi*f/1000.0)
    H = np.abs(w[:, 0, None] + w[:, 1, None]*z + w[:, 2, None]*z**2); return f, H
fig, ax = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
for r, (m, lab) in enumerate(((rnd, 'random initialisation'), (trained, 'after training'))):
    W = m.conv_in.weight.detach().cpu().numpy()          # (64, 2, 3)
    for c, ch in enumerate(('input channel 1: mixture $x_t$', r'input channel 2: measurement $\tilde x$')):
        f, H = fresp(W[:, c, :]); H = H / (H.max(axis=1, keepdims=True) + 1e-12)
        order = np.argsort(H[:, -1] - H[:, 0])            # low-pass at top, high-pass at bottom
        im = ax[r, c].imshow(H[order], aspect='auto', origin='lower', extent=[0, 500, 0, 64], cmap='magma')
        sty(ax[r, c], f'{lab} — {ch}'); ax[r, c].grid(False)
        ax[r, c].set_ylabel('64 filters (sorted)')
        if r == 1: ax[r, c].set_xlabel('frequency (Hz)')
fig.colorbar(im, ax=ax, shrink=0.6, label='|gain| (each filter scaled to its max)')
fig.savefig(OUT + 'n1_filters.png', dpi=130, bbox_inches='tight'); plt.close(fig)
W = trained.conv_in.weight.detach().cpu().numpy(); f, H = fresp(W[:, 1, :])
lp = int(((H[:, 0] > 2*H[:, -1])).sum()); hp = int(((H[:, -1] > 2*H[:, 0])).sum())
print(f'[1] trained x~ channel: {lp} filters low-pass-like, {hp} high-pass-like (gain ratio > 2), rest mixed')

# ---- 2: feature maps through the U-Net ----
acts = {}
def hook(name): return lambda mod, i, o: acts.__setitem__(name, o.detach()[0].float().cpu().numpy())
hs = [trained.enc_attns[i].register_forward_hook(hook(f'enc{i}')) for i in range(4)]
hs.append(trained.mid_block2.register_forward_hook(hook('mid')))
hs.append(trained.dec_blocks[3][1].register_forward_hook(hook('dec0')))  # last decoder level, its final ResBlock
with torch.no_grad(): eps_hat = trained(X_t, X_o, TT)[0, 0].cpu().numpy()
for h in hs: h.remove()
rows = [('enc0', 'encoder level 0:   64 channels × 10000 samples'), ('enc1', 'encoder level 1:  128 × 5000'),
        ('enc2', 'encoder level 2:  256 × 2500'), ('enc3', 'encoder level 3:  512 × 1250'),
        ('mid', 'bottleneck:  512 × 1250  (after attention)'), ('dec0', 'decoder level 0:   64 × 10000')]
fig, ax = plt.subplots(len(rows) + 2, 1, figsize=(13, 15), sharex=True)
ax[0].plot(tsec, xt, lw=0.4, color='#D2622A'); ax[0].plot(tsec, xt_obs, lw=0.7, color='#1F5FA8')
sty(ax[0], f'input: mixture $x_t$ (orange, t = {T_DEMO}) and measurement $\\tilde x$ (blue)')
for a, (k, lab) in zip(ax[1:-1], rows):
    A = acts[k]; n = A.shape[1]; tx = np.arange(n) * (10.0 / n)
    pick = np.argsort(A.std(axis=1))[::-1][:4]
    for j, ch in enumerate(pick):
        y = (A[ch] - A[ch].mean()) / (A[ch].std() + 1e-9)
        a.plot(tx, y + 4.2*j, lw=0.6, color=plt.cm.viridis(0.15 + 0.22*j))
    sty(a, lab + '   — 4 most active channels'); a.set_yticks([])
ax[-1].plot(tsec, eps, lw=0.4, color='#C9CED2', label='true injected noise ε')
ax[-1].plot(tsec, eps_hat, lw=0.4, color='#1B9A6C', alpha=0.8, label=r'output $\hat\varepsilon$')
sty(ax[-1], 'output: predicted noise'); ax[-1].legend(frameon=False, loc='upper right', fontsize=11)
ax[-1].set_xlabel('time (s)')
fig.tight_layout(); fig.savefig(OUT + 'n2_features.png', dpi=110); plt.close(fig)
print(f'[2] eps-hat rms error at t={T_DEMO}: {np.sqrt(((eps_hat-eps)**2).mean()):.4f}')

# ---- 3: bottleneck attention ----
cap = {}
hk = trained.mid_attn.register_forward_hook(lambda m, i, o: cap.__setitem__('x', i[0].detach()))
with torch.no_grad(): trained(X_t, X_o, TT)
hk.remove()
with torch.no_grad():
    h = trained.mid_attn.norm(cap['x']).permute(0, 2, 1)
    _, Wt = trained.mid_attn.attn(h, h, h, need_weights=True, average_attn_weights=True)
Wt = Wt[0].float().cpu().numpy()                             # (1250, 1250)
tk = np.arange(1250) * 0.008
q_peak = int(np.argmax(x0) / 8); q_base = int(7.0 / 0.008)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [1, 1.35]})
ax[0].imshow(np.log10(Wt + 1e-7), origin='lower', extent=[0, 10, 0, 10], cmap='magma', vmin=-6)
ax[0].set_xlabel('attended time (s)'); ax[0].set_ylabel('query time (s)')
sty(ax[0], 'attention weights (log10), all 1250 × 1250'); ax[0].grid(False)
for q, c, lab in ((q_peak, '#D2622A', f'query at pulse peak ({q_peak*0.008:.2f} s)'),
                  (q_base, '#1F5FA8', f'query on baseline ({q_base*0.008:.1f} s)')):
    ax[1].semilogy(tk, Wt[q], lw=1.2, color=c, label=lab)
ax[1].semilogy(tk, np.full_like(tk, 1/1250), color=SOFT, ls=':', lw=1.2, label='uniform (1/1250)')
ax2 = ax[1].twinx(); ax2.plot(tsec, x0, color='#B9C0C5', lw=1.5); ax2.set_yticks([]); ax2.spines[['top','right']].set_visible(False)
ax[1].set_xlabel('attended time (s)'); ax[1].set_ylabel('weight'); sty(ax[1], 'where two positions look (grey: the pulse)')
ax[1].legend(frameon=False, fontsize=10.5, loc='upper right')
fig.tight_layout(); fig.savefig(OUT + 'n3_attention.png', dpi=130); plt.close(fig)
pk = Wt[q_peak]; near = pk[max(0, q_peak-25):q_peak+25].sum()
print(f'[3] peak query: weight within ±0.2 s = {near:.3f}; max weight {pk.max():.4f} at {np.argmax(pk)*0.008:.2f} s')

# ---- 4: effective receptive field ----
n0 = 5000
Xi_t = X_t.clone().requires_grad_(True); Xi_o = X_o.clone().requires_grad_(True)
out = trained(Xi_t, Xi_o, TT); out[0, 0, n0].backward()
g_t = Xi_t.grad[0, 0].abs().cpu().numpy(); g_o = Xi_o.grad[0, 0].abs().cpu().numpy()
lag = (np.arange(L) - n0) / 1000.0
def width(g, frac):
    c = np.cumsum(np.sort(g)[::-1]); k = np.searchsorted(c, frac * c[-1]) + 1
    idx = np.argsort(g)[::-1][:k]; return np.abs(idx - n0).max()
fig, a = plt.subplots(figsize=(13, 4.6))
a.semilogy(lag, g_t + 1e-12, lw=0.8, color='#D2622A', label=r'sensitivity to the mixture $x_t$')
a.semilogy(lag, g_o + 1e-12, lw=0.8, color='#1F5FA8', label=r'sensitivity to the measurement $\tilde x$')
a.set_xlabel('input time relative to the output sample (s)'); a.set_ylabel('|∂ output / ∂ input|')
sty(a, 'how one output sample (at t = 5 s) depends on every input sample'); a.legend(frameon=False)
a.set_ylim(1e-8, None)
fig.tight_layout(); fig.savefig(OUT + 'n4_receptive.png', dpi=130); plt.close(fig)
for g, lab in ((g_t, 'x_t'), (g_o, 'x~')):
    print(f'[4] {lab}: 90% of influence within ±{width(g,0.9)} ms, 99% within ±{width(g,0.99)} ms; nonzero span {np.ptp(np.where(g>1e-9)[0])} samples')

# ---- 5: one real training step from the epoch-70 checkpoint (in memory only) ----
ck = torch.load(MD + 'checkpoint_070.pt', map_location=dev, weights_only=False)
m = UNet1D(cond_mode='step').to(dev); m.load_state_dict(ck['model_state_dict']); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=2e-4); opt.load_state_dict(ck['optimizer_state_dict'])
lr_ep71 = 1e-6 + 0.5*(2e-4-1e-6)*(1+np.cos(np.pi*70/100))
for pg in opt.param_groups: pg['lr'] = lr_ep71
diff = GaussianDiffusion(m, sch, loss_type='l1')
rng = np.random.default_rng(5); B = 8; xs, xo = [], []
for E in (5, 15, 50, 239, 511, 1461, 2615, 3034):
    with h5py.File(EVD + f'eval_efficiency_{E}keV.h5') as f:
        k = int(rng.integers(0, 5000)); nz = f['waveforms_noisy'][k].astype(np.float64); tp = f['clean_template'][:].astype(np.float64)
    sc = np.abs(nz).max(); xs.append(tp/sc); xo.append(nz/sc)
x0b = torch.from_numpy(np.array(xs, np.float32))[:, None].to(dev); xob = torch.from_numpy(np.array(xo, np.float32))[:, None].to(dev)
theta0 = torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
torch.manual_seed(1); loss = diff.training_loss(x0b, xob)
opt.zero_grad(); loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
opt.step()
theta1 = torch.cat([p.detach().flatten() for p in m.parameters()])
torch.manual_seed(1); loss_after = diff.training_loss(x0b, xob).item()
dth = (theta1 - theta0)
res = dict(loss_before=loss.item(), loss_after=loss_after, grad_norm=gnorm, lr=lr_ep71,
           rel_change=(dth.norm()/theta0.norm()).item(), median_abs_change=dth.abs().median().item(),
           frac_changed=(dth != 0).float().mean().item(), n_params=theta0.numel())
print('[5]', json.dumps(res))
json.dump(res, open(OUT + 'n5_step.json', 'w'), indent=1)
print('INSPECT_DONE')
