import os, sys, json
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
O = os.path.join(PROJ, "results", "plots", "ddpm_explainer") + "/"
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
INK, SOFT, GRID = '#18222B', '#5A6670', '#D7DCDA'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': SOFT, 'xtick.color': SOFT, 'ytick.color': SOFT,
                     'axes.labelcolor': INK, 'axes.titlecolor': INK})
def sty(a, t=None):
    a.grid(alpha=0.25, color=GRID); a.spines[['top', 'right']].set_visible(False)
    if t: a.set_title(t, fontsize=13, loc='left')
with h5py.File(EVD + 'eval_efficiency_15keV.h5') as f:
    nz = f['waveforms_noisy'][0].astype(np.float64); tp = f['clean_template'][:].astype(np.float64)
s = np.abs(nz).max(); X0 = tp/s; ts = np.arange(X0.size)/1000.0

# ---- demo training ----
d = np.load(O + 'demo_train.npz'); r = json.load(open(O + 'demo_res.json'))
fig = plt.figure(figsize=(14, 7.6)); gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.45, wspace=0.12)
a = fig.add_subplot(gs[0, :])
L = d['losses']; k = 25; sm = np.convolve(L, np.ones(k)/k, mode='valid')
a.semilogy(np.arange(1, L.size+1), L, color='#C9CED2', lw=0.6, label='loss of each batch')
a.semilogy(np.arange(k, L.size+1), sm, color='#D2622A', lw=2.2, label=f'running mean of {k} batches')
a.semilogy(d['vsteps'], d['vloss'], 'o', ms=4, color='#1F5FA8', label='fixed held-out batch')
a.set_xlabel('training step (8 samples per step)'); a.set_ylabel('L1 loss  |ε̂ − ε|')
sty(a, f'a fresh network, trained for 5.5 minutes on this laptop GPU ({r["steps"]} steps)'); a.legend(frameon=False, fontsize=11)
for j, st in enumerate(r['snap_steps']):
    b = fig.add_subplot(gs[1, j]); g = d[f'snap_{st}']
    b.plot(ts, X0, color='#B9C0C5', lw=2.2); b.plot(ts, g, color='#1B9A6C', lw=0.7)
    b.set_ylim(-0.6, 1.5); b.set_xticks([0, 5, 10]); sty(b, f'step {st}'); b.set_xlabel('time (s)')
    b.text(0.96, 0.92, f'err {r["snap_err"][str(st)]:.3f}', transform=b.transAxes, ha='right', fontsize=11, color=SOFT)
    if j: b.set_yticklabels([])
fig.savefig(O + 'n6_scratch.png', dpi=125, bbox_inches='tight'); plt.close(fig)

# ---- real checkpoints ----
c = json.load(open(O + 'ck_res.json'))
names = [x['name'] for x in c['ck']]; ep = [72 if n == 'best' else int(n) for n in names]
order = np.argsort(ep); ep = np.array(ep)[order]; ck = [c['ck'][i] for i in order]
fig = plt.figure(figsize=(14, 7.8)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.42, wspace=0.12)
show = ['010', '030', '050', 'best', '080']; lab = {'010': 'epoch 10', '030': 'epoch 30', '050': 'epoch 50', 'best': 'epoch 72 (production)', '080': 'epoch 80'}
for j, n in enumerate(show):
    a = fig.add_subplot(gs[0, j]); fin = np.load(O + f'ck_{n}_final.npy')
    a.plot(ts, fin - X0, lw=0.6, color='#D2622A' if n == '080' else '#1F5FA8'); a.axhline(0, color=SOFT, lw=0.6)
    a.set_ylim(-0.04, 0.04); a.set_xticks([0, 5, 10]); a.set_xlabel('time (s)')
    sty(a, lab[n]); a.text(0.96, 0.9, f'rms {np.sqrt(((fin-X0)**2).mean()):.4f}', transform=a.transAxes, ha='right', fontsize=11, color=SOFT)
    if j: a.set_yticklabels([])
    else: a.set_ylabel('output − truth')
b = fig.add_subplot(gs[1, :])
b.semilogy(ep, [x['final_err'] for x in ck], 'o-', color='#1F5FA8', lw=2.2, ms=7, label='full 50-step inference')
b.semilogy(ep, [x['probe_err'] for x in ck], 's--', color='#1B9A6C', lw=1.6, ms=6, label='one network call at t = 30')
b.axhline(c['meas_err'], color=SOFT, ls=':', lw=1.5, label='the measurement itself')
b.axvspan(75.5, 81, color='#F3D9CC', zorder=0); b.annotate('learning rate accidentally\nreset to its starting value', (78, 0.0021), xytext=(52, 0.0006),
          fontsize=11, color='#B4501C', arrowprops=dict(arrowstyle='->', color='#B4501C'))
b.set_xlabel('epoch (≈ 5090 parameter updates each)'); b.set_ylabel('rms deviation from the true pulse')
sty(b, 'the same held-out 15 keV event, run through the saved checkpoints of the real training'); b.legend(frameon=False, fontsize=11, loc='upper right')
fig.savefig(O + 'n7_checkpoints.png', dpi=125, bbox_inches='tight'); plt.close(fig)

# ---- attention, redrawn: where the attention goes on average ----
dev = torch.device('cuda'); sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)
m = UNet1D(cond_mode='step').to(dev); m.load_state_dict(torch.load('/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt', map_location=dev, weights_only=True)); m.eval()
eps = np.random.default_rng(3).standard_normal(X0.size); t = 30
xt = sch.sqrt_alpha_bar[t-1].item()*X0 + sch.sqrt_one_minus_alpha_bar[t-1].item()*eps
f32 = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
cap = {}; hk = m.mid_attn.register_forward_hook(lambda mod, i, o: cap.__setitem__('x', i[0].detach()))
with torch.no_grad(): m(f32(xt), f32(nz/s), torch.full((1,), t, device=dev, dtype=torch.long))
hk.remove()
with torch.no_grad():
    h = m.mid_attn.norm(cap['x']).permute(0, 2, 1); _, W = m.mid_attn.attn(h, h, h, need_weights=True, average_attn_weights=True)
W = W[0].float().cpu().numpy(); tk = np.arange(1250)*0.008
recv = W.mean(axis=0)
onset = 1.5; win = (tk >= onset-0.1) & (tk < onset+0.1)
share = recv[win].sum(); uni = win.mean()
q_peak = int(np.argmax(X0)/8); q_base = int(7.0/0.008)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].plot(tk, recv*1250, color='#1F5FA8', lw=1.3); ax[0].axhline(1, color=SOFT, ls=':', lw=1.2)
a2 = ax[0].twinx(); a2.plot(ts, X0, color='#B9C0C5', lw=1.6); a2.set_yticks([]); a2.spines[['top','right']].set_visible(False)
ax[0].set_xlabel('time being looked at (s)'); ax[0].set_ylabel('attention received\n(1 = uniform share)')
sty(ax[0], 'averaged over all 1250 positions (grey: the pulse)')
for q, col, lb in ((q_peak, '#D2622A', f'position at the pulse peak ({q_peak*0.008:.2f} s)'), (q_base, '#1F5FA8', f'position on the baseline ({q_base*0.008:.1f} s)')):
    ax[1].semilogy(tk, W[q]*1250, lw=1.1, color=col, label=lb)
ax[1].axhline(1, color=SOFT, ls=':', lw=1.2); a3 = ax[1].twinx(); a3.plot(ts, X0, color='#B9C0C5', lw=1.6); a3.set_yticks([]); a3.spines[['top','right']].set_visible(False)
ax[1].set_xlabel('time being looked at (s)'); ax[1].set_ylabel('weight (1 = uniform share)')
sty(ax[1], 'where two individual positions look'); ax[1].legend(frameon=False, fontsize=10.5, loc='upper right')
fig.tight_layout(); fig.savefig(O + 'n3_attention.png', dpi=130); plt.close(fig)
print(f'attention: ±0.1 s around onset holds {share*100:.1f}% of all attention vs uniform {uni*100:.1f}% ({share/uni:.1f}x); peak-query max at {np.argmax(W[q_peak])*0.008:.2f} s')
print('PLOTS_DONE')
