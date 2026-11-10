"""Watch training happen. (A) a fresh, randomly initialised UNet trained for a few minutes on eval
waveforms (demo only, in memory, never saved); (B) the real production checkpoints, epochs 10..80 and best.
Both evaluated on one fixed held-out example. Production files are only READ."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, time, json
PROJ = str(_REPO_ROOT)
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion
OUT = os.path.join(PROJ, "results", "plots", "ddpm_explainer") + "/"
MD = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/'
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
dev = torch.device('cuda'); sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)

# fixed demo example (held out from the demo pool)
with h5py.File(EVD + 'eval_efficiency_15keV.h5') as f:
    nz = f['waveforms_noisy'][0].astype(np.float64); tp = f['clean_template'][:].astype(np.float64)
s = np.abs(nz).max(); X0 = tp / s; XO = nz / s
eps_fix = np.random.default_rng(3).standard_normal(X0.size)
f32 = lambda a: torch.from_numpy(np.asarray(a, np.float32))
x0_t, xo_t, ep_t = (f32(a)[None, None].to(dev) for a in (X0, XO, eps_fix))

@torch.no_grad()
def probe(m, t=30):
    m.eval(); xt = sch.sqrt_alpha_bar[t-1]*x0_t + sch.sqrt_one_minus_alpha_bar[t-1]*ep_t
    e = m(xt, xo_t, torch.full((1,), t, device=dev, dtype=torch.long))
    x0h = (xt - sch.sqrt_one_minus_alpha_bar[t-1]*e) / sch.sqrt_alpha_bar[t-1]
    return x0h[0, 0].float().cpu().numpy()
@torch.no_grad()
def infer_zero(m):
    m.eval(); x = torch.zeros_like(x0_t)
    for tc in range(49, -1, -1):
        e = m(x, xo_t, torch.full((1,), tc+1, device=dev, dtype=torch.long))
        x0h = (x - sch.sqrt_one_minus_alpha_bar[tc]*e) / sch.sqrt_alpha_bar[tc]
        x = (sch.sqrt_alpha_bar[tc-1]*x0h + sch.sqrt_one_minus_alpha_bar[tc-1]*e) if tc > 0 else x0h
    return x[0, 0].float().cpu().numpy()
rms = lambda a: float(np.sqrt(((a - X0)**2).mean()))
meas_err = rms(XO)

# ---------- (B) real checkpoints ----------
ck_res = []
for name in ['010','020','030','040','050','060','070','080','best']:
    path = MD + (f'checkpoint_{name}.pt' if name != 'best' else 'best_model.pt')
    st = torch.load(path, map_location=dev, weights_only=False)
    st = st['model_state_dict'] if isinstance(st, dict) and 'model_state_dict' in st else st
    m = UNet1D(cond_mode='step').to(dev); m.load_state_dict(st)
    g = probe(m); fin = infer_zero(m)
    ck_res.append(dict(name=name, probe_err=rms(g), final_err=rms(fin)))
    np.save(OUT + f'ck_{name}_probe.npy', g); np.save(OUT + f'ck_{name}_final.npy', fin)
    print(f'[B] ckpt {name}: one-call guess err {rms(g):.5f}   full inference err {rms(fin):.5f}', flush=True)
    del m; torch.cuda.empty_cache()
json.dump(dict(ck=ck_res, meas_err=meas_err), open(OUT + 'ck_res.json', 'w'), indent=1)

# ---------- (A) training from scratch ----------
xs, xo = [], []
for E in (1, 2, 3, 5, 10, 15, 20, 30, 50, 239, 511, 727, 1461, 2615, 3034):
    with h5py.File(EVD + f'eval_efficiency_{E}keV.h5') as f:
        N = f['waveforms_noisy'][1:151].astype(np.float64); T = f['clean_template'][:].astype(np.float64)
    sc = np.abs(N).max(axis=1, keepdims=True); xs.append(T[None]/sc); xo.append(N/sc)
POOL0 = f32(np.concatenate(xs))[:, None]; POOLO = f32(np.concatenate(xo))[:, None]
P = POOL0.shape[0]; print(f'[A] demo pool: {P} pairs', flush=True)
torch.manual_seed(0); m = UNet1D(cond_mode='step').to(dev)
diff = GaussianDiffusion(m, sch, loss_type='l1')
opt = torch.optim.AdamW(m.parameters(), lr=2e-4); scaler = torch.cuda.amp.GradScaler()
rng = np.random.default_rng(0)
# fixed validation batch for a smooth curve
vidx = rng.choice(P, 16, replace=False); vgen_seed = 123
@torch.no_grad()
def vloss():
    m.eval(); torch.manual_seed(vgen_seed)
    with torch.autocast('cuda', dtype=torch.float16):
        return diff.training_loss(POOL0[vidx].to(dev), POOLO[vidx].to(dev)).item()
B = 8; BUDGET_S = 330; SNAP = {0, 25, 100, 300, 1000}
losses, vcurve, snaps = [], [], {}
snaps[0] = probe(m); vcurve.append((0, vloss())); t0 = time.time(); step = 0
while time.time() - t0 < BUDGET_S:
    m.train(); idx = rng.choice(P, B, replace=False)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = diff.training_loss(POOL0[idx].to(dev), POOLO[idx].to(dev))
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
    step += 1; losses.append(loss.item())
    if step % 25 == 0: vcurve.append((step, vloss()))
    if step in SNAP: snaps[step] = probe(m)
    if step % 100 == 0: print(f'[A] step {step}  loss {np.mean(losses[-100:]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
snaps[step] = probe(m); last = step
np.savez(OUT + 'demo_train.npz', losses=np.array(losses), vsteps=np.array([v[0] for v in vcurve]),
         vloss=np.array([v[1] for v in vcurve]), **{f'snap_{k}': v for k, v in snaps.items()})
json.dump(dict(steps=last, snap_steps=sorted(snaps), snap_err={k: rms(v) for k, v in snaps.items()},
               first_loss=losses[0], last100=float(np.mean(losses[-100:])), meas_err=meas_err),
          open(OUT + 'demo_res.json', 'w'), indent=1)
print(f'[A] done: {last} steps; one-call guess err ' + ', '.join(f'{k}:{rms(v):.4f}' for k, v in sorted(snaps.items())), flush=True)
print('TRAIN_WATCH_DONE', flush=True)
