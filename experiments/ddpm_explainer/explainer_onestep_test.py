"""Can one network call replace the 50-step loop? Compare single-call x0_hat at t=1 / t=50
with various x_t against the full Gaussian-start --no_noise run. Weights only READ."""

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
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
dev = 'cuda'
sch = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(dev)
net = UNet1D(cond_mode='step').to(dev)
st = torch.load('/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt', map_location=dev, weights_only=False)
net.load_state_dict(st.get('model_state_dict', st)); net.eval()
EVD = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV/'
N = 32
def x0hat(x, xo, t):
    e = net(x, xo, torch.full((x.shape[0],), t, device=dev, dtype=torch.long))
    return (x - sch.sqrt_one_minus_alpha_bar[t-1] * e) / sch.sqrt_alpha_bar[t-1]
R = {}
for E in (15, 2615):
    with h5py.File(EVD + f'eval_efficiency_{E}keV.h5') as f:
        nz = f['waveforms_noisy'][:N].astype(np.float32); tm = f['clean_template'][:].astype(np.float32)
    s = np.abs(nz).max(1, keepdims=True); x0 = torch.tensor(tm[None] / s, device=dev)[:, None]
    xo = torch.tensor(nz / s, device=dev)[:, None]
    torch.manual_seed(0); g = torch.randn_like(xo)
    with torch.no_grad():
        cases = {
            't=1 一步，x_t = 测量': x0hat(xo, xo, 1),
            't=1 一步，x_t = 0': x0hat(torch.zeros_like(xo), xo, 1),
            't=50 一步，x_t = 高斯噪声': x0hat(g, xo, 50),
            't=50 一步，x_t = 0': x0hat(torch.zeros_like(xo), xo, 50)}
        x = g.clone()
        for i in reversed(range(50)):
            e = net(x, xo, torch.full((N,), i + 1, device=dev, dtype=torch.long))
            x = (1.0 / sch.sqrt_alpha[i]) * (x - sch.beta[i] / sch.sqrt_one_minus_alpha_bar[i] * e)
        cases['完整 50 步（高斯起步 + no_noise）'] = x
    amp = x0.amax(-1)
    def err(y):  # rms error / pulse height, and the part left after removing a constant offset
        r = (y - x0); tot = (r.pow(2).mean(-1).sqrt() / amp).mean().item()
        ac = ((r - r.mean(-1, keepdim=True)).pow(2).mean(-1).sqrt() / amp).mean().item()
        return round(100 * tot, 4), round(100 * ac, 4)
    R[f'{E}keV'] = {'测量本身': err(xo), **{k: err(v) for k, v in cases.items()}}
for E, d in R.items():
    print(E, '(误差 % 脉高：总, 去掉恒定偏移后)')
    for k, v in d.items(): print(f'   {k:28s} {v}')
json.dump(R, open('results/plots/ddpm_explainer/onestep_test.json', 'w'), ensure_ascii=False, indent=1)
print('ALL_DONE')
