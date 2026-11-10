"""
2026-09-17: sweep the two knobs that control how much random field the
sampler injects.

  A) eta=0, x_T = s * randn, s from 1.0 down to 0   -- is 0 special, or is
     "small enough" already good? (user's question: could we use a small
     white noise instead of exactly zero?)
  B) x_T = randn, eta from 1.0 down to 0            -- does the floor track
     the per-step injected magnitude?

Metric that matters: std(a), the event-to-event amplitude scatter from
projecting the residual on the template (this is what sets FWHM), plus the
HF residual for reference.
"""
import sys, os, json
sys.path.insert(0, "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#")
import numpy as np, torch, h5py
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion

MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV'
OUT = '/tmp/claude-1000/-home-wsl-0vbb/946b7f43-5893-49fc-8189-f0872f2eda18/scratchpad/hf_analysis'
ENERGIES = [15, 2615]
N_EV, FS, K = 64, 1000.0, 780.73

device = torch.device('cuda')
schedule = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(device)
model = UNet1D(cond_mode='step').to(device)
st = torch.load(MODEL, map_location=device, weights_only=False)
if isinstance(st, dict) and 'model_state_dict' in st: st = st['model_state_dict']
model.load_state_dict(st); model.eval()
diff = GaussianDiffusion(model, schedule, cond_mode='step')
T = schedule.T


@torch.no_grad()
def sample(x_tilde, eta, xT_scale, seed=77):
    g = torch.Generator(device=device).manual_seed(seed)
    B, C, L = x_tilde.shape
    tau = torch.arange(T - 1, -1, -1, device=device)
    x_t = torch.zeros(B, 1, L, device=device) if xT_scale == 0 else \
        xT_scale * torch.randn(B, 1, L, device=device, generator=g)
    for i in range(len(tau)):
        tc = tau[i]
        cond = torch.full((B,), tc.item() + 1, device=device, dtype=torch.long)
        eps = diff._predict_noise(x_t, x_tilde, cond, None)
        ab_t = schedule.alpha_bar[tc]
        x0 = (x_t - schedule.sqrt_one_minus_alpha_bar[tc] * eps) / schedule.sqrt_alpha_bar[tc]
        if i < len(tau) - 1:
            tn = tau[i + 1]; ab_n = schedule.alpha_bar[tn]
            sig = eta * torch.sqrt((1 - ab_n) / (1 - ab_t) * (1 - ab_t / ab_n))
            x_t = schedule.sqrt_alpha_bar[tn] * x0 + torch.sqrt(torch.clamp(1 - ab_n - sig ** 2, min=0.)) * eps
            if eta > 0:
                x_t = x_t + sig * torch.randn(x_t.shape, device=device, generator=g)
        else:
            x_t = x0
    return x_t


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for E in ENERGIES:
        with h5py.File(os.path.join(EVAL, f'eval_efficiency_{E}keV.h5')) as f:
            clean = f['clean_template'][:].astype(np.float64)
            noisy = f['waveforms_noisy'][:N_EV].astype(np.float64)
            onset = int(round(float(f.attrs['onset_s']) * FS))
        scale = np.abs(noisy).max(axis=1, keepdims=True)
        x = torch.from_numpy((noisy / scale).astype(np.float32)).unsqueeze(1).to(device)
        bs = slice(0, onset - 50)
        tmpl = clean - clean[bs].mean(); tt = (tmpl * tmpl).sum()
        n = len(clean); fr = np.fft.rfftfreq(n, 1 / FS)
        hf = lambda r: float((np.abs(np.fft.rfft(r, axis=-1)) ** 2 / n)[..., fr >= 100].mean())

        def metrics(out):
            r = out - clean[None]
            a = (r * tmpl[None]).sum(axis=1) / tt
            rmean = r.mean(axis=0)
            return dict(hf=hf(r), std_a_keV=float(a.std() * K), mean_a_keV=float(a.mean() * K),
                        common=float((rmean ** 2).mean() / (r ** 2).mean()))

        # reference
        rn = metrics(np.repeat(noisy[:1], 0, 0)) if False else None
        r0 = clean[None]
        a0 = ((noisy - clean[None]) * tmpl[None]).sum(axis=1) / tt
        print(f'\n===== {E} keV  (不降噪: std(a)={a0.std()*K:.4f} keV) =====')

        print('\n  A) eta=0, x_T = s * randn   —— 起点白噪声幅度扫描')
        print(f"{'s':>6} {'HF残差':>11} {'std(a) keV':>11} {'共模占比':>9}")
        for s in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0):
            m = metrics(sample(x, 0.0, s).squeeze(1).cpu().numpy() * scale)
            m.update(E=E, mode='xT_scan', eta=0.0, xT=s); rows.append(m)
            print(f"{s:6.2f} {m['hf']:11.3e} {m['std_a_keV']:11.4f} {m['common']:8.1%}")

        print('\n  B) x_T = randn, eta 扫描   —— 每步注入量扫描')
        print(f"{'eta':>6} {'HF残差':>11} {'std(a) keV':>11} {'共模占比':>9}")
        for eta in (1.0, 0.7, 0.5, 0.3, 0.1, 0.0):
            m = metrics(sample(x, eta, 1.0).squeeze(1).cpu().numpy() * scale)
            m.update(E=E, mode='eta_scan', eta=eta, xT=1.0); rows.append(m)
            print(f"{eta:6.2f} {m['hf']:11.3e} {m['std_a_keV']:11.4f} {m['common']:8.1%}")

    with open(os.path.join(OUT, 'knob_sweep.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print('\nsaved knob_sweep.json')


if __name__ == '__main__':
    main()
