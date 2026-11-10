"""
2026-09-17: sharp test of the "cancellation-error" mechanism.

Mechanism claim: the residual floor is the sampler's OWN injected noise,
imperfectly cancelled by the network (0.1-1.4% relative error per step).
For eta=0 the mid-trajectory cancellation errors (t~20-35, where
noise_level x rel_error peaks) are carried deterministically to the end.

Counter-intuitive prediction: FEWER DDIM steps -> fewer error injections
-> LOWER floor for eta=0. (Opposite of "more steps is better".)

Also sweeps x_T scaling (1.0 / 0.5 / 0.25 / 0) x steps, to see whether
zero-init's advantage is really "nothing to cancel".

Metrics: HF residual power, and two amplitude-resolution proxies
(template-matched, max-baseline) in keV.
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
N_EV, FS, HF = 24, 1000.0, 100.0

device = torch.device('cuda')
schedule = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(device)
model = UNet1D(cond_mode='step').to(device)
st = torch.load(MODEL, map_location=device, weights_only=False)
if isinstance(st, dict) and 'model_state_dict' in st: st = st['model_state_dict']
model.load_state_dict(st); model.eval()
diff = GaussianDiffusion(model, schedule, cond_mode='step')
T = schedule.T


@torch.no_grad()
def sample(x_tilde, steps, eta, xT_scale, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    B, C, L = x_tilde.shape
    tau = torch.linspace(T - 1, 0, steps, device=device).long()
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
            if eta > 0: x_t = x_t + sig * torch.randn(x_t.shape, device=device, generator=g)
        else:
            x_t = x0
    return x_t


def hf_pow(x):
    n = x.shape[-1]; f = np.fft.rfftfreq(n, 1 / FS)
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2 / n)[..., f >= HF].mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    allrows = []
    for E in ENERGIES:
        with h5py.File(os.path.join(EVAL, f'eval_efficiency_{E}keV.h5')) as f:
            clean = f['clean_template'][:].astype(np.float64)
            noisy = f['waveforms_noisy'][:N_EV].astype(np.float64)
            onset = int(round(float(f.attrs['onset_s']) * FS))
        scale = np.abs(noisy).max(axis=1, keepdims=True)
        x = torch.from_numpy((noisy / scale).astype(np.float32)).unsqueeze(1).to(device)
        bs = slice(0, onset - 50)
        tmpl = clean - clean[bs].mean(); k = E / tmpl.max()
        amp_t = lambda w: ((w - w[:, bs].mean(axis=1, keepdims=True)) * tmpl).sum(axis=1) / (tmpl ** 2).sum()
        amp_m = lambda w: (w - w[:, bs].mean(axis=1, keepdims=True))[:, onset:onset + 400].max(axis=1)
        fw = lambda a: 2.355 * np.std(a) * k

        print(f'\n===== {E} keV (N={N_EV}) =====')
        print(f"  noisy baseline:              HF={hf_pow(noisy-clean[None]):.3e}  "
              f"FWHM_tmpl={fw(amp_t(noisy)):7.3f}  FWHM_maxbase={fw(amp_m(noisy)):7.3f} keV")
        print(f"{'eta':>4} {'x_T':>6} {'steps':>6} {'HF resid':>11} {'FWHM_tmpl':>10} {'FWHM_maxb':>10}")
        for eta in (0.0, 1.0):
            for xT in (1.0, 0.25, 0.0):
                if eta > 0 and xT == 0.0:
                    pass  # still meaningful: zero start but noise injected along the way
                for steps in (5, 10, 20, 50):
                    o = sample(x, steps, eta, xT, seed=11).squeeze(1).cpu().numpy() * scale
                    r = o - clean[None]
                    row = dict(E=E, eta=eta, xT=xT, steps=steps, hf=float(hf_pow(r)),
                               fwhm_t=float(fw(amp_t(o))), fwhm_m=float(fw(amp_m(o))))
                    allrows.append(row)
                    print(f"{eta:4.0f} {xT:6.2f} {steps:6d} {row['hf']:11.3e} {row['fwhm_t']:10.3f} {row['fwhm_m']:10.3f}")
    with open(os.path.join(OUT, 'steps_init_sweep.json'), 'w') as f:
        json.dump(allrows, f, indent=1)


if __name__ == '__main__':
    main()
