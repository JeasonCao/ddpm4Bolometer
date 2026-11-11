"""
2026-09-17: where does zero-init's REMAINING floor come from?

Decisive test: feed the network a NOISELESS input (x_tilde = clean template)
and run zero-init. A perfect denoiser would return it unchanged. Whatever
residual survives is pure model error, with no input noise involved at all.

Also sweeps input noise level (by scaling the injected noise) to see how the
remaining floor responds, and decomposes it into common-mode vs
event-varying.
"""
import sys, os, json
sys.path.insert(0, "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#")
import numpy as np, torch, h5py

MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV'
OUT = '/tmp/claude-1000/-home-wsl-0vbb/946b7f43-5893-49fc-8189-f0872f2eda18/scratchpad/hf_analysis'
FS, HF, N = 1000.0, 100.0, 48

from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion

device = torch.device('cuda')
schedule = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(device)
model = UNet1D(cond_mode='step').to(device)
st = torch.load(MODEL, map_location=device, weights_only=False)
if isinstance(st, dict) and 'model_state_dict' in st: st = st['model_state_dict']
model.load_state_dict(st); model.eval()
diff = GaussianDiffusion(model, schedule, cond_mode='step')
T = schedule.T


@torch.no_grad()
def zero_init(x_tilde, steps=50):
    B, C, L = x_tilde.shape
    tau = torch.linspace(T - 1, 0, steps, device=device).long()
    x_t = torch.zeros(B, 1, L, device=device)
    for i in range(len(tau)):
        tc = tau[i]
        cond = torch.full((B,), tc.item() + 1, device=device, dtype=torch.long)
        eps = diff._predict_noise(x_t, x_tilde, cond, None)
        x0 = (x_t - schedule.sqrt_one_minus_alpha_bar[tc] * eps) / schedule.sqrt_alpha_bar[tc]
        if i < len(tau) - 1:
            tn = tau[i + 1]
            x_t = schedule.sqrt_alpha_bar[tn] * x0 + torch.sqrt(torch.clamp(1 - schedule.alpha_bar[tn], min=0.)) * eps
        else:
            x_t = x0
    return x_t


def hf_rms_norm(r_norm):
    """rms of the >100 Hz white component, normalized units."""
    n = r_norm.shape[-1]
    f = np.fft.rfftfreq(n, 1 / FS)
    p = (np.abs(np.fft.rfft(r_norm, axis=-1)) ** 2 / n)[..., f >= HF].mean()
    return float(np.sqrt(p))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print('=== zero-init residual vs how noisy the input is ===')
    print('  noise_scale = 0 means the input IS the clean template (nothing to denoise).\n')
    for E in (239, 2615):
        with h5py.File(os.path.join(EVAL, f'eval_efficiency_{E}keV.h5')) as f:
            clean = f['clean_template'][:].astype(np.float64)
            noisy = f['waveforms_noisy'][:N].astype(np.float64)
            onset = int(round(float(f.attrs['onset_s']) * FS))
        noise = noisy - clean[None]           # the real injected noise, per event
        bs = slice(0, onset - 50)
        tmpl = clean - clean[bs].mean(); tt = (tmpl * tmpl).sum()
        K = E / tmpl.max()

        print(f'--- {E} keV ---')
        print(f"{'noise x':>9} {'sigma_norm':>11} {'HF rms (norm)':>14} {'common %':>9} {'std(a) keV':>11}")
        for ns in (1.0, 0.5, 0.25, 0.1, 0.0):
            inp = clean[None] + ns * noise
            scale = np.abs(inp).max(axis=1, keepdims=True)
            xt = torch.from_numpy((inp / scale).astype(np.float32)).unsqueeze(1).to(device)
            out = zero_init(xt).squeeze(1).cpu().numpy() * scale
            r = out - clean[None]
            rn = r / scale
            sig_norm = float(((inp - clean[None]) / scale)[:, bs].std(axis=1).mean())
            rmean = r.mean(axis=0)
            common = float((rmean ** 2).mean() / (r ** 2).mean())
            a = (r * tmpl[None]).sum(axis=1) / tt
            row = dict(E=E, noise_scale=ns, sigma_norm=sig_norm, hf_rms_norm=hf_rms_norm(rn),
                       common=common, std_a_keV=float(a.std() * K))
            rows.append(row)
            print(f"{ns:9.2f} {sig_norm:11.5f} {row['hf_rms_norm']:14.3e} {common:8.1%} {row['std_a_keV']:11.4f}")
        print()

    with open(os.path.join(OUT, 'zeroinit_floor_origin.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    print('=== reading ===')
    print('  If the floor persists at noise x 0, it is pure model error (nothing to do with input noise).')
    print('  If it scales with sigma_norm, it is the network mishandling the real noise.')


if __name__ == '__main__':
    main()
