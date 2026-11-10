"""
2026-10-08: 2x2 sampler comparison, all four cells recomputed in one run.

  rule  x {DDIM eta=0, DDPM-mean-only (--no_noise)}
  x_T   x {randn, zeros}

The two factors the collaborator's question conflates are the per-step noise
(which --no_noise removes) and the initial x_T (which it does NOT -- sample()
hardcodes torch.randn at diffusion.py:179). Separating them needs all four
cells measured the same way, so nothing here reads the stored
waveforms_denoised_* fields: same dtype, same autocast, same event indices.

Both rules run the full T=50 steps, so all four cells cost the same 50
network evaluations per event.

Metrics (no tri-exp fit -- this is a screen, not a production number):
  amp : matched-filter projection, w = a*t_hat + c, 2-param closed form.
        std(a) * k -> FWHM in keV. Same estimator for all four cells.
  hf  : mean |FFT|^2/n over f >= 100 Hz, in V^2.
"""
import os, sys, time, json, argparse
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
from scripts.run_batch_inference import load_model

EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV'
MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
K_KEV_PER_V = 782.06          # tri-exp high-energy calibration for this group
F_SAMPLE = 1000.0
HF_CUT = 100.0

CELLS = [
    ('ddim_randn', 'ddim', False),
    ('ddim_zero',  'ddim', True),
    ('nonoise_randn', 'nonoise', False),
    ('nonoise_zero',  'nonoise', True),
]


@torch.no_grad()
def run_cell(diffusion, x_tilde, rule, zero_init, gen):
    """x_tilde: (B,1,L) already normalised. Returns (B,L) normalised output."""
    B, _, L = x_tilde.shape
    dev = x_tilde.device
    x_t = (torch.zeros(B, 1, L, device=dev) if zero_init
           else torch.randn(B, 1, L, device=dev, generator=gen))
    sch = diffusion.schedule

    if rule == 'ddim':
        tau = torch.arange(diffusion.T - 1, -1, -1, device=dev)
        for i, t_cur in enumerate(tau):
            cond = torch.full((B,), int(t_cur) + 1, device=dev, dtype=torch.long)
            eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
            x0 = (x_t - sch.sqrt_one_minus_alpha_bar[t_cur] * eps) / sch.sqrt_alpha_bar[t_cur]
            if i < len(tau) - 1:
                tn = tau[i + 1]
                x_t = sch.sqrt_alpha_bar[tn] * x0 + sch.sqrt_one_minus_alpha_bar[tn] * eps
            else:
                x_t = x0
    else:  # DPPM posterior mean, no stochastic term -- diffusion.sample(stochastic=False)
        for i in reversed(range(diffusion.T)):
            cond = torch.full((B,), i + 1, device=dev, dtype=torch.long)
            eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
            coeff = sch.beta[i] / sch.sqrt_one_minus_alpha_bar[i]
            x_t = (1.0 / sch.sqrt_alpha[i]) * (x_t - coeff * eps)
    return x_t.squeeze(1)


def metrics(den_V, tmpl):
    """den_V: (N,L) in volts. Returns (sigma_a_V, hf_power_V2)."""
    th = tmpl / np.abs(tmpl).max()
    tc = th - th.mean()
    w = den_V - den_V.mean(axis=1, keepdims=True)
    a = (w @ tc) / float(tc @ tc)
    n = den_V.shape[1]
    f = np.fft.rfftfreq(n, d=1.0 / F_SAMPLE)
    P = np.abs(np.fft.rfft(den_V, axis=1)) ** 2 / n
    hf = P[:, f >= HF_CUT].mean()
    return float(a.std()), float(hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energies', type=float, nargs='+',
                    default=[239, 511, 1461, 2615, 3034])
    ap.add_argument('--n_events', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = load_model(MODEL, dev)
    print(f'model loaded on {dev}, T={diffusion.T}', flush=True)

    rows = []
    for E in args.energies:
        path = os.path.join(EVAL, f'eval_efficiency_{int(E)}keV.h5')
        with h5py.File(path, 'r') as f:
            noisy = f['waveforms_noisy'][:args.n_events].astype(np.float64)
            tmpl = f['clean_template'][:].astype(np.float64)
        sigma_n, hf_n = metrics(noisy, tmpl)
        rec = {'E': E, 'noisy': {'fwhm_keV': 2.355 * sigma_n * K_KEV_PER_V, 'hf': hf_n}}

        for name, rule, zero in CELLS:
            gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
            out = np.empty_like(noisy)
            t0 = time.time()
            for s in range(0, len(noisy), args.batch):
                b = noisy[s:s + args.batch].astype(np.float32)
                sc = np.abs(b).max(axis=1, keepdims=True)
                sc = np.where(sc == 0, 1.0, sc)
                xt = torch.from_numpy(b / sc).unsqueeze(1).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                    enabled=dev.type == 'cuda'):
                    o = run_cell(diffusion, xt, rule, zero, gen)
                out[s:s + args.batch] = o.float().cpu().numpy() * sc
            sg, hf = metrics(out, tmpl)
            rec[name] = {'fwhm_keV': 2.355 * sg * K_KEV_PER_V, 'hf': hf,
                         'sec': time.time() - t0}
            print(f'  {E:6.0f} keV  {name:15s} FWHM={rec[name]["fwhm_keV"]:7.4f} keV '
                  f' hf={hf:.3e}  ({rec[name]["sec"]:.0f}s)', flush=True)
        rows.append(rec)

    out = args.out or '/tmp/nonoise_2x2.json'
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)

    print('\n' + '=' * 92)
    print(f'{"E[keV]":>7} {"noisy":>9} {"ddim_randn":>12} {"ddim_zero":>11} '
          f'{"nonoise_randn":>14} {"nonoise_zero":>13}')
    print('-' * 92)
    for r in rows:
        print(f'{r["E"]:7.0f} {r["noisy"]["fwhm_keV"]:9.4f} '
              f'{r["ddim_randn"]["fwhm_keV"]:12.4f} {r["ddim_zero"]["fwhm_keV"]:11.4f} '
              f'{r["nonoise_randn"]["fwhm_keV"]:14.4f} {r["nonoise_zero"]["fwhm_keV"]:13.4f}')
    print('\nNONOISE_2X2_DONE')


if __name__ == '__main__':
    main()
