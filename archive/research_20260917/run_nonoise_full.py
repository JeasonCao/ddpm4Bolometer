"""
2026-10-08: full-statistics --no_noise run at one high- and one low-energy
point, for a like-for-like comparison against the production zero-init.

  sampler : DPPM posterior mean, stochastic term deleted (= sample(stochastic=False))
  x_T     : torch.randn  (the point of the test -- no_noise is supposed to
            damp the random start out by itself)
  dtype   : fp32, autocast OFF.

The autocast choice matters and is not cosmetic: measured 2026-10-08, fp16
costs 8.8% of the post-denoising resolution for zero-init (and ~0 for
cold-start, whose residual is 13x larger). The production zero-init numbers
this is being compared against were produced without fp16, so running this
under autocast would hand no_noise an artificial 9% penalty.

Writes a minimal separate file per energy -- the eval HDF5s are not touched
here; merging is a separate, verified step.
"""
import os, sys, time, argparse
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
from scripts.run_batch_inference import load_model

EVAL = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV'
OUT  = '/mnt/d/storage/research/DDPM4bolometer/nonoise_out/efficiency_1p382mV'
MODEL = '/mnt/d/storage/research/DDPM4bolometer/model_output_cloud/best_model.pt'
FIELD = 'waveforms_denoised_nonoise'


@torch.no_grad()
def nonoise(diffusion, x_tilde, gen):
    """DDPM posterior mean iterated, no stochastic term, from x_T ~ N(0,I)."""
    B, _, L = x_tilde.shape
    dev = x_tilde.device
    x_t = torch.randn(B, 1, L, device=dev, generator=gen)
    sch = diffusion.schedule
    for i in reversed(range(diffusion.T)):
        cond = torch.full((B,), i + 1, device=dev, dtype=torch.long)
        eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
        coeff = sch.beta[i] / sch.sqrt_one_minus_alpha_bar[i]
        x_t = (1.0 / sch.sqrt_alpha[i]) * (x_t - coeff * eps)
    return x_t.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energies', type=int, nargs='+', default=[15, 2615])
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cuda')
    diffusion = load_model(MODEL, dev)
    print(f'model on {dev}, T={diffusion.T}, fp32 (autocast OFF), batch={args.batch}',
          flush=True)

    for E in args.energies:
        src = os.path.join(EVAL, f'eval_efficiency_{E}keV.h5')
        dst = os.path.join(OUT, f'eval_efficiency_{E}keV.h5')
        if os.path.exists(dst):
            print(f'{E} keV: output exists, skipping'); continue
        with h5py.File(src, 'r') as f:
            noisy = f['waveforms_noisy'][:].astype(np.float64)
            attrs = {k: f.attrs[k] for k in ('energy_kev', 'n_events')}
        N = len(noisy)
        print(f'\n=== {E} keV, N={N} ===', flush=True)

        gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
        out = np.empty_like(noisy)
        t0 = time.time()
        for s in range(0, N, args.batch):
            b = noisy[s:s + args.batch].astype(np.float32)
            sc = np.abs(b).max(axis=1, keepdims=True)
            sc = np.where(sc == 0, 1.0, sc)
            xt = torch.from_numpy(b / sc).unsqueeze(1).to(dev)
            o = nonoise(diffusion, xt, gen)          # fp32, no autocast
            out[s:s + args.batch] = o.float().cpu().numpy() * sc
            if (s // args.batch) % 25 == 0:
                el = time.time() - t0
                done = min(s + args.batch, N)
                eta = el / max(done, 1) * (N - done)
                print(f'  [{done}/{N}] {100*done/N:5.1f}%  '
                      f'{el/max(done,1)*1000:.0f} ms/evt  ETA {eta/60:.0f} min', flush=True)

        part = dst + '.part'
        with h5py.File(part, 'w') as g:
            g.create_dataset(FIELD, data=out, compression='gzip')
            for k, v in attrs.items():
                g.attrs[k] = v
            g.attrs['sampler'] = 'ddpm_posterior_mean_no_stochastic'
            g.attrs['xT'] = 'randn'
            g.attrs['autocast'] = 'off_fp32'
            g.attrs['seed'] = args.seed
        os.replace(part, dst)
        print(f'  wrote {dst}  ({(time.time()-t0)/60:.1f} min)', flush=True)

    print('\nNONOISE_FULL_DONE')


if __name__ == '__main__':
    main()
