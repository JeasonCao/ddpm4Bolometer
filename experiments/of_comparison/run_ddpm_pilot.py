"""Local DDPM inference on the pilot-run windows (cut_ddpm_windows.py), production sampler.

Sampler: Gaussian start x_T ~ N(0, I), DDPM posterior mean iterated with the stochastic term
dropped (= GaussianDiffusion.sample(stochastic=False)), fp32 without autocast; same loop as the
2026-10-08 full no_noise run (scripts/_archive_scratch_20260917/run_nonoise_full.py).

The input windows are read only. Each output file is a self-contained fit input (the datasets
run_fitting.py reads + waveforms_denoised_nonoise): nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5

Usage: python experiments/of_comparison/run_ddpm_pilot.py --runs 999996 999997
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, time, argparse
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
from scripts.run_batch_inference import load_model

D = '/mnt/d/storage/research/DDPM4bolometer'
MODEL = f'{D}/model_output_cloud/best_model.pt'
FIELD = 'waveforms_denoised_nonoise'
COPY = ['waveforms_noisy', 'snr_db', 'noise_rms', 'clean_template', 'event_id', 'onset_sample']


@torch.no_grad()
def nonoise(diffusion, x_tilde, gen):
    B, _, L = x_tilde.shape
    dev = x_tilde.device
    x_t = torch.randn(B, 1, L, device=dev, generator=gen)
    sch = diffusion.schedule
    for i in reversed(range(diffusion.T)):
        cond = torch.full((B,), i + 1, device=dev, dtype=torch.long)
        eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
        x_t = (1.0 / sch.sqrt_alpha[i]) * (x_t - sch.beta[i] / sch.sqrt_one_minus_alpha_bar[i] * eps)
    return x_t.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, nargs='+', default=[999996, 999997])
    ap.add_argument('--energies', type=int, nargs='+', default=[15, 2615])
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    dev = torch.device('cuda')
    diffusion = load_model(MODEL, dev)
    print(f'model on {dev}, T={diffusion.T}, fp32, batch={a.batch}', flush=True)
    for run in a.runs:
        src_dir, out_dir = f'{D}/eval_data_OFcompare/pilot_run{run}', f'{D}/nonoise_out/pilot_run{run}'
        os.makedirs(out_dir, exist_ok=True)
        for E in a.energies:
            fn = f'eval_efficiency_{E}keV.h5'; dst = f'{out_dir}/{fn}'
            if os.path.exists(dst):
                print(f'{run} {E} keV: exists, skipping', flush=True); continue
            with h5py.File(f'{src_dir}/{fn}', 'r') as f:
                noisy = f['waveforms_noisy'][:]
            N = len(noisy); out = np.empty_like(noisy)
            gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
            print(f'\n=== run {run}, {E} keV, N={N} ===', flush=True)
            t0 = time.time()
            for s in range(0, N, a.batch):
                b = noisy[s:s + a.batch].astype(np.float32)
                sc = np.abs(b).max(axis=1, keepdims=True); sc = np.where(sc == 0, 1.0, sc)
                o = nonoise(diffusion, torch.from_numpy(b / sc).unsqueeze(1).to(dev), gen)
                out[s:s + a.batch] = o.float().cpu().numpy() * sc
                if (s // a.batch) % 50 == 0:
                    done = min(s + a.batch, N); el = time.time() - t0
                    print(f'  [{done}/{N}] {el / done * 1000:.0f} ms/evt  ETA {el / done * (N - done) / 60:.0f} min', flush=True)
            with h5py.File(f'{src_dir}/{fn}', 'r') as f, h5py.File(dst + '.part', 'w') as g:
                for k in COPY:
                    g.create_dataset(k, data=f[k][:])
                f.copy('noise_params', g)
                for k, v in f.attrs.items():
                    g.attrs[k] = v
                g.create_dataset(FIELD, data=out, compression='gzip')
                g.attrs.update(sampler='ddpm_posterior_mean_no_stochastic', xT='randn', autocast='off_fp32', seed=a.seed)
            os.replace(dst + '.part', dst)
            print(f'  wrote {dst} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    print('PILOT_DDPM_DONE', flush=True)


if __name__ == '__main__':
    main()
