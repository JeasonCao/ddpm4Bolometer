"""Compare reverse samplers on a subset of a denoised test set.

Diagnostic for "is the residual from the sampler's discretization or from the
network?": re-denoises every ``--stride``-th window of the denoised-test HDF5
(same noisy input) with deterministic DDPM (the protocol of
``scripts/denoise_test_set.py``) and DDIM eta=0 at several step counts, for
one or more checkpoints. If DDIM / fewer steps barely change the residual,
the error is in the network, not the step size, and a larger T is unlikely to
help.

Outputs (under ``--out_dir``):
  samplers.h5        idx, clean, noisy, scale, and one dataset per
                     <model>/<sampler> with the denoised windows (V)
  samplers_summary.txt  metrics per model x sampler x energy group

Usage:
    python3 -u scripts/compare_samplers.py --scale_emb --film \\
        --models ep95=.../best_model_ep095.pt ep190=.../best_model_ep190.pt
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import build_ddpm_model
from src.ddpm.diffusion import GaussianDiffusion
from scripts.plot_triexp_inference import DEFAULT_H5, N_BASELINE

# name -> (kind, steps)
SAMPLERS = {
    'ddpm_det50': ('ddpm', None),
    'ddim50': ('ddim', 50),
    'ddim25': ('ddim', 25),
    'ddim10': ('ddim', 10),
}
E_GROUPS = ((30, 215, '30-215 keV'), (215, 1537, '215-1537 keV'),
            (1537, 11000, '>1537 keV'), (30, 11000, 'all'))


def denoise(diffusion, noisy, scale, kind, steps, device, use_scale, bs, seed):
    torch.manual_seed(seed)
    out = np.empty_like(noisy)
    for s in range(0, len(noisy), bs):
        x = torch.from_numpy((noisy[s:s + bs] / scale[s:s + bs, None])
                             .astype(np.float32)).unsqueeze(1).to(device)
        sc = (torch.from_numpy(scale[s:s + bs].astype(np.float32)).to(device)
              if use_scale else None)
        with torch.no_grad():
            if kind == 'ddim':
                y = diffusion.ddim_sample(x, scale=sc, steps=steps, eta=0.0)
            else:
                y = diffusion.sample(x, scale=sc, stochastic=False)
        out[s:s + bs] = y.squeeze(1).cpu().numpy() * scale[s:s + bs, None]
    return out


def window_metrics(clean, den, scale):
    res = den - clean
    sig = np.sum((clean - clean.mean(1, keepdims=True)) ** 2, axis=1)
    return {
        'mae_uV': np.mean(np.abs(res), axis=1) * 1e6,
        'prd_pct': np.sqrt(np.sum(res ** 2, axis=1) / sig) * 100.0,
        'snr_dB': 10 * np.log10(np.sum(clean ** 2, axis=1)
                                / np.sum(res ** 2, axis=1)),
        'bl_rms_uV': np.sqrt(np.mean(den[:, :N_BASELINE] ** 2, axis=1)) * 1e6,
        'tail_bias_pct': np.mean(res[:, 4000:9000], axis=1) / scale * 100.0,
        'bl_anom': (np.sqrt(np.mean(den[:, :N_BASELINE] ** 2, axis=1))
                    / scale > 1e-3).astype(float),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoised_h5', default=DEFAULT_H5)
    p.add_argument('--models', nargs='+', required=True,
                   help='name=path pairs')
    p.add_argument('--samplers', nargs='+', default=list(SAMPLERS))
    p.add_argument('--out_dir', default=None,
                   help='Default: <dir of denoised_h5>/qa_samplers')
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--beta_1', type=float, default=1e-4)
    p.add_argument('--beta_T', type=float, default=0.5)
    p.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    p.add_argument('--scale_cond', action='store_true')
    p.add_argument('--scale_emb', action='store_true')
    p.add_argument('--film', action='store_true')
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.denoised_h5), 'qa_samplers')
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.denoised_h5, 'r') as f:
        n = f['waveforms'].shape[0]
        idx = np.arange(0, n, args.stride)
        load = lambda k: np.stack([f[k][i] for i in idx]).astype(np.float64)
        clean, noisy = load('waveforms_clean'), load('waveforms_noisy')
        scale = f['scale'][:][idx].astype(np.float64)
        E1 = f['energies_1'][:][idx]
        E2 = np.nan_to_num(f['energies_2'][:][idx])
        is_pu = f['is_pileup'][:][idx].astype(bool)
    E_win = np.where(is_pu, E1 + E2, E1)
    print(f'{len(idx)} windows (stride {args.stride})', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1,
                                 beta_T=args.beta_T).to(device)
    use_scale = args.scale_cond or args.scale_emb

    out_h5 = os.path.join(out_dir, 'samplers.h5')
    metrics = {}
    with h5py.File(out_h5, 'w') as out:
        out['idx'] = idx
        out['clean'] = clean.astype(np.float32)
        out['noisy'] = noisy.astype(np.float32)
        out['scale'] = scale
        for spec in args.models:
            name, path = spec.split('=', 1)
            model = build_ddpm_model(cond_mode=args.cond_mode,
                                     scale_cond=args.scale_emb, film=args.film,
                                     legacy_scale_cond=args.scale_cond).to(device)
            model.load_state_dict(torch.load(path, map_location=device,
                                             weights_only=True))
            model.eval()
            diffusion = GaussianDiffusion(model, schedule,
                                          cond_mode=args.cond_mode)
            for sname in args.samplers:
                kind, steps = SAMPLERS[sname]
                t0 = time.time()
                den = denoise(diffusion, noisy, scale, kind, steps, device,
                              use_scale, args.batch_size, args.seed)
                out[f'{name}/{sname}'] = den.astype(np.float32)
                metrics[(name, sname)] = window_metrics(clean, den, scale)
                print(f'{name} {sname}: {time.time() - t0:.0f}s', flush=True)

    keys = ('mae_uV', 'prd_pct', 'snr_dB', 'bl_rms_uV', 'tail_bias_pct',
            'bl_anom')
    lines = []
    for lo, hi, gname in E_GROUPS:
        sel = (E_win >= lo) & (E_win < hi)
        lines.append(f'== {gname} (n={int(sel.sum())}); median, '
                     f'bl_anom = fraction, tail_bias = mean ==')
        lines.append(f'{"model":>6} {"sampler":>11} ' + ' '.join(
            f'{k:>14}' for k in keys))
        for (name, sname), m in metrics.items():
            vals = []
            for k in keys:
                v = m[k][sel]
                vals.append(np.mean(v) if k in ('bl_anom', 'tail_bias_pct')
                            else np.median(v))
            lines.append(f'{name:>6} {sname:>11} ' + ' '.join(
                f'{v:14.4g}' for v in vals))
        lines.append('')
    lines.append('tail_bias_pct: mean residual over 4-9 s in % of scale.')
    txt = os.path.join(out_dir, 'samplers_summary.txt')
    with open(txt, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out_h5}\nwrote {txt}')


if __name__ == '__main__':
    main()
