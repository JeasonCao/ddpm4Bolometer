"""M-shot denoising of the QA example windows, compared to the saved 1-shot.

Takes the example windows picked by ``scripts/plot_triexp_inference.py``
(typical-energy singles + pileups at several Delta t), re-denoises their noisy
waveforms from the denoised-test HDF5 M times, and averages the shots. Each
shot is the same sampler as the saved 1-shot (deterministic DDPM unless
``--stochastic``); shots differ only in x_T (and, with ``--stochastic``, the
reverse-step noise).

Outputs (under ``--out_dir``):
  multishot_examples.h5           all shots, their mean, the saved 1-shot
  multishot_examples_single.png   rows: time (noisy / clean / 1-shot /
  multishot_examples_pileup.png   M-shot mean), residuals (1-shot, M-shot
                                  mean, +-1 sigma shot spread), PSD
  multishot_metrics.txt           per-window 1-shot vs M-shot metrics

Usage:
    python3 -u scripts/denoise_examples_multishot.py --scale_emb --film
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import build_ddpm_model
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.inference import compute_all_metrics
from scripts.plot_triexp_inference import (
    DEFAULT_H5, TYPICAL_E_KEV, PILEUP_DT_S, N_BASELINE,
    compute_psd, pick_single, pick_pileup)


def run_shots(diffusion, noisy, scale, M, stochastic, device, use_scale):
    """noisy (n, L) in V -> shots (M, n, L) in V."""
    x = torch.from_numpy((noisy / scale[:, None]).astype(np.float32))
    x = x.unsqueeze(1).to(device)
    sc = (torch.from_numpy(scale.astype(np.float32)).to(device)
          if use_scale else None)
    shots = []
    with torch.no_grad():
        for _ in range(M):
            out = diffusion.sample(x, scale=sc, stochastic=stochastic)
            shots.append(out.squeeze(1).cpu().numpy() * scale[:, None])
    return np.stack(shots)


def plot_rows(clean, noisy, one, shots, titles, fs, M, out_png):
    mean = shots.mean(0)
    std = shots.std(0)
    n = len(titles)
    fig, axes = plt.subplots(n, 3, figsize=(17, 2.6 * n), squeeze=False)
    t = np.arange(clean.shape[1]) / fs
    for r in range(n):
        c, y, o, m, s = (a[r] * 1e3 for a in (clean, noisy, one, mean, std))
        ax_t, ax_r, ax_p = axes[r]
        ax_t.plot(t, y, color='tab:red', lw=0.4, alpha=0.5, label='noisy')
        ax_t.plot(t, c, color='tab:blue', lw=1.2, label='clean')
        ax_t.plot(t, o, color='tab:green', lw=0.8, ls='--', label='1-shot')
        ax_t.plot(t, m, color='tab:purple', lw=0.8, ls=':',
                  label=f'{M}-shot mean')
        ax_t.set_ylabel('amplitude (mV)')
        ax_t.set_title(titles[r], fontsize=9)

        r1, rm = o - c, m - c
        ax_r.fill_between(t, rm - s, rm + s, color='tab:purple', alpha=0.2,
                          lw=0, label=r'$\pm1\sigma$ shot spread')
        ax_r.plot(t, r1, color='tab:green', lw=0.5, label='1-shot')
        ax_r.plot(t, rm, color='tab:purple', lw=0.5, label=f'{M}-shot mean')
        ax_r.axhline(0, color='k', lw=0.5)
        ax_r.set_ylabel('denoised - clean (mV)')
        rms = lambda a: np.sqrt(np.mean(a ** 2))
        ax_r.set_title(f'residual RMS: 1-shot {rms(r1):.3f}, {M}-shot '
                       f'{rms(rm):.3f} mV; mean shot sigma '
                       f'{np.mean(s):.3f} mV', fontsize=9)

        for x, color, lw, ls, label in (
                (y, 'tab:red', 1.2, '-', 'noisy'),
                (y - c, 'tab:gray', 0.8, '-', 'noise'),
                (c, 'tab:blue', 1.0, '-', 'clean'),
                (r1, 'tab:green', 0.8, '-', 'residual 1-shot'),
                (rm, 'tab:purple', 0.8, '-', f'residual {M}-shot')):
            f, p = compute_psd(x, fs)
            ax_p.loglog(f, p, color=color, lw=lw, ls=ls, label=label,
                        alpha=0.85)
        ax_p.set_ylabel('PSD (mV$^2$/Hz)')
        ax_p.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)
        for ax in (ax_t, ax_r):
            ax.grid(True, alpha=0.3)
        if r == 0:
            for ax, loc in ((ax_t, 'upper right'), (ax_r, 'upper right'),
                            (ax_p, 'lower left')):
                ax.legend(fontsize=7, loc=loc, frameon=False)
    for ax in axes[-1, :2]:
        ax.set_xlabel('time (s)')
    axes[-1, 2].set_xlabel('frequency (Hz)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--denoised_h5', default=DEFAULT_H5)
    p.add_argument('--model_path', default=None,
                   help='Default: model_path attr of the denoised h5')
    p.add_argument('--out_dir', default=None,
                   help='Default: <dir of denoised_h5>/qa_test')
    p.add_argument('--M', type=int, default=10)
    p.add_argument('--stochastic', action='store_true',
                   help='Stochastic DDPM shots (default: deterministic, as '
                        'the saved 1-shot)')
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--beta_1', type=float, default=1e-4)
    p.add_argument('--beta_T', type=float, default=0.5)
    p.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    p.add_argument('--scale_cond', action='store_true')
    p.add_argument('--scale_emb', action='store_true')
    p.add_argument('--film', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.denoised_h5), 'qa_test')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'multishot' + ('_stoch' if args.stochastic else '')

    with h5py.File(args.denoised_h5, 'r') as f:
        fs = float(f.attrs['f_sample'])
        model_path = args.model_path or f.attrs['model_path']
        E1 = f['energies_1'][:]
        E2 = np.nan_to_num(f['energies_2'][:])
        is_pu = f['is_pileup'][:].astype(bool)
        dt = f['onsets_2'][:] - f['onsets_1'][:]
        idx_s = pick_single(E1, is_pu, TYPICAL_E_KEV)
        idx_p = pick_pileup(E1, E2, dt, is_pu, PILEUP_DT_S)
        idx = idx_s + idx_p
        load = lambda k: np.stack([f[k][i] for i in idx]).astype(np.float64)
        clean, noisy, one = (load(k) for k in
                             ('waveforms_clean', 'waveforms_noisy', 'waveforms'))
        scale = f['scale'][:][idx].astype(np.float64)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1,
                                 beta_T=args.beta_T).to(device)
    model = build_ddpm_model(cond_mode=args.cond_mode, scale_cond=args.scale_emb,
                             film=args.film,
                             legacy_scale_cond=args.scale_cond).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=args.cond_mode)
    shots = run_shots(diffusion, noisy, scale, args.M, args.stochastic, device,
                      args.scale_cond or args.scale_emb)
    mean = shots.mean(0)

    with h5py.File(os.path.join(out_dir, f'{tag}_examples.h5'), 'w') as out:
        out.attrs.update(M=args.M, stochastic=args.stochastic, seed=args.seed,
                         model_path=model_path, source=args.denoised_h5)
        out['idx'] = np.array(idx)
        out['shots'] = shots.astype(np.float32)
        out['mean'] = mean.astype(np.float32)
        out['one_shot'] = one.astype(np.float32)
        out['clean'] = clean.astype(np.float32)
        out['noisy'] = noisy.astype(np.float32)

    ns = len(idx_s)
    titles_s = [f'single, E = {E1[i]:.0f} keV (window {i})' for i in idx_s]
    titles_p = [f'pileup, E1 = {E1[i]:.0f}, E2 = {E2[i]:.0f} keV, '
                f'$\\Delta t$ = {dt[i] * 1e3:.0f} ms (window {i})'
                for i in idx_p]
    plot_rows(clean[:ns], noisy[:ns], one[:ns], shots[:, :ns], titles_s, fs,
              args.M, os.path.join(out_dir, f'{tag}_examples_single.png'))
    plot_rows(clean[ns:], noisy[ns:], one[ns:], shots[:, ns:], titles_p, fs,
              args.M, os.path.join(out_dir, f'{tag}_examples_pileup.png'))

    keys = ('mae', 'mse', 'prd', 'snr', 'psnr', 'lsd', 'jasd')
    lines = [f'{"window":>7} {"pu":>3} {"E1":>6} {"metric":>7} '
             f'{"1-shot":>11} {f"{args.M}-shot":>11}']
    for k, i in enumerate(idx):
        m1 = compute_all_metrics(clean[k], one[k], clean[k] / scale[k],
                                 one[k] / scale[k])
        mm = compute_all_metrics(clean[k], mean[k], clean[k] / scale[k],
                                 mean[k] / scale[k])
        for a, b in ((m1, one[k]), (mm, mean[k])):
            a['bl_rms'] = float(np.sqrt(np.mean(b[:N_BASELINE] ** 2)))
        for key in keys + ('bl_rms',):
            scl = {'mae': 1e3, 'mse': 1e6, 'bl_rms': 1e3}.get(key, 1.0)
            lines.append(f'{i:7d} {int(is_pu[i]):3d} {E1[i]:6.0f} {key:>7} '
                         f'{m1[key] * scl:11.4g} {mm[key] * scl:11.4g}')
    lines.append('Units: MAE / bl_rms mV, MSE mV^2, PRD %, SNR/PSNR dB.')
    txt = os.path.join(out_dir, f'{tag}_metrics.txt')
    with open(txt, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {txt}')


if __name__ == '__main__':
    main()
