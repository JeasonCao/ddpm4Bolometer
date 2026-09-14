"""
Batch DDPM inference — 3-way comparison variant (added 2026-09-14).

For each eval HDF5, produces THREE denoised versions of every waveform and
writes them all into the same file (doesn't touch the plain
`waveforms_denoised` field `run_batch_inference.py` uses, so this can run
against the same files without conflicting):

  waveforms_denoised_cold       — production default: DDIM, eta=0, 1 shot,
                                   full T steps (cold start)
  waveforms_denoised_cold5shot  — DDIM, eta=1.0 (stochastic), 5 independent
                                   shots averaged, full T steps
  waveforms_denoised_warm5shot  — SDEdit-style warm start (see
                                   GaussianDiffusion.ddim_sample_warmstart in
                                   src/ddpm/diffusion.py) at an
                                   automatically-chosen t_start, eta=1.0,
                                   5 shots averaged

t_start is chosen per file automatically (no hand-tuned lookup table — this
is the "production-ready auto t_start selection" the 2026-09-11 TODO asked
for): match sqrt(1-alpha_bar[t]) to this file's noise_rms / typical
per-window peak scale, using the file's own noiseless clean_template amplitude
(max_baseline_amplitude) as the scale reference. See the 2026-09-10/11 diary
for why this is only an approximation (SDEdit re-noising doesn't perfectly
reproduce the true forward-diffusion posterior) — good enough as a starting
point for the comparison this script exists to run.

Always sub-batches (default 256, override with --batch_size) to avoid the
attention-layer OOM/thrash bug hit twice already this project (09-09 diary:
batch=256 OOM'd on a 12GB 3080Ti; 09-10: an unbatched N=300 single-call
experiment pinned a 3060 at ~100% for 5 CPU-hours before being killed). 256
is fine on a 24GB card (4090/4090D) per the batch=256 4090D precedent
recorded in the original training/inference scripts — lower it if you're on
a 12GB-class card instead.

Usage
-----
python -u scripts/run_batch_inference_compare.py \
    --model_path /root/autodl-tmp/best_model.pt \
    --input_dir  /root/autodl-tmp/eval_data_holdout/efficiency_1p382mV \
    --batch_size 256 --n_shots 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime
import h5py
import numpy as np
import torch

from src.ddpm.unet import UNet1D
from src.ddpm.unet_cond import UNet1DScaleCond
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion
from src.basics.amplitude import max_baseline_amplitude


def load_model(model_path, device, scale_cond=False, T=50, beta_1=1e-4, beta_T=0.5,
               cond_mode='step'):
    schedule = DiffusionSchedule(T=T, beta_1=beta_1, beta_T=beta_T).to(device)
    model_cls = UNet1DScaleCond if scale_cond else UNet1D
    model = model_cls(cond_mode=cond_mode).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return GaussianDiffusion(model, schedule, cond_mode=cond_mode)


def choose_t_start(schedule, sigma_normalized):
    """Analytical t_start guess: closest step whose sqrt(1-alpha_bar[t])
    matches the observed (per-window-normalized) noise std. See docstring."""
    target = torch.sqrt(1.0 - schedule.alpha_bar).cpu().numpy()
    return int(np.argmin(np.abs(target - sigma_normalized)))


def denoise_batched(diffusion, noisy_all, device, mode, t_start, n_shots, eta, batch_size):
    """mode: 'cold' (eta=0, 1 shot, full T) / 'stochastic' (eta=1.0, n_shots,
    full T, t_start ignored) / 'warmstart' (eta=1.0, n_shots, t_start)."""
    N = noisy_all.shape[0]
    out = np.empty_like(noisy_all)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = noisy_all[start:end]
        scale_vals = np.abs(chunk).max(axis=1, keepdims=True)
        scale_vals = np.where(scale_vals == 0, 1.0, scale_vals)
        x = torch.from_numpy((chunk / scale_vals).astype(np.float32)).unsqueeze(1).to(device)

        acc = None
        shots = 1 if mode == 'cold' else n_shots
        for _ in range(shots):
            with torch.no_grad():
                if mode == 'cold':
                    o = diffusion.ddim_sample(x, eta=0.0)
                elif mode == 'stochastic':
                    o = diffusion.ddim_sample(x, eta=eta)
                else:  # warmstart
                    o = diffusion.ddim_sample_warmstart(x, t_start=t_start,
                                                         steps=t_start + 1, eta=eta)
            o_np = o.squeeze(1).cpu().numpy()
            acc = o_np if acc is None else acc + o_np
        out[start:end] = (acc / shots) * scale_vals
        torch.cuda.empty_cache()
    return out


def _append_timing_csv(csv_path, row, fieldnames):
    """Structured per-file-per-method timing log (added 2026-09-14), so a
    real cloud GPU's throughput is captured as reusable data, not just log
    text -- doubles as the empirical data for the long-pending 'inference
    speed evaluation' TODO item, no extra runs needed to get it."""
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        if is_new:
            f.write(','.join(fieldnames) + '\n')
        f.write(','.join(str(row[k]) for k in fieldnames) + '\n')


def process_file(path, diffusion, device, batch_size, n_shots, overwrite, max_events,
                 timing_csv=None, gpu_name=None):
    with h5py.File(path, 'r') as f:
        if 'waveforms_denoised_warm5shot' in f and not overwrite:
            print(f'  Already has 3-way results, skipping: {path}')
            return
        noisy = f['waveforms_noisy'][:]
        clean = f['clean_template'][:]
        energy_kev = float(f.attrs['energy_kev'])
        noise_rms = float(np.median(f['noise_rms'][:]))

    if max_events > 0:
        noisy = noisy[:max_events]
    N = noisy.shape[0]

    true_amp_V = max_baseline_amplitude(clean)
    sigma_normalized = noise_rms / true_amp_V
    t_start = choose_t_start(diffusion.schedule, sigma_normalized)

    print(f'Processing {os.path.basename(path)}  (E={energy_kev:.0f} keV, N={N}, '
          f'noise_rms={noise_rms:.5f} V, sigma_norm={sigma_normalized:.4f} -> t_start={t_start})')

    fieldnames = ['timestamp', 'gpu_name', 'file', 'energy_kev', 'method', 'n_events',
                 't_start', 'n_shots', 'batch_size', 'elapsed_s', 'events_per_s']

    results = {}
    for label, mode in [('cold', 'cold'), ('cold5shot', 'stochastic'), ('warm5shot', 'warmstart')]:
        t0 = time.time()
        out = denoise_batched(diffusion, noisy, device, mode, t_start, n_shots, 1.0, batch_size)
        elapsed = time.time() - t0
        results[label] = out
        events_per_s = N / elapsed
        print(f'  [{label:10s}] {elapsed:6.1f}s  ({events_per_s:.1f} events/s)')
        if timing_csv:
            _append_timing_csv(timing_csv, dict(
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                gpu_name=gpu_name or 'unknown',
                file=os.path.basename(path), energy_kev=energy_kev, method=label,
                n_events=N, t_start=(t_start if label == 'warm5shot' else ''),
                n_shots=(1 if label == 'cold' else n_shots), batch_size=batch_size,
                elapsed_s=round(elapsed, 2), events_per_s=round(events_per_s, 2),
            ), fieldnames)

    with h5py.File(path, 'a') as f:
        for label in ('cold', 'cold5shot', 'warm5shot'):
            key = f'waveforms_denoised_{label}'
            if key in f:
                del f[key]
            f.create_dataset(key, data=results[label], compression='gzip')
        f.attrs['inference_3way_t_start'] = t_start
        f.attrs['inference_3way_n_shots'] = n_shots
        f.attrs['inference_3way_batch_size'] = batch_size
        f.attrs['inference_3way_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'  Saved 3 datasets -> {path}')


def main():
    ap = argparse.ArgumentParser(description='3-way (cold/cold5shot/warm5shot) batch inference')
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--scale_cond', action='store_true')
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--beta_1', type=float, default=1e-4)
    ap.add_argument('--beta_T', type=float, default=0.5)
    ap.add_argument('--cond_mode', default='step', choices=['step', 'sqrt_ab'])
    input_grp = ap.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input')
    input_grp.add_argument('--input_dir')
    ap.add_argument('--n_shots', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=256,
                    help='256 is safe on a 24GB card (4090/4090D); drop to '
                         '~64 on a 12GB card (see docstring for why)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max_events', type=int, default=0)
    ap.add_argument('--timing_csv', default=None,
                    help='Append per-file-per-method timing rows here (default: '
                         '<output-location>/timing_log.csv next to the input). '
                         'Also feeds the pending "inference speed evaluation" TODO '
                         '-- no separate benchmark run needed once this exists.')
    ap.add_argument('--gpu_name', default=None,
                    help='Label for the timing CSV; auto-detected from CUDA if omitted')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    gpu_name = args.gpu_name or (torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    print(f'GPU: {gpu_name}')
    diffusion = load_model(args.model_path, device, scale_cond=args.scale_cond,
                           T=args.T, beta_1=args.beta_1, beta_T=args.beta_T,
                           cond_mode=args.cond_mode)

    if args.input:
        files = [args.input]
        default_timing_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        files = sorted(os.path.join(args.input_dir, f)
                       for f in os.listdir(args.input_dir) if f.endswith('.h5'))
        default_timing_dir = os.path.abspath(args.input_dir)
    timing_csv = args.timing_csv or os.path.join(default_timing_dir, 'timing_log.csv')
    print(f'Timing log: {timing_csv}')

    t0_all = time.time()
    for path in files:
        process_file(path, diffusion, device, args.batch_size, args.n_shots,
                    args.overwrite, args.max_events,
                    timing_csv=timing_csv, gpu_name=gpu_name)
    print(f'\nAll done. Total: {(time.time()-t0_all)/60:.1f} min')


if __name__ == '__main__':
    main()
