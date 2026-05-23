"""
Inference and evaluation for trained CDiffSD pulse denoiser.

Compares two CDiffSD inference modes from the paper:
  - Direct: single-pass x_0 prediction (1 U-Net call)
  - Sampling: iterative reverse process (T U-Net calls)

Both start from x_tilde (the noisy observation).

Usage:
    python -u -m src.cdiffsd.inference \
        --model_path /media/AVFD/yunshancheng/cuore/cdiffsd_runs/best_model.pt \
        --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_000.h5 \
        --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_000.h5 \
        --output qa_cdiffsd_inference.png \
        --n 5
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.inference import compute_all_metrics
from src.cdiffsd.cold_diffusion import ColdDiffusion
from src.cdiffsd.dataset import ColdDiffusionDataset


def main():
    parser = argparse.ArgumentParser(description="CDiffSD inference and QA.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_cdiffsd_inference.png')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of examples to visualize')
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    schedule = DiffusionSchedule(T=args.T).to(device)
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = ColdDiffusion(model, schedule)

    # Load test samples
    dataset = ColdDiffusionDataset(args.clean_dir, args.noise_dir)
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.n].tolist()

    results = []
    for idx in indices:
        x_clean, x_noisy, _n_fwd, scale = dataset[idx]
        scale = scale.item()

        x_noisy_dev = x_noisy.unsqueeze(0).to(device)
        print(f"Sample {idx} (scale={scale*1e3:.1f} mV): ", end='', flush=True)

        print("direct...", end='', flush=True)
        x_direct_norm = diffusion.sample_direct(x_noisy_dev).squeeze().cpu().numpy()

        print(" sampling...", end='', flush=True)
        x_sampling_norm = diffusion.sample(x_noisy_dev).squeeze().cpu().numpy()
        print(" done")

        # Rescale to physical units
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale
        x_direct = x_direct_norm * scale
        x_sampling = x_sampling_norm * scale

        m_noisy = compute_all_metrics(x_clean_np, x_noisy_np)
        m_direct = compute_all_metrics(x_clean_np, x_direct)
        m_sampling = compute_all_metrics(x_clean_np, x_sampling)

        results.append({
            'idx': idx,
            'clean': x_clean_np,
            'noisy': x_noisy_np,
            'direct': x_direct,
            'sampling': x_sampling,
            'm_noisy': m_noisy,
            'm_direct': m_direct,
            'm_sampling': m_sampling,
        })

    # Print metrics table
    print("\n" + "=" * 110)
    print(f"{'':>10} | {'MSE':>12} {'MAD':>12} {'PRD (%)':>12} "
          f"{'Cosine':>10} {'CC':>10} {'SNR (dB)':>10}")
    print("-" * 110)

    avg = {'noisy': {}, 'direct': {}, 'sampling': {}}
    for r in results:
        for key in ['mse', 'mad', 'prd', 'cosine_sim', 'cc', 'snr']:
            for label, mdict in [('noisy', r['m_noisy']),
                                  ('direct', r['m_direct']),
                                  ('sampling', r['m_sampling'])]:
                avg[label][key] = avg[label].get(key, 0) + mdict[key]

    n = len(results)
    for label in ['noisy', 'direct', 'sampling']:
        for key in avg[label]:
            avg[label][key] /= n

    for label, name in [('noisy', 'Noisy'), ('direct', 'Direct'),
                         ('sampling', 'Sampling')]:
        m = avg[label]
        print(f"{name:>10} | {m['mse']:12.6f} {m['mad']:12.6f} {m['prd']:12.2f} "
              f"{m['cosine_sim']:10.6f} {m['cc']:10.6f} {m['snr']:10.2f}")
    print("=" * 110)

    # Plot
    fig, axes = plt.subplots(args.n, 3, figsize=(20, 3.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        t = np.arange(len(r['clean'])) / 1000.0

        ax = axes[row, 0]
        ax.plot(t, r['direct'] * 1e3, 'g-', lw=0.5, alpha=0.7, label='Direct')
        ax.plot(t, r['sampling'] * 1e3, 'm-', lw=0.5, alpha=0.7, label='Sampling')
        ax.plot(t, r['clean'] * 1e3, 'b-', lw=0.5, label='Clean')
        ax.plot(t, r['noisy'] * 1e3, 'r-', lw=0.3, alpha=0.5, label='Noisy')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_title(f"Sample {r['idx']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        ax2 = axes[row, 1]
        ax2.plot(t, (r['clean'] - r['direct']) * 1e3, 'g-', lw=0.4, alpha=0.7, label='Direct')
        ax2.plot(t, (r['clean'] - r['sampling']) * 1e3, 'm-', lw=0.4, alpha=0.7, label='Sampling')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Residual [mV]')
        ax2.set_title('Clean - Denoised', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[row, 2]
        ax3.axis('off')
        text = (
            f"{'Metric':>12}  {'Noisy':>10}  {'Direct':>10}  {'Sampling':>10}\n"
            f"{'MSE':>12}  {r['m_noisy']['mse']:10.6f}  {r['m_direct']['mse']:10.6f}  {r['m_sampling']['mse']:10.6f}\n"
            f"{'MAD':>12}  {r['m_noisy']['mad']*1e3:9.3f}m  {r['m_direct']['mad']*1e3:9.3f}m  {r['m_sampling']['mad']*1e3:9.3f}m\n"
            f"{'PRD (%)':>12}  {r['m_noisy']['prd']:10.2f}  {r['m_direct']['prd']:10.2f}  {r['m_sampling']['prd']:10.2f}\n"
            f"{'Cosine':>12}  {r['m_noisy']['cosine_sim']:10.6f}  {r['m_direct']['cosine_sim']:10.6f}  {r['m_sampling']['cosine_sim']:10.6f}\n"
            f"{'CC':>12}  {r['m_noisy']['cc']:10.6f}  {r['m_direct']['cc']:10.6f}  {r['m_sampling']['cc']:10.6f}\n"
            f"{'SNR (dB)':>12}  {r['m_noisy']['snr']:10.2f}  {r['m_direct']['snr']:10.2f}  {r['m_sampling']['snr']:10.2f}"
        )
        ax3.text(0.05, 0.5, text, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace')

    fig.suptitle('CDiffSD Inference: Direct vs Sampling (Cold Diffusion)', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
