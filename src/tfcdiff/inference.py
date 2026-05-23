"""
Inference and evaluation for trained TFCDiff pulse denoiser.

Usage:
    python -u -m src.tfcdiff.inference \
        --model_path /media/AVFD/yunshancheng/cuore/tfcdiff_runs/best_model.pt \
        --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_001.h5 \
        --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_001.h5 \
        --output qa_inference_tfcdiff.png \
        --n 5
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.tfcdiff.schedule import DiffusionSchedule
from src.tfcdiff.unet import UNet
from src.tfcdiff.diffusion import TFCDiffusion
from src.ddpm.dataset import PulseNoiseDataset
from src.ddpm.inference import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="TFCDiff inference and QA.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_inference_tfcdiff.png')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--dct_len', type=int, default=2400)
    parser.add_argument('--eta', type=float, default=27.0)
    parser.add_argument('--snr_scale', type=float, default=150.0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    attn_res = (args.dct_len // 4,)
    schedule = DiffusionSchedule(
        T=args.T, snr_scale=args.snr_scale).to(device)
    model = UNet(seq_len=args.dct_len, attn_res=attn_res).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = TFCDiffusion(
        model, schedule,
        dct_len=args.dct_len, eta=args.eta,
    )

    # Load test samples
    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.n].tolist()

    results = []
    for idx in indices:
        x_clean, x_noisy, scale = dataset[idx]
        scale = scale.item()

        x_noisy_dev = x_noisy.unsqueeze(0).to(device)
        print(f"Sample {idx} (scale={scale*1e3:.1f} mV): denoising (single-shot)...",
              end='', flush=True)
        x_single_norm = diffusion.sample(x_noisy_dev).squeeze().cpu().numpy()
        print(" (10-shot)...", end='', flush=True)
        x_multi_norm = diffusion.sample_multi_shot(x_noisy_dev, M=10).squeeze().cpu().numpy()
        print(" done")

        # Rescale back to physical units
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale
        x_single = x_single_norm * scale
        x_multi = x_multi_norm * scale

        m_noisy = compute_all_metrics(x_clean_np, x_noisy_np)
        m_single = compute_all_metrics(x_clean_np, x_single)
        m_multi = compute_all_metrics(x_clean_np, x_multi)

        results.append({
            'idx': idx,
            'clean': x_clean_np,
            'noisy': x_noisy_np,
            'single': x_single,
            'multi': x_multi,
            'm_noisy': m_noisy,
            'm_single': m_single,
            'm_multi': m_multi,
        })

    # Print metrics table
    print("\n" + "=" * 110)
    print(f"{'':>8} | {'MSE':>12} {'MAD':>12} {'PRD (%)':>12} "
          f"{'Cosine':>10} {'CC':>10} {'SNR (dB)':>10}")
    print("-" * 110)

    avg = {'noisy': {}, 'single': {}, 'multi': {}}
    for r in results:
        for key in ['mse', 'mad', 'prd', 'cosine_sim', 'cc', 'snr']:
            for label, mdict in [('noisy', r['m_noisy']),
                                  ('single', r['m_single']),
                                  ('multi', r['m_multi'])]:
                avg[label][key] = avg[label].get(key, 0) + mdict[key]

    n = len(results)
    for label in ['noisy', 'single', 'multi']:
        for key in avg[label]:
            avg[label][key] /= n

    for label, name in [('noisy', 'Noisy'), ('single', '1-shot'),
                         ('multi', '10-shot')]:
        m = avg[label]
        print(f"{name:>8} | {m['mse']:12.6f} {m['mad']:12.6f} "
              f"{m['prd']:12.2f} {m['cosine_sim']:10.6f} "
              f"{m['cc']:10.6f} {m['snr']:10.2f}")
    print("=" * 110)

    # Plot
    fig, axes = plt.subplots(args.n, 3, figsize=(20, 3.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        t = np.arange(len(r['clean'])) / 1000.0

        ax = axes[row, 0]
        ax.plot(t, r['single'] * 1e3, 'g-', lw=0.5, alpha=0.7, label='1-shot')
        ax.plot(t, r['multi'] * 1e3, 'm-', lw=0.5, alpha=0.7, label='10-shot')
        ax.plot(t, r['clean'] * 1e3, 'b-', lw=0.5, label='Clean')
        ax.plot(t, r['noisy'] * 1e3, 'r-', lw=0.3, alpha=0.5, label='Noisy')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_title(f"Sample {r['idx']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        ax2 = axes[row, 1]
        ax2.plot(t, (r['clean'] - r['single']) * 1e3, 'g-', lw=0.4,
                 alpha=0.7, label='1-shot')
        ax2.plot(t, (r['clean'] - r['multi']) * 1e3, 'm-', lw=0.4,
                 alpha=0.7, label='10-shot')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Residual [mV]')
        ax2.set_title('Clean - Denoised', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[row, 2]
        ax3.axis('off')
        text = (
            f"{'Metric':>12}  {'Noisy':>10}  {'1-shot':>10}  {'10-shot':>10}\n"
            f"{'MSE':>12}  {r['m_noisy']['mse']:10.6f}  {r['m_single']['mse']:10.6f}  {r['m_multi']['mse']:10.6f}\n"
            f"{'MAD':>12}  {r['m_noisy']['mad']*1e3:9.3f}m  {r['m_single']['mad']*1e3:9.3f}m  {r['m_multi']['mad']*1e3:9.3f}m\n"
            f"{'PRD (%)':>12}  {r['m_noisy']['prd']:10.2f}  {r['m_single']['prd']:10.2f}  {r['m_multi']['prd']:10.2f}\n"
            f"{'Cosine':>12}  {r['m_noisy']['cosine_sim']:10.6f}  {r['m_single']['cosine_sim']:10.6f}  {r['m_multi']['cosine_sim']:10.6f}\n"
            f"{'CC':>12}  {r['m_noisy']['cc']:10.6f}  {r['m_single']['cc']:10.6f}  {r['m_multi']['cc']:10.6f}\n"
            f"{'SNR (dB)':>12}  {r['m_noisy']['snr']:10.2f}  {r['m_single']['snr']:10.2f}  {r['m_multi']['snr']:10.2f}"
        )
        ax3.text(0.05, 0.5, text, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace')

    fig.suptitle('TFCDiff Inference: 1-shot vs 10-shot', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
