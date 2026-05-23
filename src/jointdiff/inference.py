"""
Joint Diffusion inference: load two score networks, run PiGDM joint
sampling, evaluate and visualize.

Usage:
    python -u -m src.jointdiff.inference \
        --model_x /path/to/score_x/best_model.pt \
        --model_n /path/to/score_n/best_model.pt \
        --clean_dir /path/to/clean \
        --noise_dir /path/to/noise \
        --output qa_jointdiff.png \
        --n 5
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.inference import compute_all_metrics
from src.ddpm.dataset import PulseNoiseDataset
from src.jointdiff.score_unet import ScoreUNet1D
from src.jointdiff.joint_sampler import JointSampler


def main():
    parser = argparse.ArgumentParser(description="Joint Diffusion inference.")
    parser.add_argument('--model_x', type=str, required=True,
                        help='Path to trained signal score network')
    parser.add_argument('--model_n', type=str, required=True,
                        help='Path to trained noise score network')
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_jointdiff.png')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Noise coefficient: y = x + alpha*n')
    parser.add_argument('--lambda_x', type=float, default=0.93)
    parser.add_argument('--lambda_n', type=float, default=0.88)
    parser.add_argument('--guidance', type=str, default='pigdm',
                        choices=['projection', 'dps', 'pigdm', 'signal_fixed'],
                        help='Guidance strategy')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    schedule = DiffusionSchedule(T=args.T).to(device)

    model_x = ScoreUNet1D().to(device)
    model_x.load_state_dict(torch.load(args.model_x, map_location=device,
                                        weights_only=True))
    model_x.eval()

    model_n = ScoreUNet1D().to(device)
    model_n.load_state_dict(torch.load(args.model_n, map_location=device,
                                        weights_only=True))
    model_n.eval()

    print(f"Guidance: {args.guidance}")
    sampler = JointSampler(
        model_x, model_n, schedule,
        guidance=args.guidance,
        alpha=args.alpha,
        lambda_x=args.lambda_x,
        lambda_n=args.lambda_n,
    )

    # Load test data (paired clean+noise for evaluation)
    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.n].tolist()

    results = []
    for idx in indices:
        x_clean, x_noisy, scale = dataset[idx]
        scale = scale.item()

        y = x_noisy.unsqueeze(0).to(device)  # (1, 1, L)
        print(f"Sample {idx} (scale={scale*1e3:.1f} mV): joint sampling...",
              end='', flush=True)

        x_0, n_0 = sampler.sample(y)
        print(" done")

        # Solve for final blend coefficients: y ≈ a*x_0 + b*n_0
        from src.jointdiff.guidance import BaseGuidance
        a, b = BaseGuidance._solve_blend(y, x_0, n_0)
        a_val, b_val = a.item(), b.item()
        print(f"  blend: a={a_val:.4f}, b={b_val:.4f}")

        # Rescale to physical units (apply blend coefficients)
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale
        x_denoised = x_0.squeeze().cpu().numpy() * a_val * scale
        n_separated = n_0.squeeze().cpu().numpy() * b_val * scale

        m_noisy = compute_all_metrics(x_clean_np, x_noisy_np)
        m_denoised = compute_all_metrics(x_clean_np, x_denoised)

        results.append({
            'idx': idx,
            'clean': x_clean_np,
            'noisy': x_noisy_np,
            'denoised': x_denoised,
            'noise_sep': n_separated,
            'm_noisy': m_noisy,
            'm_denoised': m_denoised,
        })

    # Print metrics
    print("\n" + "=" * 90)
    print(f"{'':>10} | {'MSE':>12} {'MAD':>12} {'PRD (%)':>12} "
          f"{'Cosine':>10} {'CC':>10} {'SNR (dB)':>10}")
    print("-" * 90)

    avg = {'noisy': {}, 'denoised': {}}
    for r in results:
        for key in ['mse', 'mad', 'prd', 'cosine_sim', 'cc', 'snr']:
            for label, mdict in [('noisy', r['m_noisy']),
                                  ('denoised', r['m_denoised'])]:
                avg[label][key] = avg[label].get(key, 0) + mdict[key]

    n = len(results)
    for label in avg:
        for key in avg[label]:
            avg[label][key] /= n

    for label, name in [('noisy', 'Noisy'), ('denoised', 'JointDiff')]:
        m = avg[label]
        print(f"{name:>10} | {m['mse']:12.6f} {m['mad']:12.6f} {m['prd']:12.2f} "
              f"{m['cosine_sim']:10.6f} {m['cc']:10.6f} {m['snr']:10.2f}")
    print("=" * 90)

    # Plot: 3 columns per sample (same layout as DDPM QA)
    fig, axes = plt.subplots(args.n, 3, figsize=(20, 3.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        t = np.arange(len(r['clean'])) / 1000.0

        # Col 1: waveform overlay
        ax = axes[row, 0]
        ax.plot(t, r['denoised'] * 1e3, 'g-', lw=0.5, alpha=0.7, label='JointDiff')
        ax.plot(t, r['clean'] * 1e3, 'b-', lw=0.5, label='Clean')
        ax.plot(t, r['noisy'] * 1e3, 'r-', lw=0.3, alpha=0.5, label='Noisy')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_title(f"Sample {r['idx']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Col 2: residual
        ax2 = axes[row, 1]
        ax2.plot(t, (r['clean'] - r['denoised']) * 1e3, 'g-', lw=0.4, alpha=0.7,
                 label='JointDiff')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Residual [mV]')
        ax2.set_title('Clean - Denoised', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        # Col 3: metrics
        ax3 = axes[row, 2]
        ax3.axis('off')
        text = (
            f"{'Metric':>12}  {'Noisy':>10}  {'JointDiff':>10}\n"
            f"{'MSE':>12}  {r['m_noisy']['mse']:10.6f}  {r['m_denoised']['mse']:10.6f}\n"
            f"{'MAD':>12}  {r['m_noisy']['mad']*1e3:9.3f}m  {r['m_denoised']['mad']*1e3:9.3f}m\n"
            f"{'PRD (%)':>12}  {r['m_noisy']['prd']:10.2f}  {r['m_denoised']['prd']:10.2f}\n"
            f"{'Cosine':>12}  {r['m_noisy']['cosine_sim']:10.6f}  {r['m_denoised']['cosine_sim']:10.6f}\n"
            f"{'CC':>12}  {r['m_noisy']['cc']:10.6f}  {r['m_denoised']['cc']:10.6f}\n"
            f"{'SNR (dB)':>12}  {r['m_noisy']['snr']:10.2f}  {r['m_denoised']['snr']:10.2f}"
        )
        ax3.text(0.05, 0.5, text, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace')

    fig.suptitle('Joint Diffusion (PiGDM) Inference', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
