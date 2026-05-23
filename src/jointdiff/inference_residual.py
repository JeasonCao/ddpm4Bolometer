"""
Two-stage inference: DDPM 10-shot for signal, then Score_n for noise via
residual guidance.

Stage 1: Run DDPM 10-shot → x̂_0 (clean signal estimate)
Stage 2: Compute r = y - x̂_0, run Score_n with guidance to match r

Usage:
    python -u -m src.jointdiff.inference_residual \
        --ddpm_model /media/AVFD/yunshancheng/cuore/ddpm_l1/best_model.pt \
        --model_n /media/AVFD/yunshancheng/cuore/jointdiff/score_n/best_model.pt \
        --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_000.h5 \
        --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_000.h5 \
        --output /media/AVFD/yunshancheng/cuore/jointdiff/qa_residual_000.png
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import PulseNoiseDataset
from src.ddpm.inference import compute_all_metrics
from src.jointdiff.score_unet import ScoreUNet1D
from src.jointdiff.guidance import get_guidance


def run_score_n_with_residual(residual, model_n, schedule, T=50,
                              lambda_n=0.88):
    """Run Score_n reverse diffusion guided to match the residual."""
    device = residual.device
    B, C, L = residual.shape

    guidance = get_guidance('residual', schedule, lambda_n=lambda_n)

    n_t = torch.randn(B, 1, L, device=device)

    for i in reversed(range(T)):
        # DDPM reverse step for n_t
        t_batch = torch.full((B,), i + 1, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model_n(n_t, t_batch)

        alpha_t = schedule.alpha[i]
        beta_t = schedule.beta[i]
        sqrt_alpha = schedule.sqrt_alpha[i]
        sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[i]

        coeff = beta_t / sqrt_1_ab
        n_mean = (1.0 / sqrt_alpha) * (n_t - coeff * eps_pred)

        if i > 0:
            n_t = n_mean + torch.sqrt(beta_t) * torch.randn_like(n_t)
        else:
            n_t = n_mean

        # Guidance: push n_t to match residual
        with torch.enable_grad():
            _, n_t = guidance.joint_update(
                residual, n_t, n_t, None, model_n, i,
            )
        n_t = n_t.detach()

    return n_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddpm_model', type=str, required=True)
    parser.add_argument('--model_n', type=str, required=True)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_residual.png')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load DDPM
    schedule = DiffusionSchedule(T=args.T).to(device)
    ddpm_net = UNet1D().to(device)
    ddpm_net.load_state_dict(torch.load(args.ddpm_model, map_location=device,
                                        weights_only=True))
    ddpm_net.eval()
    ddpm = GaussianDiffusion(ddpm_net, schedule)

    # Load Score_n
    model_n = ScoreUNet1D().to(device)
    model_n.load_state_dict(torch.load(args.model_n, map_location=device,
                                        weights_only=True))
    model_n.eval()

    # Load data
    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.n].tolist()

    results = []
    for idx in indices:
        x_clean, x_noisy, scale = dataset[idx]
        scale_val = scale.item()

        y = x_noisy.unsqueeze(0).to(device)

        # Stage 1: DDPM 10-shot
        print(f"Sample {idx} (scale={scale_val*1e3:.1f} mV): "
              f"DDPM 10-shot...", end='', flush=True)
        x0_ddpm = ddpm.sample_multi_shot(y, M=10)
        print(" residual guidance...", end='', flush=True)

        # Stage 2: residual → Score_n with guidance
        residual = y - x0_ddpm  # in y-normalized space
        n0_est = run_score_n_with_residual(residual, model_n, schedule,
                                           T=args.T)

        # Solve for b to get noise in y-scale
        from src.jointdiff.guidance import BaseGuidance
        b = BaseGuidance._solve_blend_1d(residual, n0_est).item()
        print(f" done (b={b:.4f})")

        # Rescale to physical units
        c = x_clean.squeeze().numpy() * scale_val
        n_phys = x_noisy.squeeze().numpy() * scale_val
        x_ddpm = x0_ddpm.squeeze().cpu().numpy() * scale_val
        n_sep = n0_est.squeeze().cpu().numpy() * b * scale_val
        # Refined signal: subtract guided noise from observation
        x_refined = n_phys - n_sep

        true_noise = n_phys - c

        results.append({
            'idx': idx,
            'clean': c,
            'noisy': n_phys,
            'ddpm': x_ddpm,
            'refined': x_refined,
            'noise_sep': n_sep,
            'true_noise': true_noise,
            'm_noisy': compute_all_metrics(c, n_phys),
            'm_ddpm': compute_all_metrics(c, x_ddpm),
            'm_refined': compute_all_metrics(c, x_refined),
        })

    # Print metrics
    print("\n" + "=" * 110)
    print(f"{'':>12} | {'MSE':>12} {'MAD':>12} {'PRD (%)':>12} "
          f"{'Cosine':>10} {'CC':>10} {'SNR (dB)':>10}")
    print("-" * 110)

    avg = {'noisy': {}, 'ddpm': {}, 'refined': {}}
    for r in results:
        for key in ['mse', 'mad', 'prd', 'cosine_sim', 'cc', 'snr']:
            for label, mdict in [('noisy', r['m_noisy']),
                                  ('ddpm', r['m_ddpm']),
                                  ('refined', r['m_refined'])]:
                avg[label][key] = avg[label].get(key, 0) + mdict[key]

    nn = len(results)
    for label in avg:
        for key in avg[label]:
            avg[label][key] /= nn

    for label, name in [('noisy', 'Noisy'), ('ddpm', 'DDPM-10shot'),
                         ('refined', 'Refined')]:
        m = avg[label]
        print(f"{name:>12} | {m['mse']:12.6f} {m['mad']:12.6f} {m['prd']:12.2f} "
              f"{m['cosine_sim']:10.6f} {m['cc']:10.6f} {m['snr']:10.2f}")
    print("=" * 110)

    # Plot: 3 columns
    fig, axes = plt.subplots(args.n, 3, figsize=(20, 3.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        t = np.arange(len(r['clean'])) / 1000.0

        # Col 1: waveforms
        ax = axes[row, 0]
        ax.plot(t, r['noisy'] * 1e3, 'r-', lw=0.3, alpha=0.4, label='Noisy')
        ax.plot(t, r['clean'] * 1e3, 'b-', lw=0.5, label='Clean')
        ax.plot(t, r['ddpm'] * 1e3, 'm-', lw=0.5, alpha=0.7, label='DDPM-10shot')
        ax.plot(t, r['refined'] * 1e3, 'g-', lw=0.5, alpha=0.7, label='Refined')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_title(f"Sample {r['idx']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Col 2: residuals
        ax2 = axes[row, 1]
        ax2.plot(t, (r['clean'] - r['ddpm']) * 1e3, 'm-', lw=0.4, alpha=0.7,
                 label='DDPM')
        ax2.plot(t, (r['clean'] - r['refined']) * 1e3, 'g-', lw=0.4, alpha=0.7,
                 label='Refined')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Residual [mV]')
        ax2.set_title('Clean - Denoised', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        # Col 3: metrics
        ax3 = axes[row, 2]
        ax3.axis('off')
        text = (
            f"{'Metric':>12}  {'Noisy':>10}  {'DDPM-10':>10}  {'Refined':>10}\n"
            f"{'MSE':>12}  {r['m_noisy']['mse']:10.6f}  {r['m_ddpm']['mse']:10.6f}  {r['m_refined']['mse']:10.6f}\n"
            f"{'MAD':>12}  {r['m_noisy']['mad']*1e3:9.3f}m  {r['m_ddpm']['mad']*1e3:9.3f}m  {r['m_refined']['mad']*1e3:9.3f}m\n"
            f"{'PRD (%)':>12}  {r['m_noisy']['prd']:10.2f}  {r['m_ddpm']['prd']:10.2f}  {r['m_refined']['prd']:10.2f}\n"
            f"{'Cosine':>12}  {r['m_noisy']['cosine_sim']:10.6f}  {r['m_ddpm']['cosine_sim']:10.6f}  {r['m_refined']['cosine_sim']:10.6f}\n"
            f"{'CC':>12}  {r['m_noisy']['cc']:10.6f}  {r['m_ddpm']['cc']:10.6f}  {r['m_refined']['cc']:10.6f}\n"
            f"{'SNR (dB)':>12}  {r['m_noisy']['snr']:10.2f}  {r['m_ddpm']['snr']:10.2f}  {r['m_refined']['snr']:10.2f}"
        )
        ax3.text(0.05, 0.5, text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='center', fontfamily='monospace')

    fig.suptitle('DDPM 10-shot + Residual Noise Guidance', fontsize=13)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
