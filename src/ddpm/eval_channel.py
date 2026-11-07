"""
Evaluate the multi-channel foundation-model denoiser on held-out channels.

Runs the comparison ladder on the held-out channels' TEST windows (disjoint
from the windows used to fine-tune their tokens):

  * zero-shot   — pretrained backbone, held-out token still zero (no adaptation)
  * fine-tuned  — fine-tuned token(s), one checkpoint per fine-tune budget K
  * oracle      — backbone fully fine-tuned per channel (ceiling, optional)

For each variant it denoises every test window (single-shot deterministic DDPM,
channel-conditioned), computes metrics (reusing src/ddpm/inference.py), and
aggregates per channel and overall. Produces:
  * eval_summary.json — metrics per variant / per channel
  * data_efficiency.png — headline metrics vs fine-tune budget K, with
    zero-shot and oracle reference lines.

Usage
-----
    python -u -m src.ddpm.eval_channel \
        --clean_dir .../channel_bank/clean \
        --noise_dir .../channel_bank/noise/noise_total \
        --backbone  .../runs/foundation/backbone.pt \
        --finetuned .../runs/foundation/finetuned_nft10.pt \
                    .../runs/foundation/finetuned_nft50.pt \
                    .../runs/foundation/finetuned_nft100.pt \
                    .../runs/foundation/finetuned_nft200.pt \
        --oracle    .../runs/foundation/oracle_nft200.pt \
        --channels 80-99 --test_start 200 --max_test_windows 100 \
        --output_dir .../runs/foundation/eval
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import ChannelBankDataset
from src.ddpm.inference import compute_all_metrics, detect_peaks
from src.ddpm.train_channel import parse_channel_spec

N_BASELINE = 1500  # samples before pulse onset (1.5 s @ 1 kHz) — noise-only region

# Headline metrics for the data-efficiency curve (lower is better for all).
HEADLINE = [
    ('peak_amp_err', 'Peak amplitude error [%]'),
    ('baseline_rms', 'Baseline RMS [V]'),
    ('norm_mse', 'Normalized MSE'),
]


def load_diffusion(ckpt_path, device):
    """Load a model+diffusion from a checkpoint produced by train_channel.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = UNet1D(cond_mode=cfg['cond_mode'], cond_scale=cfg['cond_scale'],
                   n_channels=cfg['n_channels']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    schedule = DiffusionSchedule(T=cfg['T'], beta_1=cfg['beta_1'],
                                 beta_T=cfg['beta_T']).to(device)
    diffusion = GaussianDiffusion(model, schedule, cond_mode=cfg['cond_mode'])
    return diffusion, ckpt


def fill_untrained_with_mean(diffusion, train_channels):
    """Set every channel-token row that was NOT pretrained to the mean of the
    trained tokens.

    The held-out rows are zero-initialized and never updated in pretrain, so a
    raw zero-shot feeds the backbone a token it never saw during training
    (out-of-distribution conditioning). Using the mean trained token instead
    gives a fair "average channel" prior the backbone *has* seen, which is the
    honest zero-shot baseline. Returns the number of rows filled.
    """
    emb = getattr(diffusion.model, 'channel_emb', None)
    if emb is None or not train_channels:
        return 0
    w = emb.weight.data
    train_idx = torch.tensor(sorted(set(train_channels)), device=w.device)
    mean_tok = w[train_idx].mean(0)
    mask = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
    mask[train_idx] = False
    w[mask] = mean_tok
    return int(mask.sum())


def _peak_amp_err(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Relative dominant-peak amplitude error [%] (energy proxy, single peak)."""
    (_, a_clean), = detect_peaks(clean, 1)
    (_, a_den), = detect_peaks(denoised, 1)
    if abs(a_clean) < 1e-12:
        return float('nan')
    return 100.0 * abs(a_den - a_clean) / abs(a_clean)


@torch.no_grad()
def denoise_and_score(diffusion, loader, device):
    """Denoise every window in `loader` and return per-window metric records.

    Single-shot deterministic DDPM (stochastic=False), channel-conditioned.
    Each record carries its channel_id so callers can group by channel.
    """
    records = []
    for x_clean, x_noisy, scale, channel_id in loader:
        x_noisy = x_noisy.to(device)
        cid = channel_id.to(device)
        x0 = diffusion.sample(x_noisy, channel_id=cid, stochastic=False)  # (B,1,L)

        x0 = x0.cpu().numpy()
        x_clean = x_clean.numpy()
        scale = scale.numpy()
        for b in range(x0.shape[0]):
            den_norm = x0[b, 0]
            cln_norm = x_clean[b, 0]
            s = float(scale[b])
            den = den_norm * s
            cln = cln_norm * s

            m = compute_all_metrics(cln, den, cln_norm, den_norm)
            m['baseline_rms'] = float(np.sqrt(np.mean(den[:N_BASELINE] ** 2)))
            m['baseline_rms_norm'] = float(
                np.sqrt(np.mean(den_norm[:N_BASELINE] ** 2)))
            m['peak_amp_err'] = _peak_amp_err(cln, den)
            m['channel_id'] = int(channel_id[b])
            records.append(m)
    return records


def aggregate(records, metric_keys):
    """Return {'overall': {metric: mean}, 'per_channel': {ch: {metric: mean}}}."""
    by_ch = {}
    for r in records:
        by_ch.setdefault(r['channel_id'], []).append(r)

    def mean_over(recs, key):
        vals = [r[key] for r in recs if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    per_channel = {ch: {k: mean_over(recs, k) for k in metric_keys}
                   for ch, recs in by_ch.items()}
    overall = {k: mean_over(records, k) for k in metric_keys}
    return {'overall': overall, 'per_channel': per_channel}


def build_loader(args):
    ds = ChannelBankDataset(
        args.clean_dir, args.noise_dir,
        channels=parse_channel_spec(args.channels),
        window_range=(args.test_start, args.test_start + args.max_test_windows)
        if args.max_test_windows else (args.test_start, 10 ** 9),
        max_windows=args.max_test_windows,
    )
    print(f"  eval set: {len(ds)} test windows over channels {ds.channel_list}")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)


def plot_curve(results, output_path):
    """Data-efficiency curve: headline metrics vs K, with reference lines."""
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(5 * len(HEADLINE), 4))
    if len(HEADLINE) == 1:
        axes = [axes]

    ks = sorted(results['finetuned'].keys())
    for ax, (key, label) in zip(axes, HEADLINE):
        ys = [results['finetuned'][k]['overall'][key] for k in ks]
        ax.plot(ks, ys, 'o-', color='C0', label='fine-tuned (token)')

        zs = results['zero_shot']['overall'][key]
        ax.axhline(zs, ls='--', color='C3', label='zero-shot')
        if results.get('oracle') is not None:
            orc = results['oracle']['overall'][key]
            ax.axhline(orc, ls=':', color='C2', label='oracle (full FT)')

        ax.set_xlabel('fine-tune windows K')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Data efficiency on held-out channels')
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    print(f"  curve -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Eval multi-channel denoiser")
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--noise_dir', required=True)
    p.add_argument('--backbone', required=True,
                   help='Pretrained backbone ckpt (zero-shot)')
    p.add_argument('--finetuned', nargs='*', default=[],
                   help='Fine-tuned token checkpoints, one per K')
    p.add_argument('--oracle', default=None, help='Full-fine-tune ceiling ckpt')
    p.add_argument('--channels', default='80-99', help='Held-out channels')
    p.add_argument('--test_start', type=int, default=200,
                   help='First test window index (after the fine-tune pool)')
    p.add_argument('--max_test_windows', type=int, default=100,
                   help='Cap test windows per channel (speed)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    loader = build_loader(args)
    metric_keys = [k for k, _ in HEADLINE] + ['mse', 'mae', 'snr', 'cc']

    results = {}

    print("Zero-shot (backbone, mean trained token for held-out channels)...")
    diff, bb_ckpt = load_diffusion(args.backbone, device)
    n_filled = fill_untrained_with_mean(diff, bb_ckpt.get('train_channels'))
    print(f"  filled {n_filled} untrained token rows with the mean trained token")
    results['zero_shot'] = aggregate(
        denoise_and_score(diff, loader, device), metric_keys)
    print(f"  {results['zero_shot']['overall']}")

    results['finetuned'] = {}
    for ckpt_path in args.finetuned:
        diff, ckpt = load_diffusion(ckpt_path, device)
        K = ckpt.get('n_finetune_windows') or 0
        print(f"Fine-tuned token, K={K} ({os.path.basename(ckpt_path)})...")
        results['finetuned'][K] = aggregate(
            denoise_and_score(diff, loader, device), metric_keys)
        print(f"  {results['finetuned'][K]['overall']}")

    if args.oracle:
        diff, ckpt = load_diffusion(args.oracle, device)
        print(f"Oracle (full fine-tune)...")
        results['oracle'] = aggregate(
            denoise_and_score(diff, loader, device), metric_keys)
        print(f"  {results['oracle']['overall']}")
    else:
        results['oracle'] = None

    with open(os.path.join(args.output_dir, 'eval_summary.json'), 'w') as f:
        # JSON keys must be strings
        serializable = {
            'zero_shot': results['zero_shot'],
            'finetuned': {str(k): v for k, v in results['finetuned'].items()},
            'oracle': results['oracle'],
        }
        json.dump(serializable, f, indent=2)
    print(f"  summary -> {os.path.join(args.output_dir, 'eval_summary.json')}")

    if results['finetuned']:
        plot_curve(results, os.path.join(args.output_dir, 'data_efficiency.png'))


if __name__ == '__main__':
    main()
