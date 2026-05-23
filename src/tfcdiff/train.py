"""
Training script for TFCDiff pulse denoiser.

Usage:
    python -u -m src.tfcdiff.train \
        --clean_dir /media/AVFD/yunshancheng/cuore/clean \
        --noise_dir /media/AVFD/yunshancheng/cuore/noise \
        --output_dir /media/AVFD/yunshancheng/cuore/tfcdiff_runs \
        --epochs 100 --batch_size 32 --lr 1e-3
"""

import argparse
import os
import time
import json

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.tfcdiff.schedule import DiffusionSchedule
from src.tfcdiff.unet import UNet
from src.tfcdiff.diffusion import TFCDiffusion
from src.tfcdiff.dataset import PulseNoiseDataset


def main():
    parser = argparse.ArgumentParser(description="Train TFCDiff pulse denoiser.")
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--dct_len', type=int, default=2400,
                        help='Number of DCT coefficients to keep (<120 Hz)')
    parser.add_argument('--eta', type=float, default=27.0,
                        help='DCT coefficient scaling factor')
    parser.add_argument('--snr_scale', type=float, default=150.0,
                        help='SNR scaling factor for noise schedule')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'],
                        help='Loss function (default: l1)')
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume from')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Dataset
    print("Loading dataset...")
    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    n_total = len(dataset)
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    # Attention at seq_len/4 = 600 (after 2 downsamples from 2400)
    attn_res = (args.dct_len // 4,)
    schedule = DiffusionSchedule(
        T=args.T, snr_scale=args.snr_scale).to(device)
    model = UNet(
        seq_len=args.dct_len,
        attn_res=attn_res,
    ).to(device)
    diffusion = TFCDiffusion(
        model, schedule,
        dct_len=args.dct_len, eta=args.eta,
        loss_type=args.loss,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"DCT length: {args.dct_len}, eta: {args.eta}, "
          f"SNR scale: {args.snr_scale}, loss: {args.loss}")
    print(f"Attention at resolutions: {attn_res}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': [],
               'step_loss': [], 'step_num': []}
    global_step = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Restored optimizer & scheduler from epoch {ckpt['epoch']}")
        else:
            model.load_state_dict(ckpt)

        history_path = os.path.join(args.output_dir, 'history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            if 'step_loss' not in history:
                history['step_loss'] = []
                history['step_num'] = []
            if history.get('val_loss'):
                best_val_loss = min(history['val_loss'])
                print(f"Initialized best_val_loss={best_val_loss:.6f} from history")
            if history.get('step_num'):
                global_step = history['step_num'][-1]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        dataset.set_epoch(epoch)

        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        n_total_batches = len(train_loader)
        for x_clean, x_noisy, _scale in train_loader:
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)

            loss = diffusion.training_loss(x_clean, x_noisy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                history['step_loss'].append(loss.item())
                history['step_num'].append(global_step)

            if n_batches % 100 == 0 or n_batches == n_total_batches:
                avg = train_loss_sum / n_batches
                print(f"  [{n_batches}/{n_total_batches}] loss={avg:.6f}")

        train_loss = train_loss_sum / n_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_clean, x_noisy, _scale in val_loader:
                x_clean = x_clean.to(device)
                x_noisy = x_noisy.to(device)
                loss = diffusion.training_loss(x_clean, x_noisy)
                val_loss_sum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(n_val_batches, 1)

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={lr:.2e} | step={global_step} | {elapsed:.1f}s")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> New best val_loss: {val_loss:.6f}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_{epoch:03d}.pt'))

        # Save history and loss curve
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        ax1 = axes[0]
        if history['step_num']:
            ax1.plot(history['step_num'], history['step_loss'],
                     'b-', lw=0.3, alpha=0.5, label='Train (per step)')
        val_steps = [(start_epoch + i) * n_total_batches
                     for i in range(len(history['val_loss']))]
        ax1.plot(val_steps, history['val_loss'],
                 'r-o', markersize=2, lw=1, label='Val (per epoch)')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs Step')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        epochs_so_far = range(1, len(history['train_loss']) + 1)
        ax2.plot(epochs_so_far, history['train_loss'], 'b-', label='Train')
        ax2.plot(epochs_so_far, history['val_loss'], 'r-', label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss vs Epoch')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(epochs_so_far, history['lr'], 'g-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('LR Schedule')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'), dpi=150)
        plt.close()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Output: {args.output_dir} | Total steps: {global_step}")


if __name__ == '__main__':
    main()
