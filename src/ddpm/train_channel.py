"""
Two-phase training for the multi-channel foundation-model DDPM denoiser.

Pretrain  (--phase pretrain)
    Train the shared backbone U-Net *together with* the per-channel token
    embeddings on a set of TRAIN channels. Joint training is what makes the
    backbone steerable by the token. Held-out channels are never seen here.

Fine-tune (--phase finetune)
    Load the pretrained checkpoint, FREEZE the backbone, and train ONLY the
    channel token(s) for the held-out channel(s) on a limited number of their
    windows. Each held-out token is zero-initialized (so it starts from the
    channel-agnostic behavior) and learns a delta against the frozen backbone.

The two phases are disjoint at the channel level (train channels vs held-out
channels); see src/ddpm/README.md (foundation-model section).

Usage
-----
    # Pretrain backbone + train-channel tokens on channels 0..79
    python -u -m src.ddpm.train_channel --phase pretrain \
        --clean_dir .../channel_bank/clean \
        --noise_dir .../channel_bank/noise/noise_total \
        --output_dir .../runs/foundation \
        --n_channels 100 --train_channels 0-79 \
        --epochs 100 --batch_size 16 --lr 2e-4

    # Fine-tune tokens for held-out channels 80..99 on 150 windows each
    python -u -m src.ddpm.train_channel --phase finetune \
        --clean_dir .../channel_bank/clean \
        --noise_dir .../channel_bank/noise/noise_total \
        --output_dir .../runs/foundation \
        --resume .../runs/foundation/backbone.pt \
        --finetune_channels 80-99 --n_finetune_windows 150 \
        --epochs 200 --batch_size 16 --lr 1e-3
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import ChannelBankDataset


def parse_channel_spec(spec: str):
    """Parse '0-79' or '0,1,5-7' into a sorted list of ints."""
    if spec is None:
        return None
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def build_model(args, device):
    model = UNet1D(cond_mode=args.cond_mode, cond_scale=args.cond_scale,
                   n_channels=args.n_channels).to(device)
    return model


def make_diffusion(model, args, device):
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1,
                                 beta_T=args.beta_T).to(device)
    return GaussianDiffusion(model, schedule, loss_type=args.loss,
                             cond_mode=args.cond_mode)


def run_epoch(diffusion, loader, optimizer, device, train=True,
              log_every=0, tag=''):
    """Run one epoch. If train and log_every>0, print per-step progress
    (running loss, it/s, epoch ETA) every `log_every` optimizer steps."""
    diffusion.model.train(train)
    total, n = 0.0, 0
    n_steps = len(loader)
    t0 = time.time()
    for x_clean, x_noisy, _scale, channel_id in loader:
        x_clean = x_clean.to(device)
        x_noisy = x_noisy.to(device)
        channel_id = channel_id.to(device)
        with torch.set_grad_enabled(train):
            loss = diffusion.training_loss(x_clean, x_noisy, channel_id=channel_id)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in diffusion.model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        total += loss.item()
        n += 1
        if train and log_every and (n % log_every == 0 or n == n_steps):
            rate = n / max(time.time() - t0, 1e-9)
            eta = (n_steps - n) / rate if rate > 0 else float('nan')
            print(f"    {tag}[{n}/{n_steps}] loss={total/n:.6f} "
                  f"({rate:.2f} it/s, epoch ETA {eta:.0f}s)", flush=True)
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Pretrain: backbone + train-channel tokens
# --------------------------------------------------------------------------- #
def run_pretrain(args, device):
    train_channels = parse_channel_spec(args.train_channels)
    print(f"Pretrain: backbone + tokens on {len(train_channels)} train channels: "
          f"{train_channels[0]}..{train_channels[-1]}")

    dataset = ChannelBankDataset(args.clean_dir, args.noise_dir,
                                 channels=train_channels)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"  {len(dataset)} samples -> train {n_train}, val {n_val}")

    model = build_model(args, device)
    diffusion = make_diffusion(model, args, device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'backbone.pt')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dataset.set_epoch(epoch)
        tr = run_epoch(diffusion, train_loader, optimizer, device, train=True,
                       log_every=200, tag=f"ep{epoch} ")
        with torch.no_grad():
            va = run_epoch(diffusion, val_loader, optimizer, device, train=False)
        scheduler.step()

        history['train_loss'].append(tr)
        history['val_loss'].append(va)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"  epoch {epoch}/{args.epochs}  train {tr:.6f}  val {va:.6f}  "
              f"({time.time()-t0:.1f}s)")

        if va < best_val:
            best_val = va
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'n_channels': args.n_channels,
                           'cond_mode': args.cond_mode,
                           'cond_scale': args.cond_scale,
                           'T': args.T, 'beta_1': args.beta_1,
                           'beta_T': args.beta_T},
                'train_channels': train_channels,
                'epoch': epoch,
            }, ckpt_path)
        with open(os.path.join(args.output_dir, 'history_phaseA.json'), 'w') as f:
            json.dump(history, f)

    print(f"Pretrain done. Best val {best_val:.6f}. Backbone -> {ckpt_path}")


# --------------------------------------------------------------------------- #
# Fine-tune: freeze backbone, fine-tune held-out channel tokens
# --------------------------------------------------------------------------- #
def run_finetune(args, device):
    finetune_channels = parse_channel_spec(args.finetune_channels)
    print(f"Fine-tune: held-out channels {finetune_channels}")

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    cfg = ckpt['config']
    # Model config comes from the pretrained checkpoint.
    args.n_channels = cfg['n_channels']
    args.cond_mode = cfg['cond_mode']
    args.cond_scale = cfg['cond_scale']
    args.T, args.beta_1, args.beta_T = cfg['T'], cfg['beta_1'], cfg['beta_T']

    model = build_model(args, device)
    model.load_state_dict(ckpt['model_state_dict'])
    diffusion = make_diffusion(model, args, device)

    # Sanity: the channels we fine-tune must NOT have been train channels.
    overlap = set(finetune_channels) & set(ckpt.get('train_channels', []))
    if overlap:
        print(f"  WARNING: fine-tuning channels that were in the pretrain set: "
              f"{sorted(overlap)} (not a held-out test of generalization)")

    if args.finetune_mode == 'full':
        # Oracle ceiling: fully fine-tune the backbone on the channel (warm
        # start from the pretrained backbone). Upper bound on per-channel quality.
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"  ORACLE mode: full fine-tune, trainable params: "
              f"{sum(p.numel() for p in model.parameters()):,}")
    else:
        # Token mode (the method): freeze everything, unfreeze only the channel
        # token table. Other rows get no gradient (their channels aren't in the
        # data) and weight_decay=0 leaves them exactly as the frozen backbone /
        # zero-init has them.
        for p in model.parameters():
            p.requires_grad = False
        model.channel_emb.weight.requires_grad = True
        optimizer = torch.optim.Adam([model.channel_emb.weight], lr=args.lr,
                                     weight_decay=0.0)
        print(f"  frozen backbone; trainable params: "
              f"{model.channel_emb.weight.numel():,} (channel tokens only)")

    # Optional LR schedule — oracle (full fine-tune) only. A full fine-tune at
    # constant LR tends to drift the backbone away from the calibrated
    # eps-prediction the sampler relies on; cosine annealing toward ~0 keeps the
    # ceiling stable. The token fine-tune is left at constant LR (it already
    # converges, and a 256-param fit doesn't benefit).
    scheduler = None
    if args.lr_schedule == 'cosine' and args.finetune_mode == 'full':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7)
        print(f"  LR schedule: cosine annealing (peak {args.lr:g} -> 1e-7)")
    elif args.lr_schedule == 'cosine':
        print("  LR schedule: cosine requested but ignored for token mode "
              "(constant LR)")

    dataset = ChannelBankDataset(
        args.clean_dir, args.noise_dir,
        channels=finetune_channels,
        window_range=(0, args.n_finetune_windows) if args.n_finetune_windows else None,
        max_windows=args.n_finetune_windows,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    print(f"  fine-tune samples: {len(dataset)} "
          f"({args.n_finetune_windows or 'all'} windows/channel)")

    history = {'train_loss': [], 'lr': []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dataset.set_epoch(epoch)
        tr = run_epoch(diffusion, loader, optimizer, device, train=True)
        if scheduler is not None:
            scheduler.step()
        history['train_loss'].append(tr)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch}/{args.epochs}  train {tr:.6f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"({time.time()-t0:.1f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    nwin = args.n_finetune_windows if args.n_finetune_windows else 'all'
    prefix = 'oracle' if args.finetune_mode == 'full' else 'finetuned'
    tag = f"nft{nwin}"
    out_path = os.path.join(args.output_dir, f'{prefix}_{tag}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'finetune_channels': finetune_channels,
        'n_finetune_windows': args.n_finetune_windows,
        'finetune_mode': args.finetune_mode,
        'epochs': args.epochs,
    }, out_path)
    with open(os.path.join(args.output_dir, f'history_{prefix}_{tag}.json'), 'w') as f:
        json.dump(history, f)
    print(f"Fine-tune done. Model -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Two-phase multi-channel DDPM training")
    p.add_argument('--phase', choices=['pretrain', 'finetune'], required=True,
                   help="'pretrain' (backbone + train-channel tokens) or "
                        "'finetune' (held-out channel tokens / oracle)")
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--noise_dir', required=True,
                   help='Directory of per-channel noise shards (noise_total)')
    p.add_argument('--output_dir', required=True)

    # Channel selection
    p.add_argument('--n_channels', type=int, default=100,
                   help='Total channels in the bank (sizes the token table). '
                        'Fine-tune reads this from the checkpoint.')
    p.add_argument('--train_channels', type=str, default='0-79',
                   help="Pretrain: channels to pretrain on, e.g. '0-79'")
    p.add_argument('--finetune_channels', type=str, default='80-99',
                   help="Fine-tune: held-out channels to fine-tune, e.g. '80-99'")
    p.add_argument('--n_finetune_windows', type=int, default=None,
                   help='Fine-tune: windows/channel to fine-tune on (data-efficiency)')
    p.add_argument('--finetune_mode', choices=['token', 'full'], default='token',
                   help="Fine-tune: 'token' (freeze backbone, train channel token "
                        "= the method) or 'full' (oracle, fine-tune whole backbone)")

    # Optimization
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr_schedule', choices=['none', 'cosine'], default='none',
                   help="Fine-tune: LR schedule. 'cosine' (anneal to 1e-7) "
                        "applies to the oracle (full) fine-tune only; the token "
                        "fine-tune always uses constant LR. (Pretrain always "
                        "uses cosine.)")
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None,
                   help='Fine-tune: path to the pretrained backbone checkpoint')

    # Diffusion / model
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--beta_1', type=float, default=1e-4)
    p.add_argument('--beta_T', type=float, default=0.5)
    p.add_argument('--cond_mode', type=str, default='step',
                   choices=['step', 'sqrt_ab'])
    p.add_argument('--cond_scale', type=float, default=1000.0)
    p.add_argument('--loss', type=str, default='l2', choices=['l1', 'l2'])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.phase == 'pretrain':
        run_pretrain(args, device)
    else:
        if not args.resume:
            p.error("--phase finetune requires --resume <pretrained backbone>")
        run_finetune(args, device)


if __name__ == '__main__':
    main()
