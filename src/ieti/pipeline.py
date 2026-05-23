"""
pipeline.py — Shared preprocessing + separate train/infer entry points.

Preprocessing (stages 1-2):
  Stage 1: Correlated drift removal (cross-channel, optional)
  Stage 2: Rolling median/MAD baseline

Stage 3 (periodic noise removal via 3b) is applied per-window inside
the dataset and inference code.

Usage:
  python train_evenodd.py  — train even/odd N2N model
  python train_noisier.py  — train noisier N2N model
  python infer.py           — run inference with trained model
"""

import os
import numpy as np
import torch

from .convert import process_adc_file, write_bin
from .trigger import rolling_median_chunked, rolling_mad_chunked
from .drift import remove_correlated_drift
from .model import UNet1D, count_parameters
from .train import train, load_checkpoint, plot_loss_curves
from .denoise import denoise_stream, denoise_stream_noisier


def load_and_preprocess(
    file_paths: list[str],
    use_drift_removal: bool = True,
    drift_cutoff_hz: float = 0.03,
    rolling_window_sec: float = 60.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray],
           list[np.ndarray], list[dict], list[float]]:
    """Load binary files and apply stages 1-2.

    Parameters
    ----------
    file_paths : list of .bin file paths (one per channel)
    use_drift_removal : attempt cross-channel drift removal (needs same fs/length)
    drift_cutoff_hz : low-pass cutoff for drift estimation
    rolling_window_sec : window for rolling median/MAD

    Returns
    -------
    voltages : list of preprocessed voltage arrays
    rolling_meds : list of rolling medians
    rolling_mads : list of rolling MADs
    drifts : list of drift components (for add-back after inference)
    headers : list of file headers
    sampling_rates : list of sampling rates
    """
    # Load
    print("[1] Loading data...")
    voltages = []
    headers = []
    sampling_rates = []
    for path in file_paths:
        _, v, hdr = process_adc_file(path)
        voltages.append(v)
        headers.append(hdr)
        sampling_rates.append(hdr["fs"])
        print(f"  {os.path.basename(path)}: {len(v):,} samples "
              f"({len(v)/hdr['fs']:.1f}s) @ {hdr['fs']} Hz")

    # Stage 1: Drift removal
    drifts = [np.zeros_like(v) for v in voltages]
    all_same_fs = len(set(sampling_rates)) == 1
    all_same_len = len(set(len(v) for v in voltages)) == 1

    if use_drift_removal and len(voltages) == 2 and all_same_fs and all_same_len:
        print(f"\n[2] Stage 1: Correlated drift removal (cutoff={drift_cutoff_hz} Hz)...")
        v0, v1, slope, intercept, d0, d1 = remove_correlated_drift(
            voltages[0], voltages[1], sampling_rates[0], cutoff_hz=drift_cutoff_hz)
        voltages[0], voltages[1] = v0, v1
        drifts[0], drifts[1] = d0, d1
        print(f"  slope={slope:.4f}, intercept={intercept:.4f}")
    else:
        reason = "different fs" if not all_same_fs else \
                 "different lengths" if not all_same_len else \
                 f"{len(voltages)} channels (need 2)"
        print(f"\n[2] Stage 1: SKIPPED ({reason})")

    # Stage 2: Rolling baseline
    print(f"\n[3] Stage 2: Rolling baseline (window={rolling_window_sec}s)...")
    rolling_meds = []
    rolling_mads = []
    for v, fs in zip(voltages, sampling_rates):
        w = int(rolling_window_sec * fs)
        rm = rolling_median_chunked(v, w)
        rmad = rolling_mad_chunked(v, rm, w)
        rolling_meds.append(rm)
        rolling_mads.append(rmad)
        print(f"  fs={fs}: med=[{rm.min():.4f}, {rm.max():.4f}], "
              f"MAD=[{rmad.min():.6f}, {rmad.max():.6f}]")

    return voltages, rolling_meds, rolling_mads, drifts, headers, sampling_rates


def run_training(
    voltages: list[np.ndarray],
    rolling_meds: list[np.ndarray],
    rolling_mads: list[np.ndarray],
    sampling_rates: list[float],
    mode: str = "evenodd",
    checkpoint_path: str = "checkpoints/evenodd_best.pt",
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    loss_type: str = "mse",
    psd_weight: float = 0.5,
    tv_weight: float = 0.0,
    mad_threshold: float = 3.0,
    use_periodic: bool = True,
    val_frac: float = 0.2,
    window_sec: float = 6.5,
    device: torch.device | None = None,
    channel_names: list[str] | None = None,
    precomputed_train: str | None = None,
    precomputed_val: str | None = None,
) -> dict:
    """Build dataset, train model, save checkpoint and plots.

    Parameters
    ----------
    mode : 'evenodd' or 'noisier'
    precomputed_train : path to precomputed .npy dir for training set
    precomputed_val : path to precomputed .npy dir for validation set
    """
    from .dataset import EvenOddN2NDataset, NoisierN2NDataset, plot_dataset_examples

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(len(voltages))]

    DatasetClass = EvenOddN2NDataset if mode == "evenodd" else NoisierN2NDataset

    if precomputed_train and precomputed_val:
        print(f"\n[4] Loading precomputed {mode} dataset...")
        train_dataset = DatasetClass.from_precomputed(precomputed_train)
        val_dataset = DatasetClass.from_precomputed(precomputed_val)
    else:
        # Build datasets with temporal split
        ds_kwargs = dict(
            sampling_rates=sampling_rates,
            window_sec=window_sec,
            mad_threshold=mad_threshold,
            use_periodic=use_periodic,
        )
        print(f"\n[4] Building {mode} dataset (temporal split {1-val_frac:.0%}/{val_frac:.0%})...")
        train_dataset = DatasetClass(
            voltages, rolling_meds, rolling_mads,
            time_range=(0.0, 1 - val_frac), **ds_kwargs)
        val_dataset = DatasetClass(
            voltages, rolling_meds, rolling_mads,
            time_range=(1 - val_frac, 1.0), **ds_kwargs)
    print(f"  Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")

    # QA plots
    os.makedirs("plots", exist_ok=True)
    plot_dataset_examples(
        voltages, rolling_meds, rolling_mads,
        train_dataset, channel_names=channel_names,
        output_dir=f"plots/dataset_qa_{mode}",
    )

    # Model
    model = UNet1D()
    print(f"\n[5] UNet1D: {count_parameters(model):,} parameters")
    print(f"  Device: {device}")

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"  Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Train
    tv_str = f", tv_weight={tv_weight}" if tv_weight > 0 else ""
    print(f"\n[6] Training ({n_epochs} epochs, loss={loss_type}{tv_str})...")
    history = train(
        model, train_dataset, val_dataset,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        checkpoint_path=checkpoint_path,
        loss_type=loss_type, psd_weight=psd_weight,
        tv_weight=tv_weight, device=device,
    )

    # Loss curves
    ckpt_dir = os.path.dirname(checkpoint_path) or "."
    plot_loss_curves(
        os.path.join(ckpt_dir, "train_losses.npy"),
        os.path.join(ckpt_dir, "val_losses.npy"),
        f"plots/loss_curves_{mode}.png",
    )

    print("\nTraining done.")
    return history


def run_inference(
    voltages: list[np.ndarray],
    rolling_meds: list[np.ndarray],
    rolling_mads: list[np.ndarray],
    drifts: list[np.ndarray],
    headers: list[dict],
    sampling_rates: list[float],
    output_paths: list[str],
    mode: str = "evenodd",
    checkpoint_path: str = "checkpoints/evenodd_best.pt",
    mad_threshold: float = 3.0,
    use_periodic: bool = True,
    global_mean: float = 0.0,
    global_std: float = 1.0,
    device: torch.device | None = None,
) -> None:
    """Load checkpoint and denoise all channels."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet1D()
    model = load_checkpoint(checkpoint_path, model, device)
    print(f"\n[4] Loaded {checkpoint_path}, device={device}")

    denoise_fn = denoise_stream if mode == "evenodd" else denoise_stream_noisier

    print("\n[5] Denoising...")
    for i, (v, rm, rmad, drift, hdr, out_path) in enumerate(
        zip(voltages, rolling_meds, rolling_mads, drifts, headers, output_paths)
    ):
        fs = sampling_rates[i]
        print(f"  Channel {i}: inference ({len(v)/fs:.1f}s @ {fs} Hz)...")
        denoised = denoise_fn(
            v, model, rm, rmad, device,
            fs=fs, use_periodic=use_periodic,
            mad_threshold=mad_threshold,
            global_mean=global_mean,
            global_std=global_std,
        )
        denoised = denoised + drift

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"  Writing {out_path}...")
        write_bin(out_path, denoised,
                  **{k: hdr[k] for k in ["endianness", "nbits", "fs", "full_range"]})

    print("\nInference done.")
