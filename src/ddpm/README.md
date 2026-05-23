# DDPM Pulse Denoiser

Denoising Diffusion Probabilistic Model (DDPM) for denoising cryogenic bolometer pulse signals from the CUORE experiment.

## Task

CUORE measures rare nuclear decays using TeO2 bolometers operating at ~10 mK. Each detector channel records voltage pulses proportional to deposited energy, contaminated by thermal and electronic noise. This DDPM denoiser recovers clean pulse waveforms from noisy observations to improve energy resolution.

## Approach

A conditional DDPM trained on paired (clean, noisy) data:

- **Clean signals**: Simulated via an ODE electro-thermal model. 10-second windows at 1000 Hz (10000 samples). Energy range (0, 5407] keV with 70/30 single-pulse/pileup ratio.
- **Noise**: Real noise extracted from quiet (no-pulse) windows in LUCE data.
- **Training pairs**: Clean + real noise = noisy observation. Noise-clean pairings are reshuffled each epoch for combinatorial augmentation.

### Diffusion Process

- **Forward**: Gradually adds Gaussian noise to clean signal x_0 over T steps using a quadratic beta schedule. Defaults are `T=50`, `beta_1=1e-4`, `beta_T=0.5` (DeScoD-ECG style), but all three are CLI options (`--T`, `--beta_1`, `--beta_T`). With `beta_T=0.5`, `alpha_bar_T ≈ 5e-5` — x_T is effectively a standard Gaussian, eliminating the train/inference mismatch at step T (training and sampling both see ~pure noise). The earlier default of `beta_T=0.05` left `alpha_bar_T ≈ 0.41`, retaining ~40% of the clean signal at step T.
- **Reverse**: Starting from pure Gaussian noise, iteratively denoises conditioned on the noisy observation x_tilde (concatenated as a second input channel).
- **Training objective**: Predict the noise epsilon added at each timestep. Base loss is L1 or L2 on the noise prediction, with optional spectral losses on the implied clean estimate x̂_0 (see Loss Functions below).
- **Timestep / noise-level conditioning** (`--cond_mode`):
  - `step` (default): the denoiser is conditioned on the integer diffusion step index `t ∈ {1, …, T}` via a sinusoidal embedding. This is the original DDPM behavior and is what all existing checkpoints were trained with.
  - `sqrt_ab`: the denoiser is conditioned on the continuous noise level `√ᾱ` (WaveGrad / DeScoD-ECG recipe). During training, we first pick an integer step `t`, then sample `√ᾱ ~ Uniform(S_t, S_{t-1})` inside that bin of the grid `S = [1, √ᾱ_1, …, √ᾱ_T]`, and feed that scalar into the U-Net's sinusoidal embedding. Because the network sees a continuum of noise levels rather than `T` discrete points, it generalizes across schedules and tends to work better with fewer reverse steps at inference. The scalar is multiplied by `--cond_scale` (default `1000`) before the sinusoidal encoding so that values in `(0, 1]` span the full frequency grid.
- **Inference**: Two samplers available:
  - **DDPM** (default): Full T-step stochastic reverse process. Multi-shot (M=10) aggregation of independent runs reduces variance. Supports mean or median aggregation (`--aggregation`).
  - **DDIM** (`--sampler ddim`): Deterministic reverse process (eta=0). Same trained model, no retraining needed. Supports step skipping (`--ddim_steps`) for faster inference. Partial stochasticity via `--eta` (0=deterministic, 1=DDPM-equivalent). In practice, DDPM with multi-shot averaging outperforms DDIM for this conditional denoising task — the stochasticity helps explore better solutions for low-SNR inputs.

### Normalization

Per-window normalization: both clean and noisy signals are divided by max(|noisy|) to map inputs to ~[-1, 1]. The scale factor is saved and applied to the output for physical-unit reconstruction.

## Architecture

Two U-Net variants are available:

### UNet1D (default)

1D U-Net with timestep conditioning:

```
Input: [x_t, x_tilde] concatenated (2 channels, 10000 samples)

Encoder:  4 levels (64 -> 128 -> 256 -> 512), stride-2 Conv1d downsample
          2 ResBlocks per level, GroupNorm, SiLU activation
Bottleneck: 512 channels, self-attention
Decoder:  Mirror of encoder with skip connections
Attention: At level 3 (1250 samples) and bottleneck (625 samples)
Timestep: Sinusoidal embedding -> MLP -> added into each ResBlock

Output: 1 channel (predicted noise epsilon)
```

~15.1M parameters.

### UNet1DScaleCond (`--scale_cond`)

Same architecture as UNet1D, plus scale conditioning. The normalization scale `max(|noisy|)` is encoded and injected alongside the timestep embedding so the model knows the SNR regime of each input.

```
scale -> log(scale)*10 -> SinusoidalEmbedding -> MLP -> scale_emb
cond = timestep_emb + scale_emb  (fed to every ResBlock)
```

~15.6M parameters (+525K for the scale MLP).

**Motivation**: Per-window normalization maps all inputs to [-1, 1], but a low-amplitude pileup (pulse peak ~30 mV, noise ~80 mV) looks very different from a high-amplitude pulse (peak ~5000 mV) in normalized space. Without scale information, the model cannot distinguish noise-dominated inputs from pulse-dominated ones. Since the noise level is roughly constant across windows, `max(|noisy|)` directly indicates pulse amplitude and thus SNR.

## Loss Functions

The training loss consists of a base noise-prediction loss plus optional spectral losses computed on the Tweedie estimate x̂_0:

```
Total = base_loss(eps_pred, eps) + w_l1*L1(x̂_0, x_0) + w_sc*SC + w_lsd*LSD + w_asd*J_asd
```

### Base loss (`--loss`)

- **L1**: Mean absolute error on noise prediction. Preserves sharp features (rising edges).
- **L2**: Mean squared error on noise prediction. Smoother gradients but tends to blur edges.

### Spectral losses (on x̂_0 vs x_0)

All spectral losses are optional and controlled by `--w_*` flags (default 0 = disabled).

- **SC** (`--w_sc`): Spectral Convergence. Multi-resolution STFT magnitude error normalized by target magnitude. Partially phase-aware through time-frequency localization. Catches rising edge shifts and pileup shape errors. Common in speech synthesis (HiFi-GAN, MelGAN).

- **LSD** (`--w_lsd`): Log-Spectral Distance. RMS of log-PSD difference between predicted and target. Treats all frequency decades equally on a log scale. Phase-blind.

- **J_asd** (`--w_asd`): Amplitude Spectral Density ratio. `mean over f of sqrt(PSD_residual / PSD_target)`. Measures relative spectral error per frequency bin. Standard evaluation metric in CUORE and LIGO (DeepClean). Used as training loss in gravitational-wave denoising.

- **L1 on x̂_0** (`--w_l1`): Additional L1 loss directly on the reconstructed signal (not the noise prediction). Anchors time-domain amplitude and phase.

### Choosing weights

Start with `--w_sc 0.1 --w_asd 0.1` and adjust. The spectral losses have different magnitudes; use the training log (which prints per-component losses) to balance them. Setting all `w_*` to 0 recovers the original noise-only loss.

## Project Structure

```
src/ddpm/
  unet.py             - 1D U-Net model (UNet1D)
  unet_cond.py        - Scale-conditioned U-Net (UNet1DScaleCond)
  schedule.py         - Quadratic noise schedule (beta, alpha, alpha_bar)
  diffusion.py        - Forward/reverse diffusion, training loss, sampling
  dataset.py          - HDF5 dataset with on-the-fly clean+noise pairing
  spectral_loss.py    - SC, LSD, J_asd spectral loss functions
  preprocess_index.py - Precompute per-shard category index JSONs
  train.py            - Training loop with checkpointing and resume support
  inference.py        - Single/multi-shot inference with metrics and plots
```

### Module reference

- **`schedule.py`** — `DiffusionSchedule(T, beta_1, beta_T)`: precomputes the
  quadratic beta schedule and derived constants `alpha`, `alpha_bar`,
  `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar`. `beta_t` interpolates linearly
  in `sqrt(beta)` space, then is squared and clamped to `[1e-8, 0.999]`.

- **`unet.py`** — `UNet1D`: 1D U-Net (~15.1M params) with 4 encoder levels
  (64→128→256→512), 2 ResBlocks/level, GroupNorm + SiLU, self-attention at
  level 3 and the bottleneck. Conditioning embedding (`step` or `sqrt_ab`) is
  added inside every ResBlock. Input is `[x_t, x_tilde]` (2 channels), output
  is the predicted noise `eps` (1 channel).

- **`unet_cond.py`** — `UNet1DScaleCond`: identical to `UNet1D` plus a
  log-scale conditioning branch. The per-window normalization scale
  `max(|noisy|)` is encoded as `log(scale)*10 → SinusoidalEmbedding → MLP`
  and added into the timestep embedding so the network knows the input SNR.

- **`diffusion.py`** — `GaussianDiffusion(model, schedule, loss_type,
  spectral_loss, cond_mode)`: forward `q(x_t|x_0)`, training loss (noise MSE
  / L1 with optional spectral terms on the Tweedie x̂_0), DDPM ancestral
  sampler, and DDIM sampler with `eta` and configurable step skipping.

- **`dataset.py`** — `PulseNoiseDataset(clean_dir, noise_dir, subset=None)`:
  thread-safe lazy HDF5 reader. Pairs clean window `i` with noise window
  `random(i)` reshuffled each epoch; returns `(x_clean, x_noisy)` of shape
  `(1, 10000)`. `subset='pileup'`, `'low_100'`, etc. filters to indices
  from the precomputed `*_index.json` (built by `preprocess_index.py`).

- **`spectral_loss.py`** — multi-resolution STFT helpers and three spectral
  losses (`SC`, `LSD`, `J_asd`) used on the Tweedie estimate `x̂_0` during
  training.

- **`preprocess_index.py`** — walks each `clean_XXX.h5` and writes a sibling
  `clean_XXX_index.json` grouping sample indices by category
  (`all`, `single`, `pileup`, `low_100`, `low_200`, `pileup_low_100`,
  `pileup_low_200`). Required if `--subset` / `--filter` is used at
  train/inference.

- **`train.py`** — training entry point. Builds dataset, splits train/val,
  configures U-Net + schedule + diffusion wrapper, runs the loop with cosine
  LR schedule, checkpointing every N epochs, best-val checkpointing, and
  resume support. Saves `config.json`, `history.json`, periodic
  `checkpoint_*.pt`, and `best_model.pt` into `--output_dir`.

- **`inference.py`** — inference + QA entry point. Runs DDPM (multi-shot)
  or DDIM (deterministic) sampling, computes the full metric set
  (time-domain + spectral + tri-exp peak fits), and writes per-window QA
  panels, scatter-vs-SNR grids, and a stripped pickle of metric dicts for
  downstream replotting.

## Data Format

HDF5 shards in separate directories:

```
clean/clean_000.h5 ... clean_009.h5   (simulated clean pulses)
noise/noise_000.h5 ... noise_009.h5   (real noise windows from LUCE)
```

Each shard contains a `waveforms` dataset of shape (N, 10000) and an `n_windows` attribute.

## Usage

### Training

Basic (L1 noise loss only):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean/clean_000.h5 \
    --noise_dir /path/to/noise/noise_000.h5 \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

L2 noise loss with AMP (example used for `ddpm_l2_low`, mirrors `ddpm_l1_low` except `--loss l2`):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/train \
    --noise_dir /media/Disk_YIN/yunshancheng/cuore/noise/train \
    --output_dir /media/Disk_YIN/yunshancheng/cuore/ddpm_l2_low \
    --loss l2 --amp \
    --epochs 100 --batch_size 32 --lr 2e-4 --T 50 \
    --val_fraction 0.1 --save_every 10 --seed 42 --num_workers 4 \
    > /media/Disk_YIN/yunshancheng/cuore/ddpm_l2_low/train.log 2>&1 & echo "PID: $!"
```

With spectral losses:

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --w_sc 0.1 --w_asd 0.1 \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

With scale conditioning:

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --scale_cond \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

DeScoD-ECG-style schedule + continuous `√ᾱ` conditioning (larger beta_T so x_T really is Gaussian, continuous noise-level embedding):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --beta_1 1e-4 --beta_T 0.5 --cond_mode sqrt_ab \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

Inference must pass the same `--beta_1`, `--beta_T`, `--cond_mode` (and `--cond_scale`, if non-default) as training.

With mixed precision (AMP):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --amp \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

Resume from checkpoint (restores optimizer and best_val_loss; scheduler is rebuilt for remaining epochs with no warm-up):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --w_sc 0.1 --w_asd 0.1 \
    --epochs 300 --batch_size 32 --lr 1e-5 --num_workers 4 \
    --resume /path/to/output/checkpoint_100.pt \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

On resume, `--lr` sets the starting LR and a fresh cosine schedule decays it to 1e-7 over the remaining epochs. No warm-up or cosine restart.

Use full dataset (all shards) by passing the directory:

```bash
--clean_dir /path/to/clean --noise_dir /path/to/noise
```

### Inference

```bash
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10
```

With scale-conditioned model:

```bash
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10 --scale_cond
```

Filter by sample type:

```bash
# Only low-amplitude pileup (pileup with max < 100 mV)
python3 -u -m src.ddpm.inference ... --filter low_pileup --n 10

# Only pileup events
python3 -u -m src.ddpm.inference ... --filter pileup --n 10

# Only single pulses
python3 -u -m src.ddpm.inference ... --filter single --n 10

# Specific sample indices
python3 -u -m src.ddpm.inference ... --indices 23,31,33,59,65
```

Multi-shot aggregation (default: mean):

```bash
# 10-shot with median aggregation (more robust to outlier samples)
python3 -u -m src.ddpm.inference ... --aggregation median
```

DDIM sampler (deterministic, supports step skipping):

```bash
# DDIM with full 50 steps
python3 -u -m src.ddpm.inference ... --sampler ddim

# DDIM with 10 steps (5x faster)
python3 -u -m src.ddpm.inference ... --sampler ddim --ddim_steps 10

# DDIM with partial stochasticity
python3 -u -m src.ddpm.inference ... --sampler ddim --eta 0.5

# deterministic inference
   nohup stdbuf -oL python3 -u -m src.ddpm.inference \                                                                                                            
     --model_path /media/AVFD/yunshancheng/cuore/ddpm_l1/best_model.pt \                                                                                          
     --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_001.h5 \                                                                                              
     --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_001.h5 \                                                                                              
     --T 50 --n 10 --filter low_pileup \                                                                                                                          
     --output /media/AVFD/yunshancheng/cuore/ddpm_l1/qa_low_pileup_deterministic.png \                                                                            
     --no_noise \                                                                                                                                                 
     > /media/AVFD/yunshancheng/cuore/ddpm_l1/inference_deterministic.log 2>&1 &                                                                                  
   echo $!  
```

### Evaluation Metrics

Time-domain:
- **MSE**: Mean squared error
- **MAD**: Maximum absolute distance
- **BL RMS**: Baseline RMS in the pre-pulse region t in [0, 1.5s) — measures residual noise level after denoising. Compared across noisy, 1-shot, and 10-shot.
- **PRD**: Percentage root-mean-square difference
- **Cosine Similarity**: Waveform shape agreement
- **CC**: Normalized cross-correlation (mean-subtracted)
- **SNR**: Signal-to-noise ratio in dB — `10·log10(Σ clean² / Σ (clean - denoised)²)` (uses total signal energy)
- **PSNR**: Peak signal-to-noise ratio in dB — `10·log10(max(|clean|)² / MSE)`. References error against the *peak* pulse amplitude rather than total energy, so it tracks how clean the pulse peak is relative to residual noise floor (more interpretable than SNR for sparse, peak-dominated bolometer pulses).

Spectral:
- **SC**: Spectral Convergence — STFT magnitude error (partially phase-aware)
- **LSD**: Log-Spectral Distance — RMS log-PSD difference
- **J_asd**: ASD ratio — mean sqrt(PSD_residual / PSD_target)

Peak analysis (`src/basics/fit.py`):
- **Model**: tri-exponential pulse (Carrettoni & Vignati 2011) — one rise + a fast/slow decay mixture: `phi(u) = -exp(-u/τ_r) + α·exp(-u/τ_d1) + (1-α)·exp(-u/τ_d2)`. The two-decay form matches the bolometer thermal response (fast electron-phonon relaxation + slow phonon escape); a single-decay biexponential underfits and biases χ²/ndf.
- **Constraints** (matched to QA convention): baseline B fixed = mean of first 1500 samples, first-pulse onset t01 fixed = 1.5 s, pileup separation t02 ∈ [t01+0.01, t01+2.0] s.
- **Pileup parametrization**: the two pulses share (τ_r, τ_d1, τ_d2) — one detector, one thermal response — but each gets its own (A, α). 10 model params total (8 free).
- **Initial guess**: `scipy.signal.find_peaks` on a Gaussian-smoothed signal, ranked by **prominence** to ignore ripples on the decay slope, with parabolic sub-sample refinement.
- **Peak amplitude / time**: derived from the fit — `peak_amp = A · max(phi)` and `peak_time = t0 + argmax(phi)`, both computed numerically per pulse since the tri-exp shape has no closed-form peak.
- **χ²/ndf**: plain `SSR / (N − len(popt))`, unweighted. Same value across (clean, noisy, 1-shot, 10-shot) is comparable.
- **Reported metrics**: `Pk amp` (mV), `Pk amp%` (relative error vs clean), `Pk dt ms` (timing error vs clean). Pileup pulses are sorted by peak time and matched between clean and denoised by index.

### QA Plot Layout

The inference QA plot has 4 columns per sample:
1. **Waveforms**: Clean, noisy, 1-shot, and 10-shot overlay
2. **Residuals**: Clean minus denoised (1-shot and 10-shot)
3. **PSD**: Power spectral density comparing clean, noisy, and 10-shot denoised
4. **Metrics**: Per-sample metrics table with all numeric results

## CLI Reference

### `train.py` arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--clean_dir` | required | Path to a clean shard file (`clean_XXX.h5`) or a directory containing `clean_*.h5`. Directory loads all shards. |
| `--noise_dir` | required | Same convention for noise shards (`noise_*.h5`). |
| `--output_dir` | required | Where to write `config.json`, `history.json`, `checkpoint_*.pt`, `best_model.pt`. Created if missing. |
| `--epochs` | 100 | Total epochs to run. On resume, this is the *final* target epoch (not "more epochs"). |
| `--batch_size` | 16 | Mini-batch size for the train DataLoader. Val uses the same. |
| `--lr` | 2e-4 | Peak learning rate. Cosine-decayed to `1e-7` across the (remaining) epochs. No warm-up. |
| `--T` | 50 | Number of diffusion steps in the forward/reverse process. |
| `--beta_1` | 1e-4 | Starting beta of the quadratic noise schedule. |
| `--beta_T` | 0.5 | Ending beta of the quadratic noise schedule. With 0.5, `alpha_bar_T ≈ 5e-5` (effectively standard Gaussian). |
| `--cond_mode` | `step` | `step` = condition on integer step index 1..T (sinusoidal embed). `sqrt_ab` = continuous `√ᾱ` conditioning (WaveGrad / DeScoD-ECG). Existing checkpoints use `step`. |
| `--cond_scale` | 1000.0 | Multiplier applied to `√ᾱ` before its sinusoidal embedding, so `(0, 1]` spans the frequency grid. Ignored when `cond_mode='step'`. |
| `--val_fraction` | 0.1 | Fraction of the dataset used for validation (random split, seeded). |
| `--save_every` | 10 | Save `checkpoint_<epoch>.pt` every N epochs (in addition to `best_model.pt`). |
| `--seed` | 42 | RNG seed for torch, the train/val split, and the dataset's per-epoch noise reshuffle. |
| `--num_workers` | 0 | DataLoader workers. 4 is a good default on a single GPU. |
| `--loss` | `l2` | Base noise-prediction loss: `l1` (sharper edges) or `l2` (smoother). |
| `--w_l1` | 0.0 | Extra L1 weight on x̂_0 (time-domain signal). 0 disables. |
| `--w_sc` | 0.0 | Multi-resolution Spectral Convergence weight on x̂_0. 0 disables. |
| `--w_lsd` | 0.0 | Log-Spectral Distance weight on x̂_0. 0 disables. |
| `--w_asd` | 0.0 | J_asd (ASD ratio, DeepClean / CUORE metric) weight on x̂_0. 0 disables. |
| `--resume` | None | Path to a `checkpoint_*.pt` (or `best_model.pt`) to resume from. Restores model, optimizer, epoch, and `best_val_loss`. Cosine LR is rebuilt for the remaining epochs starting at `--lr` (no warm-up). |
| `--amp` | False | Enable mixed-precision (float16) training via `torch.cuda.amp`. |
| `--compile` | False | Wrap the U-Net in `torch.compile` for kernel fusion (slower first epoch). |
| `--scale_cond` | False | Use `UNet1DScaleCond` instead of `UNet1D` (adds log-scale conditioning). Must match at inference. |
| `--subset` | None | Filter the training set to a category from the precomputed `_index.json`: `pileup`, `single`, `low_100`, `low_200`, `pileup_low_100`, `pileup_low_200`, or custom. Requires `preprocess_index.py` first. |

### `inference.py` arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_path` | required | Path to `best_model.pt` / `checkpoint_*.pt`. The sibling `config.json` is *not* read; pass schedule/cond flags manually. |
| `--clean_dir` | required | A clean shard file *or* directory (same convention as training). |
| `--noise_dir` | required | A noise shard file *or* directory. |
| `--output` | `qa_inference.png` | Per-window QA panel (waveforms / residuals / PSD / metrics). One row per sample, up to `--n_plot`. |
| `--n` | 200 | Number of samples to process for metric statistics (scatter / pickle). |
| `--n_plot` | 10 | Number of samples drawn in the QA panel (must be `≤ n`). |
| `--scatter_output` | None | If set, write a metric-vs-SNR scatter grid here. |
| `--scatter_pileup_output` | None | If set, write the pileup-vs-single scatter grid here. |
| `--results_pkl` | None | If set, dump stripped per-window metric dicts here for `scripts/replot_scatter.py` (no waveforms — small file). |
| `--T` | 50 | Diffusion steps. Must match the schedule the model was trained on. |
| `--beta_1` | 1e-4 | Starting beta. **Must match training.** |
| `--beta_T` | 0.5 | Ending beta. **Must match training.** |
| `--cond_mode` | `step` | Conditioning mode. **Must match training.** |
| `--cond_scale` | 1000.0 | `√ᾱ` multiplier. **Must match training** (ignored in `step` mode). |
| `--seed` | 123 | Selection seed for samples (when `--indices` is not given) and the reverse-process noise seed. |
| `--indices` | None | Comma-separated explicit sample indices (e.g. `23,31,33`). Overrides `--n` and `--seed`-driven random selection. |
| `--filter` | None | Restrict samples to a category. Built-in: `pileup`, `single`, `low_pileup` (= `pileup_low_100`). With `*_index.json`: `low_100`, `low_200`, `pileup_low_200`, plus any custom category. |
| `--scale_cond` | False | Use `UNet1DScaleCond`. **Must match training.** |
| `--aggregation` | `mean` | Multi-shot reducer: `mean` or `median` (median is more robust to outlier samples). |
| `--sampler` | `ddpm` | `ddpm` = stochastic ancestral sampler (multi-shot averaging is the recommended mode). `ddim` = deterministic. |
| `--ddim_steps` | None | Number of DDIM steps (skip-by-K). `None` = use full `T`. |
| `--eta` | 0.0 | DDIM stochasticity: 0 = deterministic, 1 = DDPM-equivalent. Ignored when `--sampler ddpm`. |
| `--no_noise` | False | Deterministic DDPM: drop the per-step noise term in the reverse process. Useful for paired before/after comparisons. |

## Scripts (`scripts/`)

Two groups: data generation (clean + noise shards) and post-training
analysis / paper figures. Most analysis scripts work on a single
"denoised test" HDF5 emitted by `denoise_test_set.py`, so they need no
GPU and re-run in seconds.

### Data generation

- **`generate_clean_shards.py`** — drives `src.pulse.generate.generate_dataset`
  to produce simulated clean shards.
  - `--output_dir` (required): destination directory.
  - `--n_shards` (10): number of `clean_NNN.h5` shards.
  - `--windows_per_shard` (10000): windows per shard.
  - `--base_seed` (42): RNG seed; shard `k` uses `base_seed + 1000*k`.
  - `--E_min` (1.0) / `--E_max` (5407.0): energy range in keV.
  - `--pileup_fraction` (0.3): fraction of windows that are two-pulse pileups.

- **`generate_noise_shards.py`** — generates three paired noise families
  (`noise_periodic/`, `noise_white/`, `noise_total/`). For each window
  `i` of shard `NNN`, the three folders contain the matched
  periodic / white / mixed components plus the per-window mix metadata
  (`alpha`, `beta`, `r=α²`, `scale`) used by downstream
  metrics-vs-mix scripts.
  - `--output_dir`: root for the three subdirs (default
    `/media/Disk_YIN/yunshancheng/cuore/noise_v2`).
  - `--split` (''): optional subdir under each of `noise_*/` (e.g. `test`).
  - `--n_shards` (10), `--windows_per_shard` (10000), `--base_seed` (7777):
    same semantics as for clean shards.

### Post-training analysis pipeline

Run `denoise_test_set.py` once, then point the plotting / metrics
scripts at the resulting HDF5.

- **`denoise_test_set.py`** — deterministic 1-shot DDPM over the paired
  test set (`clean[i] ↔ noise[i]`, no permutation). Writes a single
  HDF5 mirroring the clean/noise layout: `waveforms` (denoised),
  `waveforms_clean`, `waveforms_noisy`, `scale`, plus copies of
  per-window detector params, equilibrium state, noise params, and the
  `noise/mix/{alpha, beta, r, scale}` metadata.
  - `--clean_h5` (required), `--noise_h5` (required, must contain
    `/mix/`), `--model_path` (required), `--output` (required H5 path).
  - `--T` (50), `--beta_1` (1e-4), `--beta_T` (0.5), `--cond_mode`
    (`step`): must match the training schedule of `--model_path`.
  - `--batch_size` (64): inference batch size.
  - `--n_limit` (None): stop after this many windows; default processes
    everything in the file.

- **`plot_denoising_paper.py`** — 4-row paper figure (single low/high
  amplitude, pileup low/high amplitude) read directly from the denoised
  HDF5. Left column: time-domain waveform overlay. Right column:
  observation / noise / clean / denoised PSDs **plus a
  denoised-baseline trace** computed over the 0–1.5 s pre-pulse window —
  this isolates the residual noise floor of the denoiser without the
  pulse dominating the spectrum.
  - `--denoised_h5`: input from `denoise_test_set.py`.
  - `--out`: output PDF/PNG path.

- **`plot_metrics_vs_mix.py`** — scatters per-window metrics against
  the noise-mix periodic-energy fraction `r = α²` so you can see
  whether the denoiser degrades more in white-dominated or
  structured-dominated noise.
  - `--denoised_h5` (required): denoised test HDF5.
  - `--n` (500): random subset size for the scatter.
  - `--seed` (0): subset RNG.
  - `--out_pkl` (required): pickle of per-window metric dicts.
  - `--out_png` (required): output figure.
  - `--title_suffix` (''): appended to each subplot title.

- **`plot_metrics_profile.py`** — reads the pickle from
  `plot_metrics_vs_mix.py` and converts the scatter into a binned
  mean ± SEM profile vs `r` (no re-inference, no re-fit).
  - `--results_pkl` (required): pickle from the scatter script.
  - `--out_png` (required).
  - `--n_bins` (10): equal-width bins over `r ∈ [0, 1]`.
  - `--min_count` (5): bins below this count get NaN'd out.
  - `--title_suffix` (''): appended to each subplot title.

- **`replot_scatter.py`** — re-renders the inference scatter grids from
  the small pickle produced by `inference.py --results_pkl`. Iterate on
  styling without re-running inference. Supports two x-axes (noisy SNR
  or clean peak amplitude in mV).
  - `--results_pkl` (required).
  - `--scatter_output` / `--scatter_pileup_output`: SNR-axis grids.
  - `--scatter_output_vs_amp` / `--scatter_pileup_output_vs_amp`:
    clean-peak-amplitude-axis grids. Any subset of the four outputs
    may be requested per run.

### Model comparison + debugging

- **`compare_models_pileup_qa.py`** — side-by-side QA of two DDPM
  checkpoints on the same deterministic 1-shot pileup samples. Reads
  each model's training schedule (`beta_1`, `beta_T`, `cond_mode`) from
  its sibling `config.json`, so reverse processes match training. Both
  models share the same initial `x_T` per sample (controlled by
  `--init_seed`) — differences come only from weights and schedule.
  - `--model_a` / `--model_b` (required), `--label_a` / `--label_b`:
    legend labels.
  - `--clean_dir`, `--noise_dir`, `--output` (required).
  - `--n_pileup` (30): number of pileup samples drawn.
  - `--T` (50): inference steps for both models.
  - `--seed` (2026): selection seed for which pileup indices to draw
    (matches `debug_pileup_fit_fail.py` so the same windows can be
    cross-referenced).
  - `--init_seed` (12345): seed for initial `x_T`, shared across both
    models per sample.
  - `--highlight`: indices to mark as user-flagged in the figure.

- **`compare_models_focus.py`** — slimmer twin of the above for a small
  hand-picked list of indices, plotted at large size so jitter / dip
  features are visible.
  - `--model_a` / `--model_b` / `--label_a` / `--label_b`,
    `--clean_dir`, `--noise_dir`, `--output` (required).
  - `--indices` (required, `nargs='+'`): exact window indices to plot.
  - `--T` (50), `--init_seed` (12345): same meaning as above.

- **`debug_pileup_fit_fail.py`** — runs DDPM inference on a handful of
  pileup samples, fits clean / noisy / 1-shot / 10-shot with
  `triexp_double`, and flags any fit that raised in `curve_fit` or
  pinned a rail (τ bound). Plots data+fit overlays for the bad windows
  and prints params, bounds, and χ²/ndf.
  - `--model_path`, `--clean_dir`, `--noise_dir` (required).
  - `--output` (default `/tmp/pileup_fit_fail.png`).
  - `--n_pileup` (30): how many pileup samples to inspect.
  - `--T` (50), `--seed` (2026): inference + sample-selection seed.
  - `--no_noise`: deterministic DDPM (skip per-step noise).

- **`qa_triggers.py`** — QA comparison of `easytrigger` (fixed baseline)
  vs `trigger` (adaptive rolling) on a single IETI `.bin` file. No CLI
  arguments — the input path and constants are set at the top of the
  file. Emits two PNGs, each with 10 triggered windows.

### Standalone paper figures

- **`plot_trajectory_paper.py`** — renders DDPM reverse-diffusion
  trajectory snapshots (noisy + T=50, 30, 29, 0) for a single ~40 mV
  non-pileup pulse as five bare-axes PNGs.
  - `--model_path`, `--clean_h5`, `--noise_h5`: paths (sensible defaults
    in-file).
  - `--clean_idx` (1364), `--noise_idx` (0): index pair to use.
  - `--T` (50), `--seed` (0): inference settings.
  - `--out_dir`: output directory for the five PNGs.

- **`plot_datasets_paper.py`** — 4×2 overview figure of the simulation
  datasets (single / pileup × low / high energy), each row in time
  domain + PSD with clean / noisy / pure-noise overlaid.
  - `--seed` (0): index selection RNG.
  - `--out`: output figure path.

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Li et al., "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal" (arXiv:2208.00542)
- Ormiston et al., "Noise reduction in gravitational-wave data via deep learning" (Phys. Rev. Research, 2020) — DeepClean ASD loss
- Stevens et al., "Removing Structured Noise with Diffusion Models" (TMLR, 2025) — Joint diffusion framework
