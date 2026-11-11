# DDPM Pulse Denoiser

Denoising Diffusion Probabilistic Model (DDPM) for denoising cryogenic bolometer pulse signals from the CUORE experiment.

## Quick Start

Use the existing ODE clean shards and synthetic noise shards. Set `DATA` to
the cloud directory that contains `simu_data/` and `RUN` to a new model output
directory. Start from a fresh scale embedding + FiLM model.

```bash
DATA=/path/to/project-data
RUN=/path/to/new-model-run
python3 -u -m src.ddpm.train \
    --clean_dir "$DATA/simu_data/clean" \
    --noise_dir "$DATA/simu_data/noise" \
    --output_dir "$RUN" --scale_emb --film
```

Record the resolved configuration. Match the architecture and diffusion
settings from training at inference. To run a small QA sample once weights
are available, specify clean and noise shards from the held-out inputs.

```bash
python3 -u scripts/denoise_test_set.py --scale_emb --film \
    --clean_h5 "$DATA/simu_data/clean_holdout/clean_000.h5" \
    --noise_h5 "$DATA/simu_data/noise_holdout/noise_000.h5" \
    --model_path "$RUN/best_model.pt" --n_limit 32 \
    --output "$RUN/qa_smoke.h5"
```

The test script uses a seeded Gaussian start and skips per-step noise
injection, matching the `--no_noise` sampler convention. Formal resolution
comparisons use the existing held-out event windows, the `Fit_Jiaxuan` fitter,
and the same cuts and energy calibration for both models. The old batch
evaluation scripts still need a check with the new checkpoint before the
full comparison. Historical warm-start scripts can be reproduced from
`Fit_Jiaxuan` (`780d190`).

New clean shards default to ODE. New noise shards default to the original
single-file format. See [pulse generation](../pulse/README.md) and
[noise generation](../noise/README.md) for the optional tri-exponential and
paired modes and use held-out seeds for any new test set.

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

**Status: legacy.** This class is kept only so old checkpoints (e.g.
`ddpm_scale_cond/`) still load. New models should use `UNet1D` with
`--scale_emb` and optionally `--film` (next section).

### Scale embedding + AdaGN/FiLM (`--scale_emb`, `--film`)

These are two independent switches on `UNet1D`, so the 2×2 ablation (none /
scale only / FiLM only / both) uses the same code path. Both also work with
the channel token (`n_channels`).

**`--scale_emb`** (`ScaleEmbedding` in `unet.py`) conditions on the same
`scale = max(|noisy|)`. This is an observable, available at inference, so
there is no truth leak.

```
x = (log10(scale) - c) / w        # log10_range (-2, 1) -> [-1, 1]  (10 mV .. 10 V)
feats = [x, sin(pi 2^k x), cos(pi 2^k x)],  k = 0..5
scale_emb = MLP(feats)            # 13 -> 1024 -> 256
cond = time_emb (+ channel_emb) + scale_emb
```

- The measured log10(scale) spans [-1.73, -0.37] on `clean_v3/clean_low`,
  [-1.23, 1.05] on `clean_v3/clean_high` and [-1.58, 1.02] on
  `clean_triexp`. The default range covers all three, and values outside it
  are not clamped.
- The legacy encoding fed `10·ln(scale)` into the integer-timestep sinusoidal
  table, an arbitrary ×10 on frequencies designed for steps 1..T. The new
  Fourier frequencies are matched to the normalized range: the finest repeats
  every ~0.09 decades, and the coarser ones keep the code unique.
- Adds about 0.28M parameters.

**`--film`** switches every ResBlock from an additive bias to AdaGN, the
improved-DDPM / ADM block:

```
additive (default):  h = SiLU(GN1(conv1 x)) + Linear(cond);  h = SiLU(GN2(conv2 h))
FiLM / AdaGN:        h = SiLU(GN1(conv1 x));  (gamma, beta) = Linear(SiLU(cond))
                     h = SiLU(GN2(conv2 h) * (1 + gamma) + beta)
```

- **Why:** with an additive bias, `Linear(t_emb + s_emb)` is a sum of
  separate terms, so t and scale can only shift channel activations. FiLM
  lets the conditioning *scale* features. That is what SNR-dependent
  denoising needs: suppress input detail when the window is noise-dominated,
  trust it when it is pulse-dominated.
- **Initialization:** the projection is zero-initialized, so at init every
  block is a plain GroupNorm and the output does not depend on the
  conditioning (verified). Conditioning is then learned as a delta.
- **Cost:** adds about 1.1M parameters (16.5M total with `--scale_emb`). The
  FiLM projection uses the attribute name `film_proj`, not `t_proj`, so
  additive and FiLM checkpoints fail loudly rather than load into each other.

**Why both, and why separate switches.** The goal is the combination
(scale-aware *and* multiplicative). An improvement from the combined model
cannot be attributed to either part without the single-switch controls, so
the switches are kept orthogonal.

**Flag rules:**
- `--scale_cond` (legacy) cannot be combined with `--scale_emb` or `--film`.
- All three must match at inference, because `inference.py` does not read
  `config.json`.
- Model construction for both scripts goes through `build_ddpm_model()` in
  `unet.py`.

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
  Legacy: superseded by `UNet1D(scale_cond=True, film=...)`, kept for old
  checkpoints.

- **`diffusion.py`** — `GaussianDiffusion(model, schedule, loss_type,
  spectral_loss, cond_mode, out_channels=1)`: forward `q(x_t|x_0)`, training
  loss (noise MSE / L1 with optional spectral terms on the Tweedie x̂_0), DDPM
  ancestral sampler, and DDIM sampler with `eta` and configurable step
  skipping. `out_channels` is the channel count of x_0 that both samplers
  start from (default 1). The joint pileup model in `src/pileup_joint/` uses
  2, one channel per pulse; the conditioning y stays 1 channel.

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
clean/clean_*.h5                    (simulated ODE pulses)
noise/noise_*.h5                    (synthetic noise windows)
```

Each shard contains a `waveforms` dataset of shape (N, 10000) and a window-count attribute (`n_windows` or `n_pulses`).

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

The command above is the legacy `UNet1DScaleCond`. For new models, use scale
embedding + AdaGN/FiLM (drop either flag for the ablation controls):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --scale_emb --film --amp \
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

# Gaussian start, with no per-step reverse noise
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/heldout/clean_000.h5 \
    --noise_dir /path/to/heldout/noise_000.h5 \
    --scale_emb --film --no_noise --n 10 --output qa_inference.png
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
- **Model**: tri-exponential pulse with one rise and two decays. Baseline and onset are fitted within their bounds.
- **Pileup**: fit the first pulse, subtract it, and fit the second pulse on the residual. The returned vector contains 13 parameters with independent decay constants. Reconstruct a successful fit with `result.model_fn(t, *result.params)`.
- **Peak amplitude and time**: obtained from the maximum of the fitted pulse. The prefactor `A` is distinct from the peak amplitude.
- **χ²/ndf**: residuals divided by the estimated pre-trigger noise standard deviation (or the supplied noise scale), squared and summed, then divided by the degrees of freedom. The older shared-time-constant fit and its unscaled residual statistic are available in `b5cd470` for historical QA reproduction.
- **Reported metrics**: peak amplitudes and timing errors are compared only when both fits succeed.

### QA Plot Layout

The inference QA plot has 4 columns per sample:
1. **Waveforms**: Clean, noisy, 1-shot, and 10-shot overlay
2. **Residuals**: Clean minus denoised (1-shot and 10-shot)
3. **PSD**: Power spectral density comparing clean, noisy, and 10-shot denoised
4. **Metrics**: Per-sample metrics table with all numeric results

## Historical results: `ddpm_triexp_scale_film`

These results come from the tri-exponential and paired-noise experiment using its original QA. The planned ODE run uses its own training configuration and evaluation.

<details>
<summary>Original training and test commands</summary>

Historical training and test recipe from yunshan’s tri-exponential/paired-noise experiment. `M` is the original data root and `D` the model directory.

```bash
M=/media/Disk_YIN/yunshancheng/cuore
D=$M/my_new_model

# 1. Clean training + test data (tri-exp pulses). Noise: $M/noise_v2/noise_total
python3 -u scripts/generate_clean_shards.py --output_dir $M/clean_triexp \
    --model triexp --tau_energy_scaling
python3 -u scripts/generate_clean_shards.py --output_dir $M/clean_triexp/test \
    --model triexp --tau_energy_scaling --n_shards 1 --base_seed 99999

# 2. Train
mkdir -p $D
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir $M/clean_triexp --noise_dir $M/noise_v2/noise_total \
    --output_dir $D --loss l2 --scale_emb --film --amp \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > $D/train.log 2>&1 &

# 3. Denoise the test set
python3 -u scripts/denoise_test_set.py --scale_emb --film \
    --clean_h5 $M/clean_triexp/test/clean_000.h5 \
    --noise_h5 $M/noise_v2/noise_total/test/noise_000.h5 \
    --model_path $D/best_model.pt --output $D/denoised_test_000.h5

# 4. Plots + metrics -> $D/qa_test/
python3 -u scripts/plot_triexp_inference.py --denoised_h5 $D/denoised_test_000.h5
```

`--scale_emb --film` must be passed at training **and** at every inference step.
Already have data? Skip step 1.

</details>

Test set: 10000 windows, 30.6 % pileup.

Deterministic 1-shot DDPM with the epoch-95 best model:

| | noisy | denoised |
|---|---|---|
| median RMS error | 9.38 mV | 0.476 mV |
| single-pulse SNR (median) | 10.1 dB | 38.2 dB |
| pileup SNR (median) | 20.2 dB | 45.0 dB |
| median abs. amplitude error (fit) | 0.95 % | 0.48 % |

Observations that drive the next steps:

- **Baseline wiggle.** About 10 % of windows, mostly below 100 keV, get a
  model-generated 2–6 Hz baseline wiggle of about 0.3 mV before the pulse.
  It is not correlated with the input noise.
- **Longer training.** Resuming to 200 epochs lowers the validation loss
  by about 10 % (best at epoch 190). Under deterministic DDPM, though, the
  epoch-190 model is worse on the test set: single-pulse MAE goes from
  277 to 385 µV, and the 4–9 s tail is biased low. Its baseline is
  cleaner: baseline RMS 25.8 → 10.7 µV, and the share of windows with a
  wiggle drops from 10.5 % to 7.7 %.
- **Sampler vs. network** (`compare_samplers.py`, 1000 windows):
  - DDIM at 10, 25 and 50 steps gives nearly the same metrics, so step
    count is not the bottleneck and a larger T is unlikely to help.
  - The low-biased tail comes mainly from deterministic DDPM, which only
    drops the noise term and is not a consistent ODE sampler. DDIM
    shrinks it by 4–5×.
  - DDIM is much better above 1.5 MeV (MAE roughly halved, SNR +5 dB).
    It is worse at low energy for the epoch-95 model and produces more
    baseline wiggle.
  - The remaining error is network-limited. A deeper U-Net with more
    downsampling levels, for a larger receptive field, is the next
    candidate.

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
| `--preload` | False | Read clean and noise waveforms into system RAM before training. |
| `--patience` | 0 | Stop after this many epochs without validation improvement; 0 disables early stopping. |
| `--amp` | False | Enable mixed-precision (float16) training via `torch.cuda.amp`. |
| `--compile` | False | Wrap the U-Net in `torch.compile` for kernel fusion (slower first epoch). |
| `--scale_cond` | False | Legacy: use `UNet1DScaleCond` instead of `UNet1D`. Only for old checkpoints. Must match at inference. |
| `--scale_emb` | False | `UNet1D` conditioned on `max(|noisy|)` via the log10 Fourier-feature `ScaleEmbedding`. Must match at inference. |
| `--film` | False | AdaGN/FiLM conditioning in every ResBlock instead of an additive bias. Must match at inference. |
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
| `--scale_cond` | False | Legacy `UNet1DScaleCond`. **Must match training.** |
| `--scale_emb` | False | Scale-embedding `UNet1D`. **Must match training.** |
| `--film` | False | AdaGN/FiLM ResBlocks. **Must match training.** |
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
  - `--model` (`ode`): use `triexp` to select the analytic pulse template.

- **`generate_noise_shards.py`** — writes one `noise_NNN.h5` per shard by
  default with the original parameter sampling and `waveforms`/`params` layout.
  - `--mode paired`: write matching `noise_periodic/`, `noise_white/` and
    `noise_total/` shards. Only total shards contain the mix metadata.
  - `--n_channels`: generate a paired channel bank with fixed parameters per channel.
  - `--output_dir`: pass a separate output directory for the intended dataset.
  - `--split`: optional subdirectory such as `test`.
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
  `noise/mix/{alpha, beta, r, scale}` metadata when present.
  - `--clean_h5`, `--noise_h5`, `--model_path`, `--output` are required.
    `--noise_h5` accepts a legacy shard or a paired `noise_total` shard.
  - `--T` (50), `--beta_1` (1e-4), `--beta_T` (0.5), `--cond_mode`
    (`step`): must match the training schedule of `--model_path`.
  - `--scale_cond` / `--scale_emb` / `--film`: architecture flags, must
    match training.
  - `--batch_size` (64): inference batch size.
  - `--seed` (0): seeds `x_T` (deterministic DDPM still starts from random
    noise), so reruns are bit-identical.
  - `--n_limit` (None): stop after this many windows; default processes
    everything in the file.

- **`plot_triexp_inference.py`** — QA for a denoised tri-exp test set
  (full 1 keV–5.4 MeV range). Writes to `--out_dir` (default
  `<dir of denoised_h5>/qa_test`):
  - `examples_single.png`: nearest single-pulse window to each of
    50/100/300/600/1000/2615/5000 keV; columns time / residual / PSD.
  - `examples_pileup.png`: pileups at Δt ≈ 0.05/0.3/1.0/1.8 s.
  - `shape_vs_energy.png`: clean waveforms of the single examples scaled
    to max|clean| = 1, to compare pulse shape vs energy.
  - `metrics_vs_energy.png` (binned median + 16–84 % band) and
    `metrics_vs_snr.png` (scatter), single vs pileup.
  - `metrics_summary.txt`, `metrics.pkl` (cached per-window metrics,
    including `fit_pulse` amplitude / peak-time errors and pre-pulse
    baseline RMS).
  - `--workers` (20): processes for the per-window fits; `--n_ebins` (12).
  - Exports `TYPICAL_E_KEV`, `PILEUP_DT_S`, `pick_single`, `pick_pileup`,
    `compute_psd` so the scripts below reuse the same example windows.

- **`denoise_examples_multishot.py`** — re-denoises the example windows
  M times (`--M` 10) and compares the shot mean with the saved 1-shot.
  Default is deterministic DDPM (shots differ only in `x_T`);
  `--stochastic` uses stochastic DDPM shots. Needs the architecture flags.
  Writes `multishot[_stoch]_examples.h5 / _single.png / _pileup.png /
  _metrics.txt`.

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

- **`compare_denoised_examples.py`** — overlays two denoised test HDF5s
  (same clean/noise pair, e.g. two checkpoints) on the QA example windows
  of `plot_triexp_inference.py`. `--h5_a/--h5_b`, `--label_a/--label_b`,
  `--out_prefix` (writes `<prefix>_single.png`, `<prefix>_pileup.png`).

- **`compare_samplers.py`** — re-denoises every `--stride`-th (10) window
  of a denoised test HDF5 with deterministic DDPM and DDIM (eta=0) at
  50/25/10 steps, for one or more checkpoints (`--models name=path ...`).
  Answers "is the residual from the sampler's discretization or from the
  network?". Writes `qa_samplers/samplers.h5` (all denoised windows) and
  `samplers_summary.txt` (MAE, PRD, SNR, baseline RMS, 4–9 s tail bias,
  fraction of windows with baseline wiggle, per energy group).

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
  the project fitter, and flags any fit that failed or
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

## Multi-channel foundation model

Real detectors have ~1000 channels, each with its own pulse-shape and noise
identity. Rather than train one U-Net per channel, we train **one shared
backbone** and adapt to each channel with a tiny **channel token** (Tier-1
adapter: one learned `emb_dim` embedding row per channel, added to the timestep
embedding in `UNet1D`, enabled via `n_channels`). The diffusion framework is
unchanged — the token is just extra conditioning, threaded through
`diffusion.py` as `channel_id`.

### Data

Channels are a data-layout concept: one shard per channel (`channel_id == shard
index`). Detector params (pulse identity) and noise PSD params + mix `r`/`scale`
(noise identity) are **fixed per channel**; event energy/pileup and the noise
realization vary per window. See `src/pulse/README.md` and `src/noise/README.md`
(channel-bank sections) for the generators. `ChannelBankDataset` pairs clean and
noise **within the same channel** and returns `channel_id`.

### Two splits (don't conflate them)

- **Channel split**: train channels (pretrain) vs held-out channels (fine-tune +
  eval). Held-out channels are never seen during pretraining.
- **Window split** (held-out channels only): fine-tune windows `[0, K_pool)` vs
  disjoint test windows `[test_start, …)`. Train channels use all their windows.

### Workflow

The whole sweep (pretrain → token fine-tune sweep → oracle → eval) is wrapped in
`scripts/run_foundation.sh` — edit the variables at the top and run it (under
`nohup` for a long job). Equivalent step-by-step commands:

```bash
# Pretrain — backbone + train-channel tokens (channels 0-79)
python -u -m src.ddpm.train_channel --phase pretrain \
    --clean_dir .../channel_bank/clean \
    --noise_dir .../channel_bank/noise/noise_total \
    --output_dir .../runs/foundation \
    --n_channels 100 --train_channels 0-79 --epochs 100 --batch_size 16

# Fine-tune — freeze backbone, fine-tune held-out tokens (the method), per K
for K in 10 50 100 200 400 800; do
  python -u -m src.ddpm.train_channel --phase finetune \
    --clean_dir .../channel_bank/clean --noise_dir .../channel_bank/noise/noise_total \
    --output_dir .../runs/foundation --resume .../runs/foundation/backbone.pt \
    --finetune_channels 80-99 --n_finetune_windows $K --epochs 200 --lr 1e-3
done

# Oracle ceiling — full backbone fine-tune on the largest pool.
# Use a low LR + cosine annealing: a full fine-tune at constant high LR drifts
# the backbone away from the calibrated eps-prediction the sampler needs, which
# degrades reverse-diffusion sampling even while the train loss keeps dropping.
python -u -m src.ddpm.train_channel --phase finetune --finetune_mode full \
    --lr_schedule cosine \
    --clean_dir .../channel_bank/clean --noise_dir .../channel_bank/noise/noise_total \
    --output_dir .../runs/foundation --resume .../runs/foundation/backbone.pt \
    --finetune_channels 80-99 --n_finetune_windows 800 --epochs 60 --lr 2e-5

# Eval — ladder (zero-shot / fine-tuned / oracle) + data-efficiency curve
python -u -m src.ddpm.eval_channel \
    --clean_dir .../channel_bank/clean --noise_dir .../channel_bank/noise/noise_total \
    --backbone .../runs/foundation/backbone.pt \
    --finetuned .../runs/foundation/finetuned_nft*.pt \
    --oracle .../runs/foundation/oracle_nft800.pt \
    --channels 80-99 --test_start 800 --max_test_windows 200 \
    --output_dir .../runs/foundation/eval
```

The **ladder** (all on the same held-out test windows): *zero-shot* (backbone
with the **mean trained token** for held-out channels — see below), *fine-tuned
token* (256 params), *oracle* (full backbone fine-tune, ~15M params — the
ceiling). Inference is the standard
reverse-diffusion denoise plus `channel_id` (single-shot deterministic DDPM in
eval: `diffusion.sample(x_tilde, channel_id=c, stochastic=False)`). If the
cheap token approaches the oracle, one backbone + a per-channel token replaces
1000 U-Nets.

**Zero-shot token choice.** Held-out channel rows in the embedding table are
zero-initialized and never updated in pretrain, so feeding the raw zero row is
out-of-distribution conditioning (the backbone only ever saw the learned,
nonzero train tokens). `eval_channel.fill_untrained_with_mean` instead sets each
untrained row to the **mean of the trained tokens** — an "average channel" prior
the backbone has actually seen — which is the honest zero-shot baseline.

**Oracle LR schedule.** `--lr_schedule cosine` anneals the LR to `1e-7` and
applies to the **oracle (full) fine-tune only**; the token fine-tune always uses
constant LR (it converges fine, and a 256-param fit doesn't need annealing). A
constant high LR on a full fine-tune was observed to collapse the oracle below
even the zero-shot baseline despite a healthy train loss.

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Li et al., "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal" (arXiv:2208.00542)
- Ormiston et al., "Noise reduction in gravitational-wave data via deep learning" (Phys. Rev. Research, 2020) — DeepClean ASD loss
- Stevens et al., "Removing Structured Noise with Diffusion Models" (TMLR, 2025) — Joint diffusion framework
