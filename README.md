# Bolometer Pulse Denoiser

Research code organization: [code layout](docs/CODE_LAYOUT.md), [collaboration conventions](AGENTS.md), and [2026-11-10 migration/validation](docs/reorganization_20261110/README.md).

Generative-model-based denoising of cryogenic bolometer pulse signals from the [CUORE experiment](https://cuore.lngs.infn.it/). CUORE searches for neutrinoless double-beta decay using 988 TeO2 bolometers at ~10 mK. Each channel records voltage pulses proportional to deposited energy, contaminated by thermal and electronic noise. This project trains conditional generative models to recover clean pulse waveforms from noisy observations to improve energy resolution.

## Quick Start

Train the scale embedding + FiLM model on the existing ODE clean shards and
synthetic noise shards. See [DDPM training and evaluation](src/ddpm/README.md#quick-start)
for the commands and the holdout workflow. For new generated data, the default
clean model is ODE and the default noise mode writes one HDF5 per shard. The
tri-exponential pulse model and paired noise are available as explicit options.

The high-energy comparison uses identical evaluation events, sampling settings,
fit rules and calibration for the old and new model. Compare with the independent
[window OF tool](scripts/window_of_README.md).

## Data Pipeline

### Pulse Simulation (`src/pulse/`)

Physics-based simulation of clean bolometer pulses. A nonlinear electro-thermal ODE model ([arXiv:2205.04549](https://arxiv.org/abs/2205.04549)) describes the coupled thermal-electrical response of the TeO2 crystal + NTD-Ge thermistor system:

- 4-ODE perturbation system: energy deposit propagates through electron → crystal → PTFE → heat sink
- Thermistor readout: voltage pulse from temperature-dependent resistance R(T) = R0 * exp((T0/T)^0.5)
- Signal chain: 10 kHz ODE → 120 Hz Bessel low-pass → 1 kHz output (10-second windows, 10000 samples)
- Detector parameters randomized per pulse to capture channel-to-channel variation across 988 channels
- Default generation range: 1–5407 keV. The energy sampler reads `data/cuore_spectrum.npz` when available and uses a parametric spectrum otherwise.
- 70% single pulses (onset at 1.5 s), 30% pileup (second pulse 0.01–2.0 s after first)

Output: HDF5 shards with `waveforms` (N, 10000), `energies_1/2`, `onsets_1/2`, `is_pileup`, `params`, and ODE `equilibrium`.

### Noise (`src/noise/`)

Realistic synthetic noise generator combining five sources observed in real CUORE data:

1. **AC power line harmonics** — 50 Hz + 20 harmonics, power-law decay
2. **Pulse tube cooler harmonics** — 1.4 Hz + 50 harmonics
3. **Colored noise** — 1/f^alpha + linear + white, with randomized crossover frequency
4. **White noise floor** — independent Gaussian
5. **Mechanical resonances** — 3–6 narrowband peaks from seismic/equipment vibrations

All components summed, modulated by a slow amplitude envelope (<0.2 Hz), Bessel-filtered at 120 Hz, and decimated to 1 kHz. In the default mode each window is scaled to its sampled target RMS. The existing training and holdout noise shards are synthetic.

### Common Signal Processing (`src/basics/`)

Shared Bessel low-pass filter (6th order, 120 Hz) matching the CUORE DAQ anti-aliasing chain, used by both pulse and noise generators.

### Training Data

Training pairs are formed by adding noise to clean pulses: `x_noisy = x_clean + noise`. Clean-noise pairings are reshuffled each epoch for combinatorial augmentation. Per-window normalization by max(|noisy|) maps inputs to ~[-1, 1].

```
clean/clean_*.h5                    (existing simulated ODE pulses)
noise/noise_*.h5                    (existing synthetic noise shards)
```

## Denoising Models

### DDPM (`src/ddpm/`)

Denoising Diffusion Probabilistic Model — the primary denoising framework.

- **Forward process**: Gradual Gaussian noise addition over T=50 steps (quadratic beta schedule)
- **Reverse process**: Iterative denoising conditioned on noisy observation x_tilde
- **Architecture**: 1D U-Net (~15.1M params), 4-level encoder-decoder with skip connections, self-attention at bottleneck. Input: [x_t, x_tilde] concatenated (2 channels). Output: predicted noise.
- **Scale conditioning** (optional): encodes max(|noisy|) so the model knows the SNR regime
- **Loss**: L1 or L2 on noise prediction, plus optional spectral losses (SC, LSD, J_asd) on the implied clean estimate
- **Inference**: Gaussian start with no noise added during the reverse steps for the planned paired evaluation. Stochastic DDPM and DDIM are also available.

See [`src/ddpm/README.md`](src/ddpm/README.md) for full details.

### Flow Matching (`src/flowMatching/`)

Conditional Flow Matching — an alternative generative framework for direct comparison with DDPM.

- **Forward**: Linear interpolation x_t = (1-t)*x_0 + t*eps, continuous t in [0, 1]
- **Reverse**: Deterministic ODE integration from t=1 (noise) to t=0 (data)
- **Model predicts**: velocity v = eps - x_0 (vs noise prediction in DDPM)
- **No noise schedule**: interpolation path is a straight line, no beta/alpha parameters
- **Solvers**: Euler (default) or midpoint (2nd order)
- Same U-Net architecture and dataset as DDPM

See [`src/flowMatching/README.md`](src/flowMatching/README.md) for full details.

## Project Structure

```
src/
  basics/             - Shared signal processing (Bessel filter)
  pulse/              - Bolometer pulse simulator (ODE model, spectrum, HDF5 generation)
  noise/              - Synthetic noise generator (5-component model)
  ddpm/               - DDPM denoiser (model, training, inference, metrics)
  flowMatching/       - Flow matching denoiser (model, training, inference)

scripts/
  generate_clean_shards.py   - Multi-shard pulse generation
  generate_noise_shards.py   - Multi-shard noise generation
  plot_trajectory_paper.py   - Paper figure: DDPM reverse-diffusion trajectory snapshots
  plot_datasets_paper.py     - Paper figure: 4 canonical simulation datasets (time + PSD)
  plot_denoising_paper.py    - Paper figure: deterministic DDPM denoising on the same 4 examples
  replot_scatter.py          - Regenerate scatter plots from saved inference pickle
                               (supports vs-SNR or vs-clean-amplitude x-axes)
```

## Other models and tools

### Train Flow Matching

```bash
nohup stdbuf -oL python3 -u -m src.flowMatching.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

### Inference

```bash
# DDPM with the scale embedding + FiLM model
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10 --scale_emb --film --no_noise

# Flow Matching
python3 -u -m src.flowMatching.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10
```

### Paper figures

Paper plots live under `scripts/plot_*_paper.py`. The denoising example figure selects two singles and two pileups from a saved test HDF5 using the clean peak amplitude.

```bash
# Reverse-diffusion trajectory snapshots (x_T → x_0 for one 40 mV pulse)
PYTHONPATH=. python3 scripts/plot_trajectory_paper.py

# Overview of the 4 canonical datasets: time domain + PSD, clean/noisy/noise
PYTHONPATH=. python3 scripts/plot_datasets_paper.py

# Waveforms and PSD from saved denoising results
PYTHONPATH=. python3 scripts/plot_denoising_paper.py \
    --denoised_h5 /path/to/denoised_test_000.h5 --out plots/paper/denoising_examples.pdf
```

Scatter plots from existing DDPM inference runs (no GPU needed):

```bash
PYTHONPATH=. python3 scripts/replot_scatter.py \
    --results_pkl /path/to/inference/results.pkl \
    --scatter_output_vs_amp /path/to/scatter_metrics_vs_amp.png \
    --scatter_pileup_output_vs_amp /path/to/scatter_pileup_vs_amp.png
```

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Li et al., "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal" (arXiv:2208.00542)
- Ormiston et al., "Noise reduction in gravitational-wave data via deep learning" (Phys. Rev. Research, 2020)
- Adams et al., "CUORE detector model" (arXiv:2205.04549)
