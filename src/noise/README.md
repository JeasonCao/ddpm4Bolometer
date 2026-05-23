# Noise Generator

Realistic synthetic noise generator for CUORE cryogenic bolometer signals, modeling the dominant noise sources observed in real detector data.

The components are split into two **noise families** — *periodic* and *white* — so we can ablate the DDPM denoiser's behavior on each independently while still training on a realistic mix.

## Noise families

| Family | Components | Typical signature |
|---|---|---|
| **Periodic** | AC harmonics (50 Hz), PT cooler harmonics (1.4 Hz), mechanical resonances | Narrowband peaks; lowest content at 1.4 Hz, so no sub-Hz drift |
| **White**    | Colored noise (1/f^α + linear + white spectrum), white floor | Broadband; the 1/f tail produces sub-Hz baseline wander |

A single window's "total" noise is built by mixing these two families with random energy weights and a random overall scale (see *Paired generation* below).

## Noise components

### 1. AC power line harmonics (50 Hz)

Electromagnetic pickup from the AC mains at 50 Hz fundamental with 20 harmonics. Amplitudes follow a power-law decay `A_k = a_base / k^p_decay` with per-harmonic randomization (factor of ~3 up or down). Random phases.

### 2. Pulse tube cooler harmonics (1.4 Hz)

Mechanical vibrations from the cryostat's pulse tube cooler at ~1.4 Hz fundamental with 50 harmonics (covering up to ~70 Hz). Same power-law decay structure as AC harmonics but with independent parameters.

The fundamental amplitude `pt_a_base` is drawn from the **same uniform range** as `ac_a_base` (`target_rms · U(0.3, 1.0)`) so the AC and PT line strengths fluctuate at a comparable scale.

### 3. Colored noise (1/f^α + linear + white)

Broadband noise with a parametric power spectral density:

```
PSD(f) = a_pink / f^alpha + a_lin * f + a_white
```

- `alpha` in [0.8, 1.2] — 1/f slope (classic flicker noise ~1.0)
- `f_cross` in [0.5, 3.0] Hz — crossover frequency where pink meets white
- Amplitudes derived from `f_cross` and target noise RMS

Generated in frequency domain with random phases, then inverse-FFT.

### 4. White noise floor

Independent Gaussian white noise added separately from the colored noise component.

### 5. Mechanical resonances

Narrowband noise peaks from seismic and equipment vibrations. 3–6 resonances per window with:

- Base frequencies near [10, 25, 40, 70] Hz (jittered ±30%)
- Quality factors Q in [5, 20]
- Amplitudes: [0.1, 0.5] × target RMS

Each resonance is bandpass-filtered white noise (Butterworth filter).

## Amplitude envelope

All summed noise is multiplied by a slowly varying amplitude envelope (< 0.2 Hz) modeling non-stationary gain variations, bounded to ±20% modulation around unity.

The same envelope realization is shared between the periodic and white subsets, so they remain consistent under linear combination.

## Signal chain

Matches the pulse simulator signal chain (applied identically to periodic and white subsets):

1. All components generated at 10 kHz internal sampling rate
2. Per family: sum the family's components, multiply by the shared envelope
3. Bessel low-pass filtered at 120 Hz (6th order), matching CUORE DAQ anti-aliasing
4. Decimated to output rate (1 kHz)
5. Per-family RMS-rescaled to `target_rms` (default 7.5 mV)

## Paired generation

`generate_paired_noise(rng, params)` produces three windows per call from one component draw:

| Output | Definition | RMS |
|---|---|---|
| `periodic` | AC + PT + resonances → envelope → Bessel + decimate → rescale | `target_rms` |
| `white`    | colored + white floor → envelope → Bessel + decimate → rescale | `target_rms` |
| `total`    | `scale · (alpha · periodic + beta · white)` | `≈ scale · target_rms` |

Mixing rule (energy-preserving):

- `r ~ U(0, 1)` — periodic *energy* fraction
- `alpha = sqrt(r)`, `beta = sqrt(1 - r)` so `alpha² + beta² = 1`
- Since periodic and white are uncorrelated by construction, `RMS(alpha · P + beta · W) ≈ target_rms`
- `scale ~ U(0.5, 2.0)` — overall window-level amplitude variation (replaces the previous inner `noise_rms · U(0.5, 1.5)` randomization)

Per-window mix metadata `(alpha, beta, r, scale, target_rms)` is returned so the total can be exactly reconstructed from the periodic and white companions.

## Parameter sampling

Each noise window gets independently sampled parameters via `sample_noise_params()`. `target_rms` is now **fixed** at `noise_rms` (default 7.5 mV); all level variation comes from the `scale` factor in `generate_paired_noise`. Key randomized quantities:

| Parameter | Range | Description |
|-----------|-------|-------------|
| alpha | 0.8 – 1.2 | 1/f slope |
| f_cross | 0.5 – 3.0 Hz | Pink/white crossover |
| AC a_base | `target_rms · U(0.3, 1.0)` | 50 Hz harmonic amplitude |
| PT a_base | `target_rms · U(0.3, 1.0)` | 1.4 Hz harmonic amplitude (matched range) |
| max_variation | 0 – 0.2 | Envelope modulation depth |
| n_resonances | 3 – 6 | Number of mechanical peaks |
| r (mix) | 0 – 1 | Periodic energy fraction in `total` |
| scale | 0.5 – 2.0 | Overall amplitude scale of `total` |

## Output layout (shard pipeline)

`scripts/generate_noise_shards.py` writes three sibling folders sharing per-shard file names and per-window indices:

```
<OUTPUT_DIR>/
  noise_periodic/noise_NNN.h5    # P, RMS = target_rms
  noise_white/noise_NNN.h5       # W, RMS = target_rms
  noise_total/noise_NNN.h5       # scale · (alpha·P + beta·W)
```

Each H5 file stores:

- `waveforms` — `(n_windows, n_samples)`, gzip-compressed
- `params/` group — per-window scalar params (target_rms, 1/f alpha, f_cross, ac_a_base, pt_a_base, white_rms, envelope_variation, n_resonances)
- *Only on `noise_total`*: `mix/` group — per-window arrays `alpha`, `beta`, `r`, `scale`

Reconstruction identity (exact):

```
noise_total[i] = mix_scale[i] * (mix_alpha[i] * noise_periodic[i]
                                 + mix_beta[i] * noise_white[i])
```

Training uses `noise_total/`; ablation studies point `--noise_dir` at `noise_periodic/` or `noise_white/`.

## Structure

```
src/noise/
  generator.py    - Component synthesis, family assembly, paired generation
  colored.py      - 1/f^alpha + linear + white noise (frequency domain)
  harmonics.py    - Harmonic series with power-law decay
  resonances.py   - Bandpass-filtered mechanical resonance noise
  envelope.py     - Slow amplitude modulation envelope
  qa_noise.py     - QA plots: waveforms + PSD with annotations
```

## Usage

### Paired generation (used by the shard pipeline)

```python
from src.noise.generator import sample_noise_params, generate_paired_noise
import numpy as np

rng = np.random.default_rng(42)
params = sample_noise_params(rng)
periodic, white, total, meta = generate_paired_noise(rng, params)
# periodic.shape == white.shape == total.shape == (10000,)
# meta = {'alpha', 'beta', 'r', 'scale', 'target_rms'}
```

### Single combined window (back-compat for QA)

```python
from src.noise.generator import sample_noise_params, generate_noise
rng = np.random.default_rng(42)
params = sample_noise_params(rng)
noise = generate_noise(rng, params)        # all components summed, no scale
```

### QA visualization

```bash
python3 -u -m src.noise.qa_noise --n 5 --output qa_noise.png
```

Plots time-domain waveforms and log-log PSDs with annotated frequency markers (PT fundamental, AC fundamental, crossover, resonance centers).
