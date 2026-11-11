# Pulse Simulator

Physics-based simulation of CUORE cryogenic bolometer pulses for generating clean training data. The default generator uses ODE and redraws detector parameters per window. The existing ODE shards remain the training input. The analytic tri-exponential and channel-bank layouts are optional.

## Physics Model

Implements the nonlinear electro-thermal detector model from [arXiv:2205.04549](https://arxiv.org/abs/2205.04549). A CUORE bolometer is a TeO2 crystal coupled to an NTD-Ge thermistor operating at ~10 mK. When a particle deposits energy, the crystal heats up, changing the thermistor resistance, which is read out as a voltage pulse.

The model solves a 4-ODE perturbation system describing coupled thermal and electrical dynamics:

```
Thermal chain:  Energy deposit → Electrons → Crystal → PTFE → Heat sink
Electrical:     V_bol(t) from thermistor resistance change under constant bias
```

Key physics:
- **Thermal**: Three coupled heat capacities (electron, crystal, PTFE) with thermal conductances between them
- **Electrical**: NTD-Ge thermistor with temperature-dependent resistance R(T) = R0 * exp((T0/T)^0.5)
- **Signal chain**: ODE solved at 10 kHz internal rate, Bessel low-pass filtered at 120 Hz, decimated to output rate (1 kHz)

## Detector Parameters

Each simulated pulse uses a randomly sampled set of detector parameters drawn from truncated Gaussians. This captures detector-to-detector variation across the 988 CUORE channels.

Randomized per pulse:
- `R0`, `T0` — thermistor resistance model parameters
- `lambda0`, `q` — thermal conductance model parameters
- `g_ec`, `a_ec` — electron-crystal coupling
- `C_p` — parasitic heat capacity
- `T_base` — base temperature

Fixed:
- `gamma`, `g_ct`, `g_ts` — thermal chain conductances
- `c_e`, `c_c`, `c_t` — heat capacities
- `R_bias`, `V_bias`, `gain` — readout circuit
- `W`, `f_bessel`, `bessel_order` — signal conditioning

Unphysical parameter draws are rejected via equilibrium validation (resistance ratio < 0.02, loop gain < 0.8).

## Tri-exponential Template Model (`--model triexp`)

A fast analytic alternative to the ODE (`src/pulse/triexp.py`):

$$V(t)=A\left[-e^{-s/\tau_r}+a_1e^{-s/\tau_{d1}}+(1-a_1)e^{-s/\tau_{d2}}\right],\quad s=t-t_0,\qquad V=0 \text{ for } s<0$$

One rise and two decays (fast `tau_d1`, slow `tau_d2`). The bracket is 0 at
`s=0`, so the pulse is continuous at onset. It is the same shape that
`src/basics/fit.py` fits, so generated templates and fit results share one
parameterization.

**Amplitude.** `A` is not the peak height (the bracket peaks below 1). Each
channel has a sensitivity `sens` [V/keV] = peak height per keV, and
`A = E · sens / phi_peak`, where `phi_peak` is the numerical maximum of the
bracket. This keeps `sens` comparable to the ODE bank's peak/E and decoupled
from the time constants.

**Per-channel parameters** (truncated Gaussians, resampled until
`tau_r < tau_d1 < tau_d2`):

| Param | Mean | Sigma | Range |
|-------|------|-------|-------|
| `tau_r` [s]  | 0.085   | 0.025  | (0.02, 0.2) |
| `a1`         | 0.815   | 0.03   | (0.6, 0.95) |
| `tau_d1` [s] | 0.49    | 0.09   | (0.2, 1.0) |
| `tau_d2` [s] | 3.4     | 0.4    | (2.0, 5.0) |
| `sens` [V/keV] | 1.13e-3 | 0.3e-3 | (0.3e-3, 2.5e-3) |

**Where the defaults come from.** They are taken from free fits of this model
to 120 single pulses of the 41C-parameter ODE bank (`clean_v3`, E > 30 keV).
Residual RMS was about 2e-4 of peak (median), so the template reproduces the
ODE shapes closely. The ODE pulses' effective onset sits about 6.7 ms after the
nominal 1.5 s because of the Bessel filter's group delay. The template applies
no filter, so its `t0` is exactly the nominal onset.

**Differences from the ODE**:
- The response is exactly linear in energy: there is no q-factor nonlinearity.
- Pileup is a linear sum (shared time constants unless `--tau_energy_scaling`).
- It is about 100× faster (≈500 windows/s vs ≈5/s in flat mode).

### Energy-dependent decay (`--tau_energy_scaling`)

Real high-energy pulses decay more slowly. These two reference fits give the
trend:

| | 1.3 MeV | 15.3 MeV | Ratio applied |
|---|---|---|---|
| τ_r    | 80 ms  | 32 ms  | 1 (not applied) |
| τ_d1   | 261 ms | 441 ms | ×1.69 |
| τ_d2   | 1.54 s | 1.78 s | ×1.16 |

Only the **relative** change is used. The absolute values come from a
different detector than the ODE-fitted defaults above. The reference energies
are also **mapped onto the generated range** instead of being used literally:
- the 1.3 MeV shape applies at **1 keV**, which is the channel's sampled τ;
- the 15.3 MeV shape applies at **5407 keV**;
- in between, τ is interpolated **linearly in E**;
- outside [1, 5407] keV, τ is held at the end values.

    tau_k(E) = tau_k^ch * [1 + (R_k - 1) * clip((E - 1) / 5406, 0, 1)]

**Why this form:**
- *Mapped range, not literal 1.3–15.3 MeV.* Used literally, about 93% of the
  default spectrum would sit below 1.3 MeV and get no variation. The upper
  point would also never be reached, because the measured spectrum ends at
  10 MeV. The mapping spreads the full reference variation over the energies
  actually generated. The endpoints are `TAU_SCALE_E_LO/HI` in `triexp.py`.
- *Linear, not power-law.* A power law through two MeV-scale points
  extrapolates badly to keV energies; for example τ_r would come out at about
  0.5 s at 10 keV.
- *Clamped at both ends.* This keeps every τ inside the envelope set by the
  two reference fits.
- *Rise kept fixed.* The rise in the reference gets about 2.5× faster, but
  only the decay variation was asked for. Set `TAU_SCALE_RATIO['tau_r']` to
  `0.4` to enable it.
- *Ordering preserved.* The ratios satisfy R_r ≤ 1 ≤ R_d, so
  τ_r < τ_d1 < τ_d2 still holds.

The peak stays at E·sens because phi_peak is recomputed per pulse, so the
pulse integral grows slightly faster than linearly. In pileup each pulse uses
the τ of its own energy. The project fitter uses a sequential pileup fit with independent decay constants. Historical QA with shared time constants used the fitter from `b5cd470`.

Each pulse's actual τ is stored per window as `tau_r_1`, `tau_d1_1`,
`tau_d2_1` and the matching `_2` datasets (NaN when there is no second pulse).
These datasets are written for every triexp file, scaled or not. The
`tau_energy_scaling` attribute records whether scaling was on.

**Coverage:** the spectrum is dominated by low energies, so most pulses stay
close to the base τ. The full ×1.69 τ_d1 (×1.16 τ_d2) is reached only near
5.4 MeV. If you generate with a different `E_max`, the scaling endpoints do not
follow it automatically.

Use it for fast large-scale data or for controlled shape studies. Use the ODE
where the physics (nonlinearity, parameter-to-shape map) matters.

Output layout is unchanged except that:
- `params` (flat) or `param_<name>` attrs (channel bank) hold `tau_r`, `a1`,
  `tau_d1`, `tau_d2` and `sens`.
- There is no `equilibrium` group or `eq_*` attrs.
- Every file carries `pulse_model` = `ode` | `triexp`.

`scripts/denoise_test_set.py` supports both clean layouts and copies
equilibrium information when present.

## Energy Spectrum

Uses the measured CUORE spectrum from `figS2_spectrum.root` (FinalSpectrum histogram), extracted to `data/cuore_spectrum.npz`. The spectrum covers 0–10000 keV in 5 keV bins and is used directly for inverse-CDF energy sampling — no parametric fitting. A parametric fallback (exponential continuum + Gaussian peaks) is used if the data file is missing.

Energy range can be restricted via `E_min`/`E_max` parameters to generate subsets (e.g., low-energy E < 150 keV vs high-energy E >= 150 keV).

## Dataset Generation

Each pulse in the dataset is generated as:

1. Sample detector parameters (rejection sampling for physical validity)
2. Find self-consistent thermal equilibrium via fixed-point iteration
3. Decide single pulse (70%) or pileup (30%)
4. Sample energy from spectrum, fixed onset at 1.5 s
5. For pileup: sample second energy and onset (0.01–2.0 s after first)
6. Solve ODE, filter, decimate to 10-second windows at 1 kHz (10000 samples)

Output HDF5 contains:
- `waveforms` — (N, 10000) float64, voltage in Volts
- `energies_1`, `energies_2` — deposited energies [keV]
- `onsets_1`, `onsets_2` — pulse onset times [s]
- `is_pileup` — boolean flag
- `params` — detector parameters per pulse
- `equilibrium` — equilibrium state per pulse

## Structure

```
src/pulse/
  simulator.py    - ODE solver, detector model, equilibrium finder
  spectrum.py     - Energy spectrum model and inverse-CDF sampler
  triexp.py       - Analytic tri-exponential template model (--model triexp)
  generate.py     - HDF5 dataset generator (CLI entry point)
  qa_pulse.py     - QA plots: waveforms with metadata
  qa_spectrum.py  - QA plots: spectrum model vs sampled histogram
```

## Usage

### Generate dataset

Single shard:

```bash
python3 -u -m src.pulse.generate \
    --n_pulses 10000 \
    --output /path/to/clean_000.h5 \
    --duration 10.0 --f_sample 1000.0 \
    --pileup_fraction 0.3 --seed 0
```

Tri-exponential template instead of the ODE (works in both flat and
channel-bank modes):

```bash
python3 -u -m src.pulse.generate --model triexp \
    --n_pulses 10000 --output /path/to/clean_triexp.h5
python3 -u -m src.pulse.generate --model triexp \
    --n_channels 100 --windows_per_channel 200 --output_dir /path/to/triexp_bank
# Energy-dependent decay constants
python3 -u -m src.pulse.generate --model triexp --tau_energy_scaling \
    --n_pulses 10000 --output /path/to/clean_triexp_escale.h5
```

Multi-shard generation with energy range:

```bash
# Low energy (E < 150 keV)
python3 -u scripts/generate_clean_shards.py \
    --output_dir /path/to/clean_low --E_min 1 --E_max 150

# High energy (E >= 150 keV)
python3 -u scripts/generate_clean_shards.py \
    --output_dir /path/to/clean_high --E_min 150 --E_max 5407
```

### Channel bank (multi-channel foundation model)

For the foundation-model denoiser, pulses are organized into **channels**: each
channel is one shard whose detector parameters are **fixed** (sampled once and
held constant for every window), while event energy and pileup still vary per
window. One channel ↔ one shard (`clean_NNN.h5`), so `channel_id == shard index`.

```bash
# 100 channels x 200 windows each
python3 -u -m src.pulse.generate \
    --n_channels 100 --windows_per_channel 200 \
    --output_dir /media/Disk_YIN/yunshancheng/cuore/channel_bank/clean \
    --seed 42
```

Each shard records its channel identity in the file attributes: `channel_id`,
the 8 fixed detector parameters as `param_<name>` (`R0`, `T0`, `lambda0`, `q`,
`g_ec`, `a_ec`, `C_p`, `T_base`), and the equilibrium state (`eq_V_bol`,
`eq_T_e`, `eq_T_c`, `eq_T_t`). Equilibrium is solved once per channel (not per
window), so the channel-bank path is much faster per window than the flat mode.

Generation is **resumable** — existing shards are skipped — and each channel
uses an independent RNG stream spawned from `--seed`. Pair with the matching
noise bank (`scripts/generate_noise_shards.py --n_channels`, same channel
indexing). The rationale for fixing detector params per channel (the channel's
pulse-shape identity) vs. varying energy/pileup per window is documented in
`src/ddpm/README.md` (foundation-model section).

### scitra bolometer response task

`scripts/generate_scitra_bolometer_task.py` builds the response-modeling dataset
specified in `scitra-bolometer-task/spec/DATASET_SPEC.md`. It reuses this
module's logic — `sample_valid_params`, i.e. draw detector parameters from the
paper's distributions and **resample whenever the equilibrium solution is
unphysical** — but reduced to a single channel and reshaped to the spec's layout.

```bash
# Inspect the sampled channel + noise level without generating windows
python3 -u scripts/generate_scitra_bolometer_task.py --dry_run

# Full run: 13k pulse windows + 2k baseline
nohup stdbuf -oL python3 -u scripts/generate_scitra_bolometer_task.py \
    --output_dir /media/Disk_YIN/yunshancheng/cuore/scitra_bolometer_task \
    --workers 8 > gen_scitra.out 2>&1 &
```

**Two independent random sources per 10 s window**, both needed because the
detector parameters are fixed for the whole dataset:

* the **noise realization** and its amplitude `scale ~ U(--scale_lo, --scale_hi)`;
* the **injected energy**, Gaussian-smeared about the nominal line energy with
  relative sigma `--energy_smear` (default 0.1%).

The **onset is fixed at 1.5 s** in every window, agent-visible and truth alike.
An earlier version jittered it by ±50 ms so windows could not be aligned
trivially, but the task is response amplitude vs. energy, not trigger-time
recovery: a known common onset lets the agent average or template-fit windows
directly, without an alignment step that only adds variance to the thing being
measured.

The energy smearing matters more than it looks. A fixed detector reproduces
*exactly* the same pulse for a given energy — two clean waveforms at the same
energy agree to ~1e-11 V. With the onset also fixed, a calibration file without
smearing would be one single pulse shape buried under N independent noise draws. Smearing gives each event its own
amplitude, as an intrinsic detector resolution does. At 2614.5 keV, 0.1% sigma
is ~2.7 keV, comparable to CUORE's real ~5 keV FWHM. Set `--energy_smear 0` for
the strictly spec-conformant behaviour (spec §3 assumes monoenergetic injection
so that peak spread measures noise alone).

**Noise is generated independently of the signal**, at a level set by
`--noise_rms` (default 7 mV, cf. the bank's 7.5 mV) rather than derived from a
target SNR. The PSD-shape parameters and the periodic/white mix `r` are drawn
once — the channel's noise identity — while each window gets its own
realization. Because the realized RMS of the mixed window drifts a few percent
from nominal, the generator measures it over `--n_cal_windows` and folds the
ratio into a fixed correction, so the delivered baseline RMS *equals*
`--noise_rms` rather than merely tracking it.

`--q` overrides the sampled `q` (default −65; `--sample_q` keeps the draw).
Spec §1 singles it out as the one parameter departing from the bank default: it
sets the second-order term of the resistance perturbation (Eq. 4.1), i.e. how
much `amp/E` bends with energy — the effect the agent must model. It does not
enter `find_equilibrium`, so overriding it leaves the operating point and the
validity check untouched.

Sensitivity (`amp/E`) is dominated by `T_base` via the operating point:
`R = R0·exp(sqrt(T0/T_e))` gives `dlnR/dlnT_e = −0.5·sqrt(T0/T_e) ≈ −10.5`, and
since `R ≪ R_bias` the bias divider makes `V_bol` linear in `R` and the pulse
amplitude linear in `V_bol`. A 1% shift in temperature is a ~10% shift in gain —
why bank channels span 0.383–1.58 mV/keV, and why the resulting SNR at a given
noise level depends strongly on which channel is drawn.

Other spec-mandated departures from the bank: `pileup_fraction = 0`
(`response(energy)` is a single-pulse concept), discrete calibration lines
instead of a continuous spectrum, and clean+noise summed into one array.

Output splits into `calibration/` (mount this: `waveforms` plus `energy`,
`f_sample`, `duration`, `n_samples`, `channel_id` — no params, onsets, injected
energies, noise settings or seeds) and `truth/` (never mount:
`truth_params.json`, float64 noise-free waveforms at the five lines and on a
19-point evaluation grid, and per-window onsets/scales/injected energies).
Every window is seeded from a `SeedSequence` tree spawned per index, so output
is bit-identical regardless of `--workers` — verified by generating the same
windows with 1 and 4 workers.

### QA visualization

```bash
python3 -u -m src.pulse.qa_pulse --input clean_000.h5 --n 10 --output qa_pulse.png
python3 -u -m src.pulse.qa_spectrum --output qa_spectrum.png
```
