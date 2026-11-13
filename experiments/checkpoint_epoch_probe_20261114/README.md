# Epoch 70 versus epoch 100: selected low-energy events

Question: did the conspicuous low-energy output distortion appear only late in
the new model's training? The old production model's best epoch was 72 (run01
training report); the new run saved checkpoints every ten epochs, so epoch 70
is the closest available snapshot. Epoch 100 is the paired control and its
tensors are checked against the final best_model.pt.

This is a small, preselected example study. It is not a measurement of energy
resolution, tail probability, or the best checkpoint over the evaluation set.

## Inputs and controls

Reuse the four examples already selected in the 2026-11-13 tail diagnostics:
one peak-core median and the upper/median/lower tail energy quantiles per case.

| Input case | Original zero-based rows, in inference order |
|---|---|
| 1.382 mV, 2 keV | 3892, 1788, 3375, 987 |
| 7.5 mV, 10 keV | 1172, 1703, 4251, 1050 |

The original noisy waveforms, clean template, archived epoch-100 output, old
zero-init fitted amplitudes, and previous selection JSON are read-only.
The archived production source is passed explicitly via --code-root; the
current checkout can have subsequent unrelated changes.

Both new checkpoints use exactly the same local Gaussian initial arrays and
the same observations. Settings: CPU FP32, 50 DDPM no_noise steps, one shot,
seed 42, batch size 4, scale embedding and FiLM enabled, beta_1=1e-4,
beta_T=0.5, step conditioning, four CPU threads. The existing denoise_batch
entry point performs normalization and inference; no sampler is rewritten.
The expected RNG state after the initial Gaussian draw is checked after
inference for both epochs.

The archived GPU run used CUDA and batch 16. Its Gaussian draws need not match
the local CPU batch-4 draws. Therefore the locally rerun epoch 100, not the
archived waveform alone, is the main control for the weight-epoch comparison.
Archived epoch-100 waveforms are plotted to reveal whether the selected
abnormal shape persists under the local start.

## Run and storage

Run with the existing CPU PyTorch / NumPy / h5py / SciPy / Matplotlib environment:

    python run.py --code-root ARCHIVED_CODE --data-root DATA --notes-root NOTES

Optional --threads and --name control CPU threading and the independent output
directory. Existing waveform or fit outputs are refused rather than overwritten.

- DATA/model261108_epoch70_probe_20261114/paired_outputs.h5:
  source rows, clean templates, local Gaussian starts, epoch-70/100 outputs.
- NOTES/results/fit_results/model261108_epoch70_probe_20261114/paired_fits.h5:
  canonical tri-exponential fitting results and original row identities; no
  waveform datasets.
- NOTES/results/plots/model261108_epoch70_probe_20261114/:
  per-case waveform/residual panels and a paired metric figure (PNG/PDF),
  event_metrics.csv and report.json.

Ratios use each input's actual clean-template peak, verified equal to the
stored true_amp_V. No new energy calibration, Gaussian energy-spectrum fit,
event exclusion, OF run, training, or cloud action is performed.

## Checks and limits

The runner checks checkpoint epochs, epoch-100/best tensor equality, exact
paired initial-noise equality, expected RNG consumption, finite output arrays,
source-row identity, fit success, residual reconstruction from saved fit
parameters, and unchanged source/weight hashes. The JSON records code and
input hashes, runtime, source paths and every event's metrics.

The same epoch number is not a controlled comparison between old and new
architectures or losses. This study isolates epochs 70 and 100 within the new
run, at one local random start. It cannot select a general-purpose best epoch
from eight outcome-selected examples.

## Observed result (2026-11-14)

The full paired probe completed on local CPU in 190.9 seconds, with all sixteen
outputs finite and all sixteen canonical fits successful. Original inputs and
weights remained unchanged. Reopened output files preserve row identity; saved
fit parameters reproduce residual statistics. Three PNG/PDF figure pairs were
visually checked.

The six preselected tail cases still have peak deficits at epoch 70. For the
two tail-median events, fitted peak / clean peak is 0.796 versus 0.766 (2 keV)
and 0.773 versus 0.758 (10 keV), for epoch 70 versus epoch 100. Other examples
have mixed changes; the severe 10-keV event is worse at epoch 70 (0.504 versus
0.576), and was also low in the old model (0.546).

The local epoch-100 control reproduces the archived abnormal shapes. The largest
fitted-peak difference relative to archived GPU epoch 100 is 0.142 percentage
points of true peak at 2 keV and 0.674 percentage points at 10 keV.

These cases establish that the distortion was already present by epoch 70.
Rolling back to epoch 70 does not resolve it. They do not establish a population
tail rate, identify the earliest onset, prove or exclude overfitting generally,
or justify switching the production checkpoint. No sampler ablation or retraining
was started. Detailed event tables and the Chinese result guide are with the
plots; run.py is the exact executed source.
