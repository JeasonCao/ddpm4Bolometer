# Model261108 inference run

Run-specific orchestration authorized on 2026-11-12. The source snapshot is commit `ed7569e`.
The scientific comparison uses existing holdout events and the existing fitting algorithms.
Training and evaluation uploads are managed by their existing jobs.

## Components

- `server.py` waits for the 100-epoch training success and verified NAS backup, and the final 34-file upload report. It checks SHA-256, loads the best model, runs two-event inference and fit checks in each noise group, then processes all files. Priority is 1.382 mV / 2615 and 15 keV. Each result is independently published to NAS after validation.
- `client.py` runs on Windows, downloads training artifacts and completed inference files, verifies checksums, starts priority analysis in a separate process, and submits the final receipt. It keeps the local host awake while active. Connectivity failures retry every minute. Server and analysis failures remain visible in status files.
- `analyze.py` reuses `run_fitting.py` and `run_gauss_fit.py`. Input waveforms are external-linked from the existing dataset. The initial comparison uses the historical shared calibration of 782.06 keV/V and finite fitted amplitudes, as the supplied historical figure did. Fit convergence statistics are also recorded. Historical OF from stream run 999998 is recorded separately with its different event/calibration provenance.
- `priority_diagnostics.py` reads the completed 1.382 mV / 2615 and 15 keV outputs and fits. It saves `details_{energy}keV.png` and `diagnostics_{energy}keV.json` beside the comparison plots: event/mean residuals, zoomed energy spectra, the existing chi-square quality gate, and a clean-template fit residual reference. Waveform diagnostics use the first 200 events; fit-quality counts use all 5,000. It uses the run-specific paths and dependencies above, runs both energies when invoked, and is separate from the deployed inference/download/shutdown workers. It does not change fit thresholds or input files.
- `common.py` implements atomic JSON publication and hashing.

## Configuration

The fixed run paths, host, port and local SSH key path are explicit constants in the scripts. There are no embedded credentials. Runtime copies are under `D:/Claude_Cropse/model261108_inference` and `/root/autodl-tmp/model261108_inference`. `code.zip` contains `src` and the three reused scripts from `ed7569e`, extracted into `code/` beside the drivers. Windows dependencies are supplied from the existing isolated fit environment and `plot_deps/`.

Inference uses Gaussian x_T, DDPM posterior-mean updates, T=50, beta_1=1e-4, beta_T=0.5, step conditioning, scale embedding and FiLM, FP32, seed=42 and batch=16. Per-file restarts preserve these settings. Existing completed files must match source and weight hashes.

Outputs are in `/root/autodl-tmp/output/run03_model261108_inference`, with checked copies in `/root/autodl-fs/output/run03_model261108_inference`. Local inference and training copies are in the data disk's `model261108_inference` and `model_output_cloud/run03_model261108`. Fit results and plots go to the established Obsidian results directories.

Shutdown requires all 34 outputs, training NAS checksums, complete local download receipts, inference metadata/code/log backup, and completion of both priority analyses. The remote shutdown function revalidates NAS checksums before scheduling shutdown in one minute. Scientific outcomes, such as a larger FWHM, are retained as results. Technical failures leave the instance running for diagnosis.

## Validation

`validate_local.py` checks the linked-HDF5 fit path on two existing events and verifies corruption detection. `analyze.py --energy 2615 --preview` and the corresponding 15-keV command exercise plotting using historical outputs in both comparison slots, explicitly labelled as previews. `validate_server.py` uses isolated fixtures and a mocked shutdown command to check that incomplete receipts, corrupt backups and nonfinite arrays are rejected. It never shuts down the machine.

Actual-model smoke checks run after training completes. Persistent network disconnection or local host power-off delays local downloads and therefore delays cloud shutdown. Status and logs preserve the reason.

## Full-range analysis (2026-11-12)

- `full_fit.py --data-root PATH --notes-root PATH --work-dir PATH --workers 8` computes missing new-model fits using the unchanged canonical `scripts.run_fitting._fit_one`. It reuses the two priority fits and copies cached noisy-event fits from each historical zero-init fit file, with source hashes and row identity. Waveforms remain read-only; no waveform arrays are stored in fit outputs. Progress, restart manifests and logs belong outside Git.
- `full_comparison.py --data-root PATH --notes-root PATH --output PATH` compares raw, old zero-init, new Gaussian-start no_noise and historical OF for both noise levels. It reuses OF trigger matching and delay-limit exclusions from `plot_full_of_vs_ddpm.py`, without running OF again.
- The new comparison applies the canonical Gaussian fitter to successful finite tri-exponential amplitudes, without the chi-square gate, as in the existing multi-sampler comparison. Conventional gated Gaussian attributes and acceptance counts remain available. No shared scientific default or historical fit/plot is overwritten.
- Each method and noise group receives a five-line zero-intercept calibration (239, 511, 727, 1461, 2615 keV), using the established inverse-centroid-variance formula. The OF curve is recalibrated from its archived amplitudes under this same convention; historical per-peak and robust-core widths are also recorded. Error bars show width-fit statistics conditional on calibration, not detector nonlinearity.
- Outputs include full-range/high-energy FWHM plots, calibration and residual plots, selection/tail diagnostics, JSON provenance and a CSV table. OF remains a different-event reference; the old/new comparison changes architecture, training loss and sampler together. Finite fitted peaks alone do not establish scientific validity.
- Dependencies are supplied via the run environment (Python with NumPy, SciPy, h5py and Matplotlib); machine paths are CLI arguments. Local work directory: the data disk's `_development/model261108_full_analysis`. Full-range figures: the notes vault's `results/plots/model261108_full_comparison`.

Full-range completion: all 34 fits are available (32 computed locally, 2 priority fits reused), 170,000 solver-successful new-model events. Both 17-energy groups and all 136 method/energy rows were checked. Raw cached arrays, source metadata, sampled original/zero-init waveforms and output hashes were verified; historical calibration slopes were reproduced.

`spectral_diagnostics.py --notes-root PATH --output PATH` adds the 7.5 mV / 15--50 keV peak-and-tail panels. The full comparison also includes 1.382 mV / 2 keV and 7.5 mV / 10 keV outlier spectra: the canonical full-range and robust-core Gaussian widths disagree strongly. Hollow plot markers flag Gaussian spectrum chi2/ndf > 10 only as a visual diagnostic; this neither removes events nor changes the stored waveform-quality gate. Do not interpret these low-energy Gaussian widths as a complete performance measure.

To automatically plot after fitting, pass `full_comparison.py --wait-for-fit-manifest PATH/fit_manifest.json`; it verifies all 34 fit hashes, or exits if fitting fails or the wait timeout is exceeded. Final audit records, source alignment, statistics and all six PNG/PDF figure pairs are stored with the results. No new OF filtering, old-model inference, cloud start or automatic Git commit is part of this analysis.

## Revised calibration and waveform diagnostics (2026-11-13)

The later user agreement (2026-09-22, rechecked against the original conversation)
takes precedence over the former blanket calibration statement above. Full-range
resolution and reconstructed-energy figures use 239/511/727/1461/2615 keV;
low-energy reconstruction acceptance uses all available lines <=511 keV.
Each before/after pair shares the denoised-derived slope. The four-method overlay
uses the new-model slope for its single raw reference and labels it explicitly.

- `recalibrate.py --notes-root PATH --source-report PATH --output PATH` preserves
  previous results, reuses saved peak fits, fixes the raw reference, adds paired
  before/after plots and low-energy acceptance. Acceptance means successful finite
  amplitudes within 5% of truth, with all 5,000 windows in the denominator; no
  waveform chi-square cut or new DT-trigger calculation.
- `waveform_tail_diagnostics.py --data-root PATH --notes-root PATH --comparison PATH --output PATH`
  reconstructs saved tri-exponential curves and reads actual input/output waveforms.
  It inspects 2/5 keV at 1.382 mV and 10/15/20/25/30/50 keV at 7.5 mV, 5,000 each.
  It adds 2615-keV examples at chi-square P50/P95 with clean-template controls.
  No event waveform refitting, OF run or inference is performed.
- `diagnostic_summary.py --output PATH` adds high-energy closeups, a compact
  low-energy overview and fixed truth-based peak-deficit counts. It checks sampled
  noise reuse across energies/groups and records the sampled rows.
- `validate_analysis.py --notes-root PATH` verifies the source/code hashes,
  unchanged old/new/OF widths, shared raw calibration, independently reproduced
  efficiency counts, diagnostic row/amplitude/group identity and PNG/PDF pairs.

The energy-defined display groups come from a descriptive two-Gaussian mixture:
high-mean peak core within 2 sigma with posterior >=0.9; low tail below mu-3 sigma;
all other events remain gray. They never become a production event selection.
Three-component BIC is better in all eight cases, so two physical populations are
not established. Group sizes cannot be compared as fixed failure rates across
energies; the separate 10%-below-clean-peak diagnostic has a common definition.

Evidence: fitted/direct peak ranks correlate above 0.998 at 2/10 keV; tails already
exist in actual output waveforms, with broader/slower rises and suppressed peaks.
Core/tail rise medians are 55.5/108.0 ms and 56.5/106.4 ms respectively; decay
constants change much less. Reconstructed energy correlates with input 1--20 Hz
noise fraction (-0.494/-0.472) and noise projection onto the clean shape
(+0.625/+0.650). The latter is an unweighted diagnostic, not OF. This is association,
not a causal separation of architecture, loss, normalization and sampler.
The 200-ms rise band is the existing fit bound. At a fixed energy the source
uses one clean template, so fitted-parameter scatter is not a physical ODE spread.

The 2615-keV clean-template fit has 0.759 mV RMS residual; 94.46% of its squared
residual is in 1.49--1.52 s. Similar residual structure appears in model outputs
while baseline noise is much smaller, explaining why solver success and the
old chi-square gate disagree. This does not prove the OF resolution gap is
entirely caused by the fitting model. The canonical fitter and gate stay unchanged.

Results are independent directories under the notes vault's `results/plots/`:
`model261108_recalibrated_20261113` (8 PNG/PDF pairs) and
`model261108_tail_diagnostics_20261113` (60 pairs, per-event NPZ, CSV, JSON,
source/row/hash provenance, and a Chinese reading guide). Validation passed for
40,000 diagnostic rows, 136 method/energy rows and 106 source/code hashes.
Original inputs and old result directories are retained. All code runs on local
CPU using the existing analysis dependencies. This work does not change deployed
workers or cloud resources.
