# CUORE release: amplitude and tri-exponential time constants

## Purpose and inputs

Descriptive scatter plots for all 10,000 windows in Zenodo record 20721645.
Input files: cuoreTraining.h5 (9,000) and cuoreTest.h5 (1,000). Input HDF5 files
are never modified. There is no per-event energy calibration in these files.
The x coordinate is the supplied normalizationScale = raw maximum minus the
mean of the first 2,500 samples, in ADC units. Sampling is 1 kHz, supported by
https://arxiv.org/html/2607.02548v1, Section III.

Training classes use Labels. Test classes are derived from the explicit
rowOrder and cleanPulseCount HDF5 attributes, not inferred from the waveform.
This is a descriptive study, not an assessment of blind classification accuracy.

## Method

The runner freezes the current src/basics/fit.py into the output directory and
records its SHA-256. The core fitter is not changed. It uses the existing single
tri-exponential and sequential two-pulse methods on all 10,000 original samples.
For numerical compatibility, the input is (raw-offset)/scale*0.01, without
clipping. This is NOT an ADC-to-volt calibration. The original normalized
parameters are stored, and fitted peak amplitudes are also converted to ADC.
The only optimizer override is maxfev=2,000 (instead of 100,000), with failure
messages retained. BLAS thread count is one per worker.

Actual inherited constraints matter: rise 0.01–0.20 s; alpha 0.01–0.99;
both decay upper bounds 10 s. Single-pulse setup raises the first decay lower
bound to approximately 0.06708 s and the second to 0.45 s. These are optimizer
bounds, despite comments in the inherited implementation calling them hints.
For pulse 2, rise is constrained to +/-20% of pulse 1; decay constants are free
within their implementation bounds. The frozen source is authoritative.

## Plot interpretation and quality

Every successful window contributes one point. For a pileup window, the
largest fitted pulse supplies its three time constants; the x coordinate is
still the FULL WINDOW amplitude, not the separate pulse amplitude or energy.
Both fitted pulses remain in the output table. The release permits more than
two pulses, so these pileup fits are diagnostic and can be misspecified.

Exploratory diagnostic-pass criteria:
- residual RMS / window amplitude <= 2%;
- all shape parameters more than 0.1% of their allowed interval from bounds;
- rise < fast decay < slow decay, and slow/fast > 1.05;
- pileup: two detected peaks, both fitted peaks > 2% of window amplitude.

This screen is not a confidence-interval or physical-identifiability test.
All successful flagged points are plotted, and failed rows remain in HDF5.
The 16-window pilot found all fits flagged, with alpha saturating in single
events. Smooth trends in the resulting scatter plots must not be mistaken
for independently determined physical thermal constants.

## Outputs and reproducibility

Arguments configure data, fitter, output and plot paths; --pilot N selects N
evenly spaced rows per class and split; default zero processes all windows.
Use independent output directories for different settings. Checkpoint fits.jsonl
supports resuming an unchanged run; do not reuse it after changing inputs or code.

- cuoreTraining_fit.h5 / cuoreTest_fit.h5: original source row and event ID,
  class, full parameters (NaN padding), component time constants/amplitudes,
  quality fields, duration and failure messages; no waveform copies.
- manifest.json: source and code hashes, fs, settings, caveats.
- summary.json: row counts and diagnostic-pass counts.
- fitter_snapshot.py: exact algorithm used.
- tau_vs_amplitude_all_windows.png/pdf: complete successful-result overview.
- tau_vs_amplitude_single_pulses.png/pdf: single events only.
- fit_examples.png: 10th/50th/90th residual-percentile examples in each class.

2026-11-12 local outputs:
D:/Obsidian/note/课题/DDPM4Bolometer/results/fit_results/cuore_release_20721645/
D:/Obsidian/note/课题/DDPM4Bolometer/results/plots/cuore_release_20721645/


## Completed run (2026-11-12)

- Full fit wall time: 993 s (about 16.6 min), 8 CPU workers.
- Singles: 5,000/5,000 numerical successes; 4,980 near a shape bound;
  4,956 with alpha at 0.99; only 20 pass the exploratory screen.
- Pileup: 4,974/5,000 numerical successes; all 4,974 near a shape bound;
  none pass the exploratory screen. 26 failures reached maxfev=2,000
  (11 in pulse 1, 15 in pulse 2), not proof of an impossible fit.
- Median residual RMS/window amplitude: singles 0.878%, pileup 1.123%.
  A small waveform residual does not establish identifiable time constants.
- Every source row, event ID, offset and amplitude matched exactly; original
  input SHA-256 values unchanged. Failed rows retained; no waveform copies.
- verification.json records checks and statistics. runner_snapshot.py retains
  the execution code. No core fitter changes, commit or push.
- plot_results.py rerenders from fits.jsonl only, with darker markers. Full
  overview uses linear rise and logarithmic decay axes; single-only figure
  uses linear axes. No point subsampling or axis-range cuts. Final plotting
  source snapshot is saved with the figures and its hash in verification.json.

Interpretation: these are descriptive parameters of the inherited constrained
model. The bands at 0.20 s and 10 s include explicit optimizer boundaries;
other structures can be induced by sequential-fit constraints. Do not infer
physical cutoffs or energy-dependent thermal constants without first checking
model suitability, parameter identifiability, and sensitivity to bounds.
