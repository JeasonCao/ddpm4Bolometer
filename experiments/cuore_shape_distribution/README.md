# CUORE amplitude-conditioned shape distributions (2026-11-13)

## Purpose and scope

Describe the typical waveform versus amplitude, distributions within narrow amplitude bands, parameter correlations, and how much scatter can arise from noise alone. This informs future experimental-like validation data; it does not generate a new validation dataset or alter standard fitting/training.

**No per-event energy or energy calibration is available.** All horizontal axes are ADC amplitude, never keV. Narrow amplitude bands are not exactly monoenergetic samples.

Public release: https://zenodo.org/records/20721645
Release paper: https://arxiv.org/html/2607.02548v1 (Table 1 and Appendix A).
The release uses one dataset-calorimeter's calibration runs, preselected pulse classes and reconstructed trigger energies above 500 keV. It cannot constrain arbitrary detector channels or sub-500-keV pulse shapes.

## Inputs, sample selection, and provenance

- Read-only source files: cuoreTraining.h5, cuoreTest.h5.
- 4500 + 500 single events, identified by labels and explicit row-order/count metadata.
- Existing fixed-readout fits from experiments/cuore_readout_fit: 5.1 Hz (-3 dB, sixth-order causal Bessel), RC=68 ms.
- Main numerical statistics use all 5000 singles. Point scatters retain the user's previous 4980-event selection; the other 20 are not separately drawn. Their influence is tabulated, not silently discarded.
- Every event retains source filename, original row, event ID, source amplitude/offset, fit parameters and selection flags.
- Source hashes, fit hashes, sample rules and seed appear in manifest.json. Original waveform files and previous fit files are never modified.
- Pileup windows are excluded: one window may contain several pulses/energies and does not define a single-event shape distribution.
- Both supplied train/test files are used descriptively. Future generator tuning should reserve fresh independent events for evaluation.

## Features and waveform alignment

- Energy proxy: original normalizationScale = raw maximum minus first-2500-sample mean, in ADC units. It is explicitly not energy.
- Effective fit variables: tr, tf, ts and alpha, preserving their joint correlations. They are not calibrated thermal/ODE parameters.
- Direct waveform features: rising 10%-90% time and falling 90%-30% time, with linear interpolation of threshold crossings.
- Direct features use Gaussian sigma=3 samples (3 ms) only to stabilize crossings/peak height. Sigma=2 and 5 ms sensitivity is saved. This analysis smoothing is not a claimed hardware response.
- Crossing selection: last rising crossing before the maximum, first falling crossing after it.
- Raw waveforms are aligned at the smoothed rising 50% point and normalized by the smoothed peak, without clipping or time stretching. The displayed/quantiled waveform itself is raw; smoothing is only used for alignment and normalization.
- Aligned waveform quantiles are pointwise event distributions, not confidence bands for the median. Alignment and normalization themselves introduce correlated errors.
- No full raw waveform copy is stored. HDF5 stores event features, existing fit p, band waveform quantiles, noise spectra, and control templates.

## Bins and statistics

Representative bands: 6500, 10000, 18000, 28500 ADC, each +/-2.5%, containing 468, 311, 113, 263 events.

Also saved:
- 1000-ADC bins across the observed range; curves require at least 20 samples/bin.
- Narrow-band medians, 16/84 and 2.5/97.5 percentiles, standard deviations, counts and source selections. A one-event bin has undefined standard deviation (blank CSV field), not a fabricated zero.
- Band-width sensitivity at +/-0.5%, +/-1%, +/-2.5%.
- Legacy-4980 versus all-single selection sensitivity.
- Theil-Sen removal of amplitude trend inside each band before reporting detrended scatter.
- Correlations of log(tr), log(tf), log(ts), logit(alpha), after removing within-band amplitude trends. Covariance tables include both raw and detrended log parameters.
- Global rank correlations and partial rank correlations controlling baseline offset. These are descriptive associations, not causal energy effects.

## Noise controls

For each band, run 96 independent realizations for each of two FIXED shapes:

1. Representative fitted model waveform (medoid in robustly scaled log-taus/alpha coordinates).
2. Median aligned experimental waveform, lightly smoothed (3 ms), so model mismatch is present.

Each fixed template is scaled to the band's central ADC amplitude. Add stationary Gaussian noise synthesized from the band-average 2.5-second pretrigger periodograms; noise RMS is sampled from the observed band distribution. The spectrum is flat-continued below the lowest resolved positive frequency (0.4 Hz), with DC set to zero. Real and imaginary Fourier variances are scaled for a one-sided PSD. Each realization is rescaled to the sampled full-window RMS.

Recompute the synthetic pretrigger mean and observed maximum, normalize using the same procedure as the source fits, then refit with fixed shared response (generic plus representative initial guesses). Record donor index, seed, synthetic measured RMS, fit status, time constants and direct features.

These are **approximate noise controls**, not an exact model of all experimental fluctuations. The 2.5-second baseline cannot establish slower drifts, nonstationarity, signal-dependent noise or all non-Gaussian effects. No intrinsic variance subtraction is performed.

Figure 5 compares central-68% half widths for:
- raw within-band event variation;
- within-band variation after removing its amplitude trend;
- fixed fitted shape plus noise;
- fixed empirical shape plus noise.

Error bars: 400-resample bootstrap 95% intervals for these half widths, conditional on the selected band/trend/noise model. They do not include systematic noise-model uncertainty.

## Readout sensitivity

32 deterministic randomly selected events (8/band) are refitted using all four combinations:
- fc = 5.1 or 6 Hz;
- rc = 68 or 80 ms.

These are near-optimal settings in the preceding calibration study. Selection is independent of which setting gives the preferred scientific trend. The comparison is paired across the same 32 events. It is not a new calibration or a full hardware uncertainty interval.

## Main results

| ADC center | N | tr median [16%,84%], ms | tf, ms | ts, ms |
|---|---:|---:|---:|---:|
| 6500 | 468 | 78.49 [74.55,84.09] | 296.20 [262.89,325.33] | 1069.14 [956.14,1230.90] |
| 10000 | 311 | 76.38 [73.51,79.08] | 311.04 [288.82,331.45] | 1114.33 [1030.80,1229.40] |
| 18000 | 113 | 73.51 [71.83,75.89] | 326.23 [302.99,342.08] | 1166.05 [1056.38,1252.97] |
| 28500 | 263 | 70.36 [69.60,70.98] | 350.19 [340.02,356.14] | 1231.63 [1182.47,1276.93] |

- Direct rise median: 154.28 -> 149.55 ms (about -3.07%).
- Direct decay median: 561.69 -> 581.21 ms (about +3.47%).
- All 768 noise controls and all 128 readout-sensitivity refits converged.
- Observed / fixed-empirical-template-noise half-width ratios: direct rise 0.92–1.16, direct decay 1.33–3.43, effective taus 1.71–3.38. This suggests direct rise scatter is compatible with the tested noise scale, while other features show additional scatter. It does not prove that all excess is intrinsic thermal variation.
- After removal of within-band amplitude trends, the corresponding ratios are direct rise 0.88–1.09, direct decay 1.29–2.57, and effective taus 1.78–3.28. In particular, finite band width contributes noticeably to the high-amplitude direct-decay scatter.
- Strong conditional parameter correlations remain (roughly absolute Spearman 0.86–0.99 among the four effective shape variables). Independent draws of separate taus would destroy that empirical joint structure.
- Four readout settings preserve the high-versus-low direction, but absolute taus change and the small-sample slow-tau curve is not strictly monotone.
- Smoothing 2-to-5 ms changes direct rise by median 0.193 ms and decay by 0.551 ms, smaller than the cross-band median trends.
- Including the previously omitted 20 events has a small effect on the chosen-band medians; exact values are in summary.json.

## Outputs

Established Obsidian locations:
- results/fit_results/cuore_shape_distribution_20261113/
- results/plots/cuore_shape_distribution_20261113/

Key files:
- CUORE_shape_report.html: portable Chinese report with all figures embedded.
- event_features.csv: 5000 event rows, identities and features.
- narrow_band_statistics.csv, amplitude_bin_statistics.csv: summary tables.
- shape_distributions.h5: structured event table, fit parameters, waveform quantiles, spectra and templates.
- noise_controls.jsonl/.csv and noise_width_comparison.csv: complete noise-control records and width comparisons.
- response_sensitivity.jsonl/.csv, parameter_correlations.json: response sensitivity and joint structure.
- summary.json, manifest.json, verification.json: machine-readable conclusions, provenance and checks.
- Eight PNG/PDF figure pairs.

## Reproduce

    python experiments/cuore_shape_distribution/run_study.py extract --data SOURCE_DIR --fits READOUT_FIT_DIR --old OLD_FULL_FIT_DIR --output NEW_DIR
    python experiments/cuore_shape_distribution/run_study.py controls --data SOURCE_DIR --fits READOUT_FIT_DIR --old OLD_FULL_FIT_DIR --output NEW_DIR --repeats 96 --workers 6
    python experiments/cuore_shape_distribution/report.py --input NEW_DIR --plots PLOT_DIR
    python experiments/cuore_shape_distribution/verify.py --input NEW_DIR --data SOURCE_DIR --fits READOUT_FIT_DIR

Scientific dependencies: NumPy, SciPy, h5py, matplotlib. Paths are CLI inputs. Use a fresh output directory for changed scientific settings/code. Control runs reject existing output files to prevent accidental duplicate records; they are not resumable.

## Practical use for future validation data

Use observed amplitude-conditioned waveform trends and measured time features as forward-model targets. Retain joint shape variation rather than independently sampling taus. Do not add the full observed parameter spread to clean pulses and then add noise again: that would count estimator-noise variation twice.

Matching effective parameters is not a unique mapping back to thermal ODE parameters. Compare the generator's final readout waveforms and distributions of direct observables. Exact energy-conditioned generation requires separate calibration, and sub-500-keV extrapolation is unsupported by this release.
