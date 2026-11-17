# CUORE baseline noise and evaluation-set shape comparison (2026-11-17)

## Scope and provenance

Read-only analysis of all 5000 CUORE single-event baselines, existing reference calibration and fits, and original fixed-template evaluation waveforms. No DDPM, event refitting or new dataset generation. The previous 4980-event selection applies only to CUORE pulse-shape distributions; baseline RMS uses all 5000. Source IDs/RMS are checked against existing extraction, raw file and code hashes recorded. Results/figures belong in Obsidian `cuore_noise_comparison_20261117` directories.

    python analysis/cuore_noise_comparison/run_analysis.py --data DATA_ROOT --notes NOTES_ROOT --output RESULT_DIR --plots PLOT_DIR
    python analysis/cuore_noise_comparison/audit_and_report.py --data DATA_ROOT --notes NOTES_ROOT --output RESULT_DIR --plots PLOT_DIR

`--reuse-direct` explicitly reuses the 34000-row direct-time table in RESULT_DIR after checking all 34 cases have the same 1000 source-row identities. Use only with unchanged inputs/definition; it is an intra-study restart aid, not automatic cache invalidation. Run the second command to verify and produce the final common-energy zoom/report.

Dependencies: numpy, scipy, h5py, matplotlib, existing `scripts/window_of.py` and `experiments/cuore_shape_distribution/run_study.py`. Source snapshots and hashes are saved. Raw data are not normalized/clipped in place.

## RMS mapping

CUORE baseline is samples 0:2500 at 1 kHz, minus its own mean; RMS uses sample SD. Apply empirical baseline-group gain and the zero-amplitude slope of the existing two-anchor energy reference scale. Convert equivalent keV sample RMS to simulator mV using the 1-keV clean template sensitivity, 774.3484 keV/V. This is a **simulation-equivalent mV**, not a hardware ADC-to-voltage conversion. The zero-energy slope extrapolates below the calibrated region; local event-slope mapping is saved as a sensitivity check.

- A median equivalent RMS 1.897 mV, 16–84% 1.748–2.072, 2.5–97.5% 1.615–2.343.
- B median 2.115 mV, 16–84% 1.860–2.584, 2.5–97.5% 1.679–3.621; larger right tail.
- Local-slope medians are 1.963/2.201 mV.
- The simulator nominal 1.382/7.5 mV targets are for full 10-s noise windows. On matched 2.5-s pieces their RMS medians are 1.352/7.338 mV, with finite-window variation. This distinction is retained in `same_window_rms_summary.json`.

## Baseline energy-resolution proxy

Public release metadata do not provide an official channel-specific baseline resolution. A specific local estimator is therefore defined:

- 2.5-s noise windows, group-specific 2615-keV median AP, peak-normalized, rising midpoint at 0.75 s.
- Subtract mean from noise and AP, Hann taper both, exclude DC, use standard one-sided PSD weighting at fixed delay 0. No scan/maximization on noise.
- Two event-index folds: learn spectrum on one half, evaluate on the other. AP is the existing high-energy median pulse, not independently retrained per fold.
- Above 20 Hz the inverse spectral weight is suppressed. The generic OF spectral floor is explicitly disabled (`floor=0`), because out-of-band sentinel powers must not inflate its median-based floor. All in-band empirical powers are positive. Unit response to a template injection is checked.
- Report empirical amplitude SD and 2.35482*SD as **Gaussian-equivalent FWHM**, not measured half-height width of a non-Gaussian distribution.

At 2.5 s, same-data AP: A sigma=0.887 keV, equivalent FWHM=2.088 keV; B sigma=1.593, FWHM=3.752. Matched procedure on simulation yields sigma=0.0800/0.4341 and FWHM=0.1883/1.0221 keV. Using a common CUORE-A AP still gives simulation FWHM=0.1667/0.9044 keV, so the difference is not only the chosen AP shape.

10/20/40-Hz cutoff sensitivity is small. Window lengths 1.5/2.0/2.5 s give A FWHM=2.436/2.203/2.088, B=3.942/4.038/3.752 keV. This range is a sensitivity study, not a confidence interval. Do not identify these with official 10-s OF/OT performance. Selected pulse pretriggers are not unbiased random triggers, sub-0.4-Hz noise is unresolved, full calibration/shape systematics are not propagated, and zero-energy pulse shape is unavailable.

## PSD and shape distributions

Mean periodogram: same 2500 samples, Hann, keV²/Hz, no additional detrending after mean subtraction. Power below 20 Hz is A 67.5%, B 75.6%, simulator 44.2%; the detailed spectral shape also differs strongly. RMS alone does not determine optimal-filter energy resolution or pulse-time errors.

Direct shape metrics reuse the **identical existing definition**: 3-ms Gaussian smoothing; rising 10–90%, falling 90–30% crossings around the global smoothed peak. Simulation has only 1.4 s available pre-onset for mean baseline, versus CUORE 2.5 s; this difference is explicit. Use 1000 deterministic paired source rows at each of 17 energies/two levels. Missing crossings are recorded, not silently removed; all 511/727/2615 comparison rows are valid. Stored raw-noisy fits are not used for these direct metrics.

CUORE comparisons use existing +/-2 fitted-sigma peak windows at 511/727/2615 keV, original 4980 selection, and subtract the existing mean energy trend before measuring scatter. The continuum fraction is retained (B 727 is particularly contaminated). Width is central 68% half-width, with SD exported separately to reveal tails. No exact Gaussianity is assumed.

At 2615 keV:
- Rise half-width: CUORE A/B 0.223/0.286 ms; simulation 1.382/7.5 mV 0.093/0.506 ms.
- Decay half-width: CUORE 1.259/2.615 ms; simulation 1.103/5.786 ms.
- Mean rise: CUORE ~149.4–149.7 ms vs simulation ~75.3–75.6 ms; mean decay ~577–583 vs ~771–772 ms. Similar scatter scale does not imply matching central waveforms.

At the three shared energies CUORE central widths lie broadly between the two simulation levels; A decay is close to the low-noise level, B decay wider. B rise has tails, so this statement is specifically about central widths, not every SD or full distribution.

The existing fitted-tau comparison is secondary: CUORE uses shared effective RC+Bessel fitting, simulation uses standard tri-exponential fitting. At 2615 keV CUORE tau half-widths are 10–77 times the low-noise simulated values and 1.8–14 times the high-noise values, but these are **not** estimates of intrinsic physical scatter ratios or directly subtractable variances.

## Implications

Existing noise levels form useful stress tests and bracket central direct-time scatter, but do not reproduce the full measured distribution. Around 2 mV is a candidate for matching total RMS, while noise PSD, mean pulse shape, working-point variation and parameter covariance also need matching. A matched waveform/noise/fitter injection control is required before assigning intrinsic clean-waveform diversity; otherwise noise and fitting degeneracy may be counted twice. No public-data validation below 500 keV is possible here.

Primary references: https://arxiv.org/html/2607.02548v1 (release, 2500-point baseline, single dataset-calorimeter, >500-keV selected calibration pulses); https://arxiv.org/html/2510.25720v1 (OF, AP/ANPS, gain/energy calibration and random-trigger noise methods).

## Artifacts and checks

Per-event RMS conversion, RMS summaries (including matched windows), PSD/band tables, cross-AP baseline estimates and held-out amplitudes, window sensitivity, 34000 direct-time rows/68 summaries, common-energy comparisons, four PNG/PDF figure pairs, offline Chinese report, manifest and independent verification. Audit recomputes all direct-time summaries, baseline-estimator SDs and unit-response checks; source IDs and all CUORE baseline RMS values are validated. No claim that these checks validate the official detector model.

## Baseline offset groups and window structure

`plot_baseline_groups.py --data CUORE_DIR --features CALIBRATED_FEATURES_CSV --output RESULT_DIR --plots PLOT_DIR` plots the baseline-offset distribution of the existing 5000 single-event identities (including the previous 20 flagged fit cases). It independently verifies each first-2500-sample mean, stored normalizationOffset, event ID, group threshold, and available training labels against the raw files. Test identities inherit the prior selection because the public test file has no Labels.

A is offset > -70000 ADC (2716 events), B is offset <= -70000 ADC (2284). Median offsets are -62730.347 / -75023.539 ADC; central 68% intervals are [-63539.192,-62526.438] / [-76681.518,-71737.634]. A and especially B have internal structure: neither should be assumed to be one Gaussian population. The split lies in an empty gap; it does not establish two physical operating states. Baseline offset across events is distinct from noise RMS within a window.

Raw shapes are training (9000,10000), test (1000,10000). The release describes 1-kHz, 10-s triggered event windows, rise midpoint near 3 s, with possible repeated pile-up pulses across windows. They are not consecutive nonoverlapping stream blocks. The public file fields contain event IDs but no absolute acquisition timestamps; concatenation cannot reconstruct a continuous stream. The release explicitly states one dataset-calorimeter, so A/B should not be interpreted as two detectors. Output figure: `05_baseline_offset_groups.png/pdf`; per-event CSV and summary JSON include verification and hashes.

## CUORE versus evaluation fitted-tau overlays (2026-11-17)

`plot_tau_overlay.py --cuore CALIBRATED_FEATURES_CSV --simulation PER_EVENT_SCATTER_H5 --output RESULT_DIR --plots PLOT_DIR` reads existing results only. Produces full-energy log-log and 450–3150 keV linear-axis three-panel plots. Input CUORE is the original 4980 selection (A/B pooled); simulated inputs are the 170000 original noisy-window fits, before DDPM, preserving the existing valid mask and boundary solutions. No downsampling, jitter, new fitting, inference, or source modifications. High-energy plot includes all 4980 CUORE events and 25000 per simulated noise level. Median connectors are display guides, and vertical bars show central 68% intervals. Nested line widths keep coincident noise-level centers visible.

CUORE reference energies actually span 498.334–2631.377 keV; 4 are below 500 due to the approximate reference calibration, distinct from the publication's original reconstructed-energy >500 cut. 239 keV belongs to the simulated set. Outputs in Obsidian `cuore_eval_tau_overlay_20261117`: two PNG/PDF figures, quantile CSV, source snapshot and manifest recording input/code hashes, selections, counts and checks. Source row order and all finite/positive taus are checked. CUORE RC+Bessel effective parameters and simulation standard tri-exponential parameters are not physical apples-to-apples thermal constants. Strong low-energy scatter includes noise/fit degeneracy and bounds; it does not show intrinsic clean-template diversity. This fixed-template evaluation set is not the broad ODE training set.

### Optional training-clean overlay

Add `--train-fits FIT_CLEAN_DIR --train-selection SELECTED_WINDOWS_NPZ` to include the existing 61225 quality-selected ODE clean single events as green scatter. Both arguments are required together. Recompute the original single-event quality criteria and verify exact shard/window identity equality to the saved selection; verify ten completed shards and their shared fingerprint. Save selected source identities, true energies and three fitted taus in a CSV, plus fit/selection hashes and plot counts in the manifest. No fitting or inference runs. The prior three-population plotting mode remains available without these arguments.

Full view extends to 5700 keV to include the entire training range (1.0074–5406.9937 keV); high view retains the previous 450–3150 keV comparison window, containing 18039 selected clean events. Axes and legends explicitly reflect these subsets. Training ranges across the selected single events: rise 21.404–199.563 ms, fast decay 251.892–1229.677 ms, slow decay 2570.694–7076.129 ms. This selection excludes 2261 flagged singles and all pileup fits, so it is not a literal plot of all 90500 training windows. Marginal overlap of rise times does not establish joint waveform coverage; CUORE slow-decay effective fits remain mostly below the simulated training population, subject to different readout-model caveats. Artifacts are saved separately in Obsidian `cuore_eval_train_tau_overlay_20261117`; previous figures remain intact.
