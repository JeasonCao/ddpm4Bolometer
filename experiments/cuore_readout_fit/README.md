# CUORE effective readout fit

## Question and scope

Can the leading-edge and peak mismatch of direct three-exponential ADC fitting be reduced by modeling a causal readout response?

This independent experiment does not change the canonical fitter or simulation defaults. Single pulses only; pileup is not refitted here.

**This is not a calibrated reconstruction of temperature or a reproduction of the complete 2011 detector model.** Public files omit the required hardware settings. Lower residuals do not establish the actual cutoff, capacitance, resistance, or thermal constants.

## Inputs and identities

- Release: https://zenodo.org/records/20721645
- cuoreTraining.h5: first 4500 labeled singles. cuoreTest.h5: first 500 singles according to its row-order/count attributes.
- 10,000 samples/window; 1000 Hz from the release paper, not an HDF5 sampling-rate field.
- Input = (Waveform - normalizationOffset) / normalizationScale, without clipping or smoothing the target.
- Horizontal axis: observed amplitude (normalizationScale, ADC); no event energy is supplied.
- Inputs read-only. Results retain source filename, original row, event ID, offset, scale, source SHA-256 and source attributes.
- Scatter selection: the **same 4980 events** with collapsed rise/fast-decay constants in the previous bare-fit study. The other 20 remain in fitting and HDF5, not drawn as separate scatter points.
- Paths are CLI arguments. Outputs belong in Obsidian results directories, not Git.

## Model

The stable positive-coefficient core is reused from experiments/cuore_fit_ranges/stable_fitter.py:

    x(t) = C [exp(-s/tf) - exp(-s/tr)] / delta
         + D [exp(-s/ts) - exp(-s/tr)],    s = t-t0 > 0
    tf = tr exp(delta), ts = tf + gap, C,D,gap > 0, delta >= 0

At delta=0 the first term is C (s/tr) exp(-s/tr). For delta>0 this is the same positive-mixture tri-exponential family, alpha=C/(C+D*delta).

Four comparisons:

1. Bare: stable three-exponential output plus constant baseline.
2. Bessel: causal sixth-order Bessel applied to x.
3. RC+Bessel: du/dt=(x-u)/rc, then Bessel.
4. Nonlinear RC+Bessel:

       du/dt = [expm1(k*x)/k - exp(k*x)*u] / rc

   This effective constant-current-limit response follows the exponential resistance form as RB/RL -> 0. Finite load-resistance effects are omitted. k is an effective coefficient in normalized ADC units, not temperature, bias or gain. k->0 recovers linear RC; the separate linear model gives the exact zero limit.

The constant ADC baseline is added after filtering. Pulse-dependent circuit/filter initial state is zero before onset. Internal sampling is 2 kHz, sampled at 1 kHz; RC uses a midpoint piecewise-constant exact recurrence. Bessel uses SOS and norm='mag': cutoff is **-3 dB**. No filtfilt or removal of filter delay; t0 is optimized.

Assumed sixth order and effective settings are not verified release-specific hardware settings. Neither the simulator's 120 Hz phase-normalized setting nor the 2011 apparatus's 12 Hz setting is silently transferred.

## Fitting and selection

- Unweighted least squares on all 10,000 samples: baseline, two coefficients, onset and three ordered time constants per event.
- Moderate bounds: tr=0.002–0.6 s; tf/tr=1–100; ts-tf=0.002–15 s; baseline +/-0.1; onset from observed maximum minus 0.95 s to maximum minus 0.001 s.
- Free-response diagnostics: fc=1–150 Hz, rc=0.0005–0.3 s, k=1e-5–1. Free rc and tr strongly compensate; these are diagnostics, not physical parameter estimates.
- Calibration: 16 training singles selected before fitting at evenly spaced amplitude ranks from the 2nd to 98th percentiles.
- Validation: 24 test singles with the same rank rule. Their event parameters are fitted, but they do not choose shared response settings.
- The previous displayed training rows 385, 3629, 2864, 3519 remain for visual comparison; they are not added to calibration unless already selected.
- Bessel-only grid: 2,2.5,3,3.5,4,5,6,8,12 Hz.
- RC grid: fc=4,6,8,12,20 Hz; rc=20,40,60,80,100,140 ms.
- Two refinement passes multiply shared candidates by 0.85,1,1.15. Selection minimizes calibration mean squared normalized residual, requiring all 16 fits to converge.
- Grid: one start/event; free diagnostics: two; final pilot: three. Full singles: two, retaining both costs/statuses and near-optimum tau spread.
- Selected shared settings: Bessel-only fc=3 Hz; RC+Bessel fc=5.1 Hz, rc=68 ms. **Effective values conditional on this model/search, not hardware measurements.**

## Independent 24-event test evidence

Median metrics as percent of observed amplitude:

| Metric | Bare | Shared Bessel | Shared RC+Bessel |
|---|---:|---:|---:|
| Full-window RMS | 0.8677 | 0.4182 | 0.3339 |
| Leading-edge/peak-region RMS | 2.6286 | 1.0236 | 0.6550 |
| Absolute residual at observed maximum | 3.2484 | 0.7403 | 0.6388 |

Peak region = (t_peak-0.3, t_peak+0.65) seconds. Error at the observed maximum is **not** an unbiased energy/amplitude-resolution estimator.

All 24 shared RC+Bessel fits converged without the previous tr≈tf collapse. Shared nonlinear k reached its near-zero lower bound in all 16 calibration and 24 validation cases without improvement. This supports expanding the simpler linear model, **not** concluding that actual thermistor nonlinearity is absent.

Free RC fits remain ill-conditioned. Fixing shared RC removes that freedom but does not prove its value. Among tested shared settings within 1% of the best calibration mean squared residual, median tr spans 65.5–80.0 ms, tf 299–313 ms, ts 1.117–1.154 s. These are grid sensitivities, not confidence intervals.

Structured residuals remain visible in the unchanged examples. No claim of noise-limited fitting or measured thermal constants.

## Commands

From repository root, with NumPy/SciPy/h5py/matplotlib:

    python experiments/cuore_readout_fit/check_model.py --output CHECKS_JSON
    python experiments/cuore_readout_fit/run_study.py pilot --data SOURCE_DIR --old PREVIOUS_FULL_FIT_DIR --output NEW_RESULT_DIR --workers 6
    python experiments/cuore_readout_fit/run_study.py full --data SOURCE_DIR --old PREVIOUS_FULL_FIT_DIR --output NEW_RESULT_DIR --workers 6
    python experiments/cuore_readout_fit/verify.py --input NEW_RESULT_DIR --data SOURCE_DIR
    python experiments/cuore_readout_fit/report.py --input NEW_RESULT_DIR --plots NEW_PLOT_DIR --data SOURCE_DIR --old PREVIOUS_FULL_FIT_DIR

Use a fresh output directory after scientific code/settings change. JSONL resumes by source file/row/configuration tag. Manifest/full-run configuration guards detect changed inputs/settings, but do not replace a fresh directory after code changes.

## Files and validation

- model.py: causal responses, sensitivities, bounds and multistart least squares.
- run_study.py: identities, preselected samples, shared-response search, full singles and fit-only HDF5.
- check_model.py: off-grid finite-difference Jacobians, causality, bare equivalence, linear limit, oversampling and noiseless recovery. Bare onset is not differentiable exactly at t0.
- verify.py: source hashes, row/event alignment, copied normalization metadata, no waveforms, finite ordered parameters, reconstruction of saved fits and 2x/4x sampling comparison.
- report.py: held-out metrics, unchanged examples, grid sensitivity, paired full-single metrics and unchanged 4980-point selection.

Fit HDF5 contains identities, parameters, effective taus, residual metrics, convergence/boundary flags, initialization stability, timing, normalization metadata and source attributes; no waveforms. full_single.jsonl preserves start diagnostics. full_complete.json and verification.json distinguish completion from validation. Solver success alone is not a fit-quality cut.

Implementation checks establish numerical consistency, not hardware validity. No independent energies/calibrated hardware settings are available. No covariance-based physical uncertainties are claimed.

## References

- Carrettoni & Vignati, JINST 6 (2011) P08007:
  https://www.roma1.infn.it/people/vignati/curriculum/JINST_6_2011_P08007.pdf
  Eq.4.1 is temperature. Actual voltage adds thermistor/circuit/gain/filtering. This experiment implements an effective subset/limit, not that calibrated full model.
- Release and same-calorimeter provenance: https://arxiv.org/html/2607.02548v1
- DAQ context: https://arxiv.org/html/2510.25720v1
- Bessel normalization: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html

Possible subsequent work: obtain actual channel/filter/bias settings; investigate residual structure/baseline/tails; validate pileup with nonlinear inputs combined before the circuit. These are not silently enabled here.

## Completed full single-event run (2026-11-13)

- 4500 training + 500 test singles, all 5000 converged from both starts; 731 s on six local workers.
- Full-window RMS median: 0.877867% -> 0.337202%; all 5000 improved; median paired reduction 61.6739%.
- tr/tf collapse (delta<0.05): 4980 -> 6; no other bound contacts. This is conditional parameter separation, not proof of physical identifiability.
- Median effective taus: 75.917 ms, 311.650 ms, 1126.371 ms.
- Median peak-region RMS 0.6403%; median absolute residual at observed maximum 0.6685%.
- Median residual/baseline-noise RMS remains 2.047: structured mismatch persists.
- Largest tau spread among near-equal-cost starts: 0.0926%; response-grid sensitivity is substantially larger.
- Two fit-only HDF5 files total 1,120,220 bytes. All source hashes/rows/event IDs/normalization values match; sampled reconstruction residual differences are zero; actual-fit 2x/4x waveform discrepancy <=8.01e-6 of observed amplitude.
- Results: Obsidian results/fit_results/cuore_readout_20261113; figures: results/plots/cuore_readout_20261113.
- Actual executed model/runner and reused stable core snapshots are preserved. Final code additionally flags an upper delta bound; none of the 5000 fits reaches it, so all stored flags remain correct.
