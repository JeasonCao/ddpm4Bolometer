# Fixed-template evaluation-set noise and fit scatter

Started 2026-11-16; reviewed/completed 2026-11-17 (Hong Kong). Output directory retains the start date.

## Scope

Reuse the raw-noisy fit results for 17 known energies and two noise RMS levels (1.382 and 7.5 mV), 5000 events per case: 170,000 events. No DDPM outputs are analyzed, no noisy event is refitted, and original HDF5/fit files remain read-only. Only 17 clean templates are fitted to establish same-estimator reference values.

## Inputs

- Data disk: `eval_data_holdout/efficiency_{1p382mV,7p5mV}/eval_efficiency_{E}keV.h5`.
- Notes: `results/fit_results/model261108_nonoise/efficiency_{level}/fit_efficiency_{E}keV.h5`, **only `*_noisy` fields**. The directory name does not mean this analysis uses denoised data. These files preserve cached original-noisy fits and source hashes.
- Canonical `src/basics/fit.py`; generation logic in `scripts/generate_eval_dataset.py` confirms `noisy[i] = clean + scaled_noise[i]`.
- Energies: 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034 keV.

Every source file SHA-256 must equal its fit metadata; all energy/noise metadata and array lengths are checked. Same-energy templates are checked for exact equality between noise levels. Fixed 32 row indices are sampled across all cases to verify actual RMS and paired noise reuse. This is not a full 170,000-row waveform subtraction audit.

## Interpretation

At fixed energy, the underlying clean waveform has no event-to-event variation. The observed fitted-parameter scatter therefore measures the effect of this noise ensemble passed through this estimator, including nonlinear behavior, finite SNR, parameter degeneracy/constraints and interaction with model mismatch. It is not two independently measured variances called noise and fitting. Clean-fit bias is separate from scatter, and the clean fit is a reference, not ODE parameter truth.

Templates are fixed **within energy**, not identical across energy. Two noise levels share exactly the same clean at each of the 17 energies. Normalized template differences vs 2615 keV reach about 2.57% at 1 keV. Sampled normalized noise is reused across energies and levels to numerical precision; different energy points are not independent noise experiments.

These uncertainties cannot be subtracted directly from CUORE data variance: the public pulses have a different shape/readout, noise ensemble and RC+Bessel fitting model. A matched fixed-CUORE-template injection/refit control is needed for quantitative variance decomposition. No new generator or CUORE subtraction is implemented here.

## Selection and boundaries

Main distributions use solver success AND finite all three taus. They do not apply historical `converged_noisy` (chi-square/amplitude), `efficient_noisy`, energy-response or boundary cuts. All 170,000 records are valid by this numerical criterion. It does not imply all are scientifically reliable. Historical flags and counts remain available for comparison.

A code audit found effective single-pulse lower bounds differ from the top-level constants: tau_r is 10–200 ms; pre-sort decay lower bounds are about 67.082 and 450 ms (from `_setup_single`), with 10000 ms upper bounds. Comments calling these just initialization hints do not change the actual optimizer bounds. The existing fitter is not modified.

Boundary proximity uses 1e-4 of each search range. Sorting can map the original 450-ms floor to the stored fast component, so fast tau near 450 ms is additionally marked as an **ambiguous boundary candidate**, not proven active constraint. Consequently the aggregate boundary-candidate rate can include interior solutions near 450 ms and should not be interpreted as an exact saturation fraction. Alpha/baseline proximity is also saved. Boundary candidates remain in all scatter statistics.

## Statistics and figures

- Mean, median, SD, central 68% half-width, 2.5/16/84/97.5 percentiles, offsets relative to the clean fit, and relative widths for all 34 cases and three taus.
- Full scatter, median/68% envelope and clean-fit reference (display-only horizontal jitter).
- Absolute and relative width vs energy; separate center-bias plots.
- Numerical failure / boundary-candidate / historical convergence rates.
- Full-range distributions at 5, 50, 2615 keV, without a Gaussian assumption.
- Raw waveform envelopes from the fixed 32 sample rows at the same three energies. Clean-peak normalization is shared within a case; no per-noisy-event peak normalization or time stretching.
- Per-event fit parameters/offsets, source rows and all flags in an independent compressed HDF5; no raw waveforms copied.

At 2615 keV, SDs (rise/fast/slow) are 0.05509/0.22807/0.84637 ms at 1.382 mV and 0.30515/1.22849/4.58021 ms at 7.5 mV. Ratios 5.539/5.387/5.412 closely follow the noise RMS ratio 5.427. Low-SNR widths and centers are strongly affected by constraints and non-Gaussian behavior, so the same simple scaling is not assumed there.

## Reproduction

    python analysis/eval_noise_shape/run_analysis.py --data DATA_ROOT --notes NOTES_ROOT --output RESULT_DIR --plots PLOT_DIR
    python analysis/eval_noise_shape/verify.py --output RESULT_DIR

Dependencies: numpy, scipy, h5py, matplotlib. Outputs are in notes `results/analysis/eval_noise_shape_20261116` and `results/plots/eval_noise_shape_20261116`. The manifest records source and code hashes; verification independently compares all exported event taus and recomputes all 102 summary rows from the original saved fit files.
