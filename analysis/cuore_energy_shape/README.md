# CUORE energy-conditioned shape analysis (2026-11-14)

## Question and status

Can pulse shape be described by a simple energy-dependent center and spread for future validation-set generation? This analysis produces an exploratory reference calibration and continuous descriptive statistics. It does not generate new pulses or modify DDPM, raw HDF5 files, or existing fits.

## Inputs and reproducibility

- Original public CUORE training/test HDF5, read-only; all 5000 single pulses.
- `event_features.csv` and adjacent `manifest.json` from the 2026-11-13 shape distribution study. Features include the shared effective RC+Bessel three-exponential fits and directly measured rise/decay times.
- The earlier `analysis/cuore_spectrum/plot_spectrum.py` supplies integrated Gaussian + constant-background Poisson peak fits.
- Configurable CLI paths; no data or generated outputs belong in Git. Local results/plots are in the Obsidian `cuore_energy_shape_20261114` directories.

    python analysis/cuore_energy_shape/run_analysis.py --data RAW_DIR --features FEATURES_CSV --output RESULT_DIR --plots PLOT_DIR
    python analysis/cuore_energy_shape/summarize.py --data RAW_DIR --output RESULT_DIR --plots PLOT_DIR

Dependencies: numpy, scipy, h5py, matplotlib. Fixed RNG seed 261114. Inputs and code are hashed, event file/row/ID is retained, raw input hashes must match the previous feature extraction. The second command independently checks all 5000 event identities, amplitudes and baselines against HDF5.

## Reference energy calibration

1. Empirical baseline split: A > -70000 ADC (2716 events), B <= -70000 ADC (2284). These are not official run IDs.
2. In each group, use the high-amplitude region A > 27500 to fit a robust linear amplitude vs baseline relation (soft-L1 scale 40 ADC). Divide amplitudes by its relative gain, referenced to the group median baseline. Continuum contamination and residual gain nonlinearity remain possible.
3. Fit 583.191 and 2614.511 keV peaks; infer E=c1*(Acorr/10000)+c2*(Acorr/10000)^2, assuming zero intercept. Other peaks are withheld checks, not extra calibration anchors. Positive derivative and independent peak residual <10 keV are required.
4. Actual maximum residual across 511/727/911/969 checks: A 0.630 keV, B 1.513 keV. B's 727 candidate has substantial estimated continuum contamination; 969 can blend. This is not official calibration or a resolution measurement, and the residual is not a per-event accuracy guarantee.

Sources and scientific motivation: public data release Appendix A https://arxiv.org/html/2607.02548v1 ; CUORE baseline gain stabilization and energy reconstruction https://arxiv.org/html/2510.25720v1 . Exact source deployment/run labels are not provided. No official CUORE gain model is claimed to have been reproduced.

## Shape and distribution methods

- Preserve the earlier 4980-event display selection, without separately highlighting the other 20 events. Save reference energies for all 5000 and refit mean trends with all 5000 as a selection sensitivity check.
- Fit six quantities (three effective taus, fast-component weight alpha, direct 10–90% rise and 90–30% decay) separately by baseline group.
- Means use 500–2650 keV (2712 A + 2264 B), equal total weight per occupied 275-keV bin. Five-fold held-out weighted RMSE compares linear/quadratic models. Quadratic requires >2% improvement; all twelve group/feature combinations select linear. This is a practical simplicity rule, not proof of exact linear physics.
- Standard deviation and middle-68% half-width are computed after subtracting the fitted energy trend in each bin. SD intervals use 300 event bootstraps, conditional on the fitted trend/calibration; calibration and model systematic uncertainty are not propagated. Width models are exp(s0+s1*x), x=(E_keV-1500)/1000. These positive smooth descriptions have not been independently validated as generative models or optimized against every alternative functional form.
- Same-line checks select +/-2 fitted amplitude sigmas about each observed peak. Estimated continuum fractions are exported. They are approximately monoenergetic selections, not pure true-energy labels.
- D'Agostino normality tests are accompanied by skewness, excess kurtosis, empirical widths/tails, and Benjamini-Hochberg correction across 72 tests. 37 reject at q<0.05. Non-rejection does not prove Gaussianity; observations share a fitted trend and selection systematics.
- Peak histograms are centered by each peak residual mean and normalized by its own SD to test shape, not trend offset. Statistics use all selected points; histogram display spans -5 to 7 SD.
- Preserve full joint standardized residual vectors and Pearson/Spearman matrices, with energy and source identities. These retain measurement error as well as possible intrinsic variation. Joint behavior can still depend on energy; do not pool blindly to generate pulses.
- Actual waveform comparisons normalize peak height and align the rising 50% crossing, without stretching time. Medians and 16–84% envelopes are exported for 583, 911, 2615 keV selections.

## Findings and limits for simulation

- Rise becomes faster and both effective decay constants increase with energy. The bulk center trends are close to linear over the measured range.
- Spread decreases with energy; B has larger scatter than A. Several three-exponential parameter distributions have long tails, while A's direct rise/decay distributions at 583/911/2615 are compatible with Gaussianity at current sample sizes.
- Tau/alpha residual rank correlations are very strong (roughly |rho|=0.84–0.98); independent parameter Gaussians are inappropriate.
- Some residual direct times still correlate with baseline in B; gain-correcting amplitude does not remove waveform working-point dependence. Group separation is necessary but not a complete physical correction.
- Observed scatter includes noise, fitting degeneracy and working-point variation. Treating it as clean-pulse scatter and adding noise again can double-count noise. Existing controls or forward simulation must establish intrinsic variability first.
- Effective readout-fit taus are not ODE thermal parameters. Matching an ODE generator requires forward waveform validation, not assigning these taus directly to ODE parameters.
- No validated extrapolation below 511 or above 2615 keV. Energy-shape errors are correlated because the same waveform supplies both. Only a selected single dataset-calorimeter is studied.
- This commit records analysis, not approval of a new production generator.

## Outputs

Calibrated per-event CSV; calibration coefficients/peak checks; mean/SD/core-width models and coefficients; bin and peak statistics; selection sensitivity; joint residual vectors/correlations; aligned peak waveform NPZ; nine PNG/PDF figure pairs; offline Chinese HTML report; manifests, verification and source snapshots.
