# Evaluation target design: steps 1–2 draft (2026-11-17)

Question: borrow CUORE effective-tau energy trends/observed widths while placing new evaluation pulse shapes within the existing simulated training population. This script only analyzes saved parameters and proposes targets. It does not generate waveforms, infer ODE settings, estimate clean intrinsic widths, run DDPM, or change production settings.

## Run and sources

    python analysis/eval_target_design/run_design.py --notes NOTES_ROOT --output RESULT_DIR --plots PLOT_DIR
    python analysis/eval_target_design/audit_design.py --output RESULT_DIR

Dependencies: numpy, scipy, matplotlib. The notes root must contain the historical CUORE energy-shape results (20261114), selected training parameters from cuore_eval_train_tau_overlay_20261117, and eval_noise_shape_20261116. This is an explicitly dated research analysis, not a generic dataset generator. Source hashes, settings and source-row identity of the reference vector are recorded. Keep results outside Git; recommended Obsidian destinations: analysis/plots `eval_target_draft_20261117`.

CUORE A/B are empirical baseline-offset groups from one dataset-calorimeter, not two detectors. Linear center fits use the original legacy selection and 500–2650 keV (A2712, B2264), equal total weight in each occupied 275-keV bin. Recompute the six weighted linear fits and all twelve log-linear SD/central-68%-half-width fits, checking exact agreement with the existing coefficients. Width models use previously stored detrended, energy-binned residual statistics; no new event fitting occurs.

Training inputs are 61225 previously quality-selected clean single events, not all90500 training windows. Selected source rows were verified in the preceding overlay task. True energy and fit units are keV/ms. CUORE effective RC+Bessel parameters and simulator standard tri-exponential parameters have different interpretations; these are phenomenological design targets, not identified thermal constants.

## Center proposal and support guards

Reference E=2615 keV. In the training interval ±10% around2615 (433 samples), find the actual vector nearest the coordinatewise log-tau median using Mahalanobis distance. The selected vector is shard3/window6780, actual E2605.1848 keV, tau=[85.668227,490.083830,3385.076272]ms. Borrow it as a desired2615 center; this does not imply its ODE settings produce those exact values at2615. ODE center search remains step3.

Primary draft uses narrower group A; B stays separate as a sensitivity scenario. Absolute A slopes are [-3.815002,25.255924,65.656253]ms/MeV. Preserve their ratios with a common attenuation k. Check 100 uniformly spaced energies511–3034 plus the five high evaluation energies:

- Local training window is ±max(100keV,10%E).
- Each center component lies within local16–84%.
- Center ±2 observed A SD lies within local2.5–97.5%. This is a marginal design guard, not a Gaussian95% probability assertion.
- Whiten local log-three-tau vectors. The target center's distance to its10th nearest neighbor must not exceed the local95th percentile of leave-one-out10th-neighbor distances. This heuristic controls center sparsity only; it excludes alpha, waveform residuals, the width's joint distribution and ODE realizability.

k=1 passes the marginal guards but fails the joint-center guard near the low end. On the0.01 factor grid the largest passing k is0.44; round down to a simple draft k=0.4. This depends on the chosen reference vector and heuristic, and is not a physical maximum or proof that another anchor cannot retain full slopes. At k0.4 the worst distance/threshold ratio is0.8557. Effective slopes are [-1.526001,10.102370,26.262501]ms/MeV.

mu(E)=anchor + k*slope_A*(max(E,511)-2615)/1000.

B-slope alternative centers use the same anchor/k for display but have not been separately accepted as a full design. B width guards in the table are explicitly evaluated around the A center, not the alternative B center. No group blending or Gaussian assumption.

## Observed width proposal and unapproved edges

Retain both SD and central68%half-width as distinct statistics. Reuse log-linear curves in absolute ms; do not multiply widths by the ratio of new/old center values. The symmetric plotted center±half-width is a design visualization, not the actual possibly asymmetric CUORE16–84%interval. SD and half-width are not interchangeable and cannot both be interpreted as Gaussian sigma.

- 511–2615: use measured-range center trend and width curves.
- 3034: extend the attenuated center line modestly, freeze widths at2615 rather than extrapolate to ever narrower distributions.
- 1–50 and239: freeze center/width references at511; explicitly unvalidated placeholder policy. These are not promised achievable noisy distributions, and low-energy training support is not certified by the high-energy guard.

All clean-width columns remain empty. A/B widths contain experimental noise, fitting effects and operating-state variation. Old-template noise/fit widths are exported only as a warning comparison, never subtracted. A new matched center waveform and a specified reference noise condition are required before variance budgeting. The same future clean windows should be used for both noise levels; choosing different clean widths to force both outputs to match CUORE would defeat the comparison.

## Artifacts and validation

Five PNG/PDF figure pairs: A/B center lines, separate SD/half-width curves, full/high target overlays with training and original CUORE, and local training marginal guards. Offline Chinese report; seventeen-point target table (A center, A/B observed SD/half-width, B alternate center, empty clean widths), curve table, coefficients, slope scan, training support table and old-noise warning table. Input/code hashes and settings in design_summary.json.

Independent audit verifies17 rows and expected energies, linear centers, low-energy frozen references,3034 width freeze, all clean widths unset, high-energy support flags/ranks, rejection of full slope, selected factor passing, and unchanged input hashes. Visual inspection of all five plots passed. These checks establish numerical/provenance consistency, not validated physical coverage or achievable clean distributions. Outputs are a reviewable draft; next stages await user direction.
