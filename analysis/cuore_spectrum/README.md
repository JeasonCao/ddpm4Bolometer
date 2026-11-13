# CUORE calibration-spectrum inspection (2026-11-14)

## Scope and inputs

Read-only inspection of all 5000 labeled single pulses (4500 training + 500 test).
Pileup is excluded. Amplitude is the original normalizationScale, not an OF or gain-stabilized amplitude. No existing waveform/fit/shape-analysis result is changed.

The release confirms calibration-run data from one dataset-calorimeter, selected above 500 keV, but does not provide per-event energies, official calibration coefficients, run IDs, or deployment details identifying the exact source configuration.

## Findings

Multiple strong peaks match the characteristic thorium-chain calibration pattern:

| Proposed feature | Group-A ADC centroid | Reference-scale centroid, keV |
|---|---:|---:|
| Annihilation 511 | 5689.39 | 511.144 |
| Tl-208 583 | 6483.17 | 583.191 (anchor) |
| Bi-212 727 candidate | 8066.44 | 727.433 |
| Ac-228 911 | 10069.32 | 910.926 |
| Ac-228 969 region | 10690.71 | 968.086 |
| approximately 1590, possible blend | 17344.57 | 1587.060 |
| approximately 2104 single-escape candidate | 22793.59 | 2103.348 |
| Tl-208 2615 | 28102.65 | 2614.511 (anchor) |

The 969 region can include the nearby 965 line. The approximately 1590 structure is weak and not a clean resolved double-escape line; a neighboring 1588 line can contribute. These two centroids must not be treated as high-precision isolated-line measurements. This inspection does not establish Co-60 peaks or identify the deployed calibration system from metadata.

## Baseline groups and provisional energy scale

There is an empty baseline gap between -70000 and -65000 ADC. Split descriptively at -70000:
- Group A: baseline > -70000, N=2716.
- Group B: baseline <= -70000, N=2284.

These labels are not run IDs. Similar peak structures shift systematically between groups; the high-amplitude peak region also correlates with baseline within groups. This is consistent with gain/working-point variation, but the causal mechanism is not determined here.

Only group A receives an illustrative energy scale:

    x = raw amplitude / 10000
    E_reference_keV = 890.3110328*x + 14.24494116*x*x

A zero-intercept quadratic is fixed using the proposed 583.191 and 2614.511 keV peaks. The 511/911/969 peaks are NOT fit anchors: their mapped centroids differ from references by +0.145/-0.278/-0.885 keV. This corroborates the line pattern under the chosen model; it does not establish sub-keV accuracy for every event or across the full range.

The figure and event CSV label this scale explicitly as provisional. Group B has no assigned energy. No thermal-gain stabilization or official CUORE calibration is claimed.

For future same-energy shape comparisons, first handle the baseline groups/gain response. Previous amplitude-conditioned distributions remain valid descriptions of the mixed raw sample, but cannot automatically be interpreted as energy-conditioned or intrinsic shape distributions.

## Method

- Histogram 25/50/100 ADC bins; candidate peaks from a one-bin Gaussian-smoothed count histogram, prominence >=3 and spacing >=250 ADC. Detection is descriptive, not a significance test.
- Proposed line assignments are selected from the observed multi-peak pattern and literature.
- Group-A local centroids: binned Poisson likelihood, integrated Gaussian plus constant background, 10 ADC bins, explicit saved windows.
- Calibration anchors must converge away from centroid bounds. The provisional map must be monotonic; each of the three check-peak residuals must be below 10 keV before the energy figure is emitted.
- Finite-sample centroid uncertainties, systematic calibration uncertainty, and detector energy resolution are NOT estimated. Do not infer official resolution from fitted widths.
- Histograms are selected-sample counts, not rates or unbiased energy spectra.
- Checks: 5000 unique event identities, count conservation, group partition, monotonic mapping, independent anchor-excluded peak checks, unchanged source hashes, visual figure inspection.

## Outputs and reproduction

    python analysis/cuore_spectrum/plot_spectrum.py --data SOURCE_DIR --output RESULT_DIR --plots PLOT_DIR

This run uses:
- Obsidian results/analysis/cuore_spectrum_20261114/
- Obsidian results/plots/cuore_spectrum_20261114/

Files: events.csv, histogram_adc.csv, histogram_reference_energy_A.csv, peak_candidates.csv,
peak_assignments_group_A.csv, summary.json, verification.json, and three PNG/PDF figure pairs.
Original source file/row/event ID is retained. Group B's energy column is blank.

## Sources

- Release Appendix A (calibration runs and selection):
  https://arxiv.org/html/2607.02548v1
- CUORE gain stabilization and calibration methodology; modern Th-Co spectrum:
  https://arxiv.org/html/2510.25720v1
- CUORE thesis (Th calibration peaks at 583/911/969/2615; historical source systems):
  https://cuore.lngs.infn.it/sites/default/files/2021_Huang_search_0nbb_CUORE-CUPID_compressed.pdf
- NNDC evaluated Bi-212 decay data (727.330 keV):
  https://www.nndc.bnl.gov/ensnds/212/Po/beta_decay_60.55_m.pdf
