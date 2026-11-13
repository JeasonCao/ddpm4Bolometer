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
