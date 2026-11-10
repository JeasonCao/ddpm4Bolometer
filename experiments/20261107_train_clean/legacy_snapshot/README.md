# Training clean tri-exponential fit

Authorized 2026-11-07. The runner reads clean shards only. It imports an immutable
snapshot of the project's existing fitter; no DDPM or GPU is involved.

## Current state and outputs

- `status.json`: current stage, shard, completed rows; refreshed every 20s or 25 completions.
- `run.log`: append-only process output and errors.
- `process.json`: supervisor/runner PID and eventual exit code.
- `manifest.json`: source SHA256, code SHA256, exact package versions and configuration.
- `pilot_report.json`: 500-window stratified pilot, quality/rate gates and projected time.
- `final_report.json`: created only after all 90,500 windows and validation complete.
- HDF5: `D:/Obsidian/note/课题/DDPM4Bolometer/results/fit_results/train_clean_triexp/`.
- Pilot HDF5: output directory's `pilot/` subdirectory. Contains original `window_idx`.
- `smoke.h5`: one real source row from initial engineering test; not a production result.

States: `preflight`, `pilot_running`, `pilot_passed`, `full_running`,
`paused_for_scientific_review`, `stopped_error`, `complete`.
Check state rather than interpreting exit code 0 as full completion: a scientific
pause is a controlled return. Do not relax gates merely to reach full execution.

## Restart

Run `launch_background.py` using the same Python interpreter recorded in the manifest.
The runner uses an exclusive OS lock. It verifies source/code/config fingerprints,
reuses completed pilot fits, and resumes only missing rows from partial HDF5s.
Never change source snapshots/configuration or overwrite completed files to force resume.
For a new fitting algorithm, use a separate versioned output directory.

The supervisor prevents idle sleep while running, releases the request on exit,
and leaves display sleep unchanged. Manual sleep, reboot or shutdown still interrupts
the job. It does not depend on an active model conversation.

## Data conventions

Each output row maps to `(source_shard, window_idx)`. Full files preserve all rows
in original order, including failures. Metadata groups and all non-waveform source
datasets are copied exactly, preserving original dtypes and dataset attributes.
`n_pulses` inherited file attribute means SOURCE WINDOW COUNT. `n_pulses_fit` is
the per-row 1/2 multiplicity. `output_row_count` is the actual output count.

Pulse columns have shape `(N,2)`, ascending fitted onset. Single pulse column 2 is NaN.
V and seconds are used for fitted parameters. `amp` is the fitted pulse peak;
`amp_raw` is the prefactor. `amp_maxbase` measures the combined input window,
not either constituent pileup pulse. `baseline` is the combined model baseline.
`t_fall1 <= t_fall2`; `alpha` is the fast-decay coefficient. These named fields
reconstruct the waveform. `raw_params` preserves the original solver ordering,
with `n_params` 7/13 and NaN padding; never combine raw alpha with sorted taus.

`bounds_lower/upper` and `boundary_mask` use RAW parameter-vector ordering.
For pileup the combined baseline has no single-stage bound, so its bounds are NaN.
First-stage baseline bounds remain documented in source_snapshot/fit.py.
The boundary tolerance is 0.001 times the search interval; it is diagnostic.
`decay_degenerate` means decay constants differ by less than 5% of the slow one.
`initial_sensitivity` is a diagnostic maximum fractional tau change from a second
initial guess on a subset of pilot singles; it never replaces the primary fit.
Truth onsets are used only for post-fit error diagnostics. A sorted fitted pulse
index is not proof of reliable matching in unresolved pileup.

`success` reports solver completion, not physical parameter identifiability.
`relative_rms` is full-window residual RMS / input max-minus-baseline amplitude.
`chi2` is reduced chi-square with baseline sigma floored at 1e-12 V; no chi2 gate
is valid for these noiseless inputs. Full-window relative RMS can dilute local
shape errors; these gates are a conservative launch screen, not a publication cut.
No waveform, reconstructed waveform, covariance, or parameter uncertainty is saved.

`done` includes failed/timed-out attempts; consult `success`, `timeout`, `message`.
Single failures and pileup failures are retained. Overlapping pileup can yield a
successful yet unreliable decomposition; inspect onset errors/segments separately.

Current-code fixed simulator defaults are explicitly distinguished from HDF5
metadata. Historical generation-code identity is not established by these defaults.
The immutable source snapshot and dependency versions support reproducing this run.

## Gates and runtime

See `manifest.json` for exact thresholds. Pilot: 30 singles by energy and 20 pileups
by time gap per shard, 10 shards, seed 20261107. Single success >=95%; no more than
10% of successful singles with relative RMS >5%; tau-boundary and decay-degeneracy
fractions each <=20%; if >=10 sensitivity probes, <=20% with tau changes >20%.
Full projection including 50% overhead margin must be <=8h. Otherwise pause.
Pileup quality is reported separately and does not block the single-event study.
Six persistent workers, numerical libraries single-threaded, 120s per-window hard
timeout, 12h overall budget. Only the controller writes HDF5. Completed files are
validated then renamed from `.partial.h5`; completed final files are not overwritten.

Diary entries are appended automatically at launch, pilot gate, each completed
full shard, final completion or error. TODO is initially marked pending and should
be reconciled with final status at the next review.
