# Clean training-set fit experiment

Question: relate waveform fit parameters to thermal/electrical simulation parameters. This is a specific 90,500-window workflow, not a general fitter or DDPM change. Original code is in legacy_snapshot at commit 8e8e3e6; the completed data-disk _development/train_clean_triexp run remains unchanged.

Repository code separates source, dependencies and runtime state. Scientific fitter calls, thresholds, selections and saved conventions are unchanged. Use a normal Python environment; requirements.txt records the original versions. Windows is checked; POSIX locking has not been runtime-tested here. Torch is not needed.

## New run, only when authorized

Set environment variables before Python:

```powershell
$env:DDPM_DATA_ROOT = 'D:\storage\research\DDPM4bolometer'
$env:DDPM_CLEAN_RUN_DIR = 'D:\storage\research\DDPM4bolometer\runs\train_clean_new_run'
$env:DDPM_NOTES_ROOT = 'D:\Obsidian\note\课题\DDPM4Bolometer'
# Optional DDPM_CLEAN_SOURCE_DIR; default DATA_ROOT/simu_data/clean.
# DDPM_CLEAN_FIT_DIR / DDPM_CLEAN_PLOT_DIR default inside RUN_DIR.
python scripts/fit_train_clean.py --help
python scripts/fit_train_clean.py --smoke
python -m experiments.train_clean_20261107.launch_background
```

Supervisor runs as a normal process; use a background launcher if wanted. Windows inhibits idle sleep until exit. Nothing launches on import. State, snapshots, logs and locks go in RUN_DIR. Code-version changes invalidate an old fit manifest: do not force-resume the historical completed run using this refactored runner.

## Analyze completed historical results

```powershell
$env:DDPM_DATA_ROOT = 'D:\storage\research\DDPM4bolometer'
$env:DDPM_CLEAN_RUN_DIR = "$env:DDPM_DATA_ROOT\_development\train_clean_triexp"
$env:DDPM_CLEAN_FIT_DIR = 'D:\Obsidian\note\课题\DDPM4Bolometer\results\fit_results\train_clean_triexp'
$env:DDPM_CLEAN_PLOT_DIR = 'D:\Obsidian\note\课题\DDPM4Bolometer\results\plots\train_clean_replot_new'
$env:DDPM_NOTES_ROOT = 'D:\Obsidian\note\课题\DDPM4Bolometer'
python -m analysis.train_clean.plot_after_fit
python -m analysis.train_clean.plot_blue_only
python -m analysis.train_clean.plot_3d_tbase_aec
python -m analysis.train_clean.plot_tau_energy
```

First plot command waits for all ten complete fit files. Blue/3D/energy validate exact selection indices. Inputs are read-only; completed plot outputs are not overwritten. Analysis status is updated in RUN_DIR. Diary uses current date or DDPM_DIARY_DATE; automatic rewriting of a hardcoded old TODO line was removed.

## Conventions and limits

Rows map to source_shard/window_idx. Params/equilibrium/energy/onset metadata retain dtype. No waveforms saved. Pulse fields are (N,2), sorted onset; singles second column NaN. amp=peak, amp_raw=prefactor. Sorted fast/slow taus use canonical alpha; raw_params preserves solver ordering. Units V/s/keV and source SI units. Source n_pulses attr means source window count; n_pulses_fit is per-row multiplicity.

Solver success does not guarantee identifiability. No clean chi2 cut: sigma floors at 1e-12V. Pilot 300 singles+200 pileups; gates are launch screens, not publication cuts. Full-window residual RMS can dilute local errors. Code-derived fixed settings are not claimed as stored historical truth.

Original: 63486 singles, 27014 pileups, 61225 blue-selected singles; unreliable pileup decompositions excluded from parameter plots. Exact reports remain on data disk/Obsidian. This cleanup does not rerun full fitting or change the scientific conclusions.
