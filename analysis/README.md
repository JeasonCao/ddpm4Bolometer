# Result analysis

- bessel/: filtering/resolution comparison.
- templates/: pulse shape and peak linearity.
- samplers/: sampler comparisons and spectra.
- paper/: paper spectra and figure composition.
- train_clean/: fit quality and ODE/energy/time-constant scatter.

Use canonical paths for new work; old scripts/ entries forward here. Where argparse is supported: `python -m analysis.<topic>.<script> --help`. Some old scripts execute on import and require original data; do not bulk-import to inspect them. No scientific rerun is implied by organization checks.
