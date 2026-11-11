# DDPM inspection and explanation

Network inspection, single-step comparisons, reverse trajectories, teaching plots and short demonstration training. Legacy scripts/ entry files forward here.

Not production training/inference defaults. Some scripts execute immediately on invocation/import, use torch/CUDA and original data/weight paths. Read headers first; do not blindly run --help or import them for smoke tests.

Inputs: production checkpoints and eval files on the data disk. Outputs: existing results/plots/ddpm_explainer and slide/mechanism directories. Production weights are intended read-only. Demo training is not production-performance evidence; interpret no_noise/zero-init labels with the diary.
