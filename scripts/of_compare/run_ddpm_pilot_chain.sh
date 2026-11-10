#!/bin/bash
# DDPM inference + tri-exponential fit on both pilot runs (background job).
set -e
cd "$(dirname "$0")/../.."
PY=~/mambaforge3/envs/DDPM/bin/python3
D=/mnt/d/storage/research/DDPM4bolometer
$PY -u scripts/of_compare/run_ddpm_pilot.py --runs 999996 999997
for R in 999996 999997; do
    echo "[$(date +%H:%M)] fitting run $R"
    $PY -u scripts/run_fitting.py --input_dir $D/nonoise_out/pilot_run$R \
        --denoised_field waveforms_denoised_nonoise \
        --output_dir results/fit_results/pilot_run${R}_nonoise --workers 11
done
echo "CHAIN_DONE"
