#!/bin/bash
# 2026-10-27: DDPM zero-init -> simulator Bessel chain -> tri-exp fits -> Gaussian fits. CPU only.
# Filtered data: eval_data_bessel/bessel_sim/{group}; fits: results/fit_results/{level}_zeroinit_bessel.
# The pure-DDPM files and fits are only read, never written.
set -e
cd "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
PY=~/mambaforge3/envs/DDPM/bin/python3
B=/mnt/d/storage/research/DDPM4bolometer/eval_data_bessel/bessel_sim
for L in 1p382mV 7p5mV; do
  echo "[$(date +%H:%M)] filtering $L"
  $PY -u scripts/apply_bessel_to_ddpm.py --group efficiency_$L
done
for L in 1p382mV 7p5mV; do
  echo "[$(date +%H:%M)] fitting $L"
  $PY -u scripts/run_fitting.py --input_dir $B/efficiency_$L \
      --denoised_field waveforms_denoised_zeroinit_bessel_sim \
      --output_dir results/fit_results/${L}_zeroinit_bessel --workers 11 2>&1 | grep -E "Fitting 5000|Fitting done|Convergence"
done
echo "[$(date +%H:%M)] gaussian fits"
$PY -u scripts/run_gauss_fit.py --fit_dir results/fit_results/1p382mV_zeroinit_bessel results/fit_results/7p5mV_zeroinit_bessel 2>&1 | tail -5
echo BESSEL_CHAIN_DONE
