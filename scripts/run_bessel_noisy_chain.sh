#!/bin/bash
# 2026-10-27: Bessel only, no DDPM: raw noisy windows -> simulator Bessel chain -> tri-exp fits -> Gaussian fits.
# Filtering starts at once; fitting waits for run_bessel_chain.sh (DDPM + Bessel) to finish so the two
# fitting jobs do not share the CPU. Output: eval_data_bessel/bessel_sim_noisy/{group},
# results/fit_results/{level}_noisy_bessel (its "denoised" columns are the Bessel-filtered raw windows).
set -e
cd "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
PY=~/mambaforge3/envs/DDPM/bin/python3
B=/mnt/d/storage/research/DDPM4bolometer/eval_data_bessel/bessel_sim_noisy
for L in 1p382mV 7p5mV; do
  echo "[$(date +%H:%M)] filtering noisy $L"
  $PY -u scripts/apply_bessel_to_ddpm.py --group efficiency_$L --source noisy
done
echo "[$(date +%H:%M)] waiting for the DDPM + Bessel chain"
until grep -q "BESSEL_CHAIN_DONE\|Traceback\|Error" /mnt/d/storage/research/DDPM4bolometer/logs/bessel_chain.log; do sleep 30; done
for L in 1p382mV 7p5mV; do
  echo "[$(date +%H:%M)] fitting noisy+Bessel $L"
  $PY -u scripts/run_fitting.py --input_dir $B/efficiency_$L \
      --denoised_field waveforms_noisy_bessel_sim \
      --output_dir results/fit_results/${L}_noisy_bessel --workers 11 2>&1 | grep -E "Fitting 5000|Fitting done|Convergence"
done
echo "[$(date +%H:%M)] gaussian fits"
$PY -u scripts/run_gauss_fit.py --fit_dir results/fit_results/1p382mV_noisy_bessel results/fit_results/7p5mV_noisy_bessel 2>&1 | tail -5
echo BESSEL_NOISY_CHAIN_DONE
