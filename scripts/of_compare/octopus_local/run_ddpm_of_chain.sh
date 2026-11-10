#!/bin/bash
# Octopus OF on the DDPM-output stream of build_ddpm_stream.py (pilot runs: AP from 2615 keV, 2-5 V, as in the raw pilot OF).
#   run_ddpm_of_chain.sh <run>      log: OFcompare_ddpm/run<run>/chain.log
set -e
RUN=$1
HERE=$(cd "$(dirname "$0")" && pwd)
B=/mnt/d/storage/research/DDPM4bolometer/OFcompare_ddpm/run$RUN
export RAW=$B/raw TRUTH=$B/truth W=$B APMIN=2.0 APMAX=5.0 APNORM=false
for STEP in trigger ideal reco export; do
    echo "[$(date +%H:%M)] $STEP"
    bash "$HERE/run_local_of.sh" $RUN $STEP ideal
done
~/mambaforge3/envs/DDPM/bin/python "$HERE/analyze_ddpm_of.py" --run $RUN
echo "[$(date +%H:%M)] DDPM_OF_CHAIN_DONE"
