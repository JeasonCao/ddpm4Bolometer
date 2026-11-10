#!/bin/bash
# Full OF chain for one full run (999998 / 999999) with ideal triggers:
#   Octopus Trigger (metadata template) -> ideal trigger file -> reconstruct + OF -> CSV -> resolution json
# AP from all events of 239 keV and above (0.15-5.0 V), normalised, so each of those energies has equal weight.
#   run_full_of_chain.sh <run>      log: OFcompare_local/run<run>/chain.log
set -e
RUN=$1
HERE=$(cd "$(dirname "$0")" && pwd)
W=/mnt/d/storage/research/DDPM4bolometer/OFcompare_local/run$RUN
mkdir -p $W
export APMIN=0.15 APMAX=5.0 APNORM=true
for STEP in trigger ideal reco export; do
    echo "[$(date +%H:%M)] $STEP"
    bash "$HERE/run_local_of.sh" $RUN $STEP ideal
done
echo "[$(date +%H:%M)] analyze"
~/mambaforge3/envs/DDPM/bin/python "$HERE/analyze_local_of.py" --run $RUN
echo "[$(date +%H:%M)] OF_CHAIN_DONE"
