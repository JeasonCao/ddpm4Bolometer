#!/bin/bash
# Local OF test with the old Octopus container (octoreflow_gio).
#   run_local_of.sh <run> <step> [variant]
#   step: trigger  - Octopus Trigger (bufferLength 1 s) -> trig_octopus/
#         ideal    - trigger file from the true onsets + quiet periods -> trig_ideal/
#         reco     - reconstruct + OF with trig_<variant>/ -> reco_<variant>/   (variant: ideal | octopus)
#         export   - one CSV row per trigger from reco_<variant>/ -> of_<variant>.csv
# AP selection (env, defaults = pilot runs): APMIN/APMAX amplitude window in V, APNORM true|false.
#   Pilot (15 + 2615 keV): 2.0-5.0 V, not normalised -> 2615 keV only.
#   Full runs (17 energies): 0.15-5.0 V, normalised -> 239 keV and above, each energy with equal weight.
set -e
RUN=$1; STEP=$2; VAR=${3:-ideal}
APMIN=${APMIN:-2.0}; APMAX=${APMAX:-5.0}; APNORM=${APNORM:-false}
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=~/SIFs/Octopus-octoreflow_gio.sif
# RAW/TRUTH/W can be overridden (env) e.g. for the DDPM-output stream of build_ddpm_stream.py
RAW=${RAW:-/mnt/d/storage/research/DDPM4bolometer/OFcompare_octopus_staging/run$RUN/raw}
TRUTH=${TRUTH:-/mnt/d/storage/research/DDPM4bolometer/OFcompare_truth/run$RUN/truth}
W=${W:-/mnt/d/storage/research/DDPM4bolometer/OFcompare_local/run$RUN}
PREFIX=$(ls $RAW | head -1 | cut -d_ -f2)
ENV='source /home/wsl_0vbb/SIFs/home4Octopus/wsl_0vbb/LoadOctopus.sh 2>/dev/null; export PATH=/software/Octopus-build/bin:/software/root_v6.32.08/bin:$PATH LD_LIBRARY_PATH=/software/root_v6.32.08/lib:$LD_LIBRARY_PATH PYTHONPATH=/software/root_v6.32.08/lib'
mkdir -p $W/trig_octopus $W/trig_ideal $W/reco_ideal $W/reco_octopus
octo() { singularity exec -B $RAW:/data/raw -B $W:/out -B "$HERE":/scripts $SIF bash -c "$ENV; $1"; }

case $STEP in
trigger)
    cat > $W/trigger.toml <<EOF
[directories]
    rawdir       = "/data/raw/"
    triggerdir   = "/out/trig_octopus/"
    processeddir = "/out/trig_octopus/"
[runConfig]
    filenamePrefix = ["$PREFIX"]
    runNumber      = ["$RUN"]
[channels.heat]
    list         = [1]
    waveformtype = "continuous"
[trigger.heat]
    rmsList      = [5.0]
    nAboveList   = [10]
    bufferLength = [1.0]
[settings]
    rawType            = "Cupid"
    runType            = "trigger"
    draw               = false
    noiseTriggerPeriod = 60
    overwrite          = true
    verbosity          = 3
    applyFilter        = false
EOF
    octo "Trigger /out/trigger.toml" > $W/trigger.log 2>&1 ;;
ideal)
    singularity exec -B $W:/out -B $TRUTH:/truth -B "$HERE":/scripts $SIF bash -c "$ENV; python3 /scripts/make_trigger_file.py \
        --template /out/trig_octopus/Trigger_Run_${RUN}_Channel_1.root --truth /truth --out /out/trig_ideal/Trigger_Run_${RUN}_Channel_1.root" ;;
reco)
    # OF delay scan (units: samples): ideal triggers sit on the onset, Octopus ones up to ~40 samples late
    if [ $VAR = ideal ]; then TSTEP=0.5; TRANGE=10.0; else TSTEP=1.0; TRANGE=20.0; fi
    sed -e "s/@RUN@/$RUN/; s/@PREFIX@/$PREFIX/; s/@TRIGDIR@/trig_$VAR/; s/@RECODIR@/reco_$VAR/; s/@APMIN@/$APMIN/g; s/@APMAX@/$APMAX/g; s/@APNORM@/$APNORM/; s/@TSTEP@/$TSTEP/; s/@TRANGE@/$TRANGE/" \
        "$HERE/reco.toml.in" > $W/reco_$VAR.toml
    cp $W/trig_$VAR/Trigger_Run_${RUN}_Channel_1.root $W/reco_$VAR/ 2>/dev/null || true
    octo "Octopus /out/reco_$VAR.toml" > $W/reco_$VAR.log 2>&1 ;;
export)
    octo "python3 /scripts/export_of_table.py /out/reco_$VAR/Processed_${PREFIX}_${RUN}_1.root /out/of_$VAR.csv" ;;
esac
echo "done $RUN $STEP $VAR"
