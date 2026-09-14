#!/bin/bash
# AutoDL cloud inference script — 2026-09-14 rewrite for the 1.382mV
# 3-way comparison run (cold / cold-5shot / warm-5shot).
#
# Changes from the 09-09 version:
#   - Targets the new efficiency_1p382mV group, all 17 energy points (no
#     LOW_ENERGIES subset — this is a brand-new noise level, nothing to skip)
#   - Calls run_batch_inference_compare.py (3-way, auto t_start selection)
#     instead of run_batch_inference.py
#   - BATCH_SIZE defaults to 256 (safe on a 24GB 4090/4090D — see the
#     attention-layer OOM math in run_batch_inference_compare.py's
#     docstring; drop back to ~64 if renting a 12GB-class card instead)
#   - Does NOT auto-shutdown on success anymore. The old
#     "sleep 60; shutdown -h now" tail raced against a local download and
#     lost that race at least once (09-09 diary) — instead this script now
#     just finishes and returns control to the SSH session. The local
#     orchestrator (Claude, driving this over SSH) pulls the results down
#     via scp/rsync FIRST, verifies them, and only THEN sends the shutdown
#     command as an explicit separate step — so shutdown is never racing
#     against anything. The on-error auto-shutdown trap is UNCHANGED and
#     still fires immediately on any crash, to stop runaway billing.
#
# Usage (on AutoDL server):
#   chmod +x scripts/infer_cloud.sh
#   nohup bash scripts/infer_cloud.sh > /root/autodl-tmp/infer_cloud.log 2>&1 &
#
# Before this run, upload directly into instance-local storage (see
# upload_and_infer.sh):
#   /root/autodl-tmp/eval_data_holdout/efficiency_1p382mV/eval_efficiency_{1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034}keV.h5
# best_model.pt and code (including the updated src/ddpm/diffusion.py with
# ddim_sample_warmstart, and this script's sibling run_batch_inference_compare.py)
# need to be re-synced to autodl-fs for this rental — they will NOT already
# be there from a previous rental (each rental is a fresh instance).

ENERGIES=(1 2 3 5 10 15 20 25 30 40 50 239 511 727 1461 2615 3034)
NOISE_GROUP="efficiency_1p382mV"
BATCH_SIZE=256   # 24GB card default; see header comment
N_SHOTS=5

set -e

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"

on_error() {
    echo ""
    echo "ERROR — shutting down to stop billing at $(date)"
    shutdown -h now
}
trap on_error ERR

FS_DIR="/root/autodl-fs"                          # persistent file storage
TMP_EVAL="/root/autodl-tmp/eval_data_holdout"     # fast local SSD (working copy)
TMP_MODEL="/root/autodl-tmp/best_model.pt"
CODE_DIR="/root/ddpm4Bolometer"                   # symlinked from file storage

echo "============================================"
echo " DDPM Cloud Inference (3-way) — $(date)"
echo "============================================"

# ── Link code repo from file storage ──────────────────────────────────────────
if [ ! -d "$CODE_DIR" ]; then
    ln -s "$FS_DIR/ddpm4Bolometer" "$CODE_DIR"
    echo "Linked $CODE_DIR → $FS_DIR/ddpm4Bolometer"
fi

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "$FS_DIR/best_model.pt" ]; then
    echo "ERROR: model not found at $FS_DIR/best_model.pt"
    exit 1
fi
if [ ! -f "$CODE_DIR/scripts/run_batch_inference_compare.py" ]; then
    echo "ERROR: run_batch_inference_compare.py not found in synced code -- did the code re-sync pick up the 2026-09-14 changes?"
    exit 1
fi
for E in "${ENERGIES[@]}"; do
    F="$TMP_EVAL/$NOISE_GROUP/eval_efficiency_${E}keV.h5"
    if [ ! -f "$F" ]; then
        echo "ERROR: missing $F (did upload_and_infer.sh finish rsyncing to autodl-tmp?)"
        exit 1
    fi
done
echo "Local SSD $NOISE_GROUP: all ${#ENERGIES[@]} files present"
echo ""

# ── Install dependencies ───────────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."
pip3 install -q numpy scipy h5py matplotlib tqdm
echo "Done."
echo ""

# ── Copy model to fast SSD ──────────────────────────────────────────────────────
echo "[2/3] Copying model to local SSD..."
cp "$FS_DIR/best_model.pt" "$TMP_MODEL"
echo "Done."
echo ""

# ── Inference ──────────────────────────────────────────────────────────────────
echo "[3/3] Running 3-way (cold / cold5shot / warm5shot) inference on $NOISE_GROUP..."
cd "$CODE_DIR"
python3 -u scripts/run_batch_inference_compare.py \
    --model_path "$TMP_MODEL"                       \
    --input_dir  "$TMP_EVAL/$NOISE_GROUP"            \
    --batch_size "$BATCH_SIZE"                       \
    --n_shots    "$N_SHOTS"                          \
    --overwrite

echo ""
echo "Inference complete at $(date)."
echo ""

# ── Copy results back to file storage ─────────────────────────────────────────
echo "Copying denoised results back to persistent file storage..."
mkdir -p "$FS_DIR/eval_data_holdout_denoised/$NOISE_GROUP"
for E in "${ENERGIES[@]}"; do
    cp "$TMP_EVAL/$NOISE_GROUP/eval_efficiency_${E}keV.h5" \
       "$FS_DIR/eval_data_holdout_denoised/$NOISE_GROUP/"
done
# Per-file-per-method timing log (written by run_batch_inference_compare.py
# next to the input files) — doubles as real-GPU throughput data for the
# pending "inference speed evaluation" TODO, copy it back too.
if [ -f "$TMP_EVAL/$NOISE_GROUP/timing_log.csv" ]; then
    cp "$TMP_EVAL/$NOISE_GROUP/timing_log.csv" \
       "$FS_DIR/eval_data_holdout_denoised/$NOISE_GROUP/timing_log.csv"
fi
echo "Done. Results (+ timing_log.csv) saved to $FS_DIR/eval_data_holdout_denoised/$NOISE_GROUP/"
echo ""

echo "=============================================================="
echo " NOT shutting down. Waiting for the local orchestrator to pull"
echo " results and confirm before issuing shutdown explicitly."
echo "=============================================================="
