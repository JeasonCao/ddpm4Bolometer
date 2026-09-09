#!/bin/bash
# AutoDL cloud inference-only script.
# Runs DDIM inference on the fast local SSD, writes results back to
# persistent file storage, then shuts down.
#
# 2026-08-13: scoped to ONLY the 10 low-energy points (1-40 keV), which were
# just regenerated locally at N=5000 to match the 7 high-energy points
# (already N=5000, already inferred on 2026-06-23 — do not re-infer them,
# it wastes billed GPU time for a deterministic no-op result).
# The 7 high-energy files already sitting in $FS_DIR/ are left untouched.
#
# Raw eval data this run is uploaded DIRECTLY to $TMP_EVAL (instance-local
# /root/autodl-tmp, wiped on shutdown) rather than through persistent file
# storage — it already lives permanently on the local WSL machine, so there
# is no need to also persist a working copy on autodl-fs (keeps file-storage
# usage from creeping toward the 20GB free-tier limit).
#
# Usage (on AutoDL server):
#   chmod +x scripts/infer_cloud.sh
#   nohup bash scripts/infer_cloud.sh > /root/autodl-tmp/infer_cloud.log 2>&1 &
#
# Before this run, upload directly into instance-local storage (see
# upload_and_infer.sh — it now rsyncs straight to $TMP_EVAL, not autodl-fs):
#   /root/autodl-tmp/eval_data_holdout/efficiency_1p2mV/eval_efficiency_{1,2,3,5,10,15,20,25,30,40}keV.h5
#   /root/autodl-tmp/eval_data_holdout/efficiency_7p5mV/eval_efficiency_{1,2,3,5,10,15,20,25,30,40}keV.h5
# best_model.pt and code are already on autodl-fs from the previous run.

LOW_ENERGIES=(1 2 3 5 10 15 20 25 30 40)

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
echo " DDPM Cloud Inference — $(date)"
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
for SUBDIR in efficiency_1p2mV efficiency_7p5mV; do
    for E in "${LOW_ENERGIES[@]}"; do
        F="$TMP_EVAL/$SUBDIR/eval_efficiency_${E}keV.h5"
        if [ ! -f "$F" ]; then
            echo "ERROR: missing $F (did upload_and_infer.sh finish rsyncing to autodl-tmp?)"
            exit 1
        fi
    done
    echo "Local SSD $SUBDIR: all ${#LOW_ENERGIES[@]} low-energy files present"
done
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
echo "[3/3] Running DDIM inference on low-energy eval datasets..."
cd "$CODE_DIR"

for SUBDIR in efficiency_1p2mV efficiency_7p5mV; do
    for E in "${LOW_ENERGIES[@]}"; do
        echo ""
        echo "--- Inferring: $SUBDIR / ${E} keV ---"
        python3 -u scripts/run_batch_inference.py \
            --model_path "$TMP_MODEL"                                       \
            --input      "$TMP_EVAL/$SUBDIR/eval_efficiency_${E}keV.h5"     \
            --sampler ddim                                                  \
            --seed    42                                                    \
            --batch_size 64                                                 \
            --overwrite
    done
done

echo ""
echo "Inference complete at $(date)."
echo ""

# ── Copy results back to file storage ─────────────────────────────────────────
echo "Copying denoised results back to persistent file storage..."
mkdir -p "$FS_DIR/eval_data_holdout_denoised/efficiency_1p2mV" \
         "$FS_DIR/eval_data_holdout_denoised/efficiency_7p5mV"
for SUBDIR in efficiency_1p2mV efficiency_7p5mV; do
    for E in "${LOW_ENERGIES[@]}"; do
        cp "$TMP_EVAL/$SUBDIR/eval_efficiency_${E}keV.h5" \
           "$FS_DIR/eval_data_holdout_denoised/$SUBDIR/"
    done
done
echo "Done. Results saved to $FS_DIR/eval_data_holdout_denoised/ (low-energy files only;"
echo "the 7 high-energy files there from the previous run are untouched)."
echo ""

echo "Download with (from your local machine):"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<IP>:/root/autodl-fs/eval_data_holdout_denoised/ /home/wsl_0vbb/DDPM4bolometer/eval_data_holdout_denoised/"
echo ""

# ── Shutdown ───────────────────────────────────────────────────────────────────
echo "Shutting down in 60 seconds... (SSH in and kill the shutdown process to cancel)"
sleep 60
shutdown -h now
