#!/bin/bash
# AutoDL cloud training + inference script.
# Runs training, then DDIM inference on all eval datasets, then shuts down.
#
# Usage (on AutoDL server):
#   chmod +x scripts/train_cloud.sh
#   nohup bash scripts/train_cloud.sh > /root/autodl-tmp/train_cloud.log 2>&1 &
#
# Directory layout assumed on AutoDL (upload before running):
#   /root/autodl-tmp/data/clean/         — clean pulse shards    (5.4 GB)
#   /root/autodl-tmp/data/noise/         — noise shards          (7.6 GB)
#   /root/autodl-tmp/data/noise_holdout/ — holdout noise shards  (0.7 GB)
#   /root/autodl-tmp/eval_data_holdout/  — eval HDF5 files       (4.6 GB)
#   /root/autodl-tmp/output/run01/       — training output (created by script)
#   /root/ddpm4Bolometer/                — code repo (git clone here)

set -e

# On any error: shutdown immediately to stop billing
on_error() {
    echo ""
    echo "ERROR — shutting down to stop billing at $(date)"
    shutdown -h now
}
trap on_error ERR

CLEAN_DIR="/root/autodl-tmp/data/clean"
NOISE_DIR="/root/autodl-tmp/data/noise"
EVAL_DIR="/root/autodl-tmp/eval_data_holdout"
OUTPUT_DIR="/root/autodl-tmp/output/run01"
CODE_DIR="/root/ddpm4Bolometer"
MODEL_PATH="$OUTPUT_DIR/best_model.pt"

echo "============================================"
echo " DDPM Cloud Training + Inference — $(date)"
echo "============================================"

# ── Sanity checks ──────────────────────────────────────────────────────────────
for dir in "$CLEAN_DIR" "$NOISE_DIR" "$EVAL_DIR"; do
    if [ ! -d "$dir" ] || [ -z "$(ls $dir/*.h5 2>/dev/null || ls $dir/resolution/*.h5 2>/dev/null)" ]; then
        echo "ERROR: data not found at $dir — upload data first."
        exit 1
    fi
done

echo "Clean shards      : $(ls $CLEAN_DIR/*.h5 | wc -l)"
echo "Noise shards      : $(ls $NOISE_DIR/*.h5 | wc -l)"
echo "Eval resolution   : $(ls $EVAL_DIR/resolution/*.h5 | wc -l) files"
echo "Eval eff 7p5mV    : $(ls $EVAL_DIR/efficiency_7p5mV/*.h5 | wc -l) files"
echo "Eval eff 1p2mV    : $(ls $EVAL_DIR/efficiency_1p2mV/*.h5 | wc -l) files"
echo ""

# ── Install dependencies ───────────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip3 install -q numpy scipy h5py matplotlib tqdm
echo "Done."
echo ""

# ── Training ───────────────────────────────────────────────────────────────────
echo "[2/4] Starting training..."
mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"

python -u -m src.ddpm.train \
    --clean_dir  "$CLEAN_DIR"  \
    --noise_dir  "$NOISE_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    --loss    l1               \
    --epochs  100              \
    --batch_size 16            \
    --lr      2e-4             \
    --T       50               \
    --beta_1  1e-4             \
    --beta_T  0.5              \
    --num_workers 4            \
    --save_every  10           \
    --amp

echo ""
echo "Training complete at $(date)."
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo ""

# ── Inference ──────────────────────────────────────────────────────────────────
echo "[3/4] Running DDIM inference on all eval datasets..."

for SUBDIR in resolution efficiency_7p5mV efficiency_1p2mV; do
    echo ""
    echo "--- Inferring: $SUBDIR ---"
    python -u scripts/run_batch_inference.py \
        --model_path "$MODEL_PATH"           \
        --input_dir  "$EVAL_DIR/$SUBDIR"     \
        --sampler ddim                       \
        --seed    42                         \
        --batch_size 64                      \
        --overwrite
done

echo ""
echo "Inference complete at $(date)."
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "[4/4] All done. Output summary:"
echo "  Model    : $MODEL_PATH"
echo "  Checkpts : $(ls $OUTPUT_DIR/checkpoint_*.pt 2>/dev/null | wc -l) files"
echo "  Eval     : denoised waveforms written into eval h5 files"
echo ""
echo "Download with (from your local machine):"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<IP>:/root/autodl-tmp/output/run01/ /local/model_output_cloud/"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<IP>:/root/autodl-tmp/eval_data_holdout/ /local/eval_data_holdout_denoised/"
echo ""

# ── Copy results to file storage (NAS) ────────────────────────────────────────
echo "Copying results to /root/autodl-fs/ (persistent file storage)..."
mkdir -p /root/autodl-fs/output/run01
mkdir -p /root/autodl-fs/eval_data_holdout

cp -r "$OUTPUT_DIR"/. /root/autodl-fs/output/run01/
cp -r "$EVAL_DIR"/. /root/autodl-fs/eval_data_holdout/

echo "  Model + checkpoints → /root/autodl-fs/output/run01/"
echo "  Eval results        → /root/autodl-fs/eval_data_holdout/"
echo "Copy complete at $(date)."
echo ""

# ── Shutdown ───────────────────────────────────────────────────────────────────
echo "Shutting down in 60 seconds... (SSH in and kill the shutdown process to cancel)"
sleep 60
shutdown -h now
