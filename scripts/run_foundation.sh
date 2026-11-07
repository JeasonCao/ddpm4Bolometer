#!/usr/bin/env bash
#
# Driver for the multi-channel foundation-model denoiser: runs the whole
# experiment end-to-end (Steps 1-4) on an already-generated channel bank.
#
#   1. Pretrain   — backbone + train-channel tokens on the train channels
#   2. Fine-tune  — freeze backbone, fit each held-out channel's token, per K
#   3. Oracle     — full backbone fine-tune on the largest pool (the ceiling)
#   4. Eval       — zero-shot / fine-tuned / oracle ladder + data-efficiency plot
#
# Step 0 (generating the channel bank) is NOT done here — run the generators
# in src/pulse/generate.py and scripts/generate_noise_shards.py first.
#
# Usage:
#   bash scripts/run_foundation.sh
#   # long job — prefer:  nohup stdbuf -oL bash scripts/run_foundation.sh > run_foundation.out 2>&1 &
#
# Override any setting inline, e.g.:
#   EPOCHS_PRETRAIN=50 K_LIST="50 200 800" bash scripts/run_foundation.sh

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration (override via environment)
# --------------------------------------------------------------------------- #
BANK="${BANK:-/media/Disk_YIN/yunshancheng/cuore/channel_bank}"
RUN="${RUN:-/media/Disk_YIN/yunshancheng/cuore/runs/foundation}"

CLEAN_DIR="${CLEAN_DIR:-$BANK/clean}"
NOISE_DIR="${NOISE_DIR:-$BANK/noise/noise_total}"

N_CHANNELS="${N_CHANNELS:-100}"
TRAIN_CHANNELS="${TRAIN_CHANNELS:-0-79}"     # pretrain set
HELDOUT_CHANNELS="${HELDOUT_CHANNELS:-80-99}" # fine-tune + eval set

K_LIST="${K_LIST:-10 50 100 200 400 800}"    # fine-tune budgets to sweep
ORACLE_K="${ORACLE_K:-800}"                   # pool size for the oracle ceiling
TEST_START="${TEST_START:-800}"               # first test window (after the pool)
MAX_TEST_WINDOWS="${MAX_TEST_WINDOWS:-200}"   # test windows/channel scored in eval

EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-100}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_PRETRAIN="${LR_PRETRAIN:-2e-4}"
LR_TOKEN="${LR_TOKEN:-1e-3}"                   # token fine-tune lr
LR_ORACLE="${LR_ORACLE:-2e-4}"                 # full fine-tune lr
NUM_WORKERS="${NUM_WORKERS:-4}"                # DataLoader workers (overlap disk I/O with GPU)

PY="${PY:-python3 -u}"
BACKBONE="$RUN/backbone.pt"

mkdir -p "$RUN"
echo "=== Foundation-model run ==="
echo "  bank:        $BANK"
echo "  run dir:     $RUN"
echo "  train ch:    $TRAIN_CHANNELS   held-out: $HELDOUT_CHANNELS"
echo "  K sweep:     $K_LIST   oracle K: $ORACLE_K   test_start: $TEST_START"
echo

# --------------------------------------------------------------------------- #
# Step 1 — Pretrain backbone + train-channel tokens
# --------------------------------------------------------------------------- #
echo "=== [1/4] Pretrain backbone ==="
$PY -m src.ddpm.train_channel --phase pretrain \
    --clean_dir "$CLEAN_DIR" --noise_dir "$NOISE_DIR" \
    --output_dir "$RUN" \
    --n_channels "$N_CHANNELS" --train_channels "$TRAIN_CHANNELS" \
    --epochs "$EPOCHS_PRETRAIN" --batch_size "$BATCH_SIZE" --lr "$LR_PRETRAIN" \
    --num_workers "$NUM_WORKERS"

# --------------------------------------------------------------------------- #
# Step 2 — Fine-tune held-out tokens (the method), one checkpoint per K
# --------------------------------------------------------------------------- #
echo "=== [2/4] Fine-tune held-out tokens (sweep K) ==="
FINETUNED_CKPTS=()
for K in $K_LIST; do
    echo "--- token fine-tune, K=$K ---"
    $PY -m src.ddpm.train_channel --phase finetune --finetune_mode token \
        --clean_dir "$CLEAN_DIR" --noise_dir "$NOISE_DIR" \
        --output_dir "$RUN" --resume "$BACKBONE" \
        --finetune_channels "$HELDOUT_CHANNELS" --n_finetune_windows "$K" \
        --epochs "$EPOCHS_FINETUNE" --batch_size "$BATCH_SIZE" --lr "$LR_TOKEN" \
        --num_workers "$NUM_WORKERS"
    FINETUNED_CKPTS+=("$RUN/finetuned_nft${K}.pt")
done

# --------------------------------------------------------------------------- #
# Step 3 — Oracle ceiling (full backbone fine-tune on the largest pool)
# --------------------------------------------------------------------------- #
echo "=== [3/4] Oracle (full fine-tune, K=$ORACLE_K) ==="
$PY -m src.ddpm.train_channel --phase finetune --finetune_mode full \
    --clean_dir "$CLEAN_DIR" --noise_dir "$NOISE_DIR" \
    --output_dir "$RUN" --resume "$BACKBONE" \
    --finetune_channels "$HELDOUT_CHANNELS" --n_finetune_windows "$ORACLE_K" \
    --epochs "$EPOCHS_FINETUNE" --batch_size "$BATCH_SIZE" --lr "$LR_ORACLE" \
    --num_workers "$NUM_WORKERS"
ORACLE_CKPT="$RUN/oracle_nft${ORACLE_K}.pt"

# --------------------------------------------------------------------------- #
# Step 4 — Eval ladder + data-efficiency curve
# --------------------------------------------------------------------------- #
echo "=== [4/4] Eval ladder ==="
$PY -m src.ddpm.eval_channel \
    --clean_dir "$CLEAN_DIR" --noise_dir "$NOISE_DIR" \
    --backbone "$BACKBONE" \
    --finetuned "${FINETUNED_CKPTS[@]}" \
    --oracle "$ORACLE_CKPT" \
    --channels "$HELDOUT_CHANNELS" --test_start "$TEST_START" \
    --max_test_windows "$MAX_TEST_WINDOWS" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$RUN/eval"

echo
echo "=== Done. Results in $RUN/eval (eval_summary.json, data_efficiency.png) ==="
