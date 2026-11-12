#!/usr/bin/env bash
# Usage: bash train_model261108_cloud.sh --check | --train
# Run --check first. CODE_DIR can point to a checkout with the pinned src tree.
set -euo pipefail
shopt -s nullglob

expected_commit=ed7569edbd372af5b95139bbb6c65116058b3e55
code_dir=${CODE_DIR:-/root/autodl-tmp/model261108_smoke_code.hE0G5WVP}
python=${PYTHON:-/root/miniconda3/bin/python}
nas=/root/autodl-fs
local_disk=/root/autodl-tmp
clean_dir=$nas/simu_data/clean
noise_dir=$nas/simu_data/noise
run_name=run03_model261108
out=$local_disk/output/$run_name
backup_parent=$nas/output
backup=$backup_parent/$run_name

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# == 1 ]] || die 'Pass --check or --train.'
case "$1" in --check|--train) mode=$1 ;; *) die 'Pass --check or --train.' ;; esac
[[ -x "$python" ]] || die "Python not found: $python"
[[ -f "$code_dir/src/ddpm/train.py" ]] || die "Training code not found: $code_dir"
if [[ -f "$code_dir/.smoke_source_commit" ]]; then
    [[ $(<"$code_dir/.smoke_source_commit") == "$expected_commit" ]] || die 'Wrong code snapshot.'
else
    git -C "$code_dir" cat-file -e "$expected_commit^{commit}" 2>/dev/null || die 'Pinned commit unavailable.'
    git -C "$code_dir" diff --quiet "$expected_commit" -- src || die 'Source tree differs from the tested commit.'
    [[ -z $(git -C "$code_dir" ls-files --others --exclude-standard -- src) ]] || die 'Untracked source files present.'
fi

nas_device=$(stat -Lc '%d' "$nas")
[[ "$nas_device" != "$(stat -Lc '%d' /)" && "$nas_device" != "$(stat -Lc '%d' "$local_disk")" ]] \
    || die 'Network storage is not mounted separately.'
for path in "$clean_dir" "$noise_dir"; do
    [[ $(stat -Lc '%d' "$path") == "$nas_device" ]] || die "Unexpected data mount: $path"
done
clean_files=("$clean_dir"/clean_*.h5)
noise_files=("$noise_dir"/noise_*.h5)
[[ ${#clean_files[@]} == 10 && ${#noise_files[@]} == 40 ]] || die 'Expected 10 clean and 40 noise shards.'
[[ ! -e "$out" && ! -L "$out" ]] || die "Training directory already exists: $out"
[[ ! -e "$backup" && ! -L "$backup" ]] || die "Backup already exists: $backup"
for path in "$local_disk" "$nas"; do
    free=$(df -PB1 "$path" | awk 'NR == 2 {print $4}')
    [[ "$free" =~ ^[0-9]+$ ]] && (( free >= 3 * 1024 * 1024 * 1024 )) \
        || die "Need at least 3 GiB free on $path for run artifacts."
done

cd "$code_dir"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$python" -c 'import h5py, matplotlib, numpy, torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name())'
printf 'Code: %s (%s)\nData: %s clean shards, %s noise shards\nOutput: %s\nNAS backup: %s\n' \
    "$code_dir" "$expected_commit" "${#clean_files[@]}" "${#noise_files[@]}" "$out" "$backup"

train=("$python" -u -m src.ddpm.train
    --clean_dir "$clean_dir" --noise_dir "$noise_dir" --output_dir "$out"
    --loss l2 --scale_emb --film --amp --preload --epochs 100 --patience 0
    --batch_size 16 --num_workers 4 --lr 2e-4 --val_fraction 0.1 --seed 42
    --T 50 --beta_1 1e-4 --beta_T 0.5 --cond_mode step --save_every 10)
printf 'Training command: '
printf '%q ' "${train[@]}"
printf '\n'
[[ "$mode" == --train ]] || exit 0

mkdir -p -- "${out%/*}" "$backup_parent"
[[ $(stat -Lc '%d' "$backup_parent") == "$nas_device" ]] || die 'Backup parent is outside the data mount.'
mkdir -- "$out"
start=$SECONDS
status="$out/status.tsv"
record() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$((SECONDS-start))" | tee -a "$status"; }
printf 'timestamp\tstatus\telapsed_seconds\n' > "$status"
printf '%s\n' "$expected_commit" > "$out/source_commit.txt"
printf '%q ' "${train[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
record training_started

if "${train[@]}" 2>&1 | awk '{print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush()}' | tee "$out/train.log"; then
    record training_exited_ok
else
    code=$?
    record "training_failed_exit_$code"
    exit "$code"
fi

if "$python" - "$out" <<'PY' | tee -a "$out/train.log"
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
history = json.loads((out / 'history.json').read_text())
for key in ('train_loss', 'val_loss'):
    assert len(history[key]) == 100 and all(map(math.isfinite, history[key])), key
assert (out / 'best_model.pt').stat().st_size > 0
assert (out / 'checkpoint_100.pt').stat().st_size > 0
print(f'Best validation epoch: {min(range(100), key=history["val_loss"].__getitem__) + 1}')
PY
then
    record training_verified
else
    record training_artifacts_invalid
    exit 1
fi

backup_run() {
    [[ $(stat -Lc '%d' "$backup_parent") == "$nas_device" ]] || return 1
    local stage
    stage=$(mktemp -d "$backup_parent/.$run_name.partial.XXXXXXXX") || return
    local files=(best_model.pt config.json history.json loss_curve.png train.log status.tsv
                 source_commit.txt command.txt)
    local checkpoint
    for checkpoint in "$out"/checkpoint_*.pt; do files+=("${checkpoint##*/}"); done
    (
        cd "$out" || exit 1
        sha256sum -- "${files[@]}" > "$stage/SHA256SUMS" || exit 1
        cp -p -- "${files[@]}" "$stage/" || exit 1
    ) || return 1
    ( cd "$stage" && sha256sum --check --quiet SHA256SUMS ) || return 1
    date -Is > "$stage/BACKUP_VERIFIED" || return 1
    mv -T --no-clobber -- "$stage" "$backup" || return 1
    [[ ! -e "$stage" ]] || return 1
}

if backup_run; then
    record backup_verified
    printf 'Verified backup: %s\n' "$backup"
else
    record backup_failed
    die "Training finished. Local artifacts are in $out; the network backup needs attention."
fi
