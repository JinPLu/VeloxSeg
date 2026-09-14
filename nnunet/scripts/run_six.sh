#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results to a new experiment directory}"
export nnUNet_compile=false
# Preserve the GPU assignment supplied by the platform, including UUIDs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
fi
if (( ${#gpus[@]} < 6 )); then
    echo 'Six visible GPUs are required for six concurrent experiments.' >&2
    exit 2
fi
fold=$1
shift
pids=()
trap 'for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done' INT TERM
index=0
for dataset in 137 221; do
    for size in S B L; do
        echo "Starting dataset=$dataset size=$size GPU=${gpus[$index]}"
        CUDA_VISIBLE_DEVICES="${gpus[$index]}" bash "$script_dir/run_experiment.sh" "$dataset" "$size" "$fold" "$@" &
        pids+=("$!")
        index=$((index + 1))
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
