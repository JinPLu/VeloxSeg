#!/usr/bin/env bash
# Run a list of nnVeloxSeg experiments on the GPUs visible to one cluster job.
# Task file: one "dataset_id configuration fold plans_identifier" per line;
# blank lines and lines starting with # are ignored. Each task gets one GPU;
# tasks beyond the GPU count start as soon as an earlier task frees its GPU.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tasks_file=$1
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results to the experiment results directory}"
export nnUNet_compile=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# UUID selectors keep the platform's assignment when each child sees one GPU.
mapfile -t gpus < <(python - <<'PY'
import torch
for index in range(torch.cuda.device_count()):
    print(f'GPU-{torch.cuda.get_device_properties(index).uuid}')
PY
)
if (( ${#gpus[@]} == 0 )); then
    echo 'No visible GPU for this job.' >&2
    exit 2
fi
mapfile -t tasks < <(grep -vE '^[[:space:]]*(#|$)' "$tasks_file")
# Share the job's CPUs between concurrent augmentation pools.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-$(( $(nproc) / ${#gpus[@]} - 1 ))}"
log_dir="$nnUNet_results/group_logs"
mkdir -p "$log_dir"
echo "GPUs=${#gpus[@]} tasks=${#tasks[@]} nnUNet_n_proc_DA=$nnUNet_n_proc_DA"

declare -A running=()
trap 'for pid in "${running[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done' INT TERM
next=0
status=0
while (( next < ${#tasks[@]} || ${#running[@]} > 0 )); do
    for slot in "${!gpus[@]}"; do
        if [[ -z "${running[$slot]:-}" ]] && (( next < ${#tasks[@]} )); then
            read -r dataset configuration fold plans <<< "${tasks[$next]}"
            name="${dataset}_${plans}_${configuration}_fold${fold}"
            CUDA_VISIBLE_DEVICES="${gpus[$slot]}" bash "$script_dir/run_experiment.sh" \
                "$dataset" "$configuration" "$fold" "$plans" > "$log_dir/$name.log" 2>&1 &
            running[$slot]=$!
            echo "$(date '+%F %T') start $name on ${gpus[$slot]} pid=${running[$slot]}"
            next=$((next + 1))
        fi
    done
    sleep 30
    for slot in "${!running[@]}"; do
        pid=${running[$slot]}
        kill -0 "$pid" 2>/dev/null && continue
        if wait "$pid"; then result=ok; else result=failed; status=1; fi
        echo "$(date '+%F %T') $result pid=$pid on ${gpus[$slot]}"
        unset "running[$slot]"
    done
done
exit "$status"
