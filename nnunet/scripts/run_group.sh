#!/usr/bin/env bash
# Run a list of nnVeloxSeg experiments on the node of one cluster job, one model at a time.
# Task file: one "dataset_id configuration fold plans_identifier [nnUNetv2_train options]"
# per line, e.g. append --c to resume; blank lines and lines starting with # are ignored.
# CPU augmentation sets the training speed, so every task gets the whole node: DDP on
# min(visible GPUs, batch size) GPUs with the augmentation workers per rank that
# group_resources.py allows by CPU quota and measured memory. Tasks run in file order;
# a failed task is reported and the next one starts.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tasks_file=$1
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results to the experiment results directory}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

tasks=()
while IFS= read -r line; do tasks+=("$line"); done < <(grep -vE '^[[:space:]]*(#|$)' "$tasks_file")
if (( ${#tasks[@]} == 0 )); then
    echo "No task in $tasks_file" >&2
    exit 2
fi
# Reject malformed lines, missing plans and tasks without memory for one worker
# before any GPU work starts.
for line in "${tasks[@]}"; do
    read -ra fields <<< "$line"
    if (( ${#fields[@]} < 4 )) || ! python "$script_dir/group_resources.py" \
            "${fields[0]}" "${fields[1]}" "${fields[3]}" > /dev/null; then
        echo "Invalid task, missing plans or no memory for one worker: $line" >&2
        exit 2
    fi
done
log_dir="$nnUNet_results/group_logs"
mkdir -p "$log_dir"
echo "tasks=${#tasks[@]}"

# Each task runs in its own session so an interrupt reaches nnUNetv2_train, its DDP
# ranks and their augmentation workers, not only the wrapper shell.
pid=
stop() {
    if [[ -n "$pid" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    exit 130
}
trap stop INT TERM
status=0
for line in "${tasks[@]}"; do
    read -ra fields <<< "$line"
    name="${fields[0]}_${fields[3]}_${fields[1]}_fold${fields[2]}"
    if ! resources=$(python "$script_dir/group_resources.py" "${fields[0]}" "${fields[1]}" "${fields[3]}"); then
        echo "$(date '+%F %T') failed $name: no resources"
        status=1
        continue
    fi
    read -r ranks cpu_workers memory_workers workers rank_gib devices <<< "$resources"
    echo "$(date '+%F %T') start $name ranks=$ranks workers_by_cpu=$cpu_workers" \
        "workers_by_memory=$memory_workers nnUNet_n_proc_DA=$workers memory_per_rank=${rank_gib}GiB on $devices"
    # With no MASTER_PORT set, nnU-Net's run_training picks a free port for DDP.
    nnUNet_n_proc_DA=$workers CUDA_VISIBLE_DEVICES=$devices setsid bash "$script_dir/run_experiment.sh" \
        "${fields[@]}" -num_gpus "$ranks" >> "$log_dir/$name.log" 2>&1 &
    pid=$!
    if wait "$pid"; then result=ok; else result=failed; status=1; fi
    pid=
    echo "$(date '+%F %T') $result $name"
done
exit "$status"
