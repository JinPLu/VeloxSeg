#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
# Avoid costly huge-page compaction for large augmentation arrays on busy hosts.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
dataset_id=$1
configuration=$2
fold=$3
plans=$4
dataset=$(python -c '
import sys
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
print(maybe_convert_to_dataset_name(sys.argv[1]))
' "$dataset_id")
case "$fold" in 0|1|2|3|4) ;; *) echo "Expected fold 0, 1, 2, 3 or 4" >&2; exit 2 ;; esac
output="$nnUNet_results/$dataset/nnVeloxSegTrainer__${plans}__${configuration}/fold_${fold}"
mkdir -p "$output"
exec > >(tee -a "$output/stdout.log") 2>&1
shift 4
nnUNetv2_train "$dataset_id" "$configuration" "$fold" -tr nnVeloxSegTrainer -p "$plans" "$@"
if [[ ! -d "$nnUNet_raw/$dataset/imagesTs" ]]; then
    exit 0
fi
nnUNetv2_predict -i "$nnUNet_raw/$dataset/imagesTs" -o "$output/prediction" \
    -d "$dataset_id" -c "$configuration" -f "$fold" -tr nnVeloxSegTrainer \
    -p "$plans" -chk checkpoint_final.pth -npp 2 -nps 2
if [[ -d "$nnUNet_raw/$dataset/labelsTs" ]]; then
    nnUNetv2_evaluate_folder "$nnUNet_raw/$dataset/labelsTs" "$output/prediction" \
        -djfile "$nnUNet_preprocessed/$dataset/dataset.json" \
        -pfile "$nnUNet_preprocessed/$dataset/$plans.json" \
        -o "$output/test_summary.json" -np 2
fi
