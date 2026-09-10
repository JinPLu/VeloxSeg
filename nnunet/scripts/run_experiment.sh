#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# Avoid costly huge-page compaction for large augmentation arrays on busy hosts.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
dataset_id=$1
size=$2
case "$dataset_id" in
    137) dataset=Dataset137_BraTS2021 ;;
    221) dataset=Dataset221_AutoPETII_2023 ;;
    990) dataset=Dataset990_Hecktor_2022 ;;
    *) echo "Expected dataset 137, 221 or 990" >&2; exit 2 ;;
esac
case "$size" in S|B|L) ;; *) exit 2 ;; esac
configuration="3d_fullres_$size"
output="$nnUNet_results/$dataset/nnVeloxSegTrainer__nnVeloxSegPlans__${configuration}/fold_4"
mkdir -p "$output"
exec > >(tee -a "$output/stdout.log") 2>&1
shift 2
nnUNetv2_train "$dataset_id" "$configuration" 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1 "$@"
nnUNetv2_predict -i "$nnUNet_raw/$dataset/imagesTs" -o "$output/prediction" \
    -d "$dataset_id" -c "$configuration" -f 4 -tr nnVeloxSegTrainer \
    -p nnVeloxSegPlans -chk checkpoint_final.pth -npp 2 -nps 2
nnUNetv2_evaluate_folder "$nnUNet_raw/$dataset/labelsTs" "$output/prediction" \
    -djfile "$nnUNet_preprocessed/$dataset/dataset.json" \
    -pfile "$nnUNet_preprocessed/$dataset/nnVeloxSegPlans.json" \
    -o "$output/test_summary.json" -np 2
