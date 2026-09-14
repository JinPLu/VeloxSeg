# Fixed VeloxSeg reproduction with nnU-Net

This directory contains the fixed 96³ configurations used by the completed
September 2026 reproduction. Automatic parameter configuration is deferred.
The standalone entrypoints at the repository root use a different data and
training pipeline; their checkpoints are not interchangeable with this model.

## Environment and installation

The completed runs used Python 3.10, PyTorch 2.6.0+cu124, MONAI 1.5.0 and
nnUNetv2 2.6.2. Install the repository requirements and a suitable PyTorch wheel
in a dedicated environment, then run from the repository root:

```bash
python -m pip install -r nnunet/requirements.txt
python nnunet/install.py
export nnUNet_compile=false
```

The installer adds the trainer, network and FP32 predictor to that environment's
nnUNet package. It also installs the upstream foreground-sampling correction
used by these runs: nested BraTS regions must each retain sampling coordinates.
The nnUNet-derived files are covered by [Apache-2.0](LICENSE).

## Data and fixed plans

Set `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` to your own directories.
For a fresh preparation, the source must be either the 1,251 original BraTS2021
case directories or the 1,014-case `AutoPETII_spac_norm` directory produced by
the standalone preprocessing route (`imagesTr` and `labelsTr`). The AutoPET
reference ran nnUNet normalization on those already normalized inputs; using
raw PET/CT instead changes the experiment. Their historical generation details
are not fully recovered.

```bash
# Choose the dataset you want to reproduce.
python nnunet/prepare.py 137 /path/to/BraTS2021
nnUNetv2_preprocess -d 137 -c 3d_fullres -plans_name nnVeloxSegPlans -np 2

python nnunet/prepare.py 221 /path/to/AutoPETII_spac_norm
nnUNetv2_preprocess -d 221 -c 3d_fullres -plans_name nnVeloxSegPlans_B -np 2
```

`prepare.py` copies images, converts BraTS labels using nnUNet's converter,
checks the training IDs against the supplied split, and installs the fixed
plans and dataset metadata. It requires a new raw dataset directory. With
existing correctly converted data, copy the matching `nnunet/config/Dataset*/`
JSON files into its preprocessed directory instead. Regenerate preprocessing
if it predates the overlapping-region sampling fix. Do not run an automatic
planner over these fixed plans.

| Setting | BraTS2021 | AutoPET-II |
|---|---|---|
| Dataset / plans | 137 / `nnVeloxSegPlans` | 221 / `nnVeloxSegPlans_B` |
| Patch / batch | 96³ / 4 | 96³ / 8 |
| Train / validation / test | 800 / 200 / 251 | 648 / 162 / 204 |
| Fold | 4, supplied contiguous split | 4, supplied contiguous split |
| Channels | T1, T1ce, T2, FLAIR | PET, CT |
| Normalization | Four masked Z scores | Z score, CT normalization |
| Segmentation | WT / TC / ET region BCE + Dice | Foreground CE + Dice |

BraTS labels map original `1→2, 2→1, 4→3`. The regions are WT `{1,2,3}`,
TC `{2,3}` and ET `{3}`. Training keeps nnUNet's native augmentation, including
rotation, with the fixed patch geometry. The standalone ±15° setting does not
control this trainer.

Both configurations use seed 12345, one GPU, FP32 without autocast, 1,000 epochs,
250 training and 50 validation batches per epoch, AdamW (LR 0.001, weight decay
0.01), and cosine annealing to 0.000006. The objective is the trained nnUNet
deep-supervision segmentation loss (normalized weights, coarsest head zero),
plus `0.5 × reconstruction MSE + 2 × mean teacher Gram MSE`.

## Train, predict and evaluate

```bash
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0

# BraTS; for AutoPET replace 137 / nnVeloxSegPlans with 221 / nnVeloxSegPlans_B.
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 137 3d_fullres 4 \
  -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1

python -m nnunetv2.inference.predict_from_raw_data_noautocast \
  -i "$nnUNet_raw/Dataset137_BraTS2021/imagesTs" \
  -o ./prediction_brats -d 137 -c 3d_fullres -f 4 \
  -tr nnVeloxSegTrainer -p nnVeloxSegPlans -chk checkpoint_final.pth -npp 1 -nps 1

nnUNetv2_evaluate_folder "$nnUNet_raw/Dataset137_BraTS2021/labelsTs" ./prediction_brats \
  -djfile "$nnUNet_preprocessed/Dataset137_BraTS2021/dataset.json" \
  -pfile "$nnUNet_preprocessed/Dataset137_BraTS2021/nnVeloxSegPlans.json" \
  -o ./brats_test_summary.json -np 2
```

Prediction uses FP32, 50% sliding-window step, Gaussian blending and mirroring
TTA. Native evaluation counts voxels over each complete case; a region absent
from both prediction and reference receives NaN and is excluded from that
region's case mean. The mean of the three BraTS region means is reported.
For AutoPET, report the mean Dice among cases with nonzero `n_ref` separately
from all-case Dice and negative-case false positives.

## Completed reproduction

Training finished September 9–13; test inference completed September 14.
These are test results for the fixed settings above, not validation patch Dice.

| Dataset / metric | Result |
|---|---:|
| AutoPET: 102 positive cases, of 204 test cases | 69.8790% |
| AutoPET: all 204 cases, native Dice | 34.9395% |
| BraTS: 251 cases, native whole-volume region mean | 81.0976% |
| BraTS: same predictions, both-empty regions scored as 1 | 81.7284% |

The paper's BraTS 91.44% used a legacy slice-averaged test evaluator with a TC
label mismatch. Applying that exact evaluator to the new predictions gives
91.32%; it must not be compared directly with the corrected 81.10%. The
historical predictions score 80.44% under the native whole-volume evaluator.
The remaining validation/test difference has not been causally explained.
