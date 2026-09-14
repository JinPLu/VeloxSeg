# nnUNet

Fixed 96³ configurations for AutoPET-II and BraTS2021. Automatic configuration is deferred.

Install from the repository root after installing the main requirements:

```bash
python -m pip install -r nnunet/requirements.txt
python nnunet/install.py
export nnUNet_compile=false
```

Set `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` to your data/output directories.
The installer modifies the current environment's nnUNet package; use a dedicated environment.

## Data preparation

Inputs are the original 1,251 BraTS2021 cases or the 1,014-case `AutoPETII_spac_norm` directory. Preparation requires a new raw dataset directory and uses the supplied fixed plans and splits.

```bash
python nnunet/prepare.py 137 /path/to/BraTS2021
nnUNetv2_preprocess -d 137 -c 3d_fullres -plans_name nnVeloxSegPlans -np 2

python nnunet/prepare.py 221 /path/to/AutoPETII_spac_norm
nnUNetv2_preprocess -d 221 -c 3d_fullres -plans_name nnVeloxSegPlans_B -np 2
```

## Training

```bash
export nnUNet_n_proc_DA=2
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 137 3d_fullres 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 221 3d_fullres 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans_B -num_gpus 1
```

## Inference and evaluation

AutoPET example; for BraTS use dataset `137`, folder `Dataset137_BraTS2021`, and plans `nnVeloxSegPlans`.

```bash
python -m nnunetv2.inference.predict_from_raw_data_noautocast \
  -i "$nnUNet_raw/Dataset221_AutoPETII_2023/imagesTs" -o ./prediction_autopet \
  -d 221 -c 3d_fullres -f 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans_B \
  -chk checkpoint_final.pth -npp 1 -nps 1

nnUNetv2_evaluate_folder "$nnUNet_raw/Dataset221_AutoPETII_2023/labelsTs" ./prediction_autopet \
  -djfile "$nnUNet_preprocessed/Dataset221_AutoPETII_2023/dataset.json" \
  -pfile "$nnUNet_preprocessed/Dataset221_AutoPETII_2023/nnVeloxSegPlans_B.json" \
  -o ./autopet_test_summary.json -np 2
```

Inference uses FP32 with mirroring TTA. Evaluation uses whole-volume Dice; both-empty regions receive NaN. For AutoPET, report positive-case Dice separately from all-case Dice.

The nnUNet-derived files are covered by [Apache-2.0](LICENSE).
