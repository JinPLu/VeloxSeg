# nnVeloxSeg: nnU-Net v2 integration

`VeloxSegPlanner` generates S/B/L configurations, and `nnVeloxSegTrainer` runs
the public model through the native nnU-Net training and inference pipeline.

## Quick start

Follow the [root README](../README.md) to install and set the three nnU-Net paths.
Prepare standard nnU-Net data with explicit modality groups in `dataset.json`:

```json
"veloxseg_modality_channels": [1, 1]
```

Use `[1,1]` for PET/CT and `[4]` for the bundled four-channel MRI configuration.
Run from the repository root:

```bash
nnUNetv2_plan_and_preprocess -d 990 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
nnUNetv2_train 990 3d_fullres_B 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1
```

Choose `3d_fullres_S`, `3d_fullres_B` or `3d_fullres_L`. They share preprocessing;
only B needs preprocessing. The default is batch8, AdamW and 1000 epochs.
Append `--c` to resume a matching checkpoint.

## Experiments and implementation

- [Dataset configurations and multi-GPU experiments](EXPERIMENTS.md)
- [Planning rules, loss definitions and validation evidence](RULES.md)
- [Bundled metadata, fingerprints and folds](config/)
- [Planner](nnunetv2/experiment_planning/experiment_planners/veloxseg_planner.py)
- [Trainer](nnunetv2/training/nnUNetTrainer/nnVeloxSegTrainer.py)

The installer links `model` and `utils` to this checkout and installs the nnU-Net
extensions. Keep the checkout in place and rerun `python nnunet/install.py` after
changing extension files. Use a dedicated environment for the two historical
Hecktor reference implementations; their setup is described in the experiment guide.
