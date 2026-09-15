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
nnUNetv2_extract_fingerprint -d 990
python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner candidates \
  --dataset-name Dataset990_Hecktor_2022 \
  --dataset-json "$nnUNet_raw/Dataset990_Hecktor_2022/dataset.json" \
  --fingerprint "$nnUNet_preprocessed/Dataset990_Hecktor_2022/dataset_fingerprint.json" \
  --gpu-memory-target-in-gb 24 \
  --output "$nnUNet_preprocessed/Dataset990_Hecktor_2022/veloxseg_candidates.json"
python nnunet/cost_profile.py \
  --candidates "$nnUNet_preprocessed/Dataset990_Hecktor_2022/veloxseg_candidates.json" \
  --output "$nnUNet_preprocessed/Dataset990_Hecktor_2022/veloxseg_profile.json"
nnUNetv2_plan_and_preprocess -d 990 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
nnUNetv2_train 990 3d_fullres_B 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1
```

Choose `3d_fullres_S`, `3d_fullres_B` or `3d_fullres_L`. They share preprocessing;
only B needs preprocessing. The batch is planned per dataset; training uses AdamW
and 1000 epochs of 250 updates.
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

The trainer uses native nnU-Net epoch logs (training loss, validation loss,
pseudo Dice and epoch time) and logs applied and GradScaler-skipped optimizer
steps per epoch. Concurrent experiments share platform stdout;
read each fold’s `training_log_*.txt` for an individual run. The experiment
script disables stdout buffering, uses four native
augmentation workers and one BLAS/OpenMP thread per process, and disables NumPy
huge-page advice to avoid observed memory-compaction stalls on the training hosts.
For direct CLI training, export these before starting Python:

```bash
export PYTHONUNBUFFERED=1 NUMPY_MADVISE_HUGEPAGE=0
export nnUNet_n_proc_DA=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Worker count is a host-dependent starting point, not a measured optimum. Existing
processes require a restart to inherit these settings; preserve and resume their
checkpoints. See [runtime choices](RULES.md#runtime-throughput) for the scope and
evidence.
