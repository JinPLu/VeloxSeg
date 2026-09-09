# [ICLR 2026] VeloxSeg: Johnson-Lindenstrauss Lemma Guided Network for Efficient 3D Medical Segmentation

## News / Updates

- **2026-09**: Corrected the open-source loss weighting and learning-rate schedule to align with the paper configurations.
- **2026-01**: VeloxSeg is accepted by **ICLR 2026**!
- **2026-09**: The `v2` branch provides S/B/L auto-planning, native nnUNet training and six-experiment launch scripts. See the design, measured inference and commands below.

## VeloxSeg v2: S/B/L with nnUNet

This branch provides a native `VeloxSegPlanner` and `nnVeloxSegTrainer`, using
nnUNetv2 2.6.2 for preprocessing, augmentation, folds, checkpoints, sliding-window
prediction and evaluation. The public VeloxSeg model and auxiliary objective are
shared with the standalone entrypoints. Historical checkpoints are incompatible
with the new stage-based architecture. The standalone examples below describe
reference configurations, not this automatic family.

### Design and training protocol

S/B/L start at **8/16/24 channels**, capped at 160/320/480. B preserves the
established base16 reference; L uses 1.5× width. All tiers use one JLC/PWA block per
stage, expansion2 and dropout0. Stage count, per-axis strides and attention windows
follow dataset geometry. The first two nnUNet pooling transitions form a compact
stem; neither 96³ nor a fixed number of stages is imposed.

The planner fixes **training batch8**, then shrinks one shared patch until the
largest family member's memory estimate fits **24GiB**. S/B/L use that same
geometry for a controlled width comparison. Batch4 is a separate planning
comparison, not one of these six experiments. This is a calibrated heuristic,
not an accuracy-selected optimum; the training-memory proxy was calibrated at B
and its transfer to S/L remains unverified.

| Dataset | Tier / native configuration | Shared patch | Stage channels | Total / inference-path parameters | Batch1 counted GFLOPs | Estimated training reserved GiB |
|---|---|---|---|---:|---:|---:|
| BraTS2021 | S / `3d_fullres_S` | 160×192×160 | 8/16/32/64 | 1.047M / 0.825M | 14.548 | 8.416 |
| BraTS2021 | B / `3d_fullres_B` | 160×192×160 | 16/32/64/128 | 2.608M / 2.071M | 36.191 | 9.751 |
| BraTS2021 | L / `3d_fullres_L` | 160×192×160 | 24/48/96/192 | 4.246M / 3.371M | 49.002 | 10.470 |
| AutoPET-II | S / `3d_fullres_S` | 192×256×256 | 8/16/32/64/128 | 3.526M / 2.375M | 30.155 | 15.938 |
| AutoPET-II | B / `3d_fullres_B` | 192×256×256 | 16/32/64/128/256 | 9.146M / 6.366M | 62.626 | 19.667 |
| AutoPET-II | L / `3d_fullres_L` | 192×256×256 | 24/48/96/192/384 | 15.520M / 11.161M | 95.136 | 23.425 |

All six use fold4, seed12345, 1000 epochs ×250 training updates, 50 online
validation batches/epoch, foreground oversampling0.33, AdamW (LR1e-3,
weight decay0.01), cosine decay to6e-6 and sample-wise Dice. CUDA training uses
FP16 autocast/GradScaler, with FP32 Gram construction and loss reductions.
The objective combines native-scale segmentation heads (normalized 2^-level
weights), reconstruction MSE averaged over modalities (weight0.5), and Gram
transfer summed over modality teachers and averaged over batch (weight2).
Increasing batch does not automatically scale LR or equalize sample exposure.

### Measured inference and its limits

RTX3090, PyTorch2.6/cu124, FP16, batch1, random weights and synthetic inputs.
Forward/reverse candidate orders each used 5 warmups +10 single-patch calls,
and 1 warmup +3 whole-volume predictions. Ranges span the two medians.
Whole-volume prediction used step0.5 and Gaussian fusion, without TTA or fold
ensemble. It includes predictor transfers/accumulation, but excludes preprocessing
and export. All outputs were finite. Resident weights include inactive training
heads; the inference path omits reconstruction and auxiliary objectives.

| Dataset / tier | Single patch (ms) | Patch allocated / reserved GiB | Whole volume (s) | Whole-volume allocated GiB |
|---|---:|---:|---:|---:|
| BraTS / S | 15.3 | 0.169 / 0.281 | 0.064–0.093 | 0.288 |
| BraTS / B | 17.4–20.9 | 0.184 / 0.309 | 0.064–0.078 | 0.301 |
| BraTS / L | 22.6–22.8 | 0.250 / 0.371 | 0.081–0.097 | 0.365 |
| AutoPET / S | 31.2–39.7 | 0.240 / 0.344 | 0.858–0.942 | 1.271 |
| AutoPET / B | 31.1–32.6 | 0.292 / 0.408 | 0.870–0.921 | 1.324 |
| AutoPET / L | 34.5–45.0 | 0.351 / 0.506 | 1.040–1.188 | 1.384 |

AutoPET's 326×400×400 volume uses27 windows; BraTS's 140×171×136 volume is
padded to one window. **Whole-volume inference also uses batch1**; batch4/8 refer
to training planning. S lowers parameters/memory but has no established AutoPET
speed advantage over B. FLOPs alone do not rank latency.

The earlier base16 comparison showed a useful trade-off: whole-volume inference
fell from9.766s for the explicit historical96³ reference to0.480s (batch4-planned
224×320×320) or0.704s (batch8-planned224×256×256), while equal-batch8 training
updates rose from0.171s to0.951s. Those are different geometries from the shared
S/B/L table. Larger context reduced tile count, but did not mean faster updates,
faster patches, fewer parameters or proven accuracy gains. Full batch8 training
fit for the new S/L configurations, convergence and real-case accuracy remain
unverified. Do not silently lower only L's batch if it runs out of memory:
regenerate shared geometry for the family and report the changed protocol.

### Six experiments

Use a dedicated Python3.11 environment on a Linux task with **six visible GPUs**
(or eight allocated GPUs, of which the script uses the first six). Install:

```bash
python -m pip install -r nnunet/requirements.txt
python nnunet/install.py
export nnUNet_raw=/your/data/nnUNet_raw
export nnUNet_preprocessed=/your/experiment/nnUNet_preprocessed
export nnUNet_results=/your/experiment/results
export nnUNet_compile=false
```

Prepare the raw datasets in nnUNet format with the bundled dataset metadata and
folds under `nnunet/config/`. IDs137/221 denote BraTS2021/AutoPET-II here. The
bundled fold4 uses800/200 BraTS and648/162 AutoPET training/validation cases;
held-out test sets have251/204 cases. Channel grouping is `[4]` for BraTS and
`[1,1]` for PET/CT. Follow these names/order/labels when reusing the bundled
metadata. If your cases differ, generate your own fingerprint and splits.

```bash
# Generate all tiers; preprocess once per dataset because S/B/L share the cache.
nnUNetv2_plan_and_preprocess -d 137 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
nnUNetv2_plan_and_preprocess -d 221 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
# Use the supplied folds for the matching released case identifiers.
cp nnunet/config/Dataset137_BraTS2021/splits_final.json "$nnUNet_preprocessed/Dataset137_BraTS2021/"
cp nnunet/config/Dataset221_AutoPETII_2023/splits_final.json "$nnUNet_preprocessed/Dataset221_AutoPETII_2023/"
# Run six concurrent single-GPU experiments.
bash nnunet/scripts/run_six.sh
```

[run_six.sh](nnunet/scripts/run_six.sh) respects `CUDA_VISIBLE_DEVICES` from the
platform, assigning the first three GPUs to BraTS S/B/L and the next three to
AutoPET S/B/L. Each uses the standard native training lifecycle, including final
full-case validation, then predicts `imagesTs` with `checkpoint_final.pth` and
runs folder evaluation against `labelsTs`. Standard prediction uses Gaussian
fusion, step0.5 and training mirror axes (TTA), so its runtime differs from the
no-TTA benchmark. No model ensemble or unconditional largest-lesion removal is
added. This launcher is a foreground platform command; it does not submit a task.

Outputs are separated by dataset/configuration under `nnUNet_results`:
`<dataset>/nnVeloxSegTrainer__nnVeloxSegPlans__3d_fullres_<S|B|L>/fold_4/`.
Each contains `stdout.log`, native checkpoints/logs, `prediction/` and
`test_summary.json`. Use a new results directory for a new experiment. To resume
all six interrupted runs with checkpoints present, append `--c`; to resume one:

```bash
CUDA_VISIBLE_DEVICES=0 bash nnunet/scripts/run_experiment.sh 221 L --c
```

See [detailed rules, loss semantics and validation evidence](nnunet/README.md)
and the [generated plans](nnunet/config/). Dataset conversion helpers and
trained v2 weights are not yet provided.

## Overview

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="fig/Overview.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Overview of VeloxSeg. VeloxSeg employs an encoder-decoder architecture with Paired Window Attention (PWA) and Johnson-Lindenstrauss lemma-guided convolution (JLC) on the left, using 1x1 convolution as modal mixer. GC: group convolution; GA: grouped attention.</div>
</center>
VeloxSeg is a lightweight multimodal medical image segmentation framework that addresses the fundamental "efficiency / robustness conflict" in 3D medical image segmentation.

## Architecture

The framework consists of three main components:

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="fig/Method.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">(a) Overview of Paired Window Attention (PWA). (b) Intuitive difference between depth-wise (DW) convolution and Johnson-Lindenstrauss guided Convolution (JLC) in the feature space.</div>
</center>

1. **Encoder** (`Encoder.py`): Dual-branch architecture
   - Modal-Fusion Convolution Layer with JLC blocks
    <center>
        <img style="border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src="fig/PWA.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">Detailed architecture of Paired Window Attention (PWA). This figure focuses on visually showing the feature flows of PWA.</div>
    </center>
   - Modal-Cooperative Transformer Layer with PWA blocks

2. **Decoder** (`Decoder.py`): Dual-decoder architecture
   - Segmentation Decoder (Student): Primary segmentation task
   - Reconstruction Decoder (Teacher): Self-supervised texture teacher

3. **Main Model** (`VeloxSeg.py`): Integrates encoder and decoder with SDKT

## File Structure

```
VeloxSeg/
├── model/
│   ├── components/          # Core components (attention, convolution blocks, etc.)
│   ├── Encoder.py           # Dual-stream encoder implementation
│   ├── Decoder.py           # Dual-decoder with SDKT
│   └── VeloxSeg.py          # Main model class
├── config/                  # Configuration files for different datasets
├── utils/                   # Training and inference utilities
├── preprocess/              # Data preprocessing scripts
├── compared_model/          # Baseline model implementations
├── run_train.py             # Training script
├── run_test.py              # Testing script
├── train.sh                 # Training commands
├── test.sh                  # Testing commands
├── fig/                     # Method and overview figures
└── requirements.txt         # Python dependencies
```

## Installation

### Environment Requirements

- Ubuntu 22.04.4 LTS
- Python 3.10.16
- CUDA-capable runtime. The original environment used CUDA 12.2; install the PyTorch wheel that matches your driver/runtime.
- NVIDIA GeForce RTX 3090 (or compatible GPU)

### Setup

```bash
# Create conda environment
conda create -n VeloxSeg python==3.10
conda activate VeloxSeg

# Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

`requirement.txt` is kept as a legacy alias for `requirements.txt`.

## Datasets

The public training and inference entrypoints currently support:

- **AutoPET-II**: Automated Lesion Segmentation in PET/CT Challenge
- **Hecktor2022**: MICCAI Hecktor 2022 Challenge (Head & Neck)
- **BraTS2021**: RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge 2021

`config/train_config_bs4.json` also contains MSD2019 path placeholders, but MSD2019 is not wired into `run_train.py` or `run_test.py` yet.

## Data Preprocessing

Run the preprocessing scripts before training:

```bash
# Registration
python ./preprocess/registration.py

# Intensity normalization
python ./preprocess/normalization_CT_PET.py  # For PET/CT datasets
python ./preprocess/normalization_MRI.py     # For MRI datasets
```

## Training

### Quick Start

> On this `v2` branch, use the [nnUNet workflow and general planning rules](nnunet/README.md) to generate configurations from a dataset fingerprint. The standalone JSON examples below remain explicit reference configurations; they do not define the automatic planner.

```bash
# Train on AutoPET-II dataset
sh train.sh

# Or choose another supported dataset
DATASET_NAME=BraTS2021 GPU_ID=0 sh train.sh
```

### Custom Training

`config/train_config_bs4.json` is the historical default config filename. The effective batch size is read from the JSON file.

```bash
python run_train.py \
    --dataset_name AutoPETII \
    --model_name VeloxSeg \
    --train_config ./config/train_config_bs4.json \
    --model_config ./config/models_config_autopetii.json \
    --num_workers 4 \
    --gpu_id 0
```

### Supported Datasets

- **AutoPET-II**: `--dataset_name AutoPETII`
- **Hecktor2022**: `--dataset_name Hecktor2022`
- **BraTS2021**: `--dataset_name BraTS2021`

## Inference and Evaluation

```bash
# Run inference and evaluation
sh test.sh

# Override checkpoint date or dataset when needed
TRAIN_DATE=09_12 DATASET_NAME=Hecktor2022 sh test.sh
```

### Custom Inference

```bash
python run_test.py \
    --dataset_name AutoPETII \
    --model_name VeloxSeg \
    --train_config ./config/train_config_bs4.json \
    --model_config ./config/models_config_autopetii.json \
    --test_config ./config/test_config.json \
    --num_workers 4 \
    --gpu_id 0 \
    --train_date 09_12 \
    --use_hd95 1
```

## Model Configuration

The model configuration files contain hyperparameters for different datasets:

- `models_config_autopetii.json`: AutoPET-II configuration
- `models_config_hecktor2022.json`: Hecktor2022 configuration  
- `models_config_brats2021.json`: BraTS2021 configuration

Key VeloxSeg parameters:

- `input_size`: Input spatial dimensions (e.g., $[96, 96, 96]$)
- `in_ch`: Input channels per modality (e.g., $[1, 1]$ for $\langle PET,CT\rangle$, $[2]$ for $PET+CT$)
- `stages`: One ordered specification of per-axis stride, channels, JLC kernels/group width, PWA windows/heads and block depths; both encoders and decoders consume it.
- `dropout`: Shared dropout probability.

The v2 stage-based architecture changes checkpoint keys and the deep-supervision reduction. Historical checkpoints are not interchangeable with generated v2 plans.

## Performance

The following are published reference-model results, not measurements of the new automatically generated v2 configurations.

### Computational Efficiency

- **Parameters**: 1.66M (vs 88.62M for nnUNet)
- **FLOPs**: 1.79G (vs 3078.83G for nnUNet)
- **GPU Throughput**: 599.06 patches/s
- **CPU Throughput**: 6.67 patches/s

### Segmentation Performance

- **AutoPET-II**: 62.51% Dice (vs 48.35% for SuperLightNet)
- **Hecktor2022**: 56.48% Dice (vs 50.03% for SuperLightNet)
- **BraTS2021**: 91.44% Dice (vs 89.72% for SuperLightNet)
