# Experiment configurations

Commands run from the repository root. Prepare matching dataset metadata and
use a separate results directory for each new experiment.

## Six experiments

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

[run_six.sh](scripts/run_six.sh) respects `CUDA_VISIBLE_DEVICES` from the
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

See [detailed rules, loss semantics and validation evidence](RULES.md)
and the [generated plans](config/). Dataset conversion helpers and
trained nnVeloxSeg weights are not yet provided.

## Hecktor2022: five experiments

Hecktor uses dataset ID **990** (`Dataset990_Hecktor_2022`). Its bundled
metadata describes binary tumor segmentation, PET then CT, modality groups
`[1,1]`, 418 training-pool cases and 106 held-out test cases. The supplied fold4
uses 335 training / 83 validation cases. S/B/L use the same native protocol
in the [planning and training rules](RULES.md), with these automatically generated configurations:

| Experiment | Patch in the input tensor's spatial order | Effective training batch | Epochs |
|---|---|---:|---:|
| Original standalone reference | 128×128×64 | 4 patches (2 cases ×2 crops) | 300 |
| Fixed nnUNet reference | 64×128×128 | 4 patches | 1000 ×250 updates |
| nnUNet auto S | 160×256×256 | 8 patches | 1000 ×250 updates |
| nnUNet auto B | 160×256×256 | 8 patches | 1000 ×250 updates |
| nnUNet auto L | 160×256×256 | 8 patches | 1000 ×250 updates |

The auto tiers have five stages, starting at 8/16/24 channels and ending at
128/256/384. Total parameters are 3,525,167 / 9,144,326 / 15,518,173;
estimated training reserved memory is 13.515 / 17.217 / 20.947 GiB
with the FP32 input-embedding correction. Geometry and batch are unchanged.
These estimates do not establish convergence or segmentation accuracy.

To prepare and run the three auto configurations in the dedicated nnVeloxSeg environment:

```bash
nnUNetv2_plan_and_preprocess -d 990 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
cp nnunet/config/Dataset990_Hecktor_2022/splits_final.json "$nnUNet_preprocessed/Dataset990_Hecktor_2022/"
CUDA_VISIBLE_DEVICES=0 bash nnunet/scripts/run_experiment.sh 990 S &
CUDA_VISIBLE_DEVICES=1 bash nnunet/scripts/run_experiment.sh 990 B &
CUDA_VISIBLE_DEVICES=2 bash nnunet/scripts/run_experiment.sh 990 L &
wait
```

The two references use their existing pre-v2 code snapshots and separate Python
environments; the stage-based nnVeloxSeg model must not replace the original standalone
model or fixed nnUNet network. The standalone reference uses the existing
normalized 524-case dataset, sorted 60/20/20 (314/105/105), CT then PET, binary
nonzero labels, LR2.5e-4, 10-epoch warmup and the original cosine-restart schedule.
The fixed nnUNet reference preserves batch Dice, FP32 forward/loss computation,
LR1e-3 and its original auxiliary loss reductions. Auto S/B/L use sample-wise
Dice, mixed-precision training and the objective in the [planning and training rules](RULES.md). All references use
their explicit dropout0.1 configuration.

Thus the five runs compare complete configurations: splits, preprocessing,
axis order, model architecture, loss reductions and update budgets differ.
They are not a single-variable framework ablation. The same case identifiers,
channel order and binary labels must be used when reusing the bundled metadata.

The prepared five-experiment setup has bounded real-data checks for both
references and all three corrected auto tiers. Auto S/B/L passed full-patch,
batch8 training, validation, fresh-process resume and whole-case inference on
RTX3090; reserved peaks were 12.326 / 16.592 / 21.076 GiB. See the
[validation protocol](RULES.md#full-patch-rtx3090-checks). Long-run results are pending.
