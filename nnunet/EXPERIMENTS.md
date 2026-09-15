# Experiment configurations

Commands run from the repository root. Prepare matching dataset metadata and
use a separate results directory for each new experiment.

## Six experiments

Use a dedicated Python3.11 environment on a Linux task with **one or more visible
GPUs**; the launcher runs one experiment per GPU. Install:

```bash
python -m pip install -r nnunet/requirements.txt
python nnunet/install.py
export nnUNet_raw=/your/data/nnUNet_raw
export nnUNet_preprocessed=/your/experiment/nnUNet_preprocessed
export nnUNet_results=/your/experiment/results
```

Prepare the raw datasets in nnUNet format with the bundled dataset metadata and
folds under `nnunet/config/`. IDs137/221 denote BraTS2021/AutoPET-II here. BraTS
uses all 1251 cases with nnU-Net's shuffled five-fold split (seed 12345) and has no
held-out test set. The bundled AutoPET fold4 uses 648/162 training/validation cases
and 204 held-out test cases. Channel grouping is `[4]` for BraTS and
`[1,1]` for PET/CT. Follow these names/order/labels when reusing the bundled
metadata. If your cases differ, generate your own fingerprint and splits.

```bash
# Generate all tiers; preprocess once per dataset because S/B/L share the cache.
# Candidate crops are measured on the target CUDA GPU before planning.
for name in Dataset137_BraTS2021 Dataset221_AutoPETII_2023; do
  id=${name:7:3}
  nnUNetv2_extract_fingerprint -d "$id"
  python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner candidates \
    --dataset-name "$name" --dataset-json "$nnUNet_raw/$name/dataset.json" \
    --fingerprint "$nnUNet_preprocessed/$name/dataset_fingerprint.json" \
    --gpu-memory-target-in-gb 24 --output "$nnUNet_preprocessed/$name/veloxseg_candidates.json"
  python nnunet/cost_profile.py --candidates "$nnUNet_preprocessed/$name/veloxseg_candidates.json" \
    --output "$nnUNet_preprocessed/$name/veloxseg_profile.json"
  nnUNetv2_plan_and_preprocess -d "$id" -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
done
# Use the supplied AutoPET folds for the matching released case identifiers.
cp nnunet/config/Dataset221_AutoPETII_2023/splits_final.json "$nnUNet_preprocessed/Dataset221_AutoPETII_2023/"
# Run six single-GPU experiments, one per visible GPU; extra tasks wait for a free GPU.
for id in 137 221; do for size in S B L; do echo "$id 3d_fullres_$size 4 nnVeloxSegPlans"; done; done > tasks.txt
bash nnunet/scripts/run_group.sh tasks.txt
```

[run_group.sh](scripts/run_group.sh) reads one `dataset_id configuration fold
plans_identifier` task per line, runs one task on each GPU visible to the job and
starts the remaining tasks as GPUs free up. Each uses the standard native training
lifecycle, including final full-case validation. AutoPET then predicts `imagesTs`
with `checkpoint_final.pth` and runs folder evaluation against `labelsTs`; BraTS
has no held-out test set and stops after training. Standard prediction uses Gaussian
fusion, step0.5 and training mirror axes (TTA), so its runtime differs from the
no-TTA benchmark. No model ensemble or unconditional largest-lesion removal is
added. This launcher is a foreground platform command; it does not submit a task.

Outputs are separated by dataset/configuration under `nnUNet_results`:
`<dataset>/nnVeloxSegTrainer__nnVeloxSegPlans__3d_fullres_<S|B|L>/fold_4/`.
Each contains `stdout.log` and native checkpoints/logs; AutoPET adds `prediction/`
and `test_summary.json`. Use a new results directory for a new experiment. To resume
an interrupted run with its checkpoint present, append `--c` to its single-run command:

```bash
CUDA_VISIBLE_DEVICES=0 bash nnunet/scripts/run_experiment.sh 221 3d_fullres_L 4 nnVeloxSegPlans --c
```

See [detailed rules, loss semantics and validation evidence](RULES.md)
and the [generated plans](config/). [prepare.py](prepare.py) copies BraTS2021 and
AutoPET-II into nnU-Net format; trained nnVeloxSeg weights are not yet provided.

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
| nnUNet auto S | 160×256×256 | 4 patches | 1000 ×250 updates |
| nnUNet auto B | 160×256×256 | 4 patches | 1000 ×250 updates |
| nnUNet auto L | 160×256×256 | 4 patches | 1000 ×250 updates |

The auto tiers have five stages, starting at 16 channels and ending at 128; the
first-stage base PWA window is 5×4×4. S/B/L differ in convolution branches and
depth, not width. These plans do not establish convergence or segmentation accuracy.

To prepare and run the three auto configurations in the dedicated nnVeloxSeg environment:

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
cp nnunet/config/Dataset990_Hecktor_2022/splits_final.json "$nnUNet_preprocessed/Dataset990_Hecktor_2022/"
CUDA_VISIBLE_DEVICES=0 bash nnunet/scripts/run_experiment.sh 990 3d_fullres_S 4 nnVeloxSegPlans &
CUDA_VISIBLE_DEVICES=1 bash nnunet/scripts/run_experiment.sh 990 3d_fullres_B 4 nnVeloxSegPlans &
CUDA_VISIBLE_DEVICES=2 bash nnunet/scripts/run_experiment.sh 990 3d_fullres_L 4 nnVeloxSegPlans &
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
references and the previous auto plans. Those auto S/B/L passed full-patch,
batch8 training, validation, fresh-process resume and whole-case inference on
RTX3090; reserved peaks were 12.326 / 16.592 / 21.076 GiB. See the
[validation protocol](RULES.md#full-patch-rtx3090-checks). Long-run results are pending.
