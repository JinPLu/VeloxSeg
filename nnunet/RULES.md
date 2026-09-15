# VeloxSeg with nnU-Net: general planning and training rules

The v2 planner generates single-GPU **3D full-resolution** configurations from
nnU-Net dataset metadata, a training memory target and a measured batch-1 CUDA
cost profile. Inference efficiency is
budgeted separately at batch size 1; it does not include training-only losses,
gradients or optimizer state. It directly instantiates the public
[VeloxSeg model](../model/VeloxSeg.py). Dataset names do not select architecture
presets; 96³ and the historical four-stage/stride-4 setup are not defaults.

The policy is maintained in
[veloxseg_rules.py](nnunetv2/experiment_planning/experiment_planners/veloxseg_rules.py).
The [planner](nnunetv2/experiment_planning/experiment_planners/veloxseg_planner.py)
uses this same policy for native commands and exported fingerprints. Generated
plans contain complete stages, training settings and the resource-estimate method.

**Current status:** candidate crops reduce each axis of nnU-Net's initial envelope
by 0/1/2 stride-alignment units. A crop is trainable when L's reference-based tensor
estimate at batch 2 fits 22 GiB inside the 24 GiB target, reserving 2 GiB for
measured error/runtime costs. Among Pareto crops on coverage, B's measured batch-1
peak memory M1 and latency t1 that cover at least `coverage_fraction` (1.0 by
default) of the largest trainable crop, the least M1 wins. The batch is then the
largest power of two ≥2 within the memory target and the 5% dataset-voxel cap.
The S/B/L family uses base channels 16 (cap 128): S has one 3×3 convolution branch,
B 1/3/5 branches and L convolution and attention depth 2, with B as the reference.
All three share geometry and batch selected against the largest member. This is not proof of
an optimal lightweight configuration.96³ remains a historical reference only.

**Interpretation agreed on2026-09-09:** continue with the new larger-patch
direction because it is faster for the measured whole-volume sliding-window
workload. This does **not** mean faster training updates, faster individual
patches, fewer parameters or better segmentation accuracy. In the controlled
synthetic comparison below, whole-volume time drops from9.766s to0.480–0.704s,
while equal-batch8 training updates increase from0.171s to0.951s. Report this
trade-off whenever presenting the new configuration's efficiency. Accuracy and
real-case end-to-end timing remain unverified; batch4 versus8 is not yet settled.

### Runtime throughput

The training lifecycle remains nnU-Net 2.6.2: native sampling, augmentation,
worker queues, pinned memory, epoch loop, checkpoint/resume and full-case
validation. The subclass supplies the VeloxSeg network/objective, AdamW/cosine,
and train/validation steps needed for the reconstruction targets and precision
fixes. It does not implement a separate training loop. FP16 autocast/GradScaler,
cuDNN benchmarking, nonblocking transfers, zero-grad with `set_to_none=True`,
and unscale-before-gradient-clipping are already in use. FP32 embedding and
loss/count reductions are retained for the observed overflow failures.

On the September 10 training hosts, large CPU arrays triggered severe memory
stalls despite free RAM. One native Blosc2 crop took 47.66 seconds on host 25;
with `NUMPY_MADVISE_HUGEPAGE=0`, the same crop took 0.12 seconds (same shape and
mean). This supports huge-page allocation/compaction as a contributor, rather
than identifying shared-storage throughput alone as the cause. The setting is
an official [NumPy process option](https://numpy.org/doc/stable/reference/global_state.html),
read at import time; no system-wide kernel policy is changed.

`run_experiment.sh` now defaults to that setting, four native augmentation
workers and one OpenMP/BLAS thread per process. Both NumPy's option and worker
count can be set explicitly for another host. Four is a tested starting point,
not an optimum: an earlier eight-worker run reduced queue wait but raised
compute time under contention. Worker count must fit all concurrent jobs.

Fresh real batches, native augmentation, full planned patch and batch8, 24 train
calls, excluding the first call from averages:

| Host / model | Startup seconds | Mean data wait seconds | Mean compute seconds |
|---|---:|---:|---:|
| 25 / BraTS S | 18.26 | 3.974 | 0.387 |
| 23 / Hecktor S | 50.92 | 4.097 | 0.848 |
| 25 / AutoPET L | 38.18 | 3.185 | 0.862 |

These short runs had finite losses and pinned batches. Compute includes transfer
and synchronization. They do not establish whole-epoch speed, six-job sustained
throughput, convergence or accuracy. Data wait still dominates, so this is a
measured improvement in the diagnosed path, not a claim of complete optimization.

Do not replace the native loader with the tested RAM-frame implementation or
change preprocessing format: those trials did not resolve full-batch startup.
A newer upstream per-case loader is a future dependency-upgrade candidate;
its integration and gain have not been validated here. `torch.compile` remains
disabled for these experiments: it requires its own compatibility/throughput
measurement and cannot eliminate CPU queue waits. Patch, batch, augmentations,
loss, LR and the 1000 × 250 update budget are unchanged. Logging follows native epoch summaries. Additional per-step logging was removed
after user feedback: concurrent jobs interleaved those lines in platform stdout.

### S/B/L family (2026-09-09)

*Historical: previous fixed-batch planner.*

Use width multipliers0.5/1/1.5 around the established base16 B reference:
S starts at8 channels (cap160), B at16 (cap320), L at24 (cap480). Keep one
JLC/PWA block per stage, decoder block depth1, FFN expansion2, dropout0 and
identical reconstruction/SDKT/deep-supervision objectives. Group widths and
attention heads continue to use the same legal-divisor policy for each width.
This is a controlled width family, not an accuracy-selected optimum.

For each dataset and selected training batch, first generate a **shared geometry
with L** under the same24GiB rule; instantiate S/B/L on that same patch, stage
count, per-axis strides and PWA spatial windows. This keeps context and tile
counts comparable and prevents each capacity from independently changing the
architecture hierarchy. Geometry may differ across datasets. The family
uses batch8; batch4 remains a separate comparison, not a tier label.
Unused capacity in S/B does not trigger patch expansion or additional stages.

Current public-model instantiations and registered-op FP32 batch1 FLOP counts:

| Dataset / shared candidate patch | Tier | Stage channels | Total parameters | Inference-path parameters | Counted GFLOPs | Estimated batch8 reserved GiB |
|---|---|---|---:|---:|---:|---:|
| AutoPET /160×224×224 | S | 8/16/32/64 | 1,242,785 | 883,773 | 24.171 | 14.950 |
| AutoPET /160×224×224 | B | 16/32/64/128 | 3,151,996 | 2,283,064 | 44.605 | 17.805 |
| AutoPET /160×224×224 | L | 24/48/96/192 | 5,418,071 | 3,946,931 | 65.217 | 20.677 |
| BraTS /160×192×160 | S | 8/16/32/64 | 1,046,605 | 825,335 | 14.548 | 8.286 |
| BraTS /160×192×160 | B | 16/32/64/128 | 2,607,950 | 2,070,568 | 36.191 | 9.758 |
| BraTS /160×192×160 | L | 24/48/96/192 | 4,245,919 | 3,370,761 | 49.002 | 10.613 |

Before the FP32 embedding correction, the original AutoPET192×256×256 candidate estimated23.425GiB for L/batch8,
but failed its first real-data backward pass on RTX3090. A separate reproduction
reached22.888GiB allocated/23.365GiB reserved and failed to allocate a cuBLAS
handle. At192×224×256, measured reserved memory23.090GiB exceeded the22.071GiB
estimate. Reserving2GiB for error plus CUDA/runtime allocations rejects both;
the same search then selects160×224×224.192×224×224 estimates24.538GiB because
PWA geometry changes discretely, so shrinking patch volume does not guarantee
monotonically smaller memory. Regenerating after the FP32 embedding correction
retains160×224×224; its current estimates appear in the table. All tiers share
the accepted geometry.

This update also changes AutoPET from five stages to four; S/B/L widths remain
8/16/24. The earlier base24 versus32 sizing rationale motivated the width family,
not a fixed parameter count. Counts above describe current instantiated models;
old AutoPET inference timings below describe the withdrawn five-stage geometry.
Parameters are actual counts, FLOPs are registered operations, and the memory
proxy is still an estimate. Accuracy remains unverified.

Suggested evaluation: validate L's memory boundary first, then compare all three
on the shared geometry with the same sampling/update schedule, augmentation,
optimizer/objective and inference protocol. Report segmentation quality, lesion
recall, total versus executed parameters, training update time, single-patch peak
memory and whole-volume latency. Any selected batch4 alternative should regenerate
one shared geometry for the entire family. Do not choose a tier from FLOPs alone.

### Measured S/B/L batch1 inference

*Historical: previous fixed-batch planner.*

The AutoPET rows below retain the earlier192×256×256/five-stage measurements.
They do not measure the corrected160×224×224/four-stage models. All timings
also precede the FP32 input-embedding fix; BraTS geometry is unchanged, but its
precision changed too.

All previous whole-volume sliding-window times also used **inference batch1**:
the native predictor creates each window with a singleton batch axis (`d[s][None]`)
and executes windows sequentially. The labels batch4/batch8 describe **training
planning**, not the number of inference windows batched together.

The shared-geometry S/B/L models were measured on idle RTX3090GPU4,
PyTorch2.6/cu124, FP16 autocast, `eval()`/inference mode, cuDNN benchmark=True.
Run the six candidates in forward then reverse order. Each pass uses5 warmup
and10 timed single-patch calls; whole-volume measurement uses1 warmup and3 timed
native sliding-window calls. Time ranges below span the two run medians, not
confidence intervals; memory is the larger observed peak across both runs.
All outputs were finite. Single-patch inputs are already on GPU. All model weights,
including dormant reconstruction decoders/auxiliary heads, remain resident;
there is no optimizer, gradient or training-loss allocation.

| Dataset / tier | Patch | Single-patch ms | Single allocated / reserved GiB | Whole-volume seconds | Whole-volume allocated GiB |
|---|---|---:|---:|---:|---:|
| AutoPET / S | 192×256×256 | 31.2–39.7 | 0.240 / 0.344 | 0.858–0.942 | 1.271 |
| AutoPET / B | 192×256×256 | 31.1–32.6 | 0.292 / 0.408 | 0.870–0.921 | 1.324 |
| AutoPET / L | 192×256×256 | 34.5–45.0 | 0.351 / 0.506 | 1.040–1.188 | 1.384 |
| BraTS / S | 160×192×160 | 15.3–15.3 | 0.169 / 0.281 | 0.064–0.093 | 0.288 |
| BraTS / B | 160×192×160 | 17.4–20.9 | 0.184 / 0.309 | 0.064–0.078 | 0.301 |
| BraTS / L | 160×192×160 | 22.6–22.8 | 0.250 / 0.371 | 0.081–0.097 | 0.365 |

AutoPET uses the326×400×400 synthetic preprocessed median volume and27 windows.
BraTS uses140×171×136, padded to160×192×160, and1 window. Both use tile step0.5,
Gaussian fusion, no TTA or fold ensemble; timing excludes preprocessing/export
but includes the predictor's data movement, padding/cropping and accumulation.
Allocated is PyTorch tensor allocation; reserved additionally includes cached
allocator blocks. Neither is a full `nvidia-smi` process-memory accounting.

AutoPET S has lower parameters/memory, but its timing ranges overlap B and do
not establish a stable speed advantage. The timing order affected medians, so
do not rank close values or infer speedup from FLOPs alone. As a code fact, S
and B retain the same first two PWA packed Q/K/V widths (32,48) and spatial token
counts; shrinking the backbone does not shrink every attention operation. This
has not been isolated as the cause of the measured timing behavior. B remains
the main comparison; S's latency optimization and all tiers' accuracy remain open.
Passing batch1 inference does not validate L's estimated batch8 training fit.

### Fixed batch4 versus batch8

*Historical: previous fixed-batch planner.*

Same fingerprints,24GiB target, base16, architecture/training rules. Counts use
batch1 evaluation; FLOPs cover PyTorch's registered operations, not GPU latency.

| Dataset | Fixed training batch | Patch | Stage channels | Total parameters | Batch1 counted GFLOPs | Estimated training reserved GiB |
|---|---:|---|---|---:|---:|---:|
| BraTS2021 | 4 | 160×192×160 | 16/32/64/128 | 2,607,950 | 36.191 | 4.898 |
| BraTS2021 | 8 | 160×192×160 | 16/32/64/128 | 2,607,950 | 36.191 | 9.751 |
| AutoPET-II | 4 | 224×320×320 | 16/32/64/128/256 | 9,154,118 | 115.603 | 18.468 |
| AutoPET-II | 8 | 224×256×256 | 16/32/64/128/256 | 9,147,462 | 73.250 | 23.056 |

AutoPETbatch8 rejects224×320×320 (36.782GiB estimate), then224×256×320
(29.082GiB), accepting224×256×256. Patch volume drops36%, counted inference
FLOPs drop36.6%, but total parameters drop only0.073% because the stage hierarchy
remains five levels. BraTS has sufficient capacity at both batches and keeps the
same patch/network. Thus larger planning batch does not universally force a
smaller model. The estimates are not measured CUDA peaks or whole-case timing.

Training still uses250updates/epoch: batch8 presents twice the patches per epoch
as batch4, with different patch volumes where applicable. Any later accuracy
comparison must state its sample/update/time budget; it is not an isolated
architecture comparison. No automatic LR scaling or long training is launched.

### Native training-update timing

*Historical: previous fixed-batch planner.*

A separate timing pass measures through `train_step` and CUDA synchronization,
excluding the extra per-parameter finite-gradient inspection. Discard the first
cuDNN/optimizer startup update; the two subsequent updates average:

| AutoPET configuration | Training batch | Seconds / update | Training peak reserved GiB |
|---|---:|---:|---:|
| Historical96³ configuration | 8 | 0.171 | 2.492 |
|224×256×256 | 8 | 0.951 | 23.354 |
|224×320×320 | 4 | 0.739 | 18.430 |

At equal batch8, the new configuration takes about5.56 times as long per update
as the historical96³ reference under this synthetic protocol. It also processes
16.59 times as many voxels per patch and changes architecture; this is not an
accuracy-normalized or pure patch-only comparison. Data loading/augmentation is
excluded. The original memory reports' step timings included extra numerical
inspection and must not be used for this training-speed comparison.

### Native sliding-window speed versus the historical96³ reference

*Historical: previous fixed-batch planner.*

On the same RTX3090GPU4, compare one326×400×400 synthetic preprocessed volume
(the AutoPET fingerprint median), batch1, tile step0.5, Gaussian blending, no TTA,
no preprocessing/export. Each model has one whole-volume warmup and three timed
native `nnUNetPredictor` runs. The96³ row uses the explicit standalone reference
in `config/models_config_autopetii.json`, instantiated in the current public
implementation; it is not the automatic three-stage96³ model or a reproduction
of the old runtime/checkpoint. All output logits were finite.

| Configuration | Patch | Total parameters | Single patch median ms | Tiles | Sliding-window median seconds | Sliding-window peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| Historical96³ configuration | 96³ | 2,288,805 | 26.31 | 384 | 9.766 | 1.004 |
| Fixed batch4 plan | 224×320×320 | 9,154,118 | 50.35 | 8 | 0.480 | 1.805 |
| Fixed batch8 plan | 224×256×256 | 9,147,462 | 37.29 | 18 | 0.704 | 1.414 |

The historical configuration is faster per patch, but much slower for this
particular volume and tiling protocol. Batch8 reduces patch memory/operations,
yet takes46.6% longer than batch4 for this sliding-window volume. These timings
are short synthetic measurements, not end-to-end clinical case latency or an
accuracy comparison. Shape/overlap/TTA and CPU/GPU scheduling affect timing.
Do not interpret a single-patch FLOP reduction as a whole-case speedup.

### Patch growth and capacity coupling

*Historical: previous fixed-batch planner.*

All rows below use the then-current automatic rule, PET/CT input, base channels16,
identical spacing and batch1 evaluation. They do **not** compare historical
checkpoints or isolate patch size from stage changes.

| Patch | Stage channels | Training model parameters | Inference-path parameters | Counted forward GFLOPs |
|---|---|---:|---:|---:|
| 96³ | 16/32/64 | 1,003,246 | 678,316 | 4.635 |
| 160×224×192 | 16/32/64/128 | 3,150,124 | 2,281,192 | 37.496 |
| 224×320×320 | 16/32/64/128/256 | 9,154,118 | 6,373,888 | 115.603 |

Parameter counts come from instantiated public models. Inference-path counts
exclude reconstruction decoders and auxiliary segmentation heads; those weights
remain owned by the training model. FLOPs use PyTorch2.6 FlopCounterMode on fake
FP32 batch1 eval tensors, covering registered operations, not all operator costs
or measured GPU latency. Larger patches can reduce the number of sliding-window
tiles, so these single-patch counts do not establish whole-case runtime ordering.
The next planning revision must distinguish data/context requirements, S/B/L
capacity and whole-case efficiency; training VRAM is a feasibility bound.
The current fingerprint does not establish how much lesion context is sufficient.

## What determines a configuration

```text
channel groups + labels + nnUNet fingerprint + memory target
                             |
         spacing / transpose / median resampled shape
                             |
             patch / per-axis strides / stage count
                             |
         JLC capacity + PWA local-to-global windows
                             |
         mixed-precision tensor proxy + measured GPU reference
                             |
              final patch, stages, batch and plans
                             |
         preprocess -> train -> resume -> predict
```

| Decision | General rule and reason |
|---|---|
| Image geometry | Reuse nnUNet's target-spacing, transpose, normalization and resampling methods. Start the patch search from its inverse-spacing aspect ratio and 256³-volume envelope, clipped to median resampled shape. |
| Spatial hierarchy | Reuse nnUNet axis-pooling geometry, but combine its first two pooling transitions into VeloxSeg’s compact stem (normally stride 4). Keep at least one later decoder transition for small inputs. Later strides and total stage count remain geometry-driven; minimum feature edge 3 matches the released 96³ model's 3³ terminal grid. |
| Finest features | JLC and PWA both start at the compact stem grid. PixelShuffle produces full-resolution segmentation/reconstruction. Fullres preprocessing does not require a stride-1 feature stream. The earlier full-resolution stem was withdrawn after measured memory regression. |
| Capacity | Start at 16 channels, double by stage, cap at 128. S uses one 3×3 convolution branch, B parallel 1/3/5 branches, L convolution and attention depth 2 per stage; convolution/attention expansions 3/3/2/2 by stage; dropout 0.1. These are provisional template priors; lightweight performance has not been established. |
| JLC geometry | Parallel kernels follow nnUNet's anisotropic kernel axes. The group width interpolates the released 96³ model's cumulative-compression anchors (2^6, 4), (2^9, 8), (2^12, 8), (2^15, 16) on log scales, clamped beyond them, then rounds up to a channel divisor capped by stage width. This is a JL-inspired heuristic, not a fitted accuracy guarantee. |
| PWA geometry | The base big window tiles each axis of the stage grid with a power-of-two ratio and at least 2 positions. Among windows whose token mismatch to the reference 27/216/27 is within 4× of the best achievable, choose the most physically balanced, then nearest tokens, more scales, lexicographic order. Windows then double per axis to exact global coverage; the last stage attends over its whole grid. Pooling windows are 1×1×1. This bounded local grid keeps attention cost from growing quadratically with the whole input. |
| PWA channels | Heads / head dimension 1/4, 2/8, 2/8, 4/16 by stage from the released reference model; later stages reuse the last. Allow internal QK/V projection expansion; do not widen the whole backbone just for PWA divisibility. Attention uses PyTorch SDPA with the same relative-position bias. |
| Patch selection | Rebuild geometry and PWA for every candidate. Candidates reduce each axis of nnUNet's initial envelope by 0/1/2 stride-alignment units, recomputing alignment before subtracting. A crop is trainable when L fits the target at batch 2; if none is, the envelope takes nnUNet's next smaller step. `nnunet/cost_profile.py` measures B's batch-1 peak memory M1 and latency t1 on the target GPU; axis-permuted crops with equal volume, parameters and M1 count once. Among Pareto crops on (coverage, M1, t1) covering at least `coverage_fraction` (1.0) of the largest trainable crop, the least M1 wins. |
| Batch | Largest power of two ≥2 whose S/B/L estimate fits the 24 GiB target and whose batch covers at most 5% of dataset voxels; shared by S/B/L. |

The geometric foundation comes from the pinned
[nnUNet default planner](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py)
and [topology function](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/network_topology.py).
The coverage allowance is explicit in the
[ResEnc M/L/XL presets](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py).
The VeloxSeg-specific choices above are initial policy decisions subject to
real-data calibration; they are not claims that nnUNet itself specifies PWA.

## Resource model

The planner runs the real mixed-precision network and FP32 objective on fake
CPU tensors to count unique saved storage `A`. It also counts fixed storage `F`
for parameters, gradients, two AdamW moments and buffers. CPU FP16 autocast and
math SDPA are a portable **proxy**, not a CUDA peak trace.

Following nnUNet's proxy-plus-reference approach, use one shared two-term fit:

```text
O = batch * patch_volume * (segmentation_heads + input_channels) * 4
estimated_reserved = F + 0.93111870 * A + 3.62225068 * O
```

The exact coefficients are solved from `MEMORY_REFERENCES`. The two references
are RTX 3090 measurements below, PyTorch 2.6.0/cu124, CUDA FP16, FP32 losses,
AdamW and cuDNN benchmark=True. This is an empirical proxy, not a physical
allocation decomposition. Two observations determine two coefficients: agreement
at those points is calibration, **not independent predictive validation**.
No dataset-name-specific coefficient is used. Other shapes, devices and longer
runs remain unverified. Fake persistent tensors are retained during inventory
so weak converter references cannot recycle excluded storage identities;
three repeated inventories per reference returned identical counts.

The device target is **24 GiB by default**, independent of **batch-1 inference**.
The tensor acceptance budget is target minus2GiB; this reserve was introduced
after the observed L backward failure and the subsequent1.019GiB estimate error.
Candidate crops are judged at batch 2; the batch grows only after the crop is
selected. Reserved memory includes
allocator cache; external CUDA allocations and other processes are outside this
proxy. The batch therefore never changes patch, stages or PWA geometry.

For comparison, nnUNet ResEnc-L counts convolutional feature-map elements,
`C_out * spatial_volume`, through the encoder and decoder. It compares this
proxy against `2.1e9 * target_GB / 24`, then computes roughly
`round(reference / proxy * 2)` training samples, subject to its dataset cap.
Those units are not bytes and the reference cannot be copied to an attention
model with reconstruction losses. See the
[official planner](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py).

## Training policy

| Setting | Shared rule |
|---|---|
| Optimizer | AdamW, LR 0.001, weight decay 0.01; optional norm/bias/position-bias decay exclusions (off by default); no automatic batch-to-LR scaling |
| Schedule | Warmup given in updates (default 0), then cosine decay to 6e-6 over 1000 epochs (250,000 updates) |
| Sampling budget | 250 updates and 50 validation batches per epoch; 0.33 foreground oversampling |
| Data pipeline | Official nnUNet augmentation, folds, region handling and sliding-window prediction |
| Precision | CUDA FP16 autocast + inherited GradScaler; compact patch embedding executes in FP32 to avoid the observed bias-gradient overflow; unscale then clip gradient norm to 12. Gram construction and objective reductions use FP32. CPU training remains FP32. GradScaler-skipped steps are logged per epoch. Full-patch real-data RTX3090 checks passed for the six BraTS/AutoPET tiers of the previous plans; long-run convergence is unverified. |
| Segmentation | CE+Dice for class labels; BCE+Dice for regions; per-sample Dice for this fullres-only workflow |
| Deep supervision | Native-resolution logits with nearest-neighbor target resizing; supervise decoded features, omit the undecoded bottleneck; normalize `2^-level` weights to sum to 1 so adding stages does not multiply the segmentation objective |
| Reconstruction | Per-modality elementwise MSE, then average modality groups; coefficient 0.5 |
| SDKT | Gram `X Xᵀ/(C N)`; per-sample squared Frobenius difference, sum modality teachers, average batch; coefficient 2 |
| Seed | 12345 for initialization and main-process random generators; exact multiworker/resumed RNG reproducibility is not claimed |

The public model keeps a flat output list with native-resolution segmentation
heads. The shared loss resizes the full-resolution target to each head, preserving
class IDs, regions and ignore masks. Standalone auxiliary metric reporting
upsamples detached predictions only when requested. Both entry points share [model/loss.py](../model/loss.py); their
segmentation criteria and standalone scheduling are distinct.

The deep-supervision weighting is an adaptation for variable-stage models.
The selected paper RC/SDKT reduction remains explicit. Historical checkpoints
and old summed-head training are not numerically interchangeable with this
protocol. Initial synthetic gradients show strong SDKT contributions; convergence
and coefficient suitability still require real-data training.

## Install and native commands

Use a dedicated **Linux or macOS** Python environment. The complete CLI check used
Python 3.11.15, PyTorch 2.6.0 and nnUNetv2 2.6.2. From the repository root:

```bash
python -m pip install -r nnunet/requirements.txt
python nnunet/install.py
```

The installer registers the planner/trainer in nnUNet and links the public
`model` and `utils` packages to this checkout. Keep it in place and rerun the
installer after modifying nnUNet extension files. Existing unrelated packages
with those names require a separate environment.

Set `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` to your data/output
locations. In the dataset's
`dataset.json`, add explicit channel groups in channel order:

```json
"veloxseg_modality_channels": [1, 1]
```

The examples use `[1,1]` for PET/CT and `[4]` for four-channel MRI early fusion.
Labels and `regions_class_order` remain standard nnUNet metadata. Existing raw
`splits_final.json` is preserved by the upstream planning workflow.

```bash
# Both native examples use a 24 GiB training target. Planning reads
# veloxseg_profile.json, measured on the target CUDA GPU.
nnUNetv2_extract_fingerprint -d 137
python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner candidates \
  --dataset-name Dataset137_BraTS2021 \
  --dataset-json "$nnUNet_raw/Dataset137_BraTS2021/dataset.json" \
  --fingerprint "$nnUNet_preprocessed/Dataset137_BraTS2021/dataset_fingerprint.json" \
  --gpu-memory-target-in-gb 24 \
  --output "$nnUNet_preprocessed/Dataset137_BraTS2021/veloxseg_candidates.json"
python nnunet/cost_profile.py \
  --candidates "$nnUNet_preprocessed/Dataset137_BraTS2021/veloxseg_candidates.json" \
  --output "$nnUNet_preprocessed/Dataset137_BraTS2021/veloxseg_profile.json"
nnUNetv2_plan_and_preprocess -d 137 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
nnUNetv2_train 137 3d_fullres_B 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1

# 221: run the same fingerprint, candidates and profile steps first.
nnUNetv2_plan_and_preprocess -d 221 -pl VeloxSegPlanner -c 3d_fullres_B -gpu_memory_target 24
nnUNetv2_train 221 3d_fullres_B 4 -tr nnVeloxSegTrainer -p nnVeloxSegPlans -num_gpus 1
```

Add `--c` to the training command to resume. For prediction use
`nnUNetv2_predict` with matching dataset, trainer, configuration, fold and
`-p nnVeloxSegPlans`. S/B/L share one preprocessing data identifier, so preprocess only B once per
dataset. Cache reuse requires matching preprocessing geometry, normalization,
resampling, labels and sampling metadata; changing patch alone does not require
resampling the data. The multi-GPU launcher is documented in the [experiment guide](EXPERIMENTS.md#six-experiments).

Exported fingerprints can use the same policy without raw images; only the
profile step needs the target CUDA GPU:

```bash
python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner candidates \
  --dataset-name Dataset137_BraTS2021 \
  --dataset-json nnunet/config/Dataset137_BraTS2021/dataset.json \
  --fingerprint nnunet/config/Dataset137_BraTS2021/dataset_fingerprint.json \
  --gpu-memory-target-in-gb 24 \
  --output candidates.json
python nnunet/cost_profile.py --candidates candidates.json --output profile.json
python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner plan \
  --dataset-name Dataset137_BraTS2021 \
  --dataset-json nnunet/config/Dataset137_BraTS2021/dataset.json \
  --fingerprint nnunet/config/Dataset137_BraTS2021/dataset_fingerprint.json \
  --gpu-memory-target-in-gb 24 \
  --profile profile.json \
  --output nnunet/config/Dataset137_BraTS2021/nnVeloxSegPlans.json
```

Both CLI and native planning call the same `plan_family`. Without
`veloxseg_profile.json`, native planning stops and prints these steps; a profile
measured on a different architecture is rejected. Existing checkpoints/plans keep their original configuration.

## Generated examples and verification

Bundled plans contain `3d_fullres_S/B/L` regenerated on RTX 3090 with the 24 GiB
target: BraTS 160×192×160 / batch 8 (4 stages), AutoPET 256×320×256 / batch 2
(5 stages), Hecktor 160×256×256 / batch 4 (5 stages). They have not been trained yet.

### Full-patch RTX3090 checks

*Historical: previous fixed-batch planner.*

All six corrected configurations passed real-data checks at their full planned
patch and batch8, PyTorch2.6/cu124, native two-worker augmentation and default
GradScaler initial value65536. Each initial phase used20 native training calls
on three augmented batches (14/3/3 calls), followed by three online validation
batches and a further effective training update in the same process. Initial
scaler backoffs are excluded from the effective-update counts below. All losses,
effective gradients and validation counts were finite.

| Dataset / tier | Effective updates in 20 calls | Peak allocated / reserved GiB | Resume / whole-case inference |
|---|---:|---:|---|
| BraTS / S | 14 | 6.331 / 12.182 | passed / passed |
| BraTS / B | 15 | 7.811 / 13.850 | passed / passed |
| BraTS / L | 14 | 9.232 / 15.533 | passed / passed |
| AutoPET / S | 13 | 8.729 / 12.271 | passed / passed |
| AutoPET / B | 13 | 11.837 / 15.543 | passed / passed |
| AutoPET / L | 13 | 15.404 / 20.934 | passed / passed |

Hecktor S/B/L were subsequently checked on the corrected implementation with
full 160×256×256 patches, batch8 and production CLI cuDNN settings
(`benchmark=True`, `deterministic=False`). The same 20-call, three-real-batch
training protocol, three validation batches and post-validation update passed:

| Dataset / tier | Effective updates in 20 calls | Peak allocated / reserved GiB | Resume / whole-case inference |
|---|---:|---:|---|
| Hecktor / S | 15 | 10.678 / 12.326 | passed / passed |
| Hecktor / B | 15 | 14.736 / 16.592 | passed / passed |
| Hecktor / L | 14 | 18.890 / 21.076 | passed / passed |

All three fresh-process restores completed three effective updates, finite
validation and a finite native prediction on a real 91×512×512 case (two output
classes, Gaussian fusion, step0.5, no TTA or export). The unchanged standalone
reference previously passed a one-epoch, five-case train/validation/test check;
the fixed nnUNet reference passed real-data full-patch/batch4 train/validation
with 1.582 GiB peak reserved memory. These are runtime checks, not accuracy or
long-run convergence results.

Each BraTS/AutoPET checkpoint was then loaded in a fresh process at epoch1, followed by three
effective training updates, finite online validation and one native batch1
whole-case prediction. Actual preprocessed case shapes were140×176×133 for
BraTS and619×400×400 for AutoPET; predictions were finite. This inference check
used Gaussian fusion/step0.5 without TTA or export, and is not an accuracy score.
Peak memory above spans the initial training, online validation and return to
training; reserved includes allocator cache, but excludes external CUDA memory.
The direct-Trainer lifecycle above used the backend default (`benchmark=False`).
A separate check matched the production CLI (`cudnn.benchmark=True`,
`deterministic=False`) on all six full configurations, reusing one real augmented
batch for20 calls plus validation and a further effective update. All passed.
Reserved peaks for BraTS S/B/L were11.971/13.857/15.461GiB; AutoPET S/B/L were
12.264/15.559/20.912GiB. Native predictor construction enables benchmark mode,
so the whole-case inference checks also exercised it.
These bounded checks do not establish1000-epoch convergence or long-run stability.

The previous bundled plans contained `3d_fullres_S/B/L` at the shared batch8 geometry.
All six configuration names passed CPU 32³ synthetic native training, validation,
checkpoint reload, resumed updates and sliding-window inference. These reduced
CPU checks alone did not establish full-patch CUDA training fit; the real-data
RTX3090 evidence is given above.

The earlier base16 fixed-batch comparison below uses its original patches, not
the new shared AutoPET geometry. On idle RTX3090 GPU4, its four candidates
completed three effective AMP/AdamW updates with finite losses/gradients and
finite batch1 outputs:

| Dataset / fixed batch | Training peak allocated GiB | Training peak reserved GiB | Batch1 peak allocated GiB |
|---|---:|---:|---:|
| BraTS /4 | 4.071 | 4.924 | 0.201 |
| BraTS /8 | 8.089 | 9.746 | 0.201 |
| AutoPET /4 | 17.007 | 18.430 | 0.535 |
| AutoPET /8 | 21.670 | 23.354 | 0.363 |

PyTorch2.6/cu124, cuDNN benchmark=True, initial GradScaler128; inference runs
follow optimizer/gradient/input cleanup, with three warmups and five timed passes.
These are synthetic operation/memory checks, not accuracy or long-run stability
measurements. The new AutoPETbatch8 reserved estimate23.056GiB underpredicts the
observed23.354GiB by0.298GiB; the GPU has about23.69GiB visible, leaving little
headroom. The reference coefficients were not refitted to this new observation.

Historical reference configurations used to calibrate or inspect the proxy:

These use compact entry. Actual synthetic checks completed three full AMP
updates with finite losses/gradients, initialized AdamW state and batch-1 inference:

| GPU / protocol | Patch / batch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---:|---:|
| RTX 3090, BraTS, benchmark=True | 160×192×160 / 19 | 19.149 | 23.098 |
| RTX 3090, AutoPET, benchmark=True | 224×320×320 / 5 | 21.235 | 23.047 |
| GTX 1080 Ti, BraTS, benchmark=False | 160×192×160 / 8 | 8.102 | 9.734 |
| GTX 1080 Ti, AutoPET, benchmark=False | 224×320×320 / 2 | 8.588 | 9.301 |

RTX 3090 BraTS batch20 with benchmark=False passed the first update but OOMed
on the second; this does not establish a universal maximum batch. The 3090
has about 23.69 GiB visible capacity, so the chosen batches leave little headroom.
These are synthetic memory/operation checks, not accuracy or long-run stability
results. Measurements initialize GradScaler at 128 to ensure actual updates.
Production retains nnUNet's default 65536. A separate eight-attempt default-scale
check observed three skipped overflow-gradient steps followed by five finite
updates at scale8192; nonfinite gradients were not applied to weights.

### AutoPET configuration behind the 23.047 GiB measurement

*Historical: previous fixed-batch planner.*

Patch224×320×320 at spacing3×2.03642×2.03642 mm covers approximately
672×651.65×651.65 mm. It contains22,937,600 voxels, 25.93 times a96³ patch.
Input groups are PET and CT, one channel each; outputs are background/lesion.
The model has9,154,118 parameters. The initial patch follows nnUNet's
inverse-spacing256³ volume envelope, is clipped to median shape and aligned to
legal pooling; it is not a fixed96³ design.

| Stage | Stride | Feature grid | Channels | JLC group width | Heads / head dim | Base window / pool | Scales |
|---|---|---|---:|---:|---|---|---:|
| 0 | 4×4×4 | 56×80×80 | 16 | 8 | 1 / 8 | 7×10×10 / 1×2×2 | 4 |
| 1 | 2×2×2 | 28×40×40 | 32 | 16 | 1 / 16 | 7×10×10 / 1×2×2 | 3 |
| 2 | 2×2×2 | 14×20×20 | 64 | 16 | 2 / 16 | 7×10×10 / 1×2×2 | 2 |
| 3 | 2×2×2 | 7×10×10 | 128 | 32 | 4 / 32 | 7×10×10 / 1×2×2 | 1 |
| 4 | 1×2×2 | 7×5×5 | 256 | 32 | 8 / 32 | 7×5×5 / 1×1×1 | 1 |

The instantiated AutoPET PWA lists (coordinates in each stage's feature grid):

| Stage grid | All attention windows | Corresponding max-pool kernels/strides |
|---|---|---|
| 56×80×80 | 7×10×10;14×20×20;28×40×40;56×80×80 | 1×2×2;2×4×4;4×8×8;8×16×16 |
| 28×40×40 | 7×10×10;14×20×20;28×40×40 | 1×2×2;2×4×4;4×8×8 |
| 14×20×20 | 7×10×10;14×20×20 | 1×2×2;2×4×4 |
| 7×10×10 | 7×10×10 | 1×2×2 |
| 7×5×5 | 7×5×5 | 1×1×1 |

Window counts per stage are512/64/8/1,64/8/1,8/1,1,1 respectively.
Each pooled window has7×5×5=175 spatial tokens, paired to350 across PET/CT;
the final window at every stage covers that entire feature grid after pooling.
For stage0, the largest feature window56×80×80 spans the entire224×320×320
input patch, but this does not perform dense attention over all original voxels.

Every stage uses one convolution block and one attention block, parallel
1³/3³/5³ kernels, FFN expansion2 and dropout0. Each scale retains175 spatial
tokens (350 paired tokens). Q/K and V packed widths are32,48,64,128,256;
packing expands the first two stages beyond their backbone width.

Geometry reuses nnUNet's axis-wise pooling, then merges the first two transitions
into the compact entry. Width16 with doubling/cap320, minimum feature edge4,
JLC logarithmic group-width heuristic, heads=max(1,C//32), head_dim=group_width,
and exact-divisor window/pooling selection are **maintained engineering priors**,
not uniquely derived from the paper or demonstrated optimal. Stage count follows
geometry. Stride4 restored compact entry after the stride1 adaptation proved
expensive; its effect on small-lesion accuracy remains untested.

The segmentation decoder outputs full patch resolution and auxiliary grids
28×40×40,14×20×20,7×10×10, with weights8/15,4/15,2/15,1/15. Two reconstruction
decoders and FP32 Gram teaching participate in training; inference executes only
the main segmentation path. Reserved memory includes about1.812 GiB above peak
allocated memory in this check; it is not parameter memory alone.

The corrected compact architecture also passed CPU 32³ native loading,
augmentation, train/validation, checkpoint reload with prediction error zero,
resumed update and sliding-window prediction for both label protocols.

Observed CPU checks on the earlier 8 GiB-generated configurations completed official data
loading/augmentation, train/validation steps, exact checkpoint prediction reload,
a resumed update and sliding-window prediction on synthetic inputs. Separate
checks covered anisotropic strides, unequal modality groups `[2,1]`, single-group
input, SDPA value/gradient equivalence, and stage-count-independent supervision
weighting. CUDA compile, full convergence and DDP remain unverified; 2D/cascade is outside the requested scope.

A separate three-case synthetic NIfTI check completed official fingerprinting,
planning, preprocessing, one native training epoch, final-checkpoint reload and
standard `nnUNetv2_predict` export. All three outputs retained their original
shape and spacing, and the command exited successfully on Python 3.11.15.
A prior macOS Python 3.12.13 run exported the outputs but hung during interpreter
shutdown in the multiprocessing resource tracker; that runtime is not the
validated end-to-end environment.

## Upstream adaptation audit (2026-09-09)

This audit reads the installed **nnUNetv2 2.6.2** implementation and the matching
[official ResEnc documentation](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/documentation/resenc_presets.md).
It distinguishes fixed training presets, fingerprint-driven rules and choices
made from validation results. Inheriting the trainer does not automatically
implement every part of nnUNet planning.

### Separate training capacity from inference efficiency

The project inference budget is **batch 1**, `eval()` and disabled gradients.
There is no user-specified numerical inference limit yet. Count executed network
operations separately from resident parameters: reconstruction decoders and
auxiliary losses do not execute in evaluation, but the current model still owns
their parameters. Whole-case runtime additionally includes tile count, overlap,
TTA, folds, resampling and export. A small tile can increase whole-case work.

Training may use the target GPU's available capacity. The default is now 24 GiB,
with no arbitrary75% discount. The planner now requires L to train at batch 2, then
expands the batch by powers of two within the target and the 5% dataset cap. Upstream ResEnc-L targets 24 GB and is the recommended preset in this
version; the original planner uses 8, ResEnc-M targets roughly 9–11 GB actual
VRAM, and XL targets 40. These are per-GPU targets, not pooled DDP memory.

Upstream uses architecture-specific feature-map counts and empirical reference
values. The previous VeloxSeg `2 * saved_activation_bytes + fixed_bytes` rule
was withdrawn after real measurements: AutoPET 96³/batch2 estimated 16.43 GiB,
but allocated peak was 8.52 GiB. More significantly, the previous stride-1
architecture itself used much more memory than the compact reference. The
current calibrated proxy and compact stem address these two separate issues.

### What is inherited and what remains different

| Area | Upstream behavior in 2.6.2 | VeloxSeg status and implication |
|---|---|---|
| Fingerprint | Cropped shapes, spacings, crop ratios and foreground intensity statistics | Reused. These do not specify lesion-context requirements or a theoretical optimum patch. |
| Target spacing | Axis medians; strongly anisotropic spacing **and** voxel shape trigger a 10th-percentile correction on the coarse axis | Reused directly. Do not force isotropic spacing for every dataset. |
| Crop / normalization | Nonzero crop; CT foreground-statistic clipping at 0.5/99.5 percentiles and dataset-level standardization; other mapped channels commonly case-wise z-score; mask depends on crop ratio | Reused. Bundled PET is ZScore, CT is CTNormalization, MRI uses its planned nonzero mask. This is not a claim that PET z-score is universally optimal. |
| Resampling | Image interpolation and label-aware interpolation differ; strong anisotropy can use separate coarse-axis resampling; predictions reverse resample/crop/transpose | Reused through upstream preprocessor/export. Preserve geometry metadata. |
| Spatial topology | Pool axes according to relative spacing and remaining size; anisotropic kernels; minimum feature edge 4; patch padding follows resulting divisibility | Reuse axis geometry and combine the first two pooling transitions into a compact stem. A stride-1 network stem is not required by 3D fullres preprocessing. |
| Capacity | ResEnc uses base 32, 3D cap 320, encoder block pattern 1/3/4/6… and decoder depth 1 | VeloxSeg uses its JLC/PWA template, base 16 (cap 128) and one block per stage (two for L). These are model-specific priors requiring calibration, not inherited optimal settings. |
| Patch / batch | Prefer a large patch fitting at least 2 training samples, then allocate batch capacity; original 5% dataset cap is relaxed in ResEnc presets (code sets 1.0) | Search structure reused with measured VeloxSeg references, including modality branches, reconstruction and Gram objectives. Larger GPU targets remain extrapolations until measured. |
| Context coverage | If fullres patch volume is below 25% of median resampled volume, search lowres spacing in 1.03 increments; discard lowres if volume reduction is below 2×; otherwise offer lowres and cascade | **Outside user-selected scope:** only fullres is required (2026-09-09). Keep fullres spacing; evaluate its context coverage without adding lowres/cascade. |
| Configuration selection | 2D, 3D fullres and available lowres/cascade candidates; cross-validation predictions guide best configuration/ensembles | Only 3D fullres is in scope. Real-data fold/configuration validation remains open; no lowres/cascade implementation is planned for this work. |
| Sampling / augmentation | Foreground-aware crops, larger pre-augmentation crop, rotations/scales, noise/blur/intensity/gamma/low-resolution/mirroring; strongly anisotropic patches use dummy-2D augmentation | Inherited. Configured 0.33 foreground oversampling rounds to one forced-foreground sample in a batch of 2, i.e. 50% of that batch. Spatial transforms preserve channel alignment; intensity transforms need not be identical across modalities. |
| Optimization | SGD, LR 0.01, momentum 0.99/Nesterov, weight decay 3e-5, polynomial schedule; 1000 epochs ×250 updates, 50 validation batches | Update budget inherited; AdamW/cosine is a VeloxSeg policy. Changing batch changes sample exposure; nnUNet does not automatically turn this into an exposure-matched experiment. |
| Precision | CUDA autocast + GradScaler; unscale then clip gradients to norm 12 | Adapted: CUDA FP16 autocast, inherited GradScaler/checkpoint state, unscale then norm-12 clipping. Patch embedding, Gram and loss reductions stay FP32. CPU FP16 numerical checks and short actual CUDA updates passed; long-run numerical behavior remains open. |
| Segmentation loss | CE+Dice or region BCE+Dice, ignore-label handling, memory-efficient Dice; batch-Dice depends on configuration | Label handling reused; sample-wise Dice matches fullres without cascade. If lowres/cascade is introduced, revisit configuration-specific Dice rather than sharing one global boolean. |
| Deep supervision | Native-scale decoder outputs and downsampled targets, normalized powers of 1/2; final lowest-resolution output weight is zero (tiny nonzero in a DDP case) | Adapted to native scales in the shared loss. Normalized 2^-level weights retain every decoded head and exclude the undecoded bottleneck; we do not copy the extra zero weight, which would discard a different level and leave unused head parameters. |
| Validation / checkpoints | Online patch Dice is an approximate training signal; full-case validation, checkpoints and fold predictions support selection | Native lifecycle reused. CPU synthetic resume passed; no real-data convergence, exact multiworker RNG resume or DDP verification is claimed. |
| Inference | One tile per network call, default 0.5 step, Gaussian weighting, enabled mirror axes, optional fold averaging; CUDA autocast; half-precision output accumulators | Standard predictor reused. Three mirror axes can mean 8 forwards/tile; five folds can make 40. GPU whole-case arrays consume memory beyond a batch-1 forward. CPU export and GPU batch-1 forward checks passed. Synthetic whole-case CUDA memory/runtime were measured above; real-case end-to-end latency remains open. |
| Postprocessing | Largest-component rules are accepted only when validation metrics improve, with safeguards for foreground aggregation | Keep upstream validation-driven selection. Never unconditionally retain only the largest lesion for multifocal AutoPET. |

### Earlier base16 same-fingerprint planning comparison

The official `nnUNetPlannerResEncL.get_plans_for_configuration` was executed on
both bundled fingerprints with its 24 GB default. This compares planning
outputs, **not equivalent models, measured memory or accuracy**.

| Dataset | Official ResEnc-L patch / batch | Earlier base16-only 24 GiB proxy output | Actual VeloxSeg GPU check |
|---|---|---|---|
| BraTS2021 | 160×192×160 / 3 | 160×192×160 / 8 | Earlier batch19 reference passed on RTX3090 |
| AutoPET-II | 160×224×192 / 2 | 224×256×256 / 8 | Batch8 passed on RTX3090 |

BraTS median shape is 140×171×136; topology divisibility padding explains why
its patch can exceed the median. AutoPET median is 326×400×400. At step 0.5,
these batch8 patches need 1 and 18 tiles for these median shapes, before mirroring or
fold ensembles (geometry calculation, not whole-case measured runtime).
PWA is global within the patch and cannot recover anatomy outside it.

### Consequences for the maintained policy

The current scope is **3D fullres only**, as requested. AMP, native-scale supervision and compact entry are implemented. The measured
patch/batch pairs above work; 24 GiB extrapolations and accuracy remain unverified. Keep batch-1 inference and whole-case tile
cost separate. Do not coarsen fullres spacing, widen the model to fill memory,
or restore the withdrawn full-resolution feature stem without new evidence.

### Mixed-precision NaN investigation and memory attribution

The former Gram calculation formed `X Xᵀ` in the autocast dtype and divided by
`C*N` afterwards. FP16 can overflow before that division. An observed CPU FP16
reproduction with all-one features `[1,16,48,48,48]` produced infinity; the fixed
path produced about `0.0625` and finite gradients. The implementation disables
autocast, converts features to FP32 and divides each operand by `sqrt(C*N)`
before multiplication, preserving the intended Gram normalization. GradScaler
cannot repair an already-infinite forward loss. No clamping, NaN replacement
or silent precision fallback was added.

Observed: two 12-update CPU FP16-autocast/GradScaler runs (class CE and region BCE,
complete segmentation+RC+SDKT) had finite losses and unscaled gradients. The
native nnUNet lifecycle was also rechecked on 32³ synthetic patches for both
label protocols: loading/augmentation, training/validation, exact checkpoint
prediction reload, resumed update and sliding-window inference passed. These
CPU checks alone do not establish CUDA behavior; the short CUDA checks above
add direct evidence but still do not establish late-epoch convergence.
The inspected historical RC3 logs did not identify the reported AMP failure;
its exact checkpoint/batch/first failing operation remains unknown. A later
rise in feature energy crossing the FP16 sum limit is a plausible mechanism,
not a demonstrated diagnosis of that historical run.

The source of the small-patch regression was identified with controlled 96³,
batch-2 CUDA measurements (same GPU and AMP; each model's complete objective):

| Architecture | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|
| Previous automatically generated stride-1 VeloxSeg | 8.519 | 9.637 |
| ResEnc with its standard blocks, SGD and segmentation loss | 2.781 | 2.977 |
| Explicit historical compact VeloxSeg configuration, current AMP/loss | 0.392 | 0.455 |

The old automatic configuration was not representative of VeloxSeg's compact
training cost. Its full-resolution multi-branch features, changed depth and
window configuration jointly increased memory. This comparison does not isolate
all of the difference to one stride parameter. The selected correction keeps
geometry-driven stages and pooling but restores compact entry and PWA from the
first retained stage. At 160×224×192/batch2, the corrected dynamic model uses
2.753 GiB allocated, demonstrating that large patches need not inherit that
regression. Historical checkpoints remain incompatible with these dynamic plans.

The earlier CPU saved-tensor attribution (14.651 GiB with fullres heads,
14.569 with native heads, 8.157 with AMP) describes the **withdrawn stride-1
configuration**. It explains the precision improvement but is not the current
model's resource baseline. Native supervision alone saved only about 85 MiB;
architecture correction was the larger issue.

Primary implementation references: [planner](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py),
[ResEnc planner](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py),
[trainer](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py),
[preprocessor](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/preprocessing/preprocessors/default_preprocessor.py),
[predictor](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/inference/predict_from_raw_data.py),
[postprocessing](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.2/nnunetv2/postprocessing/remove_connected_components.py).
