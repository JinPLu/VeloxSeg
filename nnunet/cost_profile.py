"""CUDA cost of VeloxSeg candidate crops or final plans.

--candidates measures what the planner needs, in its order
(veloxseg_rules.measured_family): L training memory at the minimum batch for
every crop of an envelope until one trains within the manifest's budget; then,
per trainable crop, B batch-1 inference and L training at doubling batches up
to the first that does not fit or the dataset-voxel cap. --plans measures S/B/L
batch-1 inference of final plans, which are read, never modified.

Every measurement runs in a fresh process forked before CUDA is initialised, so
allocator state, cuDNN benchmark caches and out-of-memory failures stay apart
while the imports are shared. cudnn.benchmark is on, as in nnU-Net training.
Inference: FP32 weights and input, autocast as in nnU-Net's predictor (VeloxSeg
runs it in BF16), eval + inference_mode segmentation forward with the whole
model resident (reconstruction decoder weights included), warmup forwards,
CUDA-event median/p90 latency over the timed forwards, and peak allocated and
reserved memory after a reset.
Training: a random batch with nnU-Net's loader dtypes, nnVeloxSegTrainer's own
train_step (BF16 autocast forward, VeloxSegLoss with the dataset's segmentation
loss, backward, gradient clipping and AdamW), then its validation_step once
under eval and no_grad, as in online
validation; its FP32 loss, predictions and Dice counts can set the peak (BraTS
L 160x192x160 batch 8: 16.04 GiB with them, 14.30 GiB with the forward and loss
alone). The row records the process's peak reserved memory; CUDA out of
memory is a result.

  python nnunet/cost_profile.py --candidates veloxseg_candidates.json --output veloxseg_profile.json
  python nnunet/cost_profile.py --plans nnVeloxSegPlans.json --dataset-json dataset.json --output cost_report.json
"""
import argparse
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model.VeloxSeg import VeloxSeg
from model.loss import VeloxSegLoss
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import (
    TRAINING_POLICY, MissingMeasurement, measured_family, patch_key, segmentation_loss,
)
from nnunetv2.training.nnUNetTrainer.nnVeloxSegTrainer import nnVeloxSegTrainer
from nnunetv2.utilities.label_handling.label_handling import LabelManager

MEASUREMENT = {'warmup': 80, 'repeats': 120, 'training_steps': 3, 'seed': 12345}


def loader_batch(kwargs, labels, batch):
    """Random batch in nnU-Net's loader dtypes: float32 images; int16 labels, or bool regions plus ignore."""
    size = kwargs['input_size']
    if labels.has_regions:
        target = torch.rand(batch, labels.num_segmentation_heads + labels.has_ignore_label, *size) < 0.5
    else:
        # nnU-Net labels are consecutive; the ignore label, if any, is the highest.
        highest = labels.ignore_label if labels.has_ignore_label else max(labels.all_labels)
        target = torch.randint(highest + 1, (batch, 1, *size), dtype=torch.int16)
    return {'data': torch.randn(batch, sum(kwargs['in_ch']), *size), 'target': target}


def inference(network, kwargs):
    network.eval()
    x = torch.randn(1, sum(kwargs['in_ch']), *kwargs['input_size'], device='cuda')
    times = []
    with torch.inference_mode(), torch.autocast('cuda'):
        for _ in range(MEASUREMENT['warmup']):
            network(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(MEASUREMENT['repeats']):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            network(x)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20}


def training(network, kwargs, labels, batch):
    objective = VeloxSegLoss(segmentation_loss(labels, TRAINING_POLICY['batch_dice']), kwargs['in_ch'])
    optimizer = torch.optim.AdamW(network.parameters(), lr=TRAINING_POLICY['initial_lr'],
                                  weight_decay=TRAINING_POLICY['weight_decay'])
    sample = loader_batch(kwargs, labels, batch)
    # The attributes nnVeloxSegTrainer.train_step and validation_step read.
    trainer = SimpleNamespace(device=torch.device('cuda'), network=network, loss=objective, label_manager=labels,
                              optimizer=optimizer, grad_scaler=None,
                              step_records=[], consecutive_skipped_steps=0, current_epoch=0)
    try:
        network.train()
        for _ in range(MEASUREMENT['training_steps']):
            nnVeloxSegTrainer.train_step(trainer, sample)
        network.eval()
        with torch.no_grad():
            nnVeloxSegTrainer.validation_step(trainer, sample)
        oom = False
    except torch.OutOfMemoryError:
        oom = True
    except RuntimeError as error:
        # Memory exhausted outside the caching allocator surfaces as a library
        # failure in backward on an RTX 3090: Hecktor L 128x160x224 batch 8 raised
        # CUBLAS_STATUS_ALLOC_FAILED from cublasCreate, and BraTS L 160x192x96
        # batch 32 (batch 16: 16.5 GiB reserved) CUDNN_STATUS_INTERNAL_ERROR.
        if not any(status in str(error) for status in ('ALLOC_FAILED', 'CUDNN_STATUS_INTERNAL_ERROR')):
            raise
        oom = True
    return {'oom': oom, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20}


def measure(spec):
    torch.manual_seed(MEASUREMENT['seed'])
    torch.backends.cudnn.benchmark = True
    kwargs = spec['architecture']['arch_kwargs']
    labels = LabelManager(spec['labels'], spec['regions_class_order'])
    network = VeloxSeg(**kwargs, n_classes=labels.num_segmentation_heads).cuda()
    cost = (inference(network, kwargs) if spec['kind'] == 'inference'
            else training(network, kwargs, labels, spec['batch']))
    return {'kind': spec['kind'], 'patch': list(kwargs['input_size']), 'size': spec['size'], 'batch': spec['batch'],
            'architecture': kwargs, 'parameters': sum(p.numel() for p in network.parameters()), **cost,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda}


def isolated(spec):
    """measure(spec) in a fresh process; its traceback is printed if it fails."""
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=lambda: sender.send(measure(spec)))
    process.start()
    sender.close()
    row = receiver.recv()
    process.join()
    cost = ('out of memory, ' if row.get('oom') else
            f"median {row['median_ms']:.2f} ms, p90 {row['p90_ms']:.2f} ms, " if row['kind'] == 'inference' else '')
    print(f"{patch_key(row['patch'])} {row['size']} {row['kind']} batch {row['batch']}: {cost}"
          f"peak allocated {row['peak_allocated_mib']:.0f} MiB, reserved {row['peak_reserved_mib']:.0f} MiB",
          flush=True)
    return row


def profile_candidates(manifest):
    measurements = []
    while True:
        try:
            measured_family(manifest, {'measurements': measurements})
            return measurements
        except MissingMeasurement as missing:
            measurements.append(isolated({**missing.request, 'labels': manifest['labels'],
                                          'regions_class_order': manifest['regions_class_order']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--candidates', type=Path, help='veloxseg_planner candidates output')
    source.add_argument('--plans', type=Path, help='Final nnVeloxSegPlans.json; S/B/L inference is reported')
    parser.add_argument('--dataset-json', type=Path, help='dataset.json of the plans')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.plans and args.dataset_json is None:
        parser.error('--plans requires --dataset-json')
    if args.candidates:
        measurements = profile_candidates(json.loads(args.candidates.read_text()))
    else:
        plans = json.loads(args.plans.read_text())
        dataset = json.loads(args.dataset_json.read_text())
        measurements = [isolated({'kind': 'inference', 'size': size, 'batch': 1, 'labels': dataset['labels'],
                                  'regions_class_order': dataset.get('regions_class_order'),
                                  'architecture': plans['configurations'][f'3d_fullres_{size}']['architecture']})
                        for size in 'SBL']
    first = measurements[0]
    report = {'gpu': first['gpu'], 'torch': first['torch'], 'cuda': first['cuda'], 'measurement': MEASUREMENT,
              'source': str(args.candidates or args.plans), 'measurements': measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
