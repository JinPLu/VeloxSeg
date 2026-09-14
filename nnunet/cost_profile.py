"""Batch-1 CUDA inference cost of VeloxSeg candidate crops or final plans.

Every (patch, size) runs in a fresh subprocess: FP32 weights and input on the
GPU, FP16 autocast, eval + inference_mode segmentation forward with the whole
model resident (reconstruction decoder weights included), warmup forwards,
CUDA-event median/p90 latency over the timed forwards, and peak allocated and
reserved memory after a reset. Plans are read, never modified.

  python nnunet/cost_profile.py --candidates veloxseg_candidates.json --output veloxseg_profile.json
  python nnunet/cost_profile.py --plans nnVeloxSegPlans.json --dataset-json dataset.json --output cost_report.json
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

MEASUREMENT = {'warmup': 80, 'repeats': 120, 'seed': 12345}


def measure(spec):
    import numpy as np
    import torch
    from model.VeloxSeg import VeloxSeg

    torch.manual_seed(MEASUREMENT['seed'])
    torch.backends.cudnn.benchmark = True
    kwargs = spec['architecture']['arch_kwargs']
    network = VeloxSeg(**kwargs, n_classes=spec['n_classes']).cuda().eval()
    x = torch.randn(1, sum(kwargs['in_ch']), *kwargs['input_size'], device='cuda')
    times = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
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
    return {
        'patch': list(kwargs['input_size']), 'size': spec['size'], 'architecture': kwargs,
        'parameters': sum(p.numel() for p in network.parameters()),
        'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
    }


def candidate_specs(path, sizes):
    manifest = json.loads(path.read_text())
    return [{'size': size, 'architecture': row['architectures'][size], 'n_classes': manifest['n_classes']}
            for row in manifest['candidates'] if row['trainable'] for size in sizes]


def plan_specs(path, dataset_json):
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    plans = json.loads(path.read_text())
    dataset = json.loads(dataset_json.read_text())
    n_classes = LabelManager(dataset['labels'], dataset.get('regions_class_order')).num_segmentation_heads
    return [{'size': size, 'architecture': plans['configurations'][f'3d_fullres_{size}']['architecture'],
             'n_classes': n_classes} for size in 'SBL']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--candidates', type=Path, help='veloxseg_planner candidates output')
    source.add_argument('--plans', type=Path, help='Final nnVeloxSegPlans.json; S/B/L are reported')
    source.add_argument('--measure', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--size', nargs='+', choices='SBL', default=['B'], help='Candidate model sizes')
    parser.add_argument('--dataset-json', type=Path, help='dataset.json of the plans')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.measure:
        print(json.dumps(measure(json.loads(sys.stdin.read()))), flush=True)
        return
    if args.output is None:
        parser.error('--output is required')
    if args.plans and args.dataset_json is None:
        parser.error('--plans requires --dataset-json')
    specs = candidate_specs(args.candidates, args.size) if args.candidates else plan_specs(args.plans, args.dataset_json)
    rows = {}
    for spec in specs:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--measure'],
                                 input=json.dumps(spec), stdout=subprocess.PIPE, text=True, check=True)
        row = json.loads(process.stdout.splitlines()[-1])
        key = 'x'.join(str(n) for n in row['patch'])
        rows.setdefault(key, {})[row['size']] = row
        print(f"{key} {row['size']}: median {row['median_ms']:.2f} ms, p90 {row['p90_ms']:.2f} ms, "
              f"peak allocated {row['peak_allocated_mib']:.2f} MiB", flush=True)
    first = next(iter(next(iter(rows.values())).values()))
    report = {'gpu': first['gpu'], 'torch': first['torch'], 'cuda': first['cuda'],
              'measurement': {**MEASUREMENT, 'batch': 1,
                              'precision': 'FP32 resident weights and input; FP16 autocast',
                              'scope': 'eval inference_mode segmentation forward; RC decoder weights resident'},
              'source': str(args.candidates or args.plans), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
