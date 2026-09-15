"""Size one run_group.sh task on this node.

python group_resources.py DATASET_ID CONFIGURATION PLANS_IDENTIFIER prints
"ranks cpu_workers memory_workers workers rank_memory_gib cuda_visible_devices":
the DDP ranks, the augmentation workers per rank the CPUs and the memory allow,
the nnUNet_n_proc_DA to use (the smaller), the estimated peak memory of one rank
at that worker count, and the GPUs of the task.
"""
import math
import os
import sys
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnVeloxSegTrainer import NUM_CACHED_BATCHES
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

GIB = 2 ** 30
# Share of the node memory left out of the plan.
RESERVE = 0.12

# Host memory of one training rank at its peak, measured on an RTX 3090 node
# (nnU-Net 2.8.1, 2026-09-15; reported GB read as GiB). The peak is torch.compile:
# the GPU takes no batches, so every queue is full and every worker holds a batch.
MEASURED = {
    # BraTS coverage-65% plans on one GPU: batch 16 x 4 channels x 160x160x128, 12 workers.
    'brats_batch_bytes': 16 * 4 * 160 * 160 * 128 * 4,
    # Upstream num_cached of the training and validation augmenters at 12 workers.
    'brats_cached': (6, 3),
    'main_process_gib': 17.3,  # PSS, pinned batches included
    'pinned_gib': 15.6,
    'shm_gib': 7.0,  # /dev/shm: batches in the worker->main queues
    'compile_workers_gib': 2.8,  # PSS of the 33 Inductor compile workers
    # AutoPET default plans: batch 2 x 2 channels x 256x320x320. Largest
    # augmentation worker, holding a finished batch.
    'autopet_sample_bytes': 2 * 256 * 320 * 320 * 4,
    'autopet_batch': 2,
    'autopet_worker_gib': 4.55,
}


def pinned_slots(cached):
    # Per augmenter: its pinned queue plus the batch its pin thread holds; plus the batch in train_step.
    return sum(depth + 1 for depth in cached) + 1


def shm_slots(cached):
    # Per augmenter: its worker->main multiprocessing queue.
    return sum(cached)


# Per-byte factors are relative to batch bytes = batch x channels x patch voxels x 4.
_shm = MEASURED['shm_gib'] * GIB / (shm_slots(MEASURED['brats_cached']) * MEASURED['brats_batch_bytes'])
MODEL = {
    # Main process without its pinned batches, plus the compile workers.
    'rank_fixed_bytes': (MEASURED['main_process_gib'] - MEASURED['pinned_gib'] + MEASURED['compile_workers_gib']) * GIB,
    'pinned_per_batch_byte': MEASURED['pinned_gib'] * GIB / (
        pinned_slots(MEASURED['brats_cached']) * MEASURED['brats_batch_bytes']),
    'shm_per_batch_byte': _shm,
    # A worker's finished batch is the tensors it later moves to shared memory;
    # the rest of the worker scales with the sample it augments.
    'worker_per_sample_byte': (MEASURED['autopet_worker_gib'] * GIB
                               - _shm * MEASURED['autopet_batch'] * MEASURED['autopet_sample_bytes'])
                              / MEASURED['autopet_sample_bytes'],
}


def rank_memory(workers, rank_batch, sample_bytes):
    """Peak bytes of one rank running `workers` training augmentation workers."""
    cached = (NUM_CACHED_BATCHES, NUM_CACHED_BATCHES)
    batch_bytes = rank_batch * sample_bytes
    # nnUNetTrainer.get_dataloaders gives validation max(1, workers // 2) workers.
    processes = workers + max(1, workers // 2)
    return (MODEL['rank_fixed_bytes']
            + (pinned_slots(cached) * MODEL['pinned_per_batch_byte']
               + shm_slots(cached) * MODEL['shm_per_batch_byte']) * batch_bytes
            + processes * (MODEL['worker_per_sample_byte'] * sample_bytes
                           + MODEL['shm_per_batch_byte'] * batch_bytes))


def plan(batch_size, sample_bytes, gpus, cpus, memory):
    # ranks <= batch_size keeps nnU-Net's DDP requirement (global batch >= world size).
    ranks = min(gpus, batch_size)
    # One CPU per rank stays with its training process.
    cpu_workers = max(1, cpus // ranks - 1)
    # nnU-Net splits the batch over ranks; the largest share is ceil(batch / ranks).
    rank_batch = math.ceil(batch_size / ranks)
    budget = memory * (1 - RESERVE) / ranks
    memory_workers = 0
    while rank_memory(memory_workers + 1, rank_batch, sample_bytes) <= budget:
        memory_workers += 1
    workers = min(cpu_workers, memory_workers)
    return ranks, cpu_workers, memory_workers, workers, rank_memory(workers, rank_batch, sample_bytes)


def read(path):
    try:
        return Path(path).read_text().split()
    except FileNotFoundError:
        return None


def cpu_count():
    """CPUs of this process (what nproc counts, without its OMP_NUM_THREADS override), capped by a cgroup quota."""
    cpus = len(os.sched_getaffinity(0))
    quota = read('/sys/fs/cgroup/cpu.max')  # cgroup v2: "max 100000" or "<quota> <period>"
    if quota is None:
        v1 = read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        quota = v1 and v1 + read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota is None or quota[0] in ('max', '-1'):
        return cpus
    return min(cpus, max(1, int(quota[0]) // int(quota[1])))


def memory_bytes():
    """The smaller of the cgroup memory limit and the node's MemAvailable."""
    meminfo = read('/proc/meminfo')
    available = int(meminfo[meminfo.index('MemAvailable:') + 1]) * 1024
    limit = read('/sys/fs/cgroup/memory.max') or read('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if limit is None or limit[0] == 'max':
        return available
    return min(available, int(limit[0]))


def main():
    dataset_id, configuration, plans_identifier = sys.argv[1:]
    # UUID selectors keep the platform's GPU assignment.
    gpus = [f'GPU-{torch.cuda.get_device_properties(index).uuid}' for index in range(torch.cuda.device_count())]
    if not gpus:
        sys.exit('No visible GPU for this job.')
    dataset = maybe_convert_to_dataset_name(dataset_id)
    plans_manager = PlansManager(join(nnUNet_preprocessed, dataset, f'{plans_identifier}.json'))
    configuration_manager = plans_manager.get_configuration(configuration)
    dataset_json = load_json(join(nnUNet_preprocessed, dataset, 'dataset.json'))
    channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    sample_bytes = channels * math.prod(configuration_manager.patch_size) * 4
    ranks, cpu_workers, memory_workers, workers, rank_bytes = plan(
        configuration_manager.batch_size, sample_bytes, len(gpus), cpu_count(), memory_bytes())
    if memory_workers == 0:
        sys.exit(f'{dataset} {configuration}: one augmentation worker per rank needs '
                 f'{rank_memory(1, math.ceil(configuration_manager.batch_size / ranks), sample_bytes) / GIB:.1f} GiB, '
                 f'more than the memory budget of {memory_bytes() * (1 - RESERVE) / ranks / GIB:.1f} GiB per rank')
    print(ranks, cpu_workers, memory_workers, workers, f'{rank_bytes / GIB:.1f}', ','.join(gpus[:ranks]))


if __name__ == '__main__':
    main()
