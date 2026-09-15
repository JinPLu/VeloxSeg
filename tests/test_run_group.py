"""run_group.sh on a simulated Linux GPU node, and the resource plan it launches tasks with."""
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'nnunet' / 'scripts'
BUNDLED = ROOT / 'nnunet' / 'config'
_spec = importlib.util.spec_from_file_location('group_resources', SCRIPTS / 'group_resources.py')
resources = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resources)
GIB = resources.GIB

# Python started by the scripts sees a Linux node: CPU affinity, CUDA GPUs and the
# /proc and cgroup files come from FAKE_NODE.
SITECUSTOMIZE = '''
import builtins, io, json, os, types
import torch

node = os.environ['FAKE_NODE']
real_open = io.open
with real_open(os.path.join(node, 'node.json')) as file:
    spec = json.load(file)


def node_open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)) and os.fspath(file).startswith(('/proc/', '/sys/fs/cgroup/')):
        file = node + os.fspath(file)
    return real_open(file, *args, **kwargs)


io.open = builtins.open = node_open
os.sched_getaffinity = lambda pid: set(range(spec['cpus']))
torch.cuda.device_count = lambda: spec['gpus']
torch.cuda.get_device_properties = lambda index: types.SimpleNamespace(uuid=f'fake-{index}')
'''
FAKE_TRAIN = '''#!/bin/bash
echo "start $* CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nnUNet_n_proc_DA=$nnUNet_n_proc_DA" >> "$FAKE_NODE/calls"
sleep "$FAKE_TRAIN_SECONDS" &
echo "$$ $!" >> "$FAKE_NODE/pids"
wait $!
echo "end $1" >> "$FAKE_NODE/calls"
[[ $1 != "$FAKE_FAIL_DATASET" ]]
'''
# macOS has no setsid(1).
SETSID = '''#!/bin/bash
exec python -S -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"
'''


def bundled_task(dataset, plans='nnVeloxSegPlans', configuration='3d_fullres_B'):
    """Batch size and bytes of one input sample of a bundled configuration."""
    folder = next(BUNDLED.glob(f'Dataset{dataset:03d}_*'))
    config = json.loads((folder / f'{plans}.json').read_text())['configurations'][configuration]
    channels = len(json.loads((folder / 'dataset.json').read_text())['channel_names'])
    return config['batch_size'], channels * math.prod(config['patch_size']) * 4


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class GroupResourcesTests(unittest.TestCase):
    def test_ranks_follow_batch_and_cpu_workers_share_the_cpus(self):
        _, sample = bundled_task(137)
        unlimited = 10 ** 6 * GIB
        self.assertEqual(resources.plan(8, sample, 8, 104, unlimited)[:2], (8, 12))
        self.assertEqual(resources.plan(16, sample, 8, 104, unlimited)[:2], (8, 12))
        self.assertEqual(resources.plan(2, sample, 8, 104, unlimited)[:2], (2, 51))
        self.assertEqual(resources.plan(8, sample, 1, 104, unlimited)[:2], (1, 103))
        self.assertEqual(resources.plan(8, sample, 8, 8, unlimited)[:2], (8, 1))

    def test_memory_workers_are_the_most_that_fit_the_budget(self):
        batch, sample = bundled_task(221)
        memory = 100 * GIB
        ranks, cpu_workers, memory_workers, workers, rank_bytes = resources.plan(batch, sample, 8, 104, memory)
        budget = memory * (1 - resources.RESERVE) / ranks
        rank_batch = math.ceil(batch / ranks)
        self.assertLessEqual(resources.rank_memory(memory_workers, rank_batch, sample), budget)
        self.assertGreater(resources.rank_memory(memory_workers + 1, rank_batch, sample), budget)
        self.assertLess(memory_workers, cpu_workers)
        self.assertEqual(workers, memory_workers)
        self.assertEqual(rank_bytes, resources.rank_memory(workers, rank_batch, sample))


class RunGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.node = self.tmp / 'node'
        (self.node / 'proc').mkdir(parents=True)
        (self.node / 'sys' / 'fs' / 'cgroup').mkdir(parents=True)
        bin_dir = self.tmp / 'bin'
        bin_dir.mkdir()
        for name, text in (('nnUNetv2_train', FAKE_TRAIN), ('setsid', SETSID)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        site = self.tmp / 'site'
        site.mkdir()
        (site / 'sitecustomize.py').write_text(SITECUSTOMIZE)
        preprocessed = self.tmp / 'preprocessed'
        for folder in BUNDLED.glob('Dataset*'):
            shutil.copytree(folder, preprocessed / folder.name)
        for name in ('raw', 'results'):
            (self.tmp / name).mkdir()
        self.env = dict(
            os.environ, PATH=os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), '/usr/bin', '/bin']),
            PYTHONPATH=os.pathsep.join([str(site), os.environ.get('PYTHONPATH', '')]), FAKE_NODE=str(self.node),
            FAKE_TRAIN_SECONDS='0', FAKE_FAIL_DATASET='', nnUNet_raw=str(self.tmp / 'raw'),
            nnUNet_preprocessed=str(preprocessed), nnUNet_results=str(self.tmp / 'results'))

    def configure_node(self, cpus, cpu_max, available_gib, limit_gib, gpus=8):
        (self.node / 'node.json').write_text(json.dumps({'cpus': cpus, 'gpus': gpus}))
        (self.node / 'proc' / 'meminfo').write_text(
            f'MemTotal: {377 * 2 ** 20} kB\nMemFree: {10 * 2 ** 20} kB\nMemAvailable: {available_gib * 2 ** 20} kB\n')
        (self.node / 'sys' / 'fs' / 'cgroup' / 'cpu.max').write_text(f'{cpu_max}\n')
        (self.node / 'sys' / 'fs' / 'cgroup' / 'memory.max').write_text(f'{limit_gib * GIB}\n')

    def start(self, lines):
        tasks = self.tmp / 'tasks.txt'
        tasks.write_text('# dataset_id configuration fold plans_identifier\n\n' + '\n'.join(lines) + '\n')
        return subprocess.Popen(['bash', str(SCRIPTS / 'run_group.sh'), str(tasks)], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def calls(self):
        path = self.node / 'calls'
        return path.read_text().splitlines() if path.exists() else []

    def test_tasks_run_in_order_each_on_the_whole_node(self):
        # cgroup v2 quota of 52 CPUs on 104 threads; MemAvailable 317 GiB under a 320 GiB limit.
        self.configure_node(cpus=104, cpu_max='5200000 100000', available_gib=317, limit_gib=320)
        self.env.update(FAKE_TRAIN_SECONDS='1', FAKE_FAIL_DATASET='221')
        group = self.start(['137 3d_fullres_B 4 nnVeloxSegPlans --c', '221 3d_fullres_B 0 nnVeloxSegPlans',
                            '990 3d_fullres_B 4 nnVeloxSegPlans'])
        output, _ = group.communicate(timeout=600)
        self.assertEqual(group.returncode, 1, output)

        calls = self.calls()
        self.assertEqual([' '.join(call.split()[:2]) for call in calls],
                         ['start 137', 'end 137', 'start 221', 'end 221', 'start 990', 'end 990'])
        self.assertIn('-p nnVeloxSegPlans --c -num_gpus 8 ', calls[0])
        # Bundled batches: BraTS 8, AutoPET 2, Hecktor 4.
        for call, dataset, ranks, cpu_workers in zip(calls[::2], (137, 221, 990), (8, 2, 4), (5, 25, 12)):
            batch, sample = bundled_task(dataset)
            expected = resources.plan(batch, sample, 8, 52, 317 * GIB)
            self.assertEqual(expected[:2], (ranks, cpu_workers))
            devices = ','.join(f'GPU-fake-{index}' for index in range(ranks))
            self.assertIn(f'-num_gpus {ranks} CUDA_VISIBLE_DEVICES={devices} nnUNet_n_proc_DA={expected[3]}', call)
            self.assertIn(f'ranks={ranks} workers_by_cpu={cpu_workers} workers_by_memory={expected[2]} '
                          f'nnUNet_n_proc_DA={expected[3]}', output)
        # AutoPET's memory bound is below its CPU share, so both bounds were exercised.
        self.assertLess(resources.plan(*bundled_task(221), 8, 52, 317 * GIB)[2], 25)
        self.assertIn('failed 221_nnVeloxSegPlans_3d_fullres_B_fold0', output)
        self.assertEqual(sorted(path.name for path in (self.tmp / 'results' / 'group_logs').iterdir()),
                         ['137_nnVeloxSegPlans_3d_fullres_B_fold4.log', '221_nnVeloxSegPlans_3d_fullres_B_fold0.log',
                          '990_nnVeloxSegPlans_3d_fullres_B_fold4.log'])

    def test_cgroup_memory_limit_bounds_workers_without_cpu_quota(self):
        self.configure_node(cpus=104, cpu_max='max 100000', available_gib=317, limit_gib=60)
        group = self.start(['990 3d_fullres_B 4 nnVeloxSegPlans'])
        output, _ = group.communicate(timeout=300)
        self.assertEqual(group.returncode, 0, output)
        expected = resources.plan(*bundled_task(990), 8, 104, 60 * GIB)
        self.assertEqual(expected[:2], (4, 25))
        self.assertLess(expected[3], 25)
        self.assertTrue(self.calls()[0].endswith(f'nnUNet_n_proc_DA={expected[3]}'), self.calls())

    def test_invalid_task_stops_before_any_training(self):
        self.configure_node(cpus=104, cpu_max='max 100000', available_gib=317, limit_gib=320)
        group = self.start(['990 3d_fullres_B 4 nnVeloxSegPlans', '137 3d_fullres_B 4 MissingPlans'])
        output, _ = group.communicate(timeout=300)
        self.assertEqual(group.returncode, 2, output)
        self.assertIn('Invalid task, missing plans or no memory for one worker: 137 3d_fullres_B 4 MissingPlans',
                      output)
        self.assertEqual(self.calls(), [])

    def test_interrupt_terminates_the_running_task_process_group(self):
        self.configure_node(cpus=104, cpu_max='max 100000', available_gib=317, limit_gib=320)
        self.env['FAKE_TRAIN_SECONDS'] = '300'
        group = self.start(['990 3d_fullres_B 4 nnVeloxSegPlans', '990 3d_fullres_B 0 nnVeloxSegPlans'])
        pids_file = self.node / 'pids'
        deadline = time.monotonic() + 300
        while not (pids_file.exists() and len(pids_file.read_text().split()) == 2):
            self.assertIsNone(group.poll(), 'run_group.sh exited before the task started')
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.2)
        # The fake trainer and its child sleep, both inside the task's session.
        pids = [int(pid) for pid in pids_file.read_text().split()]
        group.send_signal(signal.SIGINT)
        output, _ = group.communicate(timeout=60)
        self.assertEqual(group.returncode, 130, output)
        deadline = time.monotonic() + 10
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual([pid for pid in pids if alive(pid)], [])
        self.assertEqual(len(self.calls()), 1)


if __name__ == '__main__':
    unittest.main()
