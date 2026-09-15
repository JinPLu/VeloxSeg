import math
import socket
import time
import unittest
from datetime import timedelta

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from model.loss import VeloxSegLoss
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnVeloxSegTrainer import NUMERIC_GUARD, nnVeloxSegTrainer

LIMIT = NUMERIC_GUARD['max_consecutive_skipped_steps']


def gram(features):
    flat = features.flatten(2)
    return flat @ flat.transpose(1, 2) / (flat.shape[1] * flat.shape[2])


class OutputContract(nn.Module):
    """Smallest network with VeloxSeg's one-modality output layout: two
    segmentation scales, reconstruction, decoder Gram and one teacher Gram."""

    def __init__(self):
        super().__init__()
        self.segmentation = nn.Conv3d(1, 2, 3, padding=1)
        self.reconstruction = nn.Conv3d(1, 1, 3, padding=1)
        self.student = nn.Conv3d(1, 4, 1)
        self.teacher = nn.Conv3d(1, 4, 1)

    def forward(self, x):
        segmentation = self.segmentation(x)
        return [segmentation, F.avg_pool3d(segmentation, 2), self.reconstruction(x),
                gram(self.student(x)), gram(self.teacher(x))]


def cross_entropy(prediction, target):
    return F.cross_entropy(prediction, target[:, 0].long())


def make_trainer():
    """Trainer instance without nnU-Net's dataset setup; attributes mirror __init__/initialize."""
    torch.manual_seed(0)
    trainer = object.__new__(nnVeloxSegTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = OutputContract()
    trainer.loss = VeloxSegLoss(cross_entropy, [1])
    trainer.optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=1e-2)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1.0)
    trainer.grad_scaler = None
    trainer.logger = MetaLogger(None, False)
    trainer.is_ddp = False
    trainer.current_epoch = 0
    trainer.consecutive_skipped_steps = 0
    trainer.log_lines = []
    trainer.print_to_log_file = lambda *args, **kwargs: trainer.log_lines.append(' '.join(map(str, args)))
    trainer.on_train_epoch_start()
    return trainer


def batch(nan=False):
    data = torch.randn(2, 1, 8, 8, 8, generator=torch.Generator().manual_seed(1))
    if nan:
        data[0, 0, 0, 0, 0] = math.nan
    return {'data': data, 'target': (data > 0).float()}


def snapshot(trainer):
    return {name: tensor.clone() for name, tensor in trainer.network.state_dict().items()}


def changed(before, trainer):
    return [name for name, tensor in trainer.network.state_dict().items() if not torch.equal(before[name], tensor)]


class StepGuardTests(unittest.TestCase):
    def test_finite_step_updates_weights(self):
        trainer = make_trainer()
        before = snapshot(trainer)
        output = trainer.train_step(batch())
        record = trainer.step_records[-1]
        self.assertTrue(math.isfinite(float(output['loss'])))
        self.assertTrue(math.isfinite(record['grad_norm']))
        self.assertAlmostEqual(record['loss'], record['segmentation'] + record['reconstruction'] + record['sdkt'],
                               places=5)
        self.assertEqual(sorted(changed(before, trainer)), sorted(before))
        self.assertEqual(trainer.consecutive_skipped_steps, 0)

    def test_nan_loss_is_skipped_before_backward_and_counted(self):
        trainer = make_trainer()
        outputs = [trainer.train_step(batch())]
        before = snapshot(trainer)
        outputs.append(trainer.train_step(batch(nan=True)))
        self.assertTrue(math.isnan(float(outputs[-1]['loss'])))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.network.parameters()))
        self.assertEqual(changed(before, trainer), [])
        self.assertEqual({state['step'].item() for state in trainer.optimizer.state.values()}, {1})
        self.assertEqual(trainer.consecutive_skipped_steps, 1)

        outputs.append(trainer.train_step(batch()))
        self.assertEqual(trainer.consecutive_skipped_steps, 0)
        trainer.on_train_epoch_end(outputs)
        log = '\n'.join(trainer.log_lines)
        self.assertIn('Optimizer steps applied: 2, skipped: 1 (non-finite loss: 1, non-finite gradient norm: 0), '
                      'consecutive skipped: 0', log)
        self.assertIn('Pre-clip gradient norm: median', log)
        self.assertRegex(log, r'Weighted loss terms, mean over finite steps: segmentation \S+ \(non-finite 1\), '
                              r'reconstruction \S+ \(non-finite 1\), sdkt \S+ \(non-finite 1\)')

    def test_inf_gradient_skips_update(self):
        trainer = make_trainer()
        before = snapshot(trainer)
        hook = trainer.network.reconstruction.weight.register_hook(lambda grad: torch.full_like(grad, math.inf))
        output = trainer.train_step(batch())
        self.assertTrue(math.isfinite(float(output['loss'])))
        self.assertEqual(trainer.step_records[-1]['grad_norm'], math.inf)
        self.assertEqual(changed(before, trainer), [])
        self.assertEqual(trainer.optimizer.state, {})

        hook.remove()
        trainer.train_step(batch())
        self.assertEqual(sorted(changed(before, trainer)), sorted(before))

    def test_consecutive_skips_across_epochs_raise(self):
        trainer = make_trainer()
        outputs = [trainer.train_step(batch(nan=True)) for _ in range(LIMIT // 2)]
        trainer.on_train_epoch_end(outputs)
        trainer.current_epoch = 1
        trainer.on_train_epoch_start()
        for _ in range(LIMIT - LIMIT // 2):
            trainer.train_step(batch(nan=True))
        with self.assertRaisesRegex(RuntimeError, f'{LIMIT + 1} consecutive optimizer steps were skipped'):
            trainer.train_step(batch(nan=True))
        self.assertIn('Numerically broken training', trainer.log_lines[-1])


def ddp_rank(rank, port, results):
    """One gloo rank: a NaN batch on rank 1 only, then a finite batch on both ranks."""
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    trainer = make_trainer()
    trainer.network = DistributedDataParallel(trainer.network)
    trainer.is_ddp = True
    before = snapshot(trainer)
    trainer.train_step(batch(nan=rank == 1))
    skipped = {'record': trainer.step_records[-1], 'changed': changed(before, trainer),
               'no_grads': all(parameter.grad is None for parameter in trainer.network.parameters())}
    trainer.train_step(batch())
    applied = {'record': trainer.step_records[-1], 'changed': changed(before, trainer),
               'weights': [tensor.double().sum().item() for tensor in trainer.network.state_dict().values()]}
    results.put((rank, skipped, applied))
    dist.destroy_process_group()


class DDPStepGuardTests(unittest.TestCase):
    def test_nonfinite_loss_on_one_rank_skips_the_step_on_every_rank(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        results = mp.get_context('spawn').SimpleQueue()
        ranks = mp.start_processes(ddp_rank, args=(port, results), nprocs=2, join=False, start_method='spawn')
        deadline = time.monotonic() + 120
        while not ranks.join(timeout=5):
            if time.monotonic() > deadline:
                for process in ranks.processes:
                    process.kill()
                self.fail('DDP ranks did not finish: a rank is blocked in a collective')
        outcomes = {rank: (skipped, applied) for rank, skipped, applied in (results.get() for _ in range(2))}

        self.assertTrue(math.isfinite(outcomes[0][0]['record']['loss']))
        self.assertTrue(math.isnan(outcomes[1][0]['record']['loss']))
        for skipped, applied in outcomes.values():
            self.assertFalse(skipped['record']['finite_loss'])
            self.assertTrue(math.isnan(skipped['record']['grad_norm']))
            self.assertEqual(skipped['changed'], [])
            self.assertTrue(skipped['no_grads'])
            self.assertTrue(applied['record']['finite_loss'])
            self.assertTrue(math.isfinite(applied['record']['grad_norm']))
            self.assertTrue(applied['changed'])
        self.assertEqual(outcomes[0][1]['record']['grad_norm'], outcomes[1][1]['record']['grad_norm'])
        self.assertEqual(outcomes[0][1]['weights'], outcomes[1][1]['weights'])


class EpochStateTests(unittest.TestCase):
    def test_nonfinite_weights_raise_at_epoch_end(self):
        trainer = make_trainer()
        outputs = [trainer.train_step(batch())]
        with torch.no_grad():
            trainer.network.teacher.bias[0] = math.inf
        with self.assertRaisesRegex(RuntimeError, r"1 network tensors hold non-finite values .*teacher\.bias"):
            trainer.on_train_epoch_end(outputs)

    def test_finite_weights_pass_epoch_end(self):
        trainer = make_trainer()
        trainer.on_train_epoch_end([trainer.train_step(batch())])
        self.assertIn('Optimizer steps applied: 1, skipped: 0', '\n'.join(trainer.log_lines))


if __name__ == '__main__':
    unittest.main()
