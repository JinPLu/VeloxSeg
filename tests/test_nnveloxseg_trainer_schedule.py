import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from model.VeloxSeg import VeloxSeg
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import architecture_for_patch, training_policy
from nnunetv2.training.nnUNetTrainer import nnVeloxSegTrainer as trainer_module
from nnunetv2.training.nnUNetTrainer.nnVeloxSegTrainer import NORM_MODULES, NUM_CACHED_BATCHES, nnVeloxSegTrainer

INITIAL_LR = 1e-3
MINIMUM_LR = 6e-6
# The planned length at batch 8: 250 epochs of 250 updates.
PLANNED = training_policy(8)
NUM_EPOCHS = PLANNED['num_epochs']
ITERATIONS = PLANNED['num_iterations_per_epoch']


def small_veloxseg():
    arch = architecture_for_patch([64, 64, 64], [1.0, 1.0, 1.0], [1, 1], 'B')
    return VeloxSeg(**arch['arch_kwargs'], n_classes=2, deep_supervision=True)


def configure(network, warmup_updates=0, decay_exclusions=False):
    training = {'warmup_updates': warmup_updates, 'initial_lr': INITIAL_LR, 'minimum_lr': MINIMUM_LR,
                'weight_decay': 1e-2, 'decay_exclusions': decay_exclusions}
    stand_in = SimpleNamespace(network=network, num_epochs=NUM_EPOCHS, num_iterations_per_epoch=ITERATIONS,
                               configuration_manager=SimpleNamespace(configuration={'training': training}))
    optimizer, scheduler = nnVeloxSegTrainer.configure_optimizers(stand_in)
    return stand_in, optimizer, scheduler


def lr_at(scheduler, optimizer, epoch):
    # nnU-Net's on_train_epoch_start calls step(current_epoch).
    scheduler.step(epoch)
    return optimizer.param_groups[0]['lr']


def cosine(progress):
    return MINIMUM_LR + (INITIAL_LR - MINIMUM_LR) * (1 + math.cos(math.pi * progress)) / 2


class ScheduleTests(unittest.TestCase):
    def test_no_warmup_is_cosine_from_initial_to_minimum(self):
        stand_in, optimizer, scheduler = configure(nn.Linear(2, 2), warmup_updates=0)
        self.assertEqual(stand_in.warmup_epochs, 0)
        self.assertAlmostEqual(lr_at(scheduler, optimizer, 0), INITIAL_LR)
        self.assertAlmostEqual(lr_at(scheduler, optimizer, 1), cosine(1 / (NUM_EPOCHS - 1)))
        self.assertAlmostEqual(lr_at(scheduler, optimizer, NUM_EPOCHS - 1), MINIMUM_LR)

    def test_warmup_updates_resolve_to_epochs(self):
        # 2% of the 62,500 planned updates.
        stand_in, optimizer, scheduler = configure(nn.Linear(2, 2), warmup_updates=1250)
        self.assertEqual(stand_in.warmup_epochs, 5)
        self.assertAlmostEqual(lr_at(scheduler, optimizer, 0), INITIAL_LR / 5)
        self.assertAlmostEqual(lr_at(scheduler, optimizer, 4), INITIAL_LR)
        self.assertAlmostEqual(lr_at(scheduler, optimizer, 5), cosine(1 / (NUM_EPOCHS - 5)))
        self.assertAlmostEqual(lr_at(scheduler, optimizer, NUM_EPOCHS - 1), MINIMUM_LR)

    def test_partial_warmup_epoch_rounds_up(self):
        stand_in, _, _ = configure(nn.Linear(2, 2), warmup_updates=251)
        self.assertEqual(stand_in.warmup_epochs, 2)

    def test_warmup_must_leave_decay_epochs(self):
        with self.assertRaisesRegex(ValueError, 'warmup_updates'):
            configure(nn.Linear(2, 2), warmup_updates=PLANNED['total_updates'])


class DataloaderCacheTests(unittest.TestCase):
    def test_training_and_validation_augmenters_cache_the_same_small_queue(self):
        augmenters = []

        class Augmenter:
            def __init__(self, **kwargs):
                augmenters.append(kwargs)

            def __next__(self):
                return None

        labels = SimpleNamespace(foreground_labels=[1], has_regions=False, foreground_regions=None, ignore_label=None)
        stand_in = SimpleNamespace(
            dataset_class=object, configuration_manager=SimpleNamespace(patch_size=[8, 8, 8], use_mask_for_norm=[False]),
            _get_deep_supervision_scales=lambda: None,
            configure_rotation_dummyDA_mirroring_and_inital_patch_size=lambda: (None, False, [8, 8, 8], None),
            get_training_transforms=lambda *args, **kwargs: None, get_validation_transforms=lambda *args, **kwargs: None,
            get_tr_and_val_datasets=lambda: (None, None), is_cascaded=False, label_manager=labels, batch_size=2,
            oversample_foreground_percent=0.33, probabilistic_oversampling=False, device=torch.device('cuda'))
        # 12 workers: upstream nnU-Net 2.8.1 would cache 6 training and 3 validation batches.
        with mock.patch.multiple(trainer_module, NonDetMultiThreadedAugmenter=Augmenter,
                                 nnUNetDataLoader=lambda *args, **kwargs: None, get_allowed_n_proc_DA=lambda: 12):
            nnVeloxSegTrainer.get_dataloaders(stand_in)
        self.assertEqual([(kwargs['num_processes'], kwargs['num_cached'], kwargs['pin_memory']) for kwargs in augmenters],
                         [(12, NUM_CACHED_BATCHES, True), (6, NUM_CACHED_BATCHES, True)])
        self.assertEqual(NUM_CACHED_BATCHES, 2)


class DecayGroupTests(unittest.TestCase):
    def setUp(self):
        self.network = small_veloxseg()

    def test_single_group_without_exclusions(self):
        _, optimizer, _ = configure(self.network, decay_exclusions=False)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual([id(p) for p in optimizer.param_groups[0]['params']],
                         [id(p) for p in self.network.parameters()])
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-2)

    def test_exclusions_partition_every_parameter(self):
        _, optimizer, _ = configure(self.network, decay_exclusions=True)
        decayed, excluded = optimizer.param_groups
        self.assertEqual((decayed['weight_decay'], excluded['weight_decay']), (1e-2, 0.0))
        decayed_ids = [id(p) for p in decayed['params']]
        excluded_ids = [id(p) for p in excluded['params']]
        all_ids = [id(p) for p in self.network.parameters()]
        self.assertEqual(len(decayed_ids) + len(excluded_ids), len(all_ids))
        self.assertEqual(set(decayed_ids) | set(excluded_ids), set(all_ids))
        self.assertFalse(set(decayed_ids) & set(excluded_ids))

        expected_excluded, found = set(), {'norm weight': 0, 'bias': 0, 'position table': 0}
        for module in self.network.modules():
            for name, parameter in module.named_parameters(recurse=False):
                if isinstance(module, NORM_MODULES) and name == 'weight':
                    found['norm weight'] += 1
                    expected_excluded.add(id(parameter))
                if name == 'bias':
                    found['bias'] += 1
                    expected_excluded.add(id(parameter))
                if name == 'relative_position_bias_table':
                    found['position table'] += 1
                    expected_excluded.add(id(parameter))
        self.assertTrue(all(found.values()), found)
        self.assertEqual(set(excluded_ids), expected_excluded)


if __name__ == '__main__':
    unittest.main()
