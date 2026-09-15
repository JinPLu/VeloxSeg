"""VeloxSeg planning rules on the bundled dataset metadata with synthetic measured profiles (CPU)."""
import importlib.util
import json
import tempfile
import unittest
from math import prod
from pathlib import Path
from unittest import mock

import torch

from model.VeloxSeg import VeloxSeg
from model.loss import VeloxSegLoss
from nnunetv2.experiment_planning.experiment_planners import veloxseg_planner
from nnunetv2.experiment_planning.experiment_planners.veloxseg_planner import (
    FingerprintGeometry, VeloxSegPlanner, build_plans, candidate_manifest,
)
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import (
    PATCH_POLICY, TRAINING_MEMORY_TARGET_GIB, MissingMeasurement, architecture_for_patch, measured_family,
    patch_key, segmentation_loss, select_patch,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'nnunet' / 'config'
DATASETS = ('Dataset137_BraTS2021', 'Dataset221_AutoPETII_2023', 'Dataset990_Hecktor_2022')
BRATS, AUTOPET, HECKTOR = DATASETS
# B batch-1 cost on an RTX 3090, design_review/cost_summary.json (2026-09-14).
HECKTOR_COST = {'160x256x256': (2230.00, 35.91), '128x256x256': (308.18, 29.77)}
DEVICE_MIB = 23.3 * 2 ** 10


def planner_for(dataset, dataset_json=None):
    folder = CONFIG / dataset
    return FingerprintGeometry(dataset, dataset_json or json.loads((folder / 'dataset.json').read_text()),
                               json.loads((folder / 'dataset_fingerprint.json').read_text()),
                               TRAINING_MEMORY_TARGET_GIB)


def bundled_configuration(dataset, size):
    return json.loads((CONFIG / dataset / 'nnVeloxSegPlans.json').read_text())['configurations'][f'3d_fullres_{size}']


def voxel_memory(patch, batch):
    """L training peak growing with batch voxels, out of memory (None) beyond an RTX 3090.

    The slope matches the RTX 3090 L measurement of AutoPET 256x320x256 at
    batch 2 (18.67 GiB reserved, 2026-09-15); 256x320x320 at batch 2 then
    exceeds the 22 GiB budget without running out of memory, as measured (22.50 GiB).
    """
    mib = 1024 + 4.31e-4 * batch * prod(patch)
    return None if mib > DEVICE_MIB else mib


def measured_profile(family, training_mib):
    """Answer each measurement measured_family requests, as nnunet/cost_profile.py does on a GPU."""
    profile = {'gpu': 'synthetic', 'torch': 'synthetic', 'measurements': []}
    while True:
        try:
            measured_family(family, profile)
            return profile
        except MissingMeasurement as missing:
            request = missing.request
            row = {key: request[key] for key in ('kind', 'patch', 'size', 'batch')}
            row.update(architecture=request['architecture']['arch_kwargs'], parameters=1)
            if request['kind'] == 'training':
                mib = training_mib(request['patch'], request['batch'])
                row.update(oom=mib is None, peak_reserved_mib=DEVICE_MIB if mib is None else mib)
            else:
                row.update(peak_allocated_mib=prod(request['patch']) / 2 ** 14,
                           median_ms=prod(request['patch']) / 2 ** 18)
            profile['measurements'].append(row)


def training_batches(profile, patch):
    return [row['batch'] for row in profile['measurements'] if row['kind'] == 'training' and row['patch'] == patch]


class MeasuredPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.families = {dataset: candidate_manifest(planner_for(dataset)) for dataset in DATASETS}
        cls.profiles = {dataset: measured_profile(cls.families[dataset], voxel_memory) for dataset in DATASETS}

    def test_measured_memory_keeps_bundled_patch_and_batch(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                plans = build_plans(planner_for(dataset), self.profiles[dataset])
                for size in 'SBL':
                    configuration = plans['configurations'][f'3d_fullres_{size}']
                    bundled = bundled_configuration(dataset, size)
                    self.assertEqual(configuration['patch_size'], bundled['patch_size'])
                    self.assertEqual(configuration['batch_size'], bundled['batch_size'])
                    training = configuration['training']
                    self.assertEqual(training['num_epochs'] * training['num_iterations_per_epoch'],
                                     training['total_updates'])
                    resources = configuration['resources']
                    self.assertEqual(resources['patch_selection']['profile_gpu'], 'synthetic')
                    self.assertEqual([row['patch'] for row in resources['patch_selection']['candidates']
                                      if row['selected']], [bundled['patch_size']])
                    # The recorded rows reach the chosen batch and the first batch that does not fit.
                    measured = resources['measured_training_memory']
                    self.assertEqual(measured[-1]['batch'], 2 * configuration['batch_size'])
                    self.assertTrue(measured[-1]['oom'])

    def test_untrainable_crop_is_measured_once_and_not_selected(self):
        family, profile = self.families[AUTOPET], self.profiles[AUTOPET]
        self.assertEqual(training_batches(profile, [256, 320, 320]), [2])
        envelope, batches = measured_family(family, profile)
        self.assertNotIn('256x320x320', batches)
        self.assertIn('256x320x256', batches)
        inference = [row['patch'] for row in profile['measurements'] if row['kind'] == 'inference']
        self.assertNotIn([256, 320, 320], inference)

    def test_batch_doubling_stops_at_first_batch_over_budget(self):
        family = self.families[BRATS]
        # 22.5 GiB exceeds the 22 GiB budget without running out of memory.
        profile = measured_profile(family, lambda patch, batch: 22.5 * 2 ** 10 if batch >= 16 else 2000.0)
        plans = build_plans(planner_for(BRATS), profile)
        self.assertEqual(plans['configurations']['3d_fullres_L']['batch_size'], 8)
        self.assertEqual(training_batches(profile, [160, 192, 160]), [2, 4, 8, 16])
        last = plans['configurations']['3d_fullres_L']['resources']['measured_training_memory'][-1]
        self.assertEqual((last['batch'], last['oom']), (16, False))

    def test_batch_is_capped_by_dataset_voxels(self):
        family = self.families[BRATS]
        profile = measured_profile(family, lambda patch, batch: 2000.0)
        resources = build_plans(planner_for(BRATS), profile)['configurations']['3d_fullres_B']
        self.assertEqual(resources['resources']['batch_coverage_cap'], 42)
        self.assertEqual(resources['batch_size'], 32)
        self.assertEqual(training_batches(profile, [160, 192, 160]), [2, 4, 8, 16, 32])

    def test_envelope_shrinks_until_a_crop_trains(self):
        family = self.families[HECKTOR]
        threshold = 2_000_000
        self.assertTrue(all(prod(family['candidates'][key]['patch']) > threshold for key in family['envelopes'][0]))
        profile = measured_profile(family, lambda patch, batch: None if prod(patch) > threshold else 2000.0)
        first = next(index for index, keys in enumerate(family['envelopes'])
                     if any(prod(family['candidates'][key]['patch']) <= threshold for key in keys))
        measured = {patch_key(row['patch']) for row in profile['measurements']}
        walked = {key for keys in family['envelopes'][:first + 1] for key in keys}
        self.assertEqual(measured, walked)
        patch = build_plans(planner_for(HECKTOR), profile)['configurations']['3d_fullres_B']['patch_size']
        self.assertIn(patch_key(patch), family['envelopes'][first])
        self.assertEqual(prod(patch), max(prod(family['candidates'][key]['patch']) for key in family['envelopes'][first]
                                          if prod(family['candidates'][key]['patch']) <= threshold))

    def test_coverage_fraction_prefers_cheaper_pareto_crop(self):
        family = self.families[HECKTOR]
        profile = json.loads(json.dumps(self.profiles[HECKTOR]))
        costs = {**HECKTOR_COST, '96x256x256': (400.0, 40.0)}
        for row in profile['measurements']:
            if row['kind'] == 'inference' and patch_key(row['patch']) in costs:
                row['peak_allocated_mib'], row['median_ms'] = costs[patch_key(row['patch'])]
        envelope, batches = measured_family(family, profile)
        self.assertLessEqual(set(costs), set(batches))
        self.assertEqual(select_patch(family, envelope, batches, profile)[0], '160x256x256')
        with mock.patch.dict(PATCH_POLICY, coverage_fraction=0.8):
            selected, table = select_patch(family, envelope, batches, profile)
        self.assertEqual(selected, '128x256x256')
        pareto = {patch_key(row['patch']): row['pareto'] for row in table}
        self.assertTrue(pareto['128x256x256'])
        self.assertFalse(pareto['96x256x256'])

    def test_profile_of_other_architecture_is_rejected(self):
        for kind, size in (('training', 'L'), ('inference', 'B')):
            with self.subTest(kind=kind):
                profile = json.loads(json.dumps(self.profiles[HECKTOR]))
                row = next(row for row in profile['measurements']
                           if row['kind'] == kind and row['patch'] == [160, 256, 256])
                row['architecture'] = {'input_size': [160, 256, 256], 'stages': []}
                with self.assertRaisesRegex(ValueError, f'another {size} architecture for 160x256x256'):
                    build_plans(planner_for(HECKTOR), profile)

    def test_missing_measurement_names_patch_and_command(self):
        profile = json.loads(json.dumps(self.profiles[HECKTOR]))
        profile['measurements'] = [row for row in profile['measurements']
                                   if not (row['kind'] == 'inference' and row['patch'] == [160, 256, 256])]
        with self.assertRaisesRegex(MissingMeasurement, 'lacks B inference of 160x256x256 at batch 1.*profile.py '
                                                        '--candidates'):
            build_plans(planner_for(HECKTOR), profile)

    def test_ignore_label_changes_no_planning_path(self):
        dataset_json = json.loads((CONFIG / HECKTOR / 'dataset.json').read_text())
        dataset_json['labels'] = {**dataset_json['labels'], 'ignore': 2}
        planner = planner_for(HECKTOR, dataset_json)
        plans = build_plans(planner, measured_profile(candidate_manifest(planner), voxel_memory))
        configuration = plans['configurations']['3d_fullres_B']
        self.assertEqual((configuration['patch_size'], configuration['batch_size']), ([160, 256, 256], 4))


class ProfileBatches(unittest.TestCase):
    def test_ignore_label_objective_runs_on_loader_batch(self):
        spec = importlib.util.spec_from_file_location('cost_profile', ROOT / 'nnunet' / 'cost_profile.py')
        cost_profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cost_profile)
        labels = LabelManager({'background': 0, 'tumor': 1, 'ignore': 2}, None)
        kwargs = architecture_for_patch([32, 32, 32], [1.0, 1.0, 1.0], [1, 1], 'S')['arch_kwargs']
        batch = cost_profile.loader_batch(kwargs, labels, 2)
        self.assertEqual(batch['target'].dtype, torch.int16)
        self.assertEqual(batch['target'].unique().tolist(), [0, 1, 2])
        network = VeloxSeg(**kwargs, n_classes=labels.num_segmentation_heads)
        loss = VeloxSegLoss(segmentation_loss(labels), kwargs['in_ch'])(
            network(batch['data']), batch['target'], batch['data'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


class PermutedCrops(unittest.TestCase):
    def test_axis_permuted_crops_do_not_depend_on_latency_noise(self):
        patches = ([8, 16, 12], [8, 12, 16])
        family = {'training_memory_budget_gib': 22, 'envelopes': [[patch_key(patch) for patch in patches]],
                  'candidates': {patch_key(patch): {'patch': patch, 'batch_sizes': [2], 'architectures': {
                      size: {'arch_kwargs': {'input_size': patch}} for size in 'BL'}} for patch in patches}}
        for latencies in ((20.0, 10.0), (10.0, 20.0)):
            profile = measured_profile(family, lambda patch, batch: 100.0)
            for row in profile['measurements']:
                if row['kind'] == 'inference':
                    row.update(peak_allocated_mib=100.0, parameters=7,
                               median_ms=latencies[patches.index(row['patch'])])
            envelope, batches = measured_family(family, profile)
            selected, table = select_patch(family, envelope, batches, profile)
            self.assertEqual(selected, '8x16x12')
            self.assertEqual([row['pareto'] for row in table], [True, False])


class NativePlanner(unittest.TestCase):
    def test_missing_profile_file_lists_steps(self):
        planner = VeloxSegPlanner.__new__(VeloxSegPlanner)
        planner.dataset_name = HECKTOR
        planner.UNet_vram_target_GB = TRAINING_MEMORY_TARGET_GIB
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(veloxseg_planner, 'nnUNet_preprocessed', folder), \
                mock.patch.object(veloxseg_planner, 'nnUNet_raw', folder):
            with self.assertRaisesRegex(FileNotFoundError, 'veloxseg_planner candidates(.|\n)*profile.py --candidates'
                                                           '(.|\n)*nnUNetv2_plan_and_preprocess -d 990'):
                planner.plan_experiment()


if __name__ == '__main__':
    unittest.main()
