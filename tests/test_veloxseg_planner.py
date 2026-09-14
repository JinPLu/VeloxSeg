"""VeloxSeg planning rules on the bundled dataset metadata (CPU)."""
import json
import tempfile
import unittest
from math import prod
from pathlib import Path
from unittest import mock

from nnunetv2.experiment_planning.experiment_planners import veloxseg_planner, veloxseg_rules
from nnunetv2.experiment_planning.experiment_planners.veloxseg_planner import (
    FingerprintGeometry, VeloxSegPlanner, build_plans, candidate_manifest,
)
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import (
    PATCH_POLICY, TRAINING_MEMORY_TARGET_GIB, architecture_for_patch, estimate_training_tensors, patch_key, select_patch,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager

CONFIG = Path(__file__).resolve().parents[1] / 'nnunet' / 'config'
DATASETS = ('Dataset137_BraTS2021', 'Dataset221_AutoPETII_2023', 'Dataset990_Hecktor_2022')
HECKTOR = 'Dataset990_Hecktor_2022'
# B batch-1 cost on an RTX 3090, design_review/cost_summary.json (2026-09-14).
HECKTOR_COST = {'160x256x256': (2230.00, 35.91), '128x256x256': (308.18, 29.77)}


def planner_for(dataset):
    folder = CONFIG / dataset
    return FingerprintGeometry(dataset, json.loads((folder / 'dataset.json').read_text()),
                               json.loads((folder / 'dataset_fingerprint.json').read_text()),
                               TRAINING_MEMORY_TARGET_GIB)


def bundled_configuration(dataset, size):
    return json.loads((CONFIG / dataset / 'nnVeloxSegPlans.json').read_text())['configurations'][f'3d_fullres_{size}']


def volume_profile(manifest):
    """Synthetic B profile whose memory and latency grow with crop volume."""
    rows = {row['key']: {'B': {'peak_allocated_mib': prod(row['patch']) / 2 ** 14,
                               'median_ms': prod(row['patch']) / 2 ** 18, 'parameters': 1,
                               'architecture': row['architectures']['B']['arch_kwargs']}}
            for row in manifest['candidates'] if row['trainable']}
    return {'gpu': 'synthetic', 'torch': 'synthetic', 'rows': rows}


class BundledDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = {dataset: candidate_manifest(planner_for(dataset)) for dataset in DATASETS}

    def test_full_coverage_keeps_bundled_patch(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                manifest = self.manifests[dataset]
                self.assertLessEqual(len(manifest['candidates']), 27)
                selected, _ = select_patch(manifest['candidates'], volume_profile(manifest))
                self.assertEqual(selected['patch'], bundled_configuration(dataset, 'B')['patch_size'])

    def test_plans_record_selection_and_update_budget(self):
        manifest = self.manifests[HECKTOR]
        with mock.patch.object(veloxseg_rules, 'candidate_family', return_value=manifest['candidates']):
            plans = build_plans(planner_for(HECKTOR), volume_profile(manifest))
        for size in 'SBL':
            configuration = plans['configurations'][f'3d_fullres_{size}']
            bundled = bundled_configuration(HECKTOR, size)
            self.assertEqual(configuration['patch_size'], bundled['patch_size'])
            # Batch follows the current architecture's training memory, not the older bundled plans.
            batch = configuration['batch_size']
            self.assertTrue(batch >= 2 and batch & (batch - 1) == 0)
            training = configuration['training']
            self.assertEqual(training['num_epochs'], 1000)
            self.assertEqual(training['num_epochs'] * training['num_iterations_per_epoch'], training['total_updates'])
            selection = configuration['resources']['patch_selection']
            self.assertEqual(selection['profile_gpu'], 'synthetic')
            self.assertEqual([row['patch'] for row in selection['candidates'] if row['selected']],
                             [bundled['patch_size']])

    def test_coverage_fraction_prefers_cheaper_pareto_crop(self):
        manifest = self.manifests[HECKTOR]
        trainable = {row['key'] for row in manifest['candidates'] if row['trainable']}
        self.assertLessEqual({*HECKTOR_COST, '96x256x256'}, trainable)
        profile = volume_profile(manifest)
        for key, (memory, latency) in {**HECKTOR_COST, '96x256x256': (400.0, 40.0)}.items():
            profile['rows'][key]['B'].update(peak_allocated_mib=memory, median_ms=latency)
        selected, _ = select_patch(manifest['candidates'], profile)
        self.assertEqual(selected['patch'], [160, 256, 256])
        with mock.patch.dict(PATCH_POLICY, coverage_fraction=0.8):
            selected, table = select_patch(manifest['candidates'], profile)
        self.assertEqual(selected['patch'], [128, 256, 256])
        pareto = {'x'.join(map(str, row['patch'])): row['pareto'] for row in table}
        self.assertTrue(pareto['128x256x256'])
        self.assertFalse(pareto['96x256x256'])

    def test_profile_of_other_architecture_is_rejected(self):
        manifest = self.manifests[HECKTOR]
        profile = volume_profile(manifest)
        profile['rows']['160x256x256']['B']['architecture'] = {'input_size': [160, 256, 256], 'stages': []}
        with self.assertRaisesRegex(ValueError, 'other B architectures for .*160x256x256'):
            select_patch(manifest['candidates'], profile)

    def test_axis_permuted_crops_do_not_depend_on_latency_noise(self):
        stages = [{'stride': [2, 2, 2]}]
        candidates = [{'patch': patch, 'trainable': True,
                       'architectures': {'B': {'arch_kwargs': {'input_size': patch, 'stages': stages}}}}
                      for patch in ([8, 16, 12], [8, 12, 16])]
        for first_latency, second_latency in ((20.0, 10.0), (10.0, 20.0)):
            profile = {'gpu': 'synthetic', 'torch': 'synthetic', 'rows': {
                patch_key(row['patch']): {'B': {'peak_allocated_mib': 100.0, 'median_ms': latency, 'parameters': 7,
                                                'architecture': row['architectures']['B']['arch_kwargs']}}
                for row, latency in zip(candidates, (first_latency, second_latency))}}
            selected, table = select_patch(candidates, profile)
            self.assertEqual(selected['patch'], [8, 16, 12])
            self.assertEqual([row['pareto'] for row in table], [True, False])

    def test_missing_profile_row_names_patches(self):
        manifest = self.manifests[HECKTOR]
        profile = volume_profile(manifest)
        del profile['rows']['160x256x256']
        with self.assertRaisesRegex(ValueError, '160x256x256.*profile.py --candidates'):
            select_patch(manifest['candidates'], profile)


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


class IgnoreLabelEstimate(unittest.TestCase):
    def test_class_labels_with_ignore_label(self):
        architecture = architecture_for_patch([64, 64, 64], [1.0, 1.0, 1.0], [1], 'B')
        plain = estimate_training_tensors(architecture, LabelManager({'background': 0, 'a': 1}, None), 2)
        ignored = estimate_training_tensors(
            architecture, LabelManager({'background': 0, 'a': 1, 'ignore': 2}, None), 2)
        # The ignore mask adds saved loss storage on top of the same network graph.
        self.assertGreater(ignored['saved_activation_bytes'], plain['saved_activation_bytes'])


if __name__ == '__main__':
    unittest.main()
