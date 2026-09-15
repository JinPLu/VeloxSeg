"""Native nnUNet planner and metadata-only entry points for the same policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import (
    MODEL_POLICY, MODEL_CAPACITIES, TRAINING_POLICY, TRAINING_MEMORY_TARGET_GIB,
    candidate_family, plan_family,
)
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.json_export import recursive_fix_for_json_export

PROFILE_FILE = 'veloxseg_profile.json'


def dataset_geometry(planner):
    spacing = planner.determine_fullres_target_spacing()
    forward, backward = planner.determine_transpose()
    fingerprint = planner.dataset_fingerprint
    shapes = [compute_new_shape(shape, old_spacing, spacing)
              for old_spacing, shape in zip(fingerprint['spacings'], fingerprint['shapes_after_crop'])]
    median_shape = np.median(shapes, axis=0)[forward]
    spacing = spacing[forward]
    if len(spacing) != 3 or min(median_shape) <= 1:
        raise ValueError('VeloxSegPlanner currently supports 3D full-resolution datasets')
    return spacing, median_shape, forward, backward


def candidate_manifest(planner):
    """Candidate crops along nnUNet's envelope chain, the input of nnunet/cost_profile.py."""
    spacing, median_shape, _, _ = dataset_geometry(planner)
    manifest = {
        'dataset_name': planner.dataset_name,
        'gpu_memory_target_in_gb': planner.UNet_vram_target_GB,
        **candidate_family(spacing, median_shape, planner.dataset_json, planner.UNet_vram_target_GB),
    }
    recursive_fix_for_json_export(manifest)
    return manifest


def build_plans(planner, profile):
    spacing, median_shape, forward, backward = dataset_geometry(planner)
    fingerprint = planner.dataset_fingerprint
    family = plan_family(spacing, median_shape, planner.dataset_json, planner.UNet_vram_target_GB, profile)
    normalizations, masks = planner.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
    data_fn, data_kwargs, seg_fn, seg_kwargs = planner.determine_resampling()
    probabilities_fn, probabilities_kwargs = planner.determine_segmentation_softmax_export_fn()
    configuration = {
        'data_identifier': planner.generate_data_identifier('3d_fullres'),
        'preprocessor_name': planner.preprocessor_name,
        'median_image_size_in_voxels': median_shape.tolist(),
        'spacing': spacing.tolist(),
        'normalization_schemes': normalizations,
        'use_mask_for_norm': masks,
        'resampling_fn_data': data_fn.__name__, 'resampling_fn_data_kwargs': data_kwargs,
        'resampling_fn_seg': seg_fn.__name__, 'resampling_fn_seg_kwargs': seg_kwargs,
        'resampling_fn_probabilities': probabilities_fn.__name__,
        'resampling_fn_probabilities_kwargs': probabilities_kwargs,
        'batch_dice': TRAINING_POLICY['batch_dice'],
    }
    plans = {
        'dataset_name': planner.dataset_name,
        'plans_name': planner.plans_identifier,
        'original_median_spacing_after_transp': np.median(fingerprint['spacings'], axis=0)[forward].tolist(),
        'original_median_shape_after_transp': np.median(fingerprint['shapes_after_crop'], axis=0)[forward].tolist(),
        'image_reader_writer': planner.determine_reader_writer().__name__,
        'transpose_forward': list(forward), 'transpose_backward': list(backward),
        'configurations': {
            f'3d_fullres_{size}': {**configuration, **member,
                                 'patch_size': member['architecture']['arch_kwargs']['input_size']}
            for size, member in family.items()
        },
        'experiment_planner_used': 'VeloxSegPlanner',
        'label_manager': 'LabelManager',
        'foreground_intensity_properties_per_channel': fingerprint['foreground_intensity_properties_per_channel'],
        'veloxseg_model_policy': dict(MODEL_POLICY),
        'veloxseg_model_capacities': {size: dict(capacity) for size, capacity in MODEL_CAPACITIES.items()},
    }
    recursive_fix_for_json_export(plans)
    return plans


class VeloxSegPlanner(ExperimentPlanner):
    """Discovered by nnUNetv2_plan_and_preprocess -pl VeloxSegPlanner."""
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb=TRAINING_MEMORY_TARGET_GIB,
                 preprocessor_name='DefaultPreprocessor', plans_name='nnVeloxSegPlans',
                 overwrite_target_spacing=None, suppress_transpose=False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
                         plans_name, overwrite_target_spacing, suppress_transpose)

    def plan_experiment(self):
        destination = Path(nnUNet_preprocessed) / self.dataset_name
        profile = destination / PROFILE_FILE
        if not profile.is_file():
            candidates = destination / 'veloxseg_candidates.json'
            dataset_id = int(self.dataset_name[len('Dataset'):].split('_')[0])
            raise FileNotFoundError(
                f'{profile} is missing. VeloxSeg plans the crop and batch from CUDA measurements:\n'
                f'  1. python -m nnunetv2.experiment_planning.experiment_planners.veloxseg_planner candidates '
                f'--dataset-name {self.dataset_name} --dataset-json {Path(nnUNet_raw) / self.dataset_name / "dataset.json"} '
                f'--fingerprint {destination / "dataset_fingerprint.json"} '
                f'--gpu-memory-target-in-gb {self.UNet_vram_target_GB} --output {candidates}\n'
                f'  2. python nnunet/cost_profile.py --candidates {candidates} --output {profile}   (on the CUDA device)\n'
                f'  3. nnUNetv2_plan_and_preprocess -d {dataset_id} -pl VeloxSegPlanner '
                f'-c 3d_fullres_B -gpu_memory_target {self.UNet_vram_target_GB}')
        self.plans = build_plans(self, json.loads(profile.read_text()))
        (destination / 'dataset.json').write_text(json.dumps(self.dataset_json, indent=2) + '\n')
        self.save_plans(self.plans)
        return self.plans

    def save_plans(self, plans):
        # A fresh plan must not inherit stale configurations from another model.
        destination = Path(nnUNet_preprocessed) / self.dataset_name / (self.plans_identifier + '.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plans, indent=2) + '\n')


class FingerprintGeometry(ExperimentPlanner):
    """Use upstream geometry methods on exported metadata, without raw images.

    This is the input adapter for the real saved-fingerprint use case. Every
    entry point calls candidate_family through the same rules; there is no
    alternate planning algorithm.
    """
    def __init__(self, dataset_name, dataset_json, fingerprint, memory_gb):
        self.dataset_name = dataset_name
        self.dataset_json = dataset_json
        self.dataset_fingerprint = fingerprint
        self.UNet_vram_target_GB = memory_gb
        self.plans_identifier = 'nnVeloxSegPlans'
        self.preprocessor_name = 'DefaultPreprocessor'
        self.overwrite_target_spacing = None
        self.suppress_transpose = False
        self.anisotropy_threshold = ANISO_THRESHOLD

    def determine_reader_writer(self):
        return determine_reader_writer_from_dataset_json(self.dataset_json)


def main():
    parser = argparse.ArgumentParser(description='VeloxSeg candidates and plans from an exported nnUNet fingerprint')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (('candidates', 'Write S/B/L architectures of every candidate crop for profiling'),
                            ('plan', 'Generate plans from the candidates and their measured profile')):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--dataset-name', required=True)
        command.add_argument('--dataset-json', required=True, type=Path)
        command.add_argument('--fingerprint', required=True, type=Path)
        command.add_argument('--gpu-memory-target-in-gb', type=float, default=TRAINING_MEMORY_TARGET_GIB,
                             help='Training memory target in GiB; less the runtime reserve, it bounds the '
                                  'measured L training memory of trainable crops and of the batch')
        command.add_argument('--output', required=True, type=Path)
        if name == 'plan':
            command.add_argument('--profile', required=True, type=Path,
                                 help='nnunet/cost_profile.py --candidates output for these candidates')
    args = parser.parse_args()
    planner = FingerprintGeometry(args.dataset_name, json.loads(args.dataset_json.read_text()),
                                  json.loads(args.fingerprint.read_text()), args.gpu_memory_target_in_gb)
    if args.command == 'candidates':
        result = candidate_manifest(planner)
    else:
        result = build_plans(planner, json.loads(args.profile.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
