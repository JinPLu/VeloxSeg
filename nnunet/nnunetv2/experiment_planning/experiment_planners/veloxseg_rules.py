"""One dataset-independent VeloxSeg planning policy.

Geometry methods come from nnUNet 2.6.2. Model/optimization choices below are
explicit starting priors, not claims of optimality or GPU peak measurements.
"""
from math import ceil, log2, prod

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.VeloxSeg import VeloxSeg
from model.loss import VeloxSegLoss
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


MODEL_CAPACITIES = {
    'S': {'base_channels': 8, 'maximum_channels': 160},
    'B': {'base_channels': 16, 'maximum_channels': 320},
    'L': {'base_channels': 24, 'maximum_channels': 480},
}
MODEL_POLICY = {
    'stem_pooling_transitions': 2,
    'minimum_feature_edge': 4,
    'conv_depth': 1,
    'attn_depth': 1,
    'expansion': 2,
    'channels_per_attention_head': 32,
    'minimum_group_width': 4,
    'dropout': 0.0,
}
TRAINING_POLICY = {
    'initial_lr': 1e-3,
    'weight_decay': 1e-2,
    'minimum_lr': 6e-6,
    'num_epochs': 1000,
    'num_iterations_per_epoch': 250,
    'num_val_iterations_per_epoch': 50,
    'oversample_foreground_percent': 0.33,
    'batch_dice': False,
    'seed': 12345,
}
# Training capacity is independent of batch-1 inference efficiency.
# Training target follows the upstream ResEnc-L 24 GB preset.
TRAINING_MEMORY_TARGET_GIB = 24
TRAINING_BATCH_SIZE = 8


# Compact-entry full-objective CUDA references, 2026-09-09. RTX 3090,
# torch 2.6.0/cu124, CUDA FP16 + FP32 losses, AdamW, 3 updates, benchmark=True.
# CPU autocast is only a proxy. Account separately for saved graph storage and
# full-resolution segmentation/reconstruction outputs: one ratio overestimates
# dual-modality costs while missing region/reconstruction output workspaces.
# Fit two shared coefficients; never select a coefficient by dataset name.
MEMORY_REFERENCES = (
    {'patch': (160, 192, 160), 'batch': 19,
     'saved_activation_bytes': 16417561272, 'fixed_bytes': 42447200,
     'full_resolution_output_bytes': 19 * 160 * 192 * 160 * 7 * 4,
     'peak_reserved_bytes': int(23.09765625 * 2 ** 30)},
    {'patch': (224, 320, 320), 'batch': 5,
     'saved_activation_bytes': 19279865964, 'fixed_bytes': 147690888,
     'full_resolution_output_bytes': 5 * 224 * 320 * 320 * 4 * 4,
     'peak_reserved_bytes': int(23.046875 * 2 ** 30)},
)
MEMORY_PROXY_COEFFICIENTS = np.linalg.solve(
    [[row['saved_activation_bytes'], row['full_resolution_output_bytes']]
     for row in MEMORY_REFERENCES],
    [row['peak_reserved_bytes'] - row['fixed_bytes'] for row in MEMORY_REFERENCES])


def divisors(size):
    return [value for value in range(1, size + 1) if size % value == 0]


def modality_channels(dataset_json):
    """Modality grouping is semantic metadata, never guessed from channel names."""
    channels = dataset_json['veloxseg_modality_channels']
    if not channels or any(not isinstance(value, int) or value < 1 for value in channels):
        raise ValueError('veloxseg_modality_channels must contain positive channel counts')
    if sum(channels) != len(dataset_json['channel_names']):
        raise ValueError('Modality groups must partition the channels in their dataset.json order')
    return channels


def geometry(patch, spacing):
    _, strides, kernels, aligned, alignment = get_pool_and_conv_props(
        spacing, patch, MODEL_POLICY['minimum_feature_edge'], 999999)
    if len(strides) < 2:
        raise ValueError('The data must support a 3D encoder transition')
    # Fullres describes preprocessing spacing, not an obligatory stride-1 stem.
    # Preserve VeloxSeg's compact entry: combine the first two nnUNet pooling
    # transitions (normally stride 4), respecting anisotropy and leaving a
    # decoder transition for small inputs. Stage count remains geometry-driven.
    merged = min(MODEL_POLICY['stem_pooling_transitions'], len(strides) - 2)
    stem = [prod(stride[axis] for stride in strides[:merged + 1]) for axis in range(3)]
    return ([int(value) for value in aligned], [stem, *strides[merged + 1:]],
            kernels[merged:], alignment)


def architecture_for_patch(patch, spacing, in_ch, model_size):
    capacity = MODEL_CAPACITIES[model_size]
    patch, strides, kernels, _ = geometry(patch, spacing)
    shape = list(patch)
    compression = 1
    stages = []
    for index, (stride, kernel) in enumerate(zip(strides, kernels)):
        shape = [n // step for n, step in zip(shape, stride)]
        compression *= prod(stride)
        channels = min(capacity['maximum_channels'], capacity['base_channels'] * 2 ** index)
        # JL-inspired logarithmic growth; round the lower bound to a legal
        # channel divisor, without widening the backbone for attention padding.
        lower_bound = max(MODEL_POLICY['minimum_group_width'], ceil(log2(sum(in_ch) * compression + 1)))
        group_width = next(value for value in divisors(channels) if value >= min(channels, lower_bound))
        scale = 1
        while all(n % (2 * scale) == 0 and n // (2 * scale) >= MODEL_POLICY['minimum_feature_edge']
                  for n in shape):
            scale *= 2
        big = [n // scale for n in shape]
        # Keep fine samples whenever possible. An axis uses pooling only when
        # the exact global tiling leaves more than a bottleneck-sized token grid.
        token_edge_limit = 2 * MODEL_POLICY['minimum_feature_edge'] - 1
        small = [next(step for step in divisors(n) if n // step <= token_edge_limit) for n in big]
        parallel_kernels = []
        for size in (1, 3, 5):
            item = [1 if k == 1 else size for k in kernel]
            if item not in parallel_kernels:
                parallel_kernels.append(item)
        stages.append({
            'stride': list(stride), 'channels': channels, 'kernels': parallel_kernels,
            'conv_depth': MODEL_POLICY['conv_depth'],
            'attn_depth': MODEL_POLICY['attn_depth'],
            'group_width': group_width, 'expansion': MODEL_POLICY['expansion'],
            'heads': max(1, channels // MODEL_POLICY['channels_per_attention_head']),
            'head_dim': group_width, 'big_window': big, 'small_window': small,
        })
    return {
        'network_class_name': 'model.VeloxSeg.VeloxSeg',
        'arch_kwargs': {'input_size': patch, 'in_ch': list(in_ch), 'stages': stages,
                        'dropout': MODEL_POLICY['dropout'], 'spatial_dim': 3},
        '_kw_requires_import': [],
    }


def segmentation_loss(label_manager, batch_dice=False, ddp=False):
    dice = {'batch_dice': batch_dice, 'smooth': 1e-5, 'ddp': ddp}
    if label_manager.has_regions:
        return DC_and_BCE_loss({}, {**dice, 'do_bg': True},
                              use_ignore_label=label_manager.ignore_label is not None,
                              dice_class=MemoryEfficientSoftDiceLoss)
    return DC_and_CE_loss({**dice, 'do_bg': False}, {},
                         ignore_label=label_manager.ignore_label,
                         dice_class=MemoryEfficientSoftDiceLoss)


def estimate_training_tensors(architecture, label_manager, batch_size):
    """Inventory mixed-precision network and FP32 losses without image data.

    Count saved activation storage, full-resolution outputs and fixed AdamW
    state, then apply the measured reserved-memory references above. Math SDPA and CPU FP16 autocast
    are a portable proxy, not a CUDA allocation trace. The calibration includes
    observed workspace/allocator costs but does not guarantee other devices.
    """
    kwargs = architecture['arch_kwargs']
    with torch.random.fork_rng(devices=[]):
        network = VeloxSeg(**kwargs, n_classes=label_manager.num_segmentation_heads)
    persistent = sum(t.numel() * t.element_size() for t in network.buffers())
    parameter_bytes = sum(t.numel() * t.element_size() for t in network.parameters())
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    saved = {}
    with mode, sdpa_kernel(SDPBackend.MATH):
        # FakeTensor's converter memo is weak: retain these tensors so storage
        # identities cannot be recycled while the forward graph is inventoried.
        fake_persistent = [mode.from_tensor(t)
                           for t in [*network.parameters(), *network.buffers()]]
        excluded = {t.untyped_storage()._cdata for t in fake_persistent}
        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage._cdata not in excluded:
                saved[storage._cdata] = storage.nbytes()
            return tensor
        x = torch.randn(batch_size, sum(kwargs['in_ch']), *kwargs['input_size'])
        channels = label_manager.num_segmentation_heads + int(label_manager.has_ignore_label) if label_manager.has_regions else 1
        target = torch.zeros(batch_size, channels, *kwargs['input_size'],
                             dtype=torch.float32 if label_manager.has_regions else torch.long)
        objective = VeloxSegLoss(segmentation_loss(label_manager), kwargs['in_ch'])
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor), \
                torch.autocast('cpu', dtype=torch.float16):
            objective(network(x), target, x)
    activation_bytes = sum(saved.values())
    fixed_bytes = 4 * parameter_bytes + persistent
    output_bytes = (batch_size * prod(kwargs['input_size']) *
                    (label_manager.num_segmentation_heads + sum(kwargs['in_ch'])) * 4)
    estimated = fixed_bytes + float(MEMORY_PROXY_COEFFICIENTS @ [activation_bytes, output_bytes])
    return {'saved_activation_bytes': activation_bytes, 'fixed_bytes': fixed_bytes,
            'full_resolution_output_bytes': output_bytes,
            'estimated_reserved_bytes': estimated, 'parameters': parameter_bytes // 4}


def initial_patch(spacing, median_shape):
    inverse_spacing = 1 / np.asarray(spacing)
    # nnUNet's initial physical aspect ratio and 256^3 search envelope.
    initial = np.rint(inverse_spacing * (256 ** 3 / prod(inverse_spacing)) ** (1 / 3))
    return np.minimum(initial, median_shape).astype(int).tolist()


def smaller_patch(patch, spacing, median_shape):
    # Recompute alignment before reducing, as upstream does (256 -> 224 is legal).
    axes = sorted(range(3), key=lambda index: patch[index] / median_shape[index], reverse=True)
    _, _, _, alignment = geometry(patch, spacing)
    for axis in axes:
        tentative = list(patch)
        tentative[axis] -= int(alignment[axis])
        if tentative[axis] < MODEL_POLICY['minimum_feature_edge']:
            continue
        _, _, _, next_alignment = geometry(tentative, spacing)
        reduced = list(patch)
        reduced[axis] -= int(next_alignment[axis])
        aligned, _, _, _ = geometry(reduced, spacing)
        if prod(aligned) < prod(patch):
            return aligned
    raise ValueError('The training tensor budget cannot fit the minimum legal geometry')


def plan_family(spacing, median_shape, dataset_json, label_manager, memory_gb, training_batch_size):
    """Plan one shared geometry that fits all three widths at the fixed batch."""
    if memory_gb <= 0:
        raise ValueError('gpu_memory_target_in_gb must be positive')
    if not isinstance(training_batch_size, int) or training_batch_size < 2:
        raise ValueError('training_batch_size must be an integer >= 2')
    in_ch = modality_channels(dataset_json)
    tensor_budget = memory_gb * 2 ** 30
    patch = initial_patch(spacing, median_shape)
    while True:
        architectures = {size: architecture_for_patch(patch, spacing, in_ch, size)
                         for size in MODEL_CAPACITIES}
        patch = architectures['L']['arch_kwargs']['input_size']
        inventories = {size: estimate_training_tensors(architecture, label_manager, training_batch_size)
                       for size, architecture in architectures.items()}
        largest_cost = max(row['estimated_reserved_bytes'] for row in inventories.values())
        print(f'VeloxSeg S/B/L batch{training_batch_size} candidate {patch}: '
              f'Maximum reference-based memory estimate {largest_cost / 2**30:.3f} GiB', flush=True)
        if largest_cost <= tensor_budget:
            break
        patch = smaller_patch(patch, spacing, median_shape)
    return {
        size: {
            'architecture': architecture,
            'batch_size': training_batch_size,
            'model_size': size,
            'resources': {
                'method': 'GPU-calibrated saved-tensor and fullres-output proxy; RTX3090 torch2.6/cu124 B references',
                'training_memory_target_gib': memory_gb,
                'memory_proxy_coefficients': MEMORY_PROXY_COEFFICIENTS.tolist(),
                'training_batch_size': training_batch_size,
                'inference_batch_size': 1,
                'estimated_training_reserved_gib': inventories[size]['estimated_reserved_bytes'] / 2 ** 30,
                'parameters': inventories[size]['parameters'],
            },
        }
        for size, architecture in architectures.items()
    }
