"""One dataset-independent VeloxSeg planning policy.

Geometry comes from nnUNet 2.8.1; the shared model and optimization blueprint
comes from the VeloxSeg paper and released reference configuration.
"""
from math import log2, prod

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from model.VeloxSeg import VeloxSeg
from model.loss import VeloxSegLoss
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


# Paper Appendix R: S removes convolution branches; L increases block depth.
MODEL_CAPACITIES = {
    'S': {'kernels': [3], 'conv_depth': 1, 'attn_depth': 1},
    'B': {'kernels': [1, 3, 5], 'conv_depth': 1, 'attn_depth': 1},
    'L': {'kernels': [1, 3, 5], 'conv_depth': 2, 'attn_depth': 2},
}
MODEL_POLICY = {
    'base_channels': 16,
    'maximum_channels': 128,
    'stem_pooling_transitions': 2,
    # The reference 96^3 model ends at 3^3 after its compact entry and 3 transitions.
    'minimum_feature_edge': 3,
    'dropout': 0.1,
}
# Independent JLC and PWA settings from the released reference model. Later
# stages reuse the last row.
STAGE_BLUEPRINT = (
    {'group_width': 4, 'conv_expansion': 3, 'attn_expansion': 3, 'heads': 1, 'head_dim': 4},
    {'group_width': 8, 'conv_expansion': 3, 'attn_expansion': 3, 'heads': 2, 'head_dim': 8},
    {'group_width': 8, 'conv_expansion': 2, 'attn_expansion': 2, 'heads': 2, 'head_dim': 8},
    {'group_width': 16, 'conv_expansion': 2, 'attn_expansion': 2, 'heads': 4, 'head_dim': 16},
)
# Per-branch tokens of the smallest paired big window in the released 96^3
# model (3^3, 6^3, 3^3); its last stage attends over the whole feature map.
# Further intermediate stages reuse the last value, a transfer hypothesis.
PWA_REFERENCE_TOKENS = (27, 216, 27)
# Paper optimizer/300-epoch recipe, with the released 10-epoch warmup.
TRAINING_POLICY = {
    'initial_lr': 2.5e-4,
    'weight_decay': 1e-2,
    'minimum_lr': 6e-6,
    'warmup_epochs': 10,
    'num_epochs': 300,
    'num_val_iterations_per_epoch': 50,
    'oversample_foreground_percent': 0.33,
    'batch_dice': False,
    'seed': 12345,
}
# nnUNet: at least batch 2, spare memory raises the batch, and one batch covers
# at most 5% of the dataset voxels. Batches are powers of two. Epochs carry the
# learning-rate schedule; iterations per epoch keep the recipe's crops per
# epoch (250 iterations at the paper's batch 4) without rescaling the LR.
BATCH_POLICY = {
    'minimum_batch_size': 2,
    'max_dataset_covered': 0.05,
    'recipe_batch_size': 4,
    'recipe_iterations_per_epoch': 250,
}
TRAINING_MEMORY_TARGET_GIB = 24
# AutoPET L at 192x256x256/batch8 estimated 23.425 GiB but OOMed on
# a 23.69-GiB RTX3090 during backward (768-MiB allocation). The tensor
# proxy is not a device-capacity bound: leave room for CUDA/runtime allocations
# and its measured prediction error before accepting a family geometry. The
# next 192x224x256 candidate reserved 23.09 GiB versus a 22.07-GiB estimate,
# before accounting for non-PyTorch CUDA memory, so a 1-GiB reserve is too small.
TRAINING_RUNTIME_RESERVE_GIB = 2


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


def attention_window(shape, index, final):
    """Smallest paired big window whose token count is nearest the reference.

    PWA windows tile the feature grid with one power-of-two ratio up to global
    coverage. Anchoring local tokens, rather than the feature/window ratio,
    keeps attention size from growing with the crop; a larger feature map adds
    scales instead. Each axis keeps at least two positions per window.
    """
    if final:
        return list(shape)
    reference = PWA_REFERENCE_TOKENS[min(index, len(PWA_REFERENCE_TOKENS) - 1)]
    ratios = [1]
    while all(n % (2 * ratios[-1]) == 0 and n // (2 * ratios[-1]) >= 2 for n in shape):
        ratios.append(2 * ratios[-1])
    # Equal multiplicative distance treats twice and half the reference alike;
    # ascending ratios keep the larger window on an exact tie.
    ratio = min(ratios, key=lambda r: abs(log2(prod(shape) / (r ** 3 * reference))))
    return [n // ratio for n in shape]


def architecture_for_patch(patch, spacing, in_ch, model_size):
    capacity = MODEL_CAPACITIES[model_size]
    patch, strides, kernels, _ = geometry(patch, spacing)
    shape = list(patch)
    stages = []
    for index, (stride, kernel) in enumerate(zip(strides, kernels)):
        shape = [n // step for n, step in zip(shape, stride)]
        blueprint = STAGE_BLUEPRINT[min(index, len(STAGE_BLUEPRINT) - 1)]
        channels = min(MODEL_POLICY['maximum_channels'], MODEL_POLICY['base_channels'] * 2 ** index)
        group_width = max(value for value in divisors(channels) if value <= blueprint['group_width'])
        parallel_kernels = []
        for size in capacity['kernels']:
            item = [1 if k == 1 else size for k in kernel]
            if item not in parallel_kernels:
                parallel_kernels.append(item)
        stages.append({
            'stride': list(stride), 'channels': channels, 'kernels': parallel_kernels,
            'conv_depth': capacity['conv_depth'], 'attn_depth': capacity['attn_depth'],
            'group_width': group_width,
            'conv_expansion': blueprint['conv_expansion'], 'attn_expansion': blueprint['attn_expansion'],
            'heads': blueprint['heads'], 'head_dim': blueprint['head_dim'],
            'big_window': attention_window(shape, index, index == len(strides) - 1),
            'small_window': [1, 1, 1],
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
                FlopCounterMode(display=False) as counter, \
                torch.autocast('cpu', dtype=torch.float16):
            loss = objective(network(x), target, x)
            loss.backward()
        training_flops = counter.get_total_flops()
        network.eval()
        with FlopCounterMode(display=False) as counter, torch.no_grad(), \
                torch.autocast('cpu', dtype=torch.float16):
            network(x)
        inference_flops = counter.get_total_flops()
    activation_bytes = sum(saved.values())
    fixed_bytes = 4 * parameter_bytes + persistent
    output_bytes = (batch_size * prod(kwargs['input_size']) *
                    (label_manager.num_segmentation_heads + sum(kwargs['in_ch'])) * 4)
    estimated = fixed_bytes + float(MEMORY_PROXY_COEFFICIENTS @ [activation_bytes, output_bytes])
    return {'saved_activation_bytes': activation_bytes, 'fixed_bytes': fixed_bytes,
            'full_resolution_output_bytes': output_bytes,
            'estimated_reserved_bytes': estimated, 'parameters': parameter_bytes // 4,
            'training_flops': training_flops, 'inference_flops': inference_flops}


def initial_patch(spacing, median_shape):
    inverse_spacing = 1 / np.asarray(spacing)
    # nnUNet's initial physical aspect ratio and 256^3 search envelope.
    initial = np.rint(inverse_spacing * (256 ** 3 / prod(inverse_spacing)) ** (1 / 3))
    return np.minimum(initial, median_shape).astype(int).tolist()


def smaller_patch(patch, spacing, median_shape):
    # Upstream relative-coverage axis order and recomputed stride divisibility.
    axes = np.argsort(np.asarray(patch) / np.asarray(median_shape))[::-1]
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


def family_memory(architectures, label_manager, batch_size):
    inventories = {size: estimate_training_tensors(architecture, label_manager, batch_size)
                   for size, architecture in architectures.items()}
    return inventories, max(row['estimated_reserved_bytes'] for row in inventories.values())


def plan_family(spacing, median_shape, dataset_json, label_manager, memory_gb):
    """Plan shared S/B/L geometry from nnUNet data rules; memory sets the batch.

    The crop is nnUNet's initial physical patch and shrinks only when the
    largest member cannot train at the minimum batch. Memory the model saves
    therefore becomes batch size, never a larger crop or a deeper network.
    """
    if memory_gb <= TRAINING_RUNTIME_RESERVE_GIB:
        raise ValueError('gpu_memory_target_in_gb must exceed the runtime reserve')
    in_ch = modality_channels(dataset_json)
    tensor_budget = (memory_gb - TRAINING_RUNTIME_RESERVE_GIB) * 2 ** 30
    batch = BATCH_POLICY['minimum_batch_size']
    patch = initial_patch(spacing, median_shape)
    while True:
        architectures = {size: architecture_for_patch(patch, spacing, in_ch, size)
                         for size in MODEL_CAPACITIES}
        patch = architectures['L']['arch_kwargs']['input_size']
        inventories, largest_cost = family_memory(architectures, label_manager, batch)
        print(f'VeloxSeg S/B/L batch{batch} candidate {patch}: '
              f'maximum estimated memory {largest_cost / 2**30:.3f} GiB', flush=True)
        if largest_cost <= tensor_budget:
            break
        patch = smaller_patch(patch, spacing, median_shape)
    dataset_voxels = float(np.prod(median_shape, dtype=np.float64)) * dataset_json['numTraining']
    coverage_cap = round(dataset_voxels * BATCH_POLICY['max_dataset_covered'] / prod(patch))
    while 2 * batch <= coverage_cap:
        candidate, largest_cost = family_memory(architectures, label_manager, 2 * batch)
        if largest_cost > tensor_budget:
            break
        batch, inventories = 2 * batch, candidate
    iterations = round(BATCH_POLICY['recipe_iterations_per_epoch'] * BATCH_POLICY['recipe_batch_size'] / batch)
    # nnUNet forces foreground crops only into the last batch slots.
    forced = batch - round(batch * (1 - TRAINING_POLICY['oversample_foreground_percent']))
    return {
        size: {
            'architecture': architecture,
            'batch_size': batch,
            'model_size': size,
            'training': {**TRAINING_POLICY, 'num_iterations_per_epoch': iterations},
            'resources': {
                'method': 'GPU-calibrated saved-tensor and fullres-output proxy; RTX3090 torch2.6/cu124 B references',
                'training_memory_target_gib': memory_gb,
                'runtime_reserve_gib': TRAINING_RUNTIME_RESERVE_GIB,
                'training_tensor_budget_gib': tensor_budget / 2 ** 30,
                'memory_proxy_coefficients': MEMORY_PROXY_COEFFICIENTS.tolist(),
                'batch_coverage_cap': coverage_cap,
                'forced_foreground_fraction': forced / batch,
                'estimated_training_reserved_gib': inventories[size]['estimated_reserved_bytes'] / 2 ** 30,
                'parameters': inventories[size]['parameters'],
                'training_flops_per_batch': inventories[size]['training_flops'],
                'inference_flops_per_patch': inventories[size]['inference_flops'] / batch,
            },
        }
        for size, architecture in architectures.items()
    }
