"""One dataset-independent VeloxSeg planning policy.

Geometry comes from nnUNet 2.8.1; the shared model and optimization blueprint
comes from the VeloxSeg paper and released reference configuration.
"""
from fractions import Fraction
from itertools import product
from math import log2, prod

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
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
    {'conv_expansion': 3, 'attn_expansion': 3, 'heads': 1, 'head_dim': 4},
    {'conv_expansion': 3, 'attn_expansion': 3, 'heads': 2, 'head_dim': 8},
    {'conv_expansion': 2, 'attn_expansion': 2, 'heads': 2, 'head_dim': 8},
    {'conv_expansion': 2, 'attn_expansion': 2, 'heads': 4, 'head_dim': 16},
)
# JLC group width follows cumulative compression, not the stage index: the
# released 96^3 stages compress 2^6, 2^9, 2^12 and 2^15 voxels into one
# position and use group widths 4, 8, 8, 16. Anchors are (log2 compression,
# log2 group width), interpolated linearly; beyond the anchors 'clamp' holds
# the end value and 'extrapolate' continues the end segment's slope.
JLC_GROUP_WIDTH = {
    'anchors': ((6, 2), (9, 3), (12, 3), (15, 4)),
    'beyond_anchors': 'clamp',
}
# Per-branch tokens of the smallest paired big window in the released 96^3
# model (3^3, 6^3, 3^3); its last stage attends over the whole feature map.
# Further intermediate stages reuse the last value, a transfer hypothesis.
PWA_REFERENCE_TOKENS = (27, 216, 27)
# Windows whose token mismatch stays within this factor of the best achievable
# mismatch compete on physical balance. A factor two made isotropic Hecktor
# 40x64x64 take 5x2x2 (20x8x8 mm) over 5x4x4 (80 tokens, 20x16x16 mm).
PWA_TOKEN_BAND = 4
# The update budget is independent of the batch: 250,000 updates at nnUNet's
# native 250 iterations per epoch. Plans derive num_epochs from these two.
TRAINING_POLICY = {
    'total_updates': 250000,
    'num_iterations_per_epoch': 250,
    'warmup_updates': 0,
    'initial_lr': 1e-3,
    'minimum_lr': 6e-6,
    'weight_decay': 1e-2,
    'decay_exclusions': False,
    'num_val_iterations_per_epoch': 50,
    'oversample_foreground_percent': 0.33,
    'batch_dice': False,
    'seed': 12345,
}
# nnUNet: at least batch 2, spare memory raises the batch, and one batch covers
# at most 5% of the dataset voxels. Batches are powers of two shared by S/B/L.
BATCH_POLICY = {
    'minimum_batch_size': 2,
    'max_dataset_covered': 0.05,
}
# Candidate crops reduce each envelope axis by these stride-alignment units.
# Among Pareto crops covering at least coverage_fraction of the largest
# trainable crop, the least batch-1 memory wins. The fraction awaits
# calibration by full training; 1.0 keeps the largest trainable crop.
PATCH_POLICY = {'reduction_units': (0, 1, 2), 'coverage_fraction': 1.0}
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


def attention_window(shape, feature_spacing, index, final):
    """Smallest paired big window of a stage; PWA doubles it per axis to the grid.

    Rule. A legal window b0 divides the feature grid F on every axis with a
    power-of-two ratio and keeps at least two positions per axis. Its token
    count T = prod(b0) is compared with the stage reference Tref through the
    mismatch max(T, Tref) / min(T, Tref). Keep the windows within PWA_TOKEN_BAND
    of the smallest achievable mismatch; among them choose, in order, the most
    physically balanced window (smallest max/min over axes of
    b0_a * feature_spacing_a), the nearest token count, the most scales
    (largest max ratio) and the lexicographically smallest window. The final
    encoder stage attends over its whole grid.

    Anchoring local tokens, rather than the feature/window ratio, keeps
    attention size from growing with the crop; a larger feature map adds
    scales instead. The band is relative to the best achievable count because
    a band around the reference alone would fall back to pure nearest-token
    selection on grids where no window reaches it (40x48x40 with reference 27
    would take 5x3x5 over 5x6x5), which favours in-plane-anisotropic windows
    on in-plane-isotropic grids.
    """
    if final:
        return list(shape)
    reference = PWA_REFERENCE_TOKENS[min(index, len(PWA_REFERENCE_TOKENS) - 1)]
    candidates = [[]]
    for n in shape:
        candidates = [window + [n // ratio] for window in candidates
                      for ratio in (2 ** q for q in range(n.bit_length()))
                      if n % ratio == 0 and n // ratio >= 2]
    # Exact ratios: a candidate exactly at the band edge must not depend on
    # floating-point rounding of a logarithm.
    mismatch = {tuple(window): Fraction(max(prod(window), reference), min(prod(window), reference))
                for window in candidates}
    nearest = min(mismatch.values())
    kept = [window for window in candidates if mismatch[tuple(window)] <= PWA_TOKEN_BAND * nearest]

    def preference(window):
        extents = [size * step for size, step in zip(window, feature_spacing)]
        scales = max(log2(n // size) for n, size in zip(shape, window))
        return max(extents) / min(extents), mismatch[tuple(window)], -scales, window
    return min(kept, key=preference)


def jlc_group_width(cumulative_stride, channels):
    """Group width from the voxels compressed into one position at this stage."""
    exponents, widths = zip(*JLC_GROUP_WIDTH['anchors'])
    beyond = JLC_GROUP_WIDTH['beyond_anchors']
    if beyond not in ('clamp', 'extrapolate'):
        raise ValueError("JLC_GROUP_WIDTH['beyond_anchors'] must be 'clamp' or 'extrapolate'")
    compression = log2(prod(cumulative_stride))
    # np.interp holds the end values beyond the anchors.
    value = float(np.interp(compression, exponents, widths))
    if beyond == 'extrapolate' and not exponents[0] <= compression <= exponents[-1]:
        end = slice(0, 2) if compression < exponents[0] else slice(-2, None)
        (u0, w0), (u1, w1) = JLC_GROUP_WIDTH['anchors'][end]
        value = w0 + (w1 - w0) * (compression - u0) / (u1 - u0)
    # Round up to a divisor of the stage channels, capped at the channel count.
    return min(width for width in divisors(channels) if width >= min(2 ** value, channels))


def architecture_for_patch(patch, spacing, in_ch, model_size):
    capacity = MODEL_CAPACITIES[model_size]
    patch, strides, kernels, _ = geometry(patch, spacing)
    shape = list(patch)
    cumulative_stride = [1, 1, 1]
    stages = []
    for index, (stride, kernel) in enumerate(zip(strides, kernels)):
        shape = [n // step for n, step in zip(shape, stride)]
        cumulative_stride = [total * step for total, step in zip(cumulative_stride, stride)]
        feature_spacing = [float(value) * step for value, step in zip(spacing, cumulative_stride)]
        blueprint = STAGE_BLUEPRINT[min(index, len(STAGE_BLUEPRINT) - 1)]
        channels = min(MODEL_POLICY['maximum_channels'], MODEL_POLICY['base_channels'] * 2 ** index)
        parallel_kernels = []
        for size in capacity['kernels']:
            item = [1 if k == 1 else size for k in kernel]
            if item not in parallel_kernels:
                parallel_kernels.append(item)
        stages.append({
            'stride': list(stride), 'channels': channels, 'kernels': parallel_kernels,
            'conv_depth': capacity['conv_depth'], 'attn_depth': capacity['attn_depth'],
            'group_width': jlc_group_width(cumulative_stride, channels),
            'conv_expansion': blueprint['conv_expansion'], 'attn_expansion': blueprint['attn_expansion'],
            'heads': blueprint['heads'], 'head_dim': blueprint['head_dim'],
            'big_window': attention_window(shape, feature_spacing, index, index == len(strides) - 1),
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


class SyntheticTargetReads(TorchDispatchMode):
    """Answer the objective's data-dependent read for the synthetic target.

    nnUNet's DC_and_CE_loss with an ignore label evaluates `num_fg > 0` in
    Python, which a FakeTensor cannot answer. The all-background estimation
    target has no ignored voxel, so that read is True, as in training batches;
    the loss module and its graph are the real ones.
    """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            return True
        return func(*args, **(kwargs or {}))


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
                FlopCounterMode(display=False) as counter, SyntheticTargetReads(), \
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


def reduce_axis(patch, spacing, axis):
    """One upstream stride-alignment step along an axis, or None if illegal.

    Like nnUNet, the step uses the divisibility recomputed after a tentative
    reduction, then re-aligns the reduced crop.
    """
    _, _, _, alignment = geometry(patch, spacing)
    tentative = list(patch)
    tentative[axis] -= int(alignment[axis])
    if tentative[axis] < MODEL_POLICY['minimum_feature_edge']:
        return None
    try:
        _, _, _, next_alignment = geometry(tentative, spacing)
        reduced = list(patch)
        reduced[axis] -= int(next_alignment[axis])
        aligned, _, _, _ = geometry(reduced, spacing)
    except ValueError:
        return None
    return aligned if prod(aligned) < prod(patch) else None


def smaller_patch(patch, spacing, median_shape):
    # Upstream relative-coverage axis order.
    for axis in np.argsort(np.asarray(patch) / np.asarray(median_shape))[::-1]:
        reduced = reduce_axis(patch, spacing, axis)
        if reduced is not None:
            return reduced
    raise ValueError('The training tensor budget cannot fit the minimum legal geometry')


def candidate_patches(envelope, spacing):
    """Aligned crops reducing each envelope axis by PATCH_POLICY units, deduplicated."""
    aligned, _, _, _ = geometry(envelope, spacing)
    candidates = []
    for units in product(PATCH_POLICY['reduction_units'], repeat=len(aligned)):
        patch = aligned
        for axis, count in enumerate(units):
            for _ in range(count):
                if patch is not None:
                    patch = reduce_axis(patch, spacing, axis)
        if patch is not None and patch not in candidates:
            candidates.append(patch)
    return candidates


def training_tensor_budget(memory_gb):
    if memory_gb <= TRAINING_RUNTIME_RESERVE_GIB:
        raise ValueError('gpu_memory_target_in_gb must exceed the runtime reserve')
    return (memory_gb - TRAINING_RUNTIME_RESERVE_GIB) * 2 ** 30


def candidate_family(spacing, median_shape, dataset_json, label_manager, memory_gb):
    """S/B/L candidates around the first envelope with a trainable crop.

    A crop is trainable when L fits the tensor budget at the minimum batch.
    Without one, the envelope takes nnUNet's next smaller step and the
    candidates are regenerated around it.
    """
    in_ch = modality_channels(dataset_json)
    budget = training_tensor_budget(memory_gb)
    envelope = initial_patch(spacing, median_shape)
    while True:
        candidates = []
        for patch in candidate_patches(envelope, spacing):
            architectures = {size: architecture_for_patch(patch, spacing, in_ch, size)
                             for size in MODEL_CAPACITIES}
            cost = estimate_training_tensors(architectures['L'], label_manager,
                                             BATCH_POLICY['minimum_batch_size'])['estimated_reserved_bytes']
            print(f'VeloxSeg candidate {patch_key(patch)}: L batch{BATCH_POLICY["minimum_batch_size"]} '
                  f'estimated memory {cost / 2**30:.3f} GiB', flush=True)
            candidates.append({'patch': patch, 'architectures': architectures, 'trainable': cost <= budget})
        if any(row['trainable'] for row in candidates):
            return candidates
        envelope = smaller_patch(geometry(envelope, spacing)[0], spacing, median_shape)


def patch_key(patch):
    return 'x'.join(str(n) for n in patch)


def select_patch(candidates, profile):
    """Choose a trainable crop from measured B batch-1 memory (M1) and latency (t1).

    Crops that are the same model up to an axis permutation count once. Drop
    crops dominated on (coverage >=, M1 <=, t1 <=, one strict). Among the
    survivors covering at least PATCH_POLICY['coverage_fraction'] of the largest
    trainable crop, take the least M1, then lower t1, then larger coverage.
    Returns the selected candidate and the candidate table.
    """
    trainable = [patch_key(row['patch']) for row in candidates if row['trainable']]
    missing = [key for key in trainable if 'B' not in profile['rows'].get(key, {})]
    if missing:
        raise ValueError(f'The VeloxSeg profile lacks B rows for trainable candidates {missing}. '
                         'Write the candidates with `veloxseg_planner candidates` and measure them with '
                         '`python nnunet/cost_profile.py --candidates <candidates.json> --output <profile.json>`')
    by_key = {patch_key(row['patch']): row for row in candidates}
    stale = [key for key in trainable
             if profile['rows'][key]['B']['architecture'] != by_key[key]['architectures']['B']['arch_kwargs']]
    if stale:
        raise ValueError(f'The VeloxSeg profile measured other B architectures for {stale}; the planning rules '
                         'changed since profiling. Regenerate the candidates and measure them again.')
    costs = {}
    for key in trainable:
        row = profile['rows'][key]['B']
        costs[key] = (prod(int(n) for n in key.split('x')), row['peak_allocated_mib'], row['median_ms'])
    # Crops with equal volume, parameters and measured memory are one model up to
    # an axis permutation. Latency of separate runs differs by noise, so keep the
    # first such crop in generation order (nnUNet's reduction order) instead.
    order = {patch_key(row['patch']): index for index, row in enumerate(candidates)}
    distinct = {}
    for key in sorted(trainable, key=order.get):
        distinct.setdefault((costs[key][0], profile['rows'][key]['B']['parameters'], costs[key][1]), key)
    distinct = sorted(distinct.values(), key=order.get)

    def dominated(key):
        cost = costs[key]
        return any(other != cost and other[0] >= cost[0] and other[1] <= cost[1] and other[2] <= cost[2]
                   for other in (costs[k] for k in distinct))

    pareto = [key for key in distinct if not dominated(key)]
    largest = max(cost[0] for cost in costs.values())
    eligible = [key for key in pareto if costs[key][0] / largest >= PATCH_POLICY['coverage_fraction']]
    selected = min(eligible, key=lambda key: (costs[key][1], costs[key][2], -costs[key][0]))
    table = [{'patch': row['patch'],
              'coverage_fraction': prod(row['patch']) / largest,
              'm1_mib': costs[key][1] if key in costs else None,
              't1_ms': costs[key][2] if key in costs else None,
              'trainable': row['trainable'], 'pareto': key in pareto, 'selected': key == selected}
             for row, key in ((row, patch_key(row['patch'])) for row in candidates)]
    return next(row for row in candidates if patch_key(row['patch']) == selected), table


def family_memory(architectures, label_manager, batch_size):
    inventories = {size: estimate_training_tensors(architecture, label_manager, batch_size)
                   for size, architecture in architectures.items()}
    return inventories, max(row['estimated_reserved_bytes'] for row in inventories.values())


def plan_family(spacing, median_shape, dataset_json, label_manager, memory_gb, profile):
    """Plan shared S/B/L geometry from nnUNet data rules and measured cost.

    Candidates around nnUNet's physical envelope are filtered by L training
    memory at the minimum batch, then chosen by select_patch from the B
    batch-1 profile. The chosen crop's spare training memory becomes batch.
    """
    if TRAINING_POLICY['total_updates'] % TRAINING_POLICY['num_iterations_per_epoch']:
        raise ValueError('total_updates must be a whole number of epochs')
    candidates = candidate_family(spacing, median_shape, dataset_json, label_manager, memory_gb)
    selected, table = select_patch(candidates, profile)
    architectures, patch = selected['architectures'], selected['patch']
    tensor_budget = training_tensor_budget(memory_gb)
    batch = BATCH_POLICY['minimum_batch_size']
    inventories, _ = family_memory(architectures, label_manager, batch)
    dataset_voxels = float(np.prod(median_shape, dtype=np.float64)) * dataset_json['numTraining']
    coverage_cap = round(dataset_voxels * BATCH_POLICY['max_dataset_covered'] / prod(patch))
    while 2 * batch <= coverage_cap:
        candidate, largest_cost = family_memory(architectures, label_manager, 2 * batch)
        if largest_cost > tensor_budget:
            break
        batch, inventories = 2 * batch, candidate
    epochs = TRAINING_POLICY['total_updates'] // TRAINING_POLICY['num_iterations_per_epoch']
    # nnUNet forces foreground crops only into the last batch slots.
    forced = batch - round(batch * (1 - TRAINING_POLICY['oversample_foreground_percent']))
    return {
        size: {
            'architecture': architecture,
            'batch_size': batch,
            'model_size': size,
            'training': {**TRAINING_POLICY, 'num_epochs': epochs},
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
                'patch_selection': {
                    'policy': PATCH_POLICY,
                    'profile_gpu': profile['gpu'],
                    'profile_torch': profile['torch'],
                    'candidates': table,
                },
            },
        }
        for size, architecture in architectures.items()
    }
