"""One dataset-independent VeloxSeg planning policy.

Geometry comes from nnUNet 2.8.1; the shared model and optimization blueprint
comes from the VeloxSeg paper and released reference configuration. Training
memory and inference cost are measured on the target GPU by
nnunet/cost_profile.py; this module decides what to measure and what the
measurements imply.
"""
from fractions import Fraction
from itertools import product
from math import ceil, log2, prod

import numpy as np

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
# Training length is a sample budget: nnUNet's default 1000 epochs of 250
# iterations at batch 2 present 500,000 training samples. training_policy
# derives the epochs of a batch from it at the native 250 iterations per epoch.
TRAINING_POLICY = {
    'total_samples': 500000,
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
# The model size each measurement uses: L's training memory decides
# trainability and the shared batch, B's batch-1 inference cost the crop.
MEASURED_SIZES = {'training': 'L', 'inference': 'B'}
TRAINING_MEMORY_TARGET_GIB = 24
# Measured peaks count PyTorch's caching allocator only. The CUDA context and
# library handles live outside it: L training OOMed on a 23.69-GiB RTX 3090 with
# 23.34 GiB reserved. The reserve also covers a long run's allocations beyond
# the few measured steps.
TRAINING_RUNTIME_RESERVE_GIB = 2


def training_policy(batch_size):
    """The training block of a configuration at this batch.

    num_epochs = ceil(total_samples / (batch_size * num_iterations_per_epoch))
    and total_updates = num_epochs * num_iterations_per_epoch: batch 2 trains
    1000 epochs (250,000 updates), batch 8 250 (62,500), batch 32 63 (15,750).
    """
    iterations = TRAINING_POLICY['num_iterations_per_epoch']
    epochs = ceil(TRAINING_POLICY['total_samples'] / (batch_size * iterations))
    return {'total_samples': TRAINING_POLICY['total_samples'], 'num_epochs': epochs,
            'total_updates': epochs * iterations, **TRAINING_POLICY}


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
    """nnUNet's next smaller envelope, or None at the minimum legal geometry."""
    # Upstream relative-coverage axis order; unlike upstream, an axis that cannot
    # shrink legally passes the step to the next axis.
    for axis in np.argsort(np.asarray(patch) / np.asarray(median_shape))[::-1]:
        reduced = reduce_axis(patch, spacing, axis)
        if reduced is not None:
            return reduced
    return None


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


def patch_key(patch):
    return 'x'.join(str(n) for n in patch)


def training_memory_budget_gib(memory_gb):
    if memory_gb <= TRAINING_RUNTIME_RESERVE_GIB:
        raise ValueError('gpu_memory_target_in_gb must exceed the runtime reserve')
    return memory_gb - TRAINING_RUNTIME_RESERVE_GIB


def candidate_family(spacing, median_shape, dataset_json, memory_gb):
    """Candidate crops along nnUNet's envelope chain, the input of nnunet/cost_profile.py.

    The chain starts at nnUNet's initial envelope and takes its next smaller
    step down to the minimum legal geometry; each envelope lists its candidate
    crops. A crop carries its S/B/L architectures and the power-of-two batches
    from the minimum up to the dataset-voxel cap. measured_family decides which
    of them are measured and which envelope the plan uses.
    """
    in_ch = modality_channels(dataset_json)
    dataset_voxels = float(np.prod(median_shape, dtype=np.float64)) * dataset_json['numTraining']
    envelopes, candidates = [], {}
    envelope = geometry(initial_patch(spacing, median_shape), spacing)[0]
    while envelope is not None:
        patches = candidate_patches(envelope, spacing)
        for patch in patches:
            if patch_key(patch) in candidates:
                continue
            cap = round(dataset_voxels * BATCH_POLICY['max_dataset_covered'] / prod(patch))
            batches = [BATCH_POLICY['minimum_batch_size']]
            while 2 * batches[-1] <= cap:
                batches.append(2 * batches[-1])
            candidates[patch_key(patch)] = {
                'patch': patch, 'batch_coverage_cap': cap, 'batch_sizes': batches,
                'architectures': {size: architecture_for_patch(patch, spacing, in_ch, size)
                                  for size in MODEL_CAPACITIES}}
        envelopes.append([patch_key(patch) for patch in patches])
        envelope = smaller_patch(envelope, spacing, median_shape)
    return {'labels': dataset_json['labels'], 'regions_class_order': dataset_json.get('regions_class_order'),
            'training_memory_budget_gib': training_memory_budget_gib(memory_gb),
            'envelopes': envelopes, 'candidates': candidates}


class MissingMeasurement(ValueError):
    """The first measurement, in profiling order, that a profile lacks."""
    def __init__(self, request):
        self.request = request
        super().__init__(
            f"The VeloxSeg profile lacks {request['size']} {request['kind']} of {patch_key(request['patch'])} "
            f"at batch {request['batch']}. Write the candidates with `veloxseg_planner candidates` and measure "
            'them with `python nnunet/cost_profile.py --candidates <candidates.json> --output <profile.json>`')


def measurement(family, profile, kind, key, batch):
    """The profile row of one measurement; a row of another architecture is stale."""
    candidate = family['candidates'][key]
    size = MEASURED_SIZES[kind]
    architecture = candidate['architectures'][size]
    for row in profile['measurements']:
        if (row['kind'], row['patch'], row['size'], row['batch']) == (kind, candidate['patch'], size, batch):
            if row['architecture'] != architecture['arch_kwargs']:
                raise ValueError(f'The VeloxSeg profile measured another {size} architecture for {key}; the planning '
                                 'rules changed since profiling. Regenerate the candidates and measure them again.')
            return row
    raise MissingMeasurement({'kind': kind, 'patch': candidate['patch'], 'size': size, 'batch': batch,
                              'architecture': architecture})


def measured_family(family, profile):
    """The envelope the plan uses and the batch of each of its trainable crops.

    A crop trains at a batch when L's measured training peak reserved memory
    fits the budget without running out of memory. Envelopes are walked in
    chain order, measuring L at the minimum batch for each crop, until one has
    a trainable crop. Each trainable crop then needs B batch-1 inference and L
    at doubling batches until one does not fit or the coverage cap is reached;
    its batch is the largest that fits. The first measurement the profile lacks
    raises MissingMeasurement, which is how nnunet/cost_profile.py decides
    what to measure next.
    """
    budget_mib = family['training_memory_budget_gib'] * 2 ** 10

    def fits(key, batch):
        row = measurement(family, profile, 'training', key, batch)
        return not row['oom'] and row['peak_reserved_mib'] <= budget_mib

    for envelope in family['envelopes']:
        trainable = [key for key in envelope if fits(key, BATCH_POLICY['minimum_batch_size'])]
        if trainable:
            break
    else:
        raise ValueError('No candidate crop down to the minimum legal geometry trains L at the minimum batch '
                         'within the training memory budget')
    batches = {}
    for key in trainable:
        measurement(family, profile, 'inference', key, 1)
        batches[key] = BATCH_POLICY['minimum_batch_size']
        for batch in family['candidates'][key]['batch_sizes'][1:]:
            if not fits(key, batch):
                break
            batches[key] = batch
    return envelope, batches


def select_patch(family, envelope, batches, profile):
    """Choose a trainable crop from measured B batch-1 memory (M1) and latency (t1).

    Crops that are the same model up to an axis permutation count once. Drop
    crops dominated on (coverage >=, M1 <=, t1 <=, one strict). Among the
    survivors covering at least PATCH_POLICY['coverage_fraction'] of the largest
    trainable crop, take the least M1, then lower t1, then larger coverage.
    Returns the selected key and the candidate table of the envelope.
    """
    rows = {key: measurement(family, profile, 'inference', key, 1) for key in batches}
    costs = {key: (prod(family['candidates'][key]['patch']), row['peak_allocated_mib'], row['median_ms'])
             for key, row in rows.items()}
    # Crops with equal volume, parameters and measured memory are one model up to
    # an axis permutation. Latency of separate runs differs by noise, so keep the
    # first such crop in generation order (nnUNet's reduction order) instead.
    distinct = {}
    for key in batches:
        distinct.setdefault((costs[key][0], rows[key]['parameters'], costs[key][1]), key)
    distinct = list(distinct.values())

    def dominated(key):
        cost = costs[key]
        return any(other != cost and other[0] >= cost[0] and other[1] <= cost[1] and other[2] <= cost[2]
                   for other in (costs[k] for k in distinct))

    pareto = [key for key in distinct if not dominated(key)]
    largest = max(cost[0] for cost in costs.values())
    eligible = [key for key in pareto if costs[key][0] / largest >= PATCH_POLICY['coverage_fraction']]
    selected = min(eligible, key=lambda key: (costs[key][1], costs[key][2], -costs[key][0]))
    table = [{'patch': family['candidates'][key]['patch'],
              'coverage_fraction': prod(family['candidates'][key]['patch']) / largest,
              'm1_mib': costs[key][1] if key in costs else None,
              't1_ms': costs[key][2] if key in costs else None,
              'trainable': key in batches, 'pareto': key in pareto, 'selected': key == selected}
             for key in envelope]
    return selected, table


def plan_family(spacing, median_shape, dataset_json, memory_gb, profile):
    """Plan shared S/B/L geometry from nnUNet data rules and measured cost.

    measured_family finds the trainable crops and their L batches, select_patch
    chooses among them from the B batch-1 profile, and S/B/L share the chosen
    crop and its batch.
    """
    family = candidate_family(spacing, median_shape, dataset_json, memory_gb)
    envelope, batches = measured_family(family, profile)
    key, table = select_patch(family, envelope, batches, profile)
    candidate, batch = family['candidates'][key], batches[key]
    measured = [measurement(family, profile, 'training', key, size)
                for size in candidate['batch_sizes'] if size <= 2 * batch]
    # nnUNet forces foreground crops only into the last batch slots.
    forced = batch - round(batch * (1 - TRAINING_POLICY['oversample_foreground_percent']))
    return {
        size: {
            'architecture': architecture,
            'batch_size': batch,
            'model_size': size,
            'training': training_policy(batch),
            'resources': {
                'method': 'Measured on the profiling GPU: L training peak reserved memory bounds the crop and '
                          'batch; B batch-1 inference memory and latency select the crop',
                'training_memory_target_gib': memory_gb,
                'runtime_reserve_gib': TRAINING_RUNTIME_RESERVE_GIB,
                'training_memory_budget_gib': family['training_memory_budget_gib'],
                'batch_coverage_cap': candidate['batch_coverage_cap'],
                'forced_foreground_fraction': forced / batch,
                'measured_training_memory': [
                    {'model_size': row['size'], 'batch': row['batch'], 'oom': row['oom'],
                     'peak_reserved_gib': row['peak_reserved_mib'] / 2 ** 10} for row in measured],
                'patch_selection': {
                    'policy': PATCH_POLICY,
                    'profile_gpu': profile['gpu'],
                    'profile_torch': profile['torch'],
                    'candidates': table,
                },
            },
        }
        for size, architecture in candidate['architectures'].items()
    }
