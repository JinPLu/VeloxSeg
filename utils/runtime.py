import math


def get_torch_device(gpu_id):
    import torch

    if str(gpu_id).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def set_cuda_device_if_available(device):
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.set_device(device)


def resolve_modal_index(select_modal, raw_modal_count):
    if raw_modal_count <= 0:
        raise ValueError("raw_modal_count must be greater than 0")
    if select_modal is None:
        return [1 for _ in range(raw_modal_count)]

    try:
        selected = int(select_modal)
    except (TypeError, ValueError) as exc:
        raise ValueError("--select_modal must be an integer index") from exc

    if selected < 0 or selected >= raw_modal_count:
        raise ValueError(
            f"--select_modal index {selected} is out of range for "
            f"{raw_modal_count} modalities"
        )

    modal_index = [0 for _ in range(raw_modal_count)]
    modal_index[selected] = 1
    return modal_index


def expected_input_channels(model_name, model_config):
    config = model_config.get(model_name)
    if config is None:
        return None

    in_ch = config.get("in_ch")
    if isinstance(in_ch, list):
        return sum(int(channel) for channel in in_ch)
    if isinstance(in_ch, int):
        return in_ch

    for key in (
        "in_channels",
        "input_channel",
        "num_input_channels",
        "input_channels",
        "init_channels",
        "model_num",
        "modality_num",
    ):
        value = config.get(key)
        if isinstance(value, int):
            return value

    return None


def validate_selected_modal(model_name, model_config, raw_modal_count, select_modal):
    modal_index = resolve_modal_index(select_modal, raw_modal_count)
    selected_channels = sum(modal_index)
    expected_channels = expected_input_channels(model_name, model_config)

    if expected_channels is not None and selected_channels != expected_channels:
        raise ValueError(
            f"Model {model_name} expects {expected_channels} input channel(s), "
            f"but the selected modalities provide {selected_channels}. Use a "
            "matching model config/checkpoint or omit --select_modal."
        )

    return modal_index


def select_modal_items(items, modal_index):
    if len(items) != len(modal_index):
        raise ValueError(
            f"modal_index length {len(modal_index)} must match item count {len(items)}"
        )
    selected = [item for item, enabled in zip(items, modal_index) if enabled]
    if not selected:
        raise ValueError("At least one modality must be selected")
    return selected


def validate_file_groups(dataset_name, file_groups):
    counts = {name: len(paths) for name, paths in file_groups.items()}
    empty_groups = [name for name, count in counts.items() if count == 0]
    if empty_groups:
        raise ValueError(
            f"No files matched for {dataset_name}: {', '.join(empty_groups)}"
        )

    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(
            f"Dataset {dataset_name} modalities/labels must have the same "
            f"number of files before pairing; got {details}."
        )

    return next(iter(unique_counts))


def rotation_range_from_degrees(degrees):
    return math.radians(float(degrees))


def image_label_modes(image_key_count):
    if image_key_count <= 0:
        raise ValueError("image_key_count must be greater than 0")
    return tuple(["bilinear"] * image_key_count + ["nearest"])


def normalized_deep_loss_weights(configured_weights, output_count):
    if output_count <= 0:
        raise ValueError("output_count must be greater than 0")

    weights = [float(weight) for weight in configured_weights]
    if not weights:
        raise ValueError("deep_Loss_weight must contain at least one value")
    if sum(weights) == 0:
        raise ValueError("deep_Loss_weight sum must be non-zero")

    if len(weights) != output_count:
        if all(weight == weights[0] for weight in weights):
            return [1.0 / output_count for _ in range(output_count)]
        raise ValueError(
            "deep_Loss_weight length must match model deep-supervision outputs "
            "unless all configured weights are equal"
        )

    total = sum(weights)
    return [weight / total for weight in weights]


def a2fseg_deep_output_groups(output_count, group_size=5):
    if output_count <= 1 or (output_count - 1) % group_size != 0:
        raise ValueError(
            f"A2FSeg output count {output_count} must be 1 + N * {group_size}"
        )
    return [
        (start, start + group_size)
        for start in range(1, output_count, group_size)
    ]


def veloxseg_output_layout(output_count, num_modal):
    tail_count = 2 + int(num_modal)
    if output_count <= tail_count:
        raise ValueError(
            f"VeloxSeg output count {output_count} is too small for "
            f"{num_modal} modality reconstruction outputs"
        )

    seg_output_count = output_count - tail_count
    return {
        "seg": (0, seg_output_count),
        "reconstruction": seg_output_count,
        "decoder_gram": seg_output_count + 1,
        "teacher_grams": tuple(
            range(seg_output_count + 2, seg_output_count + 2 + int(num_modal))
        ),
    }
