from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


MAGNITUDE_SUFFIX = ".magnitude"
DORA_SCALE_SUFFIX = ".dora_scale"
LORA_UP_SUFFIXES = (
    ".lora_B.weight",
    ".lora_up.weight",
)


def _new_mapping_like(state_dict: Mapping[str, torch.Tensor]):
    if isinstance(state_dict, OrderedDict):
        return OrderedDict()
    return {}


def _find_lora_up_weight(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> torch.Tensor | None:
    for suffix in LORA_UP_SUFFIXES:
        weight = state_dict.get(f"{prefix}{suffix}")
        if weight is not None:
            return weight
    return None


def _reshape_dora_scale_for_comfy(
    value: torch.Tensor,
    up_weight: torch.Tensor | None,
) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("DoRA magnitude must have an output dimension")
    if any(dim != 1 for dim in value.shape[1:]):
        raise ValueError(
            "DoRA magnitude must contain only singleton trailing dimensions, "
            f"got {tuple(value.shape)}"
        )

    target_ndim = up_weight.ndim if up_weight is not None else max(value.ndim, 2)
    target_ndim = max(target_ndim, 2)
    return value.reshape(value.shape[0], *([1] * (target_ndim - 1)))


def convert_dora_magnitude_to_comfy(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert ai-toolkit DoRA magnitude entries to ComfyUI/LyCORIS format."""
    converted = _new_mapping_like(state_dict)

    for key, value in state_dict.items():
        output_key = key
        output_value = value

        if key.endswith(MAGNITUDE_SUFFIX):
            prefix = key[: -len(MAGNITUDE_SUFFIX)]
            output_key = f"{prefix}{DORA_SCALE_SUFFIX}"
            output_value = _reshape_dora_scale_for_comfy(
                value,
                _find_lora_up_weight(state_dict, prefix),
            )

        if output_key in converted:
            raise ValueError(
                f"Duplicate DoRA key after ComfyUI conversion: {output_key}"
            )
        converted[output_key] = output_value

    return converted


def convert_comfy_dora_scale_to_internal(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert ComfyUI/LyCORIS DoRA scale entries to ai-toolkit magnitude format."""
    converted = _new_mapping_like(state_dict)

    for key, value in state_dict.items():
        output_key = key
        output_value = value

        if key.endswith(DORA_SCALE_SUFFIX):
            prefix = key[: -len(DORA_SCALE_SUFFIX)]
            output_key = f"{prefix}{MAGNITUDE_SUFFIX}"
            if value.ndim == 0 or any(dim != 1 for dim in value.shape[1:]):
                raise ValueError(
                    "DoRA scale must have shape [out, 1, ...], "
                    f"got {tuple(value.shape)}"
                )
            output_value = value.reshape(value.shape[0])

        if output_key in converted:
            raise ValueError(
                f"Duplicate DoRA key after ai-toolkit conversion: {output_key}"
            )
        converted[output_key] = output_value

    return converted
