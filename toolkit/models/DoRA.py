# based off https://github.com/catid/dora/blob/main/dora.py
import math
from typing import TYPE_CHECKING, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.quanto import QBytesTensor, QTensor

from toolkit.network_mixins import (
    ExtractableModuleMixin,
    ToolkitModuleMixin,
    broadcast_and_multiply,
)

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork


# diffusers specific stuff
LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "OstrisLinear",
    # "GroupNorm",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
]


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


class DoRAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        network: "LoRASpecialNetwork" = None,
        use_bias: bool = False,
        **kwargs,
    ):
        self.can_merge_in = False
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in CONV_MODULES:
            raise NotImplementedError("Convolutional layers are not supported yet")

        if type(alpha) == torch.Tensor:
            alpha = float(alpha.detach().float().item())
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / self.lora_dim

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

        d_out = org_module.out_features
        d_in = org_module.in_features

        std_dev = 1 / torch.sqrt(torch.tensor(self.lora_dim).float())
        self.lora_up = nn.Linear(self.lora_dim, d_out, bias=False)  # lora_B
        self.lora_up.weight.data = torch.zeros_like(self.lora_up.weight.data)
        self.lora_down = nn.Linear(d_in, self.lora_dim, bias=False)  # lora_A
        self.lora_down.weight.data = torch.randn_like(self.lora_down.weight.data) * std_dev

        # m = Magnitude column-wise across output dimension
        weight = self.get_orig_weight()
        weight = weight.to(self.lora_up.weight.device, dtype=self.lora_up.weight.dtype)
        lora_weight = self.lora_up.weight @ self.lora_down.weight
        weight_norm = self._get_weight_norm(weight, lora_weight)
        self.magnitude = nn.Parameter(weight_norm.detach().clone(), requires_grad=True)

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, (QTensor, QBytesTensor)):
            return weight.dequantize().data.detach()
        return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], "bias") and self.org_module[0].bias is not None:
            return self.org_module[0].bias.data.detach()
        return None

    def _get_weight_norm(self, weight, scaled_lora_weight) -> torch.Tensor:
        weight = weight + scaled_lora_weight.to(weight.device)
        return torch.linalg.norm(weight, dim=1)

    @torch.no_grad()
    def _get_weight_norm_factorized(
        self,
        multiplier: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
        use_autocast: bool,
    ) -> torch.Tensor:
        """Return ||W + scale * B @ A|| per output row without forming B @ A."""
        weight = self.get_orig_weight().to(device=device, dtype=compute_dtype)

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=compute_dtype):
                weight_a_t = F.linear(weight, self.lora_down.weight)
        else:
            weight_a_t = F.linear(weight, self.lora_down.weight)

        # Keep the large frozen weight in the model compute dtype. Only the
        # small rank-space reductions are accumulated in FP32 for stability.
        base_norm_sq = torch.linalg.vector_norm(
            weight,
            dim=1,
            dtype=torch.float32,
        ).square()
        lora_a = self.lora_down.weight.float()
        lora_b = self.lora_up.weight.float()
        weight_a_t = weight_a_t.float()
        gram = lora_a @ lora_a.transpose(0, 1)
        cross = (weight_a_t * lora_b).sum(dim=1)
        delta_norm_sq = ((lora_b @ gram) * lora_b).sum(dim=1)

        adapter_scale = multiplier.float() * self.scale
        norm_sq = (
            base_norm_sq
            + 2.0 * adapter_scale * cross
            + adapter_scale.square() * delta_norm_sq
        )
        return norm_sq.clamp_min(0.0).sqrt()

    def forward(self, x, *args, **kwargs):
        network = self.network_ref()
        if (
            network.is_lorm
            or not network.is_active
            or network.is_merged_in
            or network._multiplier == 0
        ):
            return self.org_forward(x, *args, **kwargs)

        base_output = self.org_forward(x, *args, **kwargs)

        if isinstance(x, (QTensor, QBytesTensor)):
            x = x.dequantize()

        compute_dtype = base_output.dtype
        use_mixed_precision = (
            x.device.type == "cuda"
            and compute_dtype in (torch.float16, torch.bfloat16)
        ) or (x.device.type == "cpu" and compute_dtype == torch.bfloat16)

        if use_mixed_precision:
            lora_input = x if x.dtype == compute_dtype else x.to(compute_dtype)
            with torch.autocast(device_type=x.device.type, dtype=compute_dtype):
                lora_output = self._call_forward(lora_input)
        else:
            lora_input = x.to(self.lora_down.weight.dtype)
            lora_output = self._call_forward(lora_input)

        # Module dropout skips the complete adapter, including magnitude.
        if not isinstance(lora_output, torch.Tensor):
            return base_output

        multiplier = network.torch_multiplier.to(
            device=lora_output.device,
            dtype=lora_output.dtype,
        )
        if lora_output.size(0) != multiplier.size(0):
            num_interleaves = lora_output.size(0) // multiplier.size(0)
            multiplier = multiplier.repeat_interleave(num_interleaves)

        scaled_lora_output = broadcast_and_multiply(lora_output, multiplier)
        scaled_lora_output = scaled_lora_output.to(base_output.dtype)

        # Weight-space DoRA normalization cannot vary per sample. Preserve the
        # existing mean-multiplier behavior for split-batch adapter strengths.
        norm_multiplier = multiplier.mean()
        norm_compute_dtype = (
            compute_dtype if use_mixed_precision else self.lora_down.weight.dtype
        )
        weight_norm = self._get_weight_norm_factorized(
            multiplier=norm_multiplier,
            device=lora_output.device,
            compute_dtype=norm_compute_dtype,
            use_autocast=use_mixed_precision,
        )
        weight_norm = weight_norm.clamp_min(torch.finfo(weight_norm.dtype).eps)

        magnitude_scale = self.magnitude.to(
            device=weight_norm.device,
            dtype=weight_norm.dtype,
        ) / weight_norm
        magnitude_scale = magnitude_scale.to(base_output.dtype)
        view_shape = (1,) * (base_output.dim() - 1) + (-1,)
        magnitude_scale = magnitude_scale.view(view_shape)

        # The base result is already available. Reuse it exactly as PEFT DoRA
        # does instead of running a second full-rank F.linear over the activation.
        # Bias is not part of the directional weight and must remain unscaled.
        base_direction = base_output
        bias = self.get_orig_bias()
        if bias is not None:
            bias = bias.to(device=base_output.device, dtype=base_output.dtype)
            base_direction = base_direction - bias.view(view_shape)

        dora_delta = (
            (magnitude_scale - 1.0) * base_direction
            + magnitude_scale * scaled_lora_output
        )
        return base_output + dora_delta

    def apply_dora(self, x, scaled_lora_weight):
        # Retained for compatibility with older external callers. The normal
        # forward path above avoids this extra full-rank activation GEMM.
        weight = self.get_orig_weight()
        weight = weight.to(
            scaled_lora_weight.device,
            dtype=scaled_lora_weight.dtype,
        )
        weight_norm = self._get_weight_norm(weight, scaled_lora_weight).detach()
        dora_weight = transpose(weight + scaled_lora_weight, False)
        return (self.magnitude / weight_norm - 1).view(1, -1) * F.linear(
            x.to(dora_weight.dtype),
            dora_weight,
        )
