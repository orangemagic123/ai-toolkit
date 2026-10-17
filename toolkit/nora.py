"""NoRA: Normalized Low-Rank Adaptation (arXiv 2608.31036).

NoRA scales every column of the LoRA down projection A ([rank, input_dim]) to
unit L2 norm along the rank dimension, so each input coordinate is projected
through a unit-length latent direction.

- ``init``: normalize A once after the module's default initialization, then
  train it as a plain LoRA (the paper's NoRA-init).
- ``continuous``: keep the raw A at the default kaiming scale and use Norm(A)
  in the forward pass, so gradients flow through the normalization (the
  paper's NoRA). Exports store Norm(A), which makes them ordinary LoRA files.
"""
import math
from typing import Literal, Optional

import torch

NoraMode = Literal["off", "init", "continuous"]
NORA_MODES = ("off", "init", "continuous")
NORA_EPS = 1e-6
# bf16 exports round each element to ~0.4%, so a saved Norm(A) column can sit
# slightly off 1. Anything beyond this was not trained with continuous NoRA.
NORA_LOAD_TOLERANCE = 0.02


def parse_nora_mode(value) -> NoraMode:
    # YAML's SafeLoader reads an unquoted `off` as False, so both mean off.
    if value is None or value is False:
        return "off"
    if value is True:
        raise ValueError("network.nora must name a mode: 'init' or 'continuous'")
    if isinstance(value, str) and value.strip().lower() in NORA_MODES:
        return value.strip().lower()
    raise ValueError(
        f"network.nora must be one of 'off', 'init' or 'continuous', got {value!r}"
    )


def normalize_down_weight(weight: torch.Tensor, eps: float = NORA_EPS) -> torch.Tensor:
    """Return A with every input column scaled to unit norm over the rank dim.

    Handles Linear ([rank, in]) and Conv ([rank, in, kh, kw]) weights. The norm
    is accumulated in FP32, the result keeps the weight dtype, and autograd
    flows through the normalization.
    """
    flat = weight.reshape(weight.shape[0], -1).float()
    norm = torch.linalg.vector_norm(flat, dim=0, keepdim=True)
    return (flat / norm.clamp_min(eps)).reshape(weight.shape).to(weight.dtype)


@torch.no_grad()
def apply_nora_init_(weight: torch.Tensor) -> None:
    weight.copy_(normalize_down_weight(weight))


@torch.no_grad()
def column_norm_deviation(weight: torch.Tensor) -> float:
    """Largest |‖A[:, j]‖ - 1| over the non-zero columns of A."""
    norms = torch.linalg.vector_norm(
        weight.detach().reshape(weight.shape[0], -1).float(), dim=0
    )
    norms = norms[norms > 0]
    if norms.numel() == 0:
        return 0.0
    return (norms - 1.0).abs().max().item()


def nora_gain(weight: torch.Tensor) -> float:
    """sqrt(3k / r): the unit NoRA column norm over the expected column norm
    sqrt(r / 3k) of the default kaiming_uniform_(a=sqrt(5)) LoRA init, where k
    is the input dimension (in_channels * kh * kw for conv)."""
    return math.sqrt(3.0 * weight[0].numel() / weight.shape[0])


def check_nora_resume_mode(saved_mode, current_mode, checkpoint) -> None:
    """Refuse to resume across continuous and non-continuous NoRA.

    A continuous checkpoint exports Norm(A), but its paired training state and
    optimizer.pt hold the raw A, which a plain LoRA would apply as is. In the
    other direction, continuous NoRA renormalizes a plain LoRA's A and changes
    its output. init and off read A the same way, so switching between them is
    an ordinary resume.
    """
    saved_mode = parse_nora_mode(saved_mode)
    if (saved_mode == "continuous") == (current_mode == "continuous"):
        return
    raise ValueError(
        f"{checkpoint} was trained with network.nora: {saved_mode}, but the config now sets "
        f"network.nora: {current_mode}. Resuming across continuous and non-continuous NoRA would "
        f"reinterpret the saved raw weights and optimizer state. Use a new job name to start over, "
        f"or warm-start a new job by setting network.pretrained_lora_path to this checkpoint."
    )


def nora_resume_warning(network) -> Optional[str]:
    if getattr(network, "nora_mode", "off") != "continuous":
        return None
    return (
        "WARNING: Continuous NoRA resumed without its paired training state. Training "
        "continues from the exported Norm(A) (unit columns) instead of the raw A, so "
        "the adapter output is unchanged but A's direction now updates more slowly."
    )
