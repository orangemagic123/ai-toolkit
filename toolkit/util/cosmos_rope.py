from typing import Any


_DYNAMIC_ROPE_HOOK_FLAG = "_aitk_dynamic_rope_max_size"


def expand_cosmos_rope_max_size(module: Any, args: tuple[Any, ...]) -> None:
    """Expand diffusers' Cosmos RoPE coordinate buffer for the current input.

    ``CosmosRotaryPosEmbed.forward`` creates one coordinate sequence whose
    length is ``max(module.max_size)``. Slicing that sequence for a larger
    height or width silently returns a shorter tensor and later fails in
    ``torch.cat``. Growing the limits before the forward keeps the original
    frequency calculation while allowing valid rectangular inputs.
    """
    if not args:
        return

    hidden_states = args[0]
    if getattr(hidden_states, "ndim", None) != 5:
        return

    patch_size = getattr(module, "patch_size", None)
    max_size = getattr(module, "max_size", None)
    if patch_size is None or max_size is None or len(patch_size) != 3 or len(max_size) != 3:
        return

    input_size = hidden_states.shape[-3:]
    required_size = [
        int(size) // int(patch)
        for size, patch in zip(input_size, patch_size)
    ]
    expanded_size = [
        max(int(current), required)
        for current, required in zip(max_size, required_size)
    ]

    if expanded_size != list(max_size):
        module.max_size = expanded_size


def install_dynamic_cosmos_rope(transformer: Any) -> None:
    """Install the dynamic-size fix once on a Cosmos transformer."""
    rope = getattr(transformer, "rope", None)
    if rope is None or getattr(rope, _DYNAMIC_ROPE_HOOK_FLAG, False):
        return

    rope.register_forward_pre_hook(expand_cosmos_rope_max_size)
    setattr(rope, _DYNAMIC_ROPE_HOOK_FLAG, True)

