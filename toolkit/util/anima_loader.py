"""Load Diffusers Anima pipelines and original (including expanded) checkpoints."""

import os

from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


ANIMA_BASE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ANIMA_29B_REPO = "Gazingstars123/Anima-2.9B"
ANIMA_29B_FILENAME = "Anima-2.9B-preview-v1.safetensors"


def resolve_anima_checkpoint(name_or_path):
    """Return an original checkpoint path, or None for a Diffusers pipeline."""
    name_or_path = str(name_or_path)
    local_path = os.path.abspath(os.path.expanduser(name_or_path))
    if os.path.isfile(local_path):
        if not local_path.lower().endswith(".safetensors"):
            raise ValueError("Anima single-file checkpoints must use .safetensors")
        return local_path
    if os.path.isdir(local_path):
        checkpoint = os.path.join(local_path, ANIMA_29B_FILENAME)
        if os.path.isfile(checkpoint):
            return checkpoint
        return None
    if name_or_path.rstrip("/") == ANIMA_29B_REPO:
        # Always choose the trainable BF16 release, never the INT8 inference file.
        return hf_hub_download(repo_id=ANIMA_29B_REPO, filename=ANIMA_29B_FILENAME)
    if name_or_path.lower().endswith(".safetensors"):
        raise FileNotFoundError(f"Anima checkpoint does not exist: {name_or_path}")
    return None


def split_anima_checkpoint(state_dict):
    """Separate the DiT and LLM adapter before Cosmos key conversion."""
    transformer = {}
    conditioner = {}
    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        if key.endswith(".weight") and not value.is_floating_point():
            raise ValueError("Anima training requires floating-point weights; use the BF16 checkpoint.")
        if key.startswith("llm_adapter."):
            target, key = conditioner, key.removeprefix("llm_adapter.")
        else:
            target = transformer
        if key in target:
            raise ValueError(f"Duplicate Anima checkpoint key: {key}")
        target[key] = value
    if not transformer or not conditioner:
        raise ValueError("Anima checkpoint must contain both the transformer and llm_adapter weights.")
    return transformer, conditioner


def get_anima_layer_count(state_dict, prefix):
    indices = {
        int(key[len(prefix):].split(".", 1)[0])
        for key in state_dict
        if key.startswith(prefix) and key[len(prefix):].split(".", 1)[0].isdigit()
    }
    if not indices or indices != set(range(max(indices) + 1)):
        raise ValueError(f"Anima checkpoint has missing or non-contiguous {prefix} layers.")
    return len(indices)


def load_anima_single_file(checkpoint_path, config_path, dtype):
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    transformer_state, conditioner_state = split_anima_checkpoint(load_file(checkpoint_path, device="cpu"))
    transformer_config = CosmosTransformer3DModel.load_config(config_path, subfolder="transformer")
    # Base Anima has 28 layers; Anima-2.9B has 40. Never silently drop extra blocks.
    transformer_config["num_layers"] = get_anima_layer_count(transformer_state, "blocks.")
    transformer_state = convert_cosmos_transformer_checkpoint_to_diffusers(transformer_state)
    conditioner_config = AnimaTextConditioner.load_config(config_path, subfolder="text_conditioner")
    conditioner_config["num_layers"] = get_anima_layer_count(conditioner_state, "blocks.")

    with init_empty_weights():
        transformer = CosmosTransformer3DModel.from_config(transformer_config)
        conditioner = AnimaTextConditioner.from_config(conditioner_config)
    # Strict loading catches incompatible checkpoints instead of training random weights.
    transformer.load_state_dict(transformer_state, strict=True, assign=True)
    conditioner.load_state_dict(conditioner_state, strict=True, assign=True)
    del transformer_state, conditioner_state
    return transformer.to(dtype=dtype).eval(), conditioner.to(dtype=dtype).eval()


def load_anima_pipeline(name_or_path, dtype, extras_name_or_path=None):
    from diffusers import AnimaAutoBlocks

    checkpoint_path = resolve_anima_checkpoint(name_or_path)
    if checkpoint_path is None:
        pipe = AnimaAutoBlocks().init_pipeline(name_or_path)
        load_kwargs = {"torch_dtype": dtype}
        model_path = os.path.abspath(os.path.expanduser(str(name_or_path)))
        if os.path.isdir(model_path):
            load_kwargs["pretrained_model_name_or_path"] = model_path
        pipe.load_components(**load_kwargs)
        return pipe

    # Single-file releases contain the DiT and adapter; reuse Anima's Qwen encoder,
    # tokenizers and VAE without downloading the base 28-layer transformer weights.
    extras_path = extras_name_or_path
    if not extras_path or extras_path == name_or_path:
        extras_path = ANIMA_BASE_REPO
    extras_path = str(extras_path)
    local_extras_path = os.path.abspath(os.path.expanduser(extras_path))
    if os.path.isdir(local_extras_path):
        extras_path = local_extras_path
    transformer, conditioner = load_anima_single_file(checkpoint_path, extras_path, dtype)
    pipe = AnimaAutoBlocks().init_pipeline(extras_path)
    pipe.update_components(transformer=transformer, text_conditioner=conditioner)
    pipe.load_components(
        names=["text_encoder", "tokenizer", "t5_tokenizer", "vae", "scheduler"],
        pretrained_model_name_or_path=extras_path,
        torch_dtype=dtype,
    )
    return pipe
