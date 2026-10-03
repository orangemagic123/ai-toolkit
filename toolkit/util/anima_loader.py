"""Load Anima components from Diffusers sources and original (including expanded) checkpoints."""

import os

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
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
    from toolkit.models.v2.diffusion_models.cosmos import CosmosTransformer3DModel
    from toolkit.models.v2.text_encoders.anima import AnimaTextConditioner

    transformer_state, conditioner_state = split_anima_checkpoint(load_file(checkpoint_path, device="cpu"))
    transformer_config = CosmosTransformer3DModel.aitk_load_config(config_path, subfolder="transformer")
    # Base Anima has 28 layers; Anima-2.9B has 40. Never silently drop extra blocks.
    transformer_config["num_layers"] = get_anima_layer_count(transformer_state, "blocks.")
    transformer_state = convert_cosmos_transformer_checkpoint_to_diffusers(transformer_state)
    conditioner_config = AnimaTextConditioner.aitk_load_config(config_path, subfolder="text_conditioner")
    conditioner_config["num_layers"] = get_anima_layer_count(conditioner_state, "blocks.")

    # Strict loading catches incompatible checkpoints instead of training random weights.
    transformer = CosmosTransformer3DModel.load_from_state_dict(transformer_state, dtype, config=transformer_config)
    conditioner = AnimaTextConditioner.load_from_state_dict(conditioner_state, dtype, config=conditioner_config)
    del transformer_state, conditioner_state
    return transformer.eval(), conditioner.eval()


def _local_or_hub(name_or_path):
    local_path = os.path.abspath(os.path.expanduser(str(name_or_path)))
    return local_path if os.path.isdir(local_path) else str(name_or_path)


def load_anima_components(name_or_path, dtype, extras_name_or_path=None):
    """Load Anima's modules as v2 classes.

    Returns the components for AnimaModularPipeline.update_components() and the
    path that supplies the pipeline definition, VAE, text encoder and tokenizers.
    """
    from transformers import AutoTokenizer
    from toolkit.models.v2.diffusion_models.cosmos import CosmosTransformer3DModel
    from toolkit.models.v2.text_encoders.anima import AnimaTextConditioner
    from toolkit.models.v2.text_encoders.qwen3 import Qwen3ModelEncoder
    from toolkit.models.v2.vae.qwen_image import QwenImageVAE

    checkpoint_path = resolve_anima_checkpoint(name_or_path)
    if checkpoint_path is None:
        components_path = _local_or_hub(name_or_path)
        transformer = CosmosTransformer3DModel.load_model(components_path, dtype=dtype)
        text_conditioner = AnimaTextConditioner.load_model(components_path, dtype=dtype)
    else:
        # Single-file releases contain the DiT and adapter; reuse Anima's Qwen encoder,
        # tokenizers and VAE without downloading the base 28-layer transformer weights.
        extras_path = extras_name_or_path
        if not extras_path or extras_path == name_or_path:
            extras_path = ANIMA_BASE_REPO
        components_path = _local_or_hub(extras_path)
        transformer, text_conditioner = load_anima_single_file(checkpoint_path, components_path, dtype)

    components = {
        "transformer": transformer,
        "text_conditioner": text_conditioner,
        "vae": QwenImageVAE.load_model(components_path, dtype=dtype),
        "text_encoder": Qwen3ModelEncoder.load_model(components_path, dtype=dtype),
        "tokenizer": AutoTokenizer.from_pretrained(components_path, subfolder="tokenizer"),
        "t5_tokenizer": AutoTokenizer.from_pretrained(components_path, subfolder="t5_tokenizer"),
    }
    return components, components_path
