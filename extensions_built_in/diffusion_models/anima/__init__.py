from toolkit.util.dora_serialization import (
    convert_comfy_dora_scale_to_internal,
    convert_dora_magnitude_to_comfy,
)

from .anima import AnimaModel as _AnimaModel, AnimaPromptEmbeds


class AnimaModel(_AnimaModel):
    """Anima model with ComfyUI-compatible DoRA checkpoint serialization."""

    def convert_lora_weights_before_save(self, state_dict):
        state_dict = super().convert_lora_weights_before_save(state_dict)
        return convert_dora_magnitude_to_comfy(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = convert_comfy_dora_scale_to_internal(state_dict)
        return super().convert_lora_weights_before_load(state_dict)
