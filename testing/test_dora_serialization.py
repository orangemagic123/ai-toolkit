import unittest
from collections import OrderedDict

import torch

from toolkit.util.dora_serialization import (
    convert_comfy_dora_scale_to_internal,
    convert_dora_magnitude_to_comfy,
)


class DoRASerializationTest(unittest.TestCase):
    def test_save_conversion_uses_comfy_key_and_linear_shape(self):
        prefix = "diffusion_model.blocks.0.self_attn.q_proj"
        magnitude = torch.arange(8, dtype=torch.float32)
        state_dict = OrderedDict(
            {
                f"{prefix}.lora_A.weight": torch.randn(2, 8),
                f"{prefix}.lora_B.weight": torch.randn(8, 2),
                f"{prefix}.magnitude": magnitude,
            }
        )

        converted = convert_dora_magnitude_to_comfy(state_dict)

        self.assertNotIn(f"{prefix}.magnitude", converted)
        self.assertEqual(converted[f"{prefix}.dora_scale"].shape, (8, 1))
        torch.testing.assert_close(
            converted[f"{prefix}.dora_scale"].flatten(),
            magnitude,
        )
        self.assertIsInstance(converted, OrderedDict)

    def test_save_conversion_matches_up_weight_dimensions(self):
        prefix = "diffusion_model.conv"
        state_dict = {
            f"{prefix}.lora_B.weight": torch.randn(8, 2, 1, 1),
            f"{prefix}.magnitude": torch.randn(8),
        }

        converted = convert_dora_magnitude_to_comfy(state_dict)

        self.assertEqual(
            converted[f"{prefix}.dora_scale"].shape,
            (8, 1, 1, 1),
        )

    def test_load_conversion_accepts_comfy_shape_and_legacy_key(self):
        comfy_prefix = "diffusion_model.blocks.0.mlp.layer1"
        legacy_prefix = "diffusion_model.blocks.1.mlp.layer1"
        comfy_scale = torch.randn(16, 1)
        legacy_magnitude = torch.randn(16)
        state_dict = {
            f"{comfy_prefix}.dora_scale": comfy_scale,
            f"{legacy_prefix}.magnitude": legacy_magnitude,
        }

        converted = convert_comfy_dora_scale_to_internal(state_dict)

        self.assertNotIn(f"{comfy_prefix}.dora_scale", converted)
        self.assertEqual(converted[f"{comfy_prefix}.magnitude"].shape, (16,))
        torch.testing.assert_close(
            converted[f"{comfy_prefix}.magnitude"],
            comfy_scale.flatten(),
        )
        self.assertIs(
            converted[f"{legacy_prefix}.magnitude"],
            legacy_magnitude,
        )

    def test_round_trip_preserves_magnitude_values(self):
        prefix = "diffusion_model.blocks.0.cross_attn.k_proj"
        original = {
            f"{prefix}.lora_B.weight": torch.randn(12, 3),
            f"{prefix}.magnitude": torch.randn(12),
        }

        restored = convert_comfy_dora_scale_to_internal(
            convert_dora_magnitude_to_comfy(original)
        )

        torch.testing.assert_close(
            restored[f"{prefix}.magnitude"],
            original[f"{prefix}.magnitude"],
        )

    def test_invalid_or_duplicate_scale_is_rejected(self):
        prefix = "diffusion_model.blocks.0.self_attn.v_proj"
        with self.assertRaisesRegex(ValueError, "shape"):
            convert_comfy_dora_scale_to_internal(
                {f"{prefix}.dora_scale": torch.randn(8, 2)}
            )

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            convert_dora_magnitude_to_comfy(
                {
                    f"{prefix}.magnitude": torch.randn(8),
                    f"{prefix}.dora_scale": torch.randn(8, 1),
                }
            )


if __name__ == "__main__":
    unittest.main()
