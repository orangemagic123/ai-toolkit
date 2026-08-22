import unittest

import torch

from toolkit.lora_special import LoRAModule


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        # The training process keeps the adapter multiplier with the FP32
        # master parameters. The forward path must not promote the activation.
        self.torch_multiplier = torch.ones(1, dtype=torch.float32)


def _capture_input_dtype(target, name):
    def hook(_module, args):
        target[name] = args[0].dtype

    return hook


class LoRAMixedPrecisionTest(unittest.TestCase):
    def test_bfloat16_activation_uses_autocast_with_float32_master_weights(self):
        torch.manual_seed(0)
        network = _Network()
        original = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = LoRAModule(
            "mixed_precision",
            original,
            lora_dim=4,
            alpha=4,
            network=network,
        )
        module.org_forward = original.forward

        # LoRA initializes the up projection to zero. Use non-zero values so
        # both projections receive gradients in this regression test.
        with torch.no_grad():
            module.lora_up.weight.normal_()

        seen_dtypes = {}
        module.lora_down.register_forward_pre_hook(
            _capture_input_dtype(seen_dtypes, "down")
        )
        module.lora_up.register_forward_pre_hook(
            _capture_input_dtype(seen_dtypes, "up")
        )

        value = torch.randn(2, 8, dtype=torch.bfloat16)
        output = module(value)
        output.float().square().mean().backward()

        self.assertEqual(seen_dtypes["down"], torch.bfloat16)
        self.assertEqual(seen_dtypes["up"], torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(module.lora_down.weight.dtype, torch.float32)
        self.assertEqual(module.lora_up.weight.dtype, torch.float32)
        self.assertEqual(module.lora_down.weight.grad.dtype, torch.float32)
        self.assertEqual(module.lora_up.weight.grad.dtype, torch.float32)

    def test_float32_forward_keeps_existing_behavior(self):
        network = _Network()
        original = torch.nn.Linear(8, 8, bias=False, dtype=torch.float32)
        original.requires_grad_(False)
        module = LoRAModule(
            "float32",
            original,
            lora_dim=4,
            alpha=4,
            network=network,
        )
        module.org_forward = original.forward

        seen_dtypes = {}
        module.lora_down.register_forward_pre_hook(
            _capture_input_dtype(seen_dtypes, "down")
        )

        output = module(torch.randn(2, 8, dtype=torch.float32))

        self.assertEqual(seen_dtypes["down"], torch.float32)
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
