import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from toolkit.models.DoRA import DoRAModule


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self, multiplier=1.0):
        self.torch_multiplier = torch.tensor([multiplier], dtype=torch.float32)


def _capture_input_dtype(target, name):
    def hook(_module, args):
        target[name] = args[0].dtype

    return hook


class DoRAFastForwardTest(unittest.TestCase):
    def test_forward_and_gradients_match_explicit_dora(self):
        torch.manual_seed(0)
        network = _Network(multiplier=0.7)
        original = torch.nn.Linear(8, 6, bias=True)
        original.requires_grad_(False)
        module = DoRAModule(
            "dora_equivalence",
            original,
            lora_dim=3,
            alpha=6,
            network=network,
        )
        module.org_forward = original.forward

        with torch.no_grad():
            module.lora_up.weight.normal_()
            module.magnitude.mul_(1.1)

        value = torch.randn(2, 5, 8)
        actual = module(value)

        adapter_scale = module.scale * network.torch_multiplier.mean()
        lora_weight = module.lora_up.weight @ module.lora_down.weight
        adapted_weight = original.weight.detach() + adapter_scale * lora_weight
        weight_norm = torch.linalg.vector_norm(
            adapted_weight.detach(),
            dim=1,
        )
        magnitude_scale = module.magnitude / weight_norm
        expected = F.linear(
            value,
            magnitude_scale[:, None] * adapted_weight,
            original.bias,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        parameters = [
            module.lora_down.weight,
            module.lora_up.weight,
            module.magnitude,
        ]
        actual_grads = torch.autograd.grad(
            actual.square().mean(),
            parameters,
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(
            expected.square().mean(),
            parameters,
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_forward_uses_only_one_full_activation_gemm(self):
        network = _Network()
        original = torch.nn.Linear(8, 6, bias=False)
        original.requires_grad_(False)
        module = DoRAModule(
            "dora_gemm_count",
            original,
            lora_dim=3,
            alpha=3,
            network=network,
        )
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(2, 5, 8)
        calls = []
        original_linear = F.linear

        def record_linear(input_tensor, weight, bias=None):
            calls.append((tuple(input_tensor.shape), tuple(weight.shape)))
            return original_linear(input_tensor, weight, bias)

        with mock.patch("torch.nn.functional.linear", side_effect=record_linear):
            module(value)

        full_activation_calls = sum(
            input_shape == tuple(value.shape)
            and weight_shape == tuple(original.weight.shape)
            for input_shape, weight_shape in calls
        )
        self.assertEqual(full_activation_calls, 1)

    def test_bfloat16_activations_keep_float32_master_gradients(self):
        torch.manual_seed(0)
        network = _Network()
        original = torch.nn.Linear(
            8,
            6,
            bias=False,
            dtype=torch.bfloat16,
        )
        original.requires_grad_(False)
        module = DoRAModule(
            "dora_bfloat16",
            original,
            lora_dim=3,
            alpha=3,
            network=network,
        )
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        seen_dtypes = {}
        module.lora_down.register_forward_pre_hook(
            _capture_input_dtype(seen_dtypes, "down")
        )
        module.lora_up.register_forward_pre_hook(
            _capture_input_dtype(seen_dtypes, "up")
        )

        value = torch.randn(2, 5, 8, dtype=torch.bfloat16)
        output = module(value)
        output.float().square().mean().backward()

        self.assertEqual(seen_dtypes["down"], torch.bfloat16)
        self.assertEqual(seen_dtypes["up"], torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(module.lora_down.weight.dtype, torch.float32)
        self.assertEqual(module.lora_up.weight.dtype, torch.float32)
        self.assertEqual(module.lora_down.weight.grad.dtype, torch.float32)
        self.assertEqual(module.lora_up.weight.grad.dtype, torch.float32)
        self.assertEqual(module.magnitude.grad.dtype, torch.float32)

    def test_module_dropout_returns_unmodified_base_output(self):
        network = _Network()
        original = torch.nn.Linear(8, 6, bias=True)
        original.requires_grad_(False)
        module = DoRAModule(
            "dora_module_dropout",
            original,
            lora_dim=3,
            alpha=3,
            module_dropout=1.0,
            network=network,
        )
        module.org_forward = original.forward
        module.train()

        value = torch.randn(2, 5, 8)
        torch.testing.assert_close(module(value), original(value))


if __name__ == "__main__":
    unittest.main()
