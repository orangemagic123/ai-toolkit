import unittest

import torch

from toolkit.util.cosmos_rope import install_dynamic_cosmos_rope


class FakeCosmosRotaryPosEmbed(torch.nn.Module):
    """Minimal reproduction of diffusers' coordinate sequence behavior."""

    def __init__(self):
        super().__init__()
        self.max_size = [128, 120, 120]
        self.patch_size = (1, 2, 2)

    def forward(self, hidden_states):
        _, _, frames, height, width = hidden_states.shape
        pe_size = [
            frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]
        seq = torch.arange(max(self.max_size))

        emb_t = torch.empty(pe_size[0], pe_size[1], pe_size[2], 1)
        emb_h = torch.empty(pe_size[0], len(seq[: pe_size[1]]), pe_size[2], 1)
        emb_w = torch.empty(pe_size[0], pe_size[1], len(seq[: pe_size[2]]), 1)
        return torch.cat([emb_t, emb_h, emb_w], dim=-1)


class DynamicCosmosRopeTest(unittest.TestCase):
    def test_rectangular_grid_larger_than_initial_sequence_is_supported(self):
        rope = FakeCosmosRotaryPosEmbed()
        hidden_states = torch.empty(1, 16, 1, 192, 336)

        with self.assertRaisesRegex(RuntimeError, "Expected size 168 but got size 128"):
            rope(hidden_states)

        install_dynamic_cosmos_rope(type("Transformer", (), {"rope": rope})())
        output = rope(hidden_states)

        self.assertEqual(output.shape, (1, 96, 168, 3))
        self.assertEqual(rope.max_size, [128, 120, 168])

    def test_hook_is_idempotent_and_keeps_existing_capacity(self):
        rope = FakeCosmosRotaryPosEmbed()
        transformer = type("Transformer", (), {"rope": rope})()

        install_dynamic_cosmos_rope(transformer)
        install_dynamic_cosmos_rope(transformer)
        rope(torch.empty(1, 16, 1, 64, 64))

        self.assertEqual(len(rope._forward_pre_hooks), 1)
        self.assertEqual(rope.max_size, [128, 120, 120])


if __name__ == "__main__":
    unittest.main()

