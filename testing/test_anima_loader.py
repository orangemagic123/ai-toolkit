"""CPU loading/backprop regressions; uses tiny Diffusers models, no downloads.

Run: python -m unittest testing.test_anima_loader testing.test_anima_rope
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
from safetensors.torch import save_file

from toolkit.util import anima_loader as loader


def original_transformer_keys(state_dict):
    # Original Cosmos/Anima checkpoint names, independent of the loader's
    # Diffusers conversion function. Include attention norms and output weights.
    names = {
        "transformer_blocks.": "blocks.",
        "norm1.linear_": "adaln_modulation_self_attn.",
        "norm2.linear_": "adaln_modulation_cross_attn.",
        "norm3.linear_": "adaln_modulation_mlp.",
        "attn1.": "self_attn.",
        "attn2.": "cross_attn.",
        "to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj",
        "to_out.0": "output_proj", "norm_q": "q_norm", "norm_k": "k_norm",
        "ff.net.0.proj": "mlp.layer1", "ff.net.2": "mlp.layer2",
        "norm_out.linear_": "final_layer.adaln_modulation.",
        "proj_out": "final_layer.linear",
        "time_embed.t_embedder": "t_embedder.1",
        "time_embed.norm": "t_embedding_norm",
        "patch_embed.proj": "x_embedder.proj.1",
    }
    result = {}
    for key, tensor in state_dict.items():
        for old, new in names.items():
            key = key.replace(old, new)
        result[f"net.{key}"] = tensor
    return result


class AnimaCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.transformer = CosmosTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=16,
            num_layers=40, mlp_ratio=2, text_embed_dim=16, adaln_lora_dim=8,
            max_size=(2, 8, 8), patch_size=(1, 2, 2), extra_pos_embed_type=None,
        )
        cls.conditioner = AnimaTextConditioner(
            source_dim=16, target_dim=16, model_dim=16, num_layers=2,
            num_attention_heads=2, target_vocab_size=32, min_sequence_length=4,
        )
        cls.transformer.save_config(cls.root / "transformer")
        cls.conditioner.save_config(cls.root / "text_conditioner")
        # Reproduce the actual mismatch: base config has 28, checkpoint has 40.
        import json
        config_path = cls.root / "transformer" / "config.json"
        config = json.loads(config_path.read_text())
        config["num_layers"] = 28
        config_path.write_text(json.dumps(config))
        cls.state = original_transformer_keys(cls.transformer.state_dict())
        cls.state.update({f"net.llm_adapter.{k}": v for k, v in cls.conditioner.state_dict().items()})
        cls.path = cls.root / loader.ANIMA_29B_FILENAME
        save_file(cls.state, str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_40_blocks_and_conditioner_load_and_train(self):
        transformer, conditioner = loader.load_anima_single_file(str(self.path), str(self.root), torch.float32)
        self.assertEqual(len(transformer.transformer_blocks), 40)
        self.assertEqual(transformer.config.num_layers, 40)
        for source, loaded in [(self.transformer, transformer), (self.conditioner, conditioner)]:
            self.assertEqual(source.state_dict().keys(), loaded.state_dict().keys())
            for key, tensor in source.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
        transformer.enable_gradient_checkpointing()
        conditioner.enable_gradient_checkpointing()
        context = conditioner(torch.randn(1, 4, 16), torch.tensor([[1, 2, 3, 4]]))
        output = transformer(
            hidden_states=torch.randn(1, 4, 1, 4, 4), timestep=torch.tensor([0.5]),
            encoder_hidden_states=context, padding_mask=torch.zeros(1, 1, 4, 4),
        ).sample
        self.assertEqual(output.shape, (1, 4, 1, 4, 4))
        output.square().mean().backward()
        grad = transformer.transformer_blocks[39].attn1.to_q.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertIsNotNone(conditioner.embed.weight.grad)

    def test_missing_weights_fail_instead_of_random_initialization(self):
        broken = dict(self.state)
        del broken["net.blocks.39.self_attn.q_proj.weight"]
        path = self.root / "broken.safetensors"
        save_file(broken, str(path))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            loader.load_anima_single_file(str(path), str(self.root), torch.float32)

    def test_repo_selects_bf16_and_local_files_need_no_download(self):
        with patch.object(loader, "hf_hub_download", return_value=str(self.path)) as download:
            self.assertEqual(loader.resolve_anima_checkpoint(loader.ANIMA_29B_REPO), str(self.path))
            download.assert_called_once_with(repo_id=loader.ANIMA_29B_REPO, filename=loader.ANIMA_29B_FILENAME)
            download.reset_mock()
            self.assertEqual(loader.resolve_anima_checkpoint(self.path), str(self.path))
            self.assertEqual(loader.resolve_anima_checkpoint(self.root), str(self.path))
            self.assertIsNone(loader.resolve_anima_checkpoint(loader.ANIMA_BASE_REPO))
            download.assert_not_called()

    def test_missing_local_checkpoint_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            loader.resolve_anima_checkpoint(self.root / "missing.safetensors")

    def test_prefixes_and_invalid_checkpoints(self):
        for prefix in ("", "net.", "diffusion_model.", "model.diffusion_model."):
            with self.subTest(prefix=prefix):
                transformer, conditioner = loader.split_anima_checkpoint({
                    prefix + "blocks.0.weight": torch.ones(1),
                    prefix + "llm_adapter.blocks.0.weight": torch.ones(1),
                })
                self.assertEqual(loader.get_anima_layer_count(transformer, "blocks."), 1)
                self.assertEqual(loader.get_anima_layer_count(conditioner, "blocks."), 1)
        with self.assertRaisesRegex(ValueError, "BF16"):
            loader.split_anima_checkpoint({"net.blocks.0.weight": torch.ones(1, dtype=torch.int8)})
        with self.assertRaisesRegex(ValueError, "llm_adapter"):
            loader.split_anima_checkpoint({"net.blocks.0.weight": torch.ones(1)})
        for keys in ({}, {"blocks.0.weight": None, "blocks.2.weight": None}):
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                loader.get_anima_layer_count(keys, "blocks.")

    def test_pipeline_uses_checkpoint_components_and_only_downloads_extras(self):
        transformer, conditioner = Mock(), Mock()
        with patch("diffusers.AnimaAutoBlocks") as blocks, patch.object(
            loader, "load_anima_single_file", return_value=(transformer, conditioner)
        ) as load:
            pipe = loader.load_anima_pipeline(str(self.path), torch.bfloat16, str(self.path))
            blocks.return_value.init_pipeline.assert_called_once_with(loader.ANIMA_BASE_REPO)
            load.assert_called_once_with(str(self.path), loader.ANIMA_BASE_REPO, torch.bfloat16)
            pipe.update_components.assert_called_once_with(transformer=transformer, text_conditioner=conditioner)
            self.assertEqual(set(pipe.load_components.call_args.kwargs["names"]),
                             {"text_encoder", "tokenizer", "t5_tokenizer", "vae", "scheduler"})

    def test_existing_diffusers_pipeline_loading_is_preserved(self):
        with patch("diffusers.AnimaAutoBlocks") as blocks:
            pipe = loader.load_anima_pipeline(loader.ANIMA_BASE_REPO, torch.bfloat16)
            blocks.return_value.init_pipeline.assert_called_once_with(loader.ANIMA_BASE_REPO)
            pipe.load_components.assert_called_once_with(torch_dtype=torch.bfloat16)

    def test_local_extras_path_is_used_for_all_components(self):
        with patch("diffusers.AnimaAutoBlocks") as blocks, patch.object(
            loader, "load_anima_single_file", return_value=(Mock(), Mock())
        ) as load:
            pipe = loader.load_anima_pipeline(str(self.path), torch.bfloat16, str(self.root))
            blocks.return_value.init_pipeline.assert_called_once_with(str(self.root))
            load.assert_called_once_with(str(self.path), str(self.root), torch.bfloat16)
            self.assertEqual(pipe.load_components.call_args.kwargs["pretrained_model_name_or_path"], str(self.root))


if __name__ == "__main__":
    unittest.main()
