"""CPU loading/backprop regressions; uses tiny Diffusers/v2 models, no downloads.

Run: python -m unittest testing.test_anima_loader testing.test_anima_rope
"""

import tempfile
import unittest
from contextlib import ExitStack
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
        # v2 modules carry the quantize/offload/placement policy used by AnimaModel.load_model.
        self.assertTrue(hasattr(transformer, "aitk_post_load"))
        self.assertTrue(hasattr(conditioner, "aitk_post_load"))
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

    def load_components(self, name_or_path, extras=None):
        """Run load_anima_components with every hub/disk load mocked out."""
        from toolkit.models.v2.diffusion_models.cosmos import CosmosTransformer3DModel as V2Transformer
        from toolkit.models.v2.text_encoders.anima import AnimaTextConditioner as V2Conditioner
        from toolkit.models.v2.text_encoders.qwen3 import Qwen3ModelEncoder
        from toolkit.models.v2.vae.qwen_image import QwenImageVAE

        with ExitStack() as stack:
            mocks = {
                name: stack.enter_context(patch.object(cls, "load_model"))
                for name, cls in [("transformer", V2Transformer), ("text_conditioner", V2Conditioner),
                                  ("text_encoder", Qwen3ModelEncoder), ("vae", QwenImageVAE)]
            }
            mocks["tokenizer"] = stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained"))
            mocks["single_file"] = stack.enter_context(
                patch.object(loader, "load_anima_single_file", return_value=(Mock(), Mock())))
            components, components_path = loader.load_anima_components(name_or_path, torch.bfloat16, extras)
        return components, components_path, mocks

    def assert_shared_components_from(self, mocks, components_path):
        for name in ("vae", "text_encoder"):
            mocks[name].assert_called_once_with(components_path, dtype=torch.bfloat16)
        self.assertEqual({call.args for call in mocks["tokenizer"].call_args_list},
                         {(components_path,)})
        self.assertEqual({call.kwargs["subfolder"] for call in mocks["tokenizer"].call_args_list},
                         {"tokenizer", "t5_tokenizer"})

    def test_checkpoint_uses_its_dit_and_only_loads_extras(self):
        components, components_path, mocks = self.load_components(str(self.path), str(self.path))
        self.assertEqual(components_path, loader.ANIMA_BASE_REPO)
        mocks["single_file"].assert_called_once_with(str(self.path), loader.ANIMA_BASE_REPO, torch.bfloat16)
        transformer, conditioner = mocks["single_file"].return_value
        self.assertIs(components["transformer"], transformer)
        self.assertIs(components["text_conditioner"], conditioner)
        mocks["transformer"].assert_not_called()
        mocks["text_conditioner"].assert_not_called()
        self.assert_shared_components_from(mocks, loader.ANIMA_BASE_REPO)

    def test_existing_diffusers_source_loads_every_component(self):
        components, components_path, mocks = self.load_components(loader.ANIMA_BASE_REPO)
        self.assertEqual(components_path, loader.ANIMA_BASE_REPO)
        mocks["single_file"].assert_not_called()
        for name in ("transformer", "text_conditioner"):
            mocks[name].assert_called_once_with(loader.ANIMA_BASE_REPO, dtype=torch.bfloat16)
            self.assertIs(components[name], mocks[name].return_value)
        self.assert_shared_components_from(mocks, loader.ANIMA_BASE_REPO)

    def test_local_extras_path_is_used_for_all_components(self):
        _, components_path, mocks = self.load_components(str(self.path), str(self.root))
        self.assertEqual(components_path, str(self.root))
        mocks["single_file"].assert_called_once_with(str(self.path), str(self.root), torch.bfloat16)
        self.assert_shared_components_from(mocks, str(self.root))

if __name__ == "__main__":
    unittest.main()
