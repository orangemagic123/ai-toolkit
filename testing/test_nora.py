import ast
import contextlib
import copy
import io
import math
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRAModule, LoRASpecialNetwork
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.models.DoRA import DoRAModule
from toolkit.nora import (
    check_nora_resume_mode,
    column_norm_deviation,
    nora_gain,
    nora_resume_warning,
    normalize_down_weight,
    parse_nora_mode,
)
from toolkit.util.training_state import (
    load_training_state,
    restore_training_parameters,
    save_training_state,
)

TRAINER = Path(__file__).resolve().parents[1] / "jobs/process/BaseSDTrainProcess.py"


def _trainer_method(name, **namespace):
    """Run a BaseSDTrainProcess method from source without importing every backend."""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(TRAINER), "exec"), namespace)
    return namespace[name]


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self, multiplier=1.0):
        self.torch_multiplier = torch.tensor([multiplier], dtype=torch.float32)


def _column_norms(weight):
    return torch.linalg.vector_norm(weight.detach().reshape(weight.shape[0], -1).float(), dim=0)


def _lora(original, nora_mode, network, lora_dim=3, alpha=3, seed=0, module_class=LoRAModule):
    torch.manual_seed(seed)
    module = module_class(
        f"nora_{nora_mode}",
        original,
        lora_dim=lora_dim,
        alpha=alpha,
        network=network,
        nora_mode=nora_mode,
    )
    module.org_forward = original.forward
    # modules only keep a weakref to their network
    module._test_network = network
    return module


class NoraToyModel(torch.nn.Module):
    def __init__(self, conv=True):
        super().__init__()
        self.q = torch.nn.Linear(12, 10, bias=True)
        self.proj = torch.nn.Linear(10, 6, bias=False)
        self.conv = torch.nn.Conv2d(3, 5, 3, padding=1) if conv else None
        self.pw = torch.nn.Conv2d(5, 4, 1) if conv else None

    def forward(self, tokens, image=None):
        out = self.proj(self.q(tokens))
        if self.conv is None:
            return out
        return out, self.pw(self.conv(image))


def _build_network(model, nora="off", use_dora=False, conv=True, seed=0, train_unet=True):
    torch.manual_seed(seed)
    config = NetworkConfig(type="lora", linear=4, linear_alpha=4, nora=nora, use_dora=use_dora)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model,
        lora_dim=config.linear,
        alpha=config.linear_alpha,
        conv_lora_dim=3 if conv else None,
        conv_alpha=3 if conv else None,
        train_unet=train_unet,
        train_text_encoder=False,
        network_config=config,
        network_type=config.type,
        transformer_only=False,
        is_transformer=True,
        target_lin_modules=["NoraToyModel"],
        target_conv_modules=["NoraToyModel"],
    )
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network._update_torch_multiplier()
    network.is_active = True
    return network


def _toy_pair(conv=True):
    torch.manual_seed(123)
    model = NoraToyModel(conv=conv)
    model.requires_grad_(False)
    return model, copy.deepcopy(model)


def _inputs(conv=True):
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(2, 5, 12, generator=generator)
    image = torch.randn(2, 3, 6, 6, generator=generator) if conv else None
    return tokens, image


def _outputs(model, tokens, image):
    out = model(tokens, image)
    return out if isinstance(out, tuple) else (out,)


@torch.no_grad()
def _randomize(network, raw_scale=True):
    for module in network.get_all_modules():
        module.lora_up.weight.normal_()
        if raw_scale:
            # move the raw columns off the unit sphere like training does
            flat = module.lora_down.weight.view(module.lora_down.weight.shape[0], -1)
            flat.mul_(torch.rand(1, flat.shape[1]) * 3 + 0.2)


def _quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


class NoraConfigTest(unittest.TestCase):
    def test_mode_parsing(self):
        for value in (None, False, "off", "OFF", yaml.safe_load("nora: off")["nora"]):
            self.assertEqual(parse_nora_mode(value), "off")
        self.assertEqual(parse_nora_mode("init"), "init")
        self.assertEqual(parse_nora_mode("continuous"), "continuous")
        for value in (True, "nroa", 1):
            with self.assertRaises(ValueError):
                parse_nora_mode(value)

    def test_network_config(self):
        self.assertEqual(NetworkConfig(type="lora").nora, "off")
        self.assertEqual(NetworkConfig(type="lora", nora="continuous").nora, "continuous")
        legacy_dora = NetworkConfig(type="dora", nora="init")
        self.assertEqual((legacy_dora.type, legacy_dora.use_dora, legacy_dora.nora), ("lora", True, "init"))
        for network_type in ("lokr", "locon", "lycoris", "lorm", "fullrank"):
            with self.subTest(network_type=network_type), self.assertRaises(ValueError):
                NetworkConfig(type=network_type, nora="init")
        with self.assertRaisesRegex(ValueError, "directly under network"):
            NetworkConfig(type="lora", network_kwargs={"nora": "init"})


class NormalizeDownWeightTest(unittest.TestCase):
    def test_normalizes_columns_over_rank(self):
        weight = torch.randn(3, 7) * 5
        normalized = normalize_down_weight(weight)
        torch.testing.assert_close(_column_norms(normalized), torch.ones(7))
        self.assertGreater((torch.linalg.vector_norm(normalized, dim=1) - 1).abs().max(), 0.1)

        conv = torch.randn(3, 2, 3, 3)
        normalized_conv = normalize_down_weight(conv)
        self.assertEqual(normalized_conv.shape, conv.shape)
        torch.testing.assert_close(_column_norms(normalized_conv), torch.ones(18))

    def test_zero_columns_and_bfloat16_stay_finite(self):
        weight = torch.randn(4, 6)
        weight[:, 2] = 0
        normalized = normalize_down_weight(weight)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(normalized[:, 2].abs().sum().item(), 0.0)

        bf16 = normalize_down_weight(torch.randn(4, 6, dtype=torch.bfloat16))
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertLess(column_norm_deviation(bf16), 0.02)

    def test_gain_matches_default_init_column_norm(self):
        weight = torch.empty(8, 2048)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        mean_norm = _column_norms(weight).mean().item()
        self.assertAlmostEqual(nora_gain(weight), 1 / math.sqrt(8 / (3 * 2048)))
        self.assertAlmostEqual(mean_norm * nora_gain(weight), 1.0, delta=0.05)


class NoraLoRAModuleTest(unittest.TestCase):
    def test_init_mode_normalizes_the_default_init(self):
        original = torch.nn.Linear(9, 7, bias=False)
        default = _lora(original, "off", _Network())
        init = _lora(original, "init", _Network())
        torch.testing.assert_close(init.lora_down.weight, normalize_down_weight(default.lora_down.weight))
        self.assertEqual(init.lora_up.weight.abs().sum().item(), 0.0)

        value = torch.randn(2, 9)
        torch.testing.assert_close(init(value), original(value))

        conv = _lora(torch.nn.Conv2d(2, 4, 3, padding=1), "init", _Network())
        torch.testing.assert_close(_column_norms(conv.lora_down.weight), torch.ones(18))

    def test_continuous_keeps_kaiming_raw_weight_and_matches_reference(self):
        torch.manual_seed(1)
        original = torch.nn.Linear(9, 7, bias=True)
        original.requires_grad_(False)
        network = _Network(multiplier=0.7)
        default = _lora(original, "off", network, alpha=6)
        module = _lora(original, "continuous", network, alpha=6)
        torch.testing.assert_close(module.lora_down.weight, default.lora_down.weight)

        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 5, 9)
        actual = module(value)
        normalized = normalize_down_weight(module.lora_down.weight)
        expected = original(value) + 0.7 * module.scale * F.linear(F.linear(value, normalized), module.lora_up.weight)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        # loss is invariant to the raw column scale, so the A gradient is tangential
        (grad,) = torch.autograd.grad(actual.square().mean(), [module.lora_down.weight])
        radial = (grad * module.lora_down.weight).sum(dim=0)
        self.assertLess(radial.abs().max().item(), 1e-5)

        with torch.no_grad():
            module.lora_down.weight.mul_(torch.rand(1, 9) * 4 + 0.1)
        torch.testing.assert_close(module(value), actual, rtol=1e-5, atol=1e-6)

    def test_continuous_conv_matches_reference(self):
        torch.manual_seed(2)
        for kernel, padding in ((3, 1), (1, 0)):
            with self.subTest(kernel=kernel):
                original = torch.nn.Conv2d(3, 5, kernel, padding=padding)
                original.requires_grad_(False)
                module = _lora(original, "continuous", _Network())
                with torch.no_grad():
                    module.lora_up.weight.normal_()
                value = torch.randn(2, 3, 6, 6)
                normalized = normalize_down_weight(module.lora_down.weight)
                latent = F.conv2d(value, normalized, padding=padding)
                expected = original(value) + module.scale * F.conv2d(latent, module.lora_up.weight)
                torch.testing.assert_close(module(value), expected, rtol=1e-5, atol=1e-5)

    def test_bfloat16_activations_stay_finite(self):
        torch.manual_seed(3)
        original = torch.nn.Linear(9, 7, bias=False, dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = _lora(original, "continuous", _Network())
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 9, dtype=torch.bfloat16)
        actual = module(value)
        actual.float().square().mean().backward()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(module.lora_down.weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(module.lora_down.weight.grad).all())

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            _lora(torch.nn.Linear(4, 4), "sometimes", _Network())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_continuous_compiles(self):
        torch.manual_seed(0)
        torch._dynamo.reset()
        network = _Network()
        network.torch_multiplier = torch.ones(1, device="cuda")
        original = torch.nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = _lora(original, "continuous", network, lora_dim=4, alpha=4).to("cuda")
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        eager = module(value)
        eager.float().square().mean().backward()
        eager_grad = module.lora_down.weight.grad.detach().clone()
        module.zero_grad(set_to_none=True)

        compiled = torch.compile(module, fullgraph=False, dynamic=True)
        actual = compiled(value)
        actual.float().square().mean().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(module.lora_down.weight.grad, eager_grad, rtol=2e-2, atol=5e-2)


class NoraDoRAModuleTest(unittest.TestCase):
    def test_initialization(self):
        original = torch.nn.Linear(64, 6, bias=False)
        continuous = _lora(original, "continuous", _Network(), module_class=DoRAModule)
        self.assertLessEqual(continuous.lora_down.weight.abs().max().item(), 1 / math.sqrt(64))
        # randn/sqrt(rank) would put the mean column norm near 1
        self.assertLess(_column_norms(continuous.lora_down.weight).mean().item(), 0.3)

        default = _lora(original, "off", _Network(), module_class=DoRAModule)
        init = _lora(original, "init", _Network(), module_class=DoRAModule)
        torch.testing.assert_close(init.lora_down.weight, normalize_down_weight(default.lora_down.weight))
        torch.testing.assert_close(init.magnitude, default.magnitude)

    def test_continuous_forward_and_weight_norm_match_reference(self):
        torch.manual_seed(4)
        network = _Network(multiplier=0.7)
        original = torch.nn.Linear(8, 6, bias=True)
        original.requires_grad_(False)
        module = _lora(original, "continuous", network, alpha=6, module_class=DoRAModule)
        with torch.no_grad():
            module.lora_up.weight.normal_()
            module.magnitude.mul_(1.1)

        value = torch.randn(2, 5, 8)
        actual = module(value)

        adapter_scale = module.scale * network.torch_multiplier.mean()
        normalized = normalize_down_weight(module.lora_down.weight)
        adapted = original.weight + adapter_scale * (module.lora_up.weight @ normalized)
        weight_norm = torch.linalg.vector_norm(adapted.detach(), dim=1)
        torch.testing.assert_close(
            module._get_weight_norm_factorized(
                network.torch_multiplier.mean(), value.device, torch.float32, use_autocast=False
            ),
            weight_norm,
            rtol=1e-5,
            atol=1e-6,
        )
        expected = F.linear(value, (module.magnitude / weight_norm)[:, None] * adapted, original.bias)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        parameters = [module.lora_down.weight, module.lora_up.weight, module.magnitude]
        actual_grads = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_bfloat16_matches_float32_reference(self):
        torch.manual_seed(5)
        original = torch.nn.Linear(8, 6, bias=False)
        original.requires_grad_(False)
        module = _lora(original, "continuous", _Network(), module_class=DoRAModule)
        with torch.no_grad():
            module.lora_up.weight.normal_()
        value = torch.randn(2, 8)
        reference = module(value)

        original_bf16 = copy.deepcopy(original).to(torch.bfloat16)
        module.org_module = [original_bf16]
        module.org_forward = original_bf16.forward
        actual = module(value.to(torch.bfloat16))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), reference, rtol=5e-2, atol=5e-2)


class NoraNetworkTest(unittest.TestCase):
    def test_modules_receive_mode_and_summary_is_logged(self):
        model, _ = _toy_pair()
        network, log = _quiet(_build_network, model, nora="continuous")
        modes = {module.lora_name: module.nora_mode for module in network.get_all_modules()}
        self.assertEqual(len(modes), 4)
        self.assertEqual(set(modes.values()), {"continuous"})
        self.assertIn("NoRA (continuous): 2 Linear + 2 Conv LoRA modules", log)

        _, dora_log = _quiet(_build_network, _toy_pair()[0], nora="continuous", use_dora=True)
        self.assertIn("DoRA raw A uses kaiming init", dora_log)

    def test_init_mode_matches_normalized_default_network(self):
        default_model, init_model = _toy_pair()
        default = _quiet(_build_network, default_model, nora="off")[0]
        init = _quiet(_build_network, init_model, nora="init")[0]
        for default_module, init_module in zip(default.get_all_modules(), init.get_all_modules()):
            torch.testing.assert_close(
                init_module.lora_down.weight, normalize_down_weight(default_module.lora_down.weight)
            )

    def test_no_modules_is_an_error(self):
        model, _ = _toy_pair()
        with self.assertRaisesRegex(ValueError, "no LoRA modules"):
            _quiet(_build_network, model, nora="init", train_unet=False)

    def test_export_is_a_plain_lora_for_any_loader(self):
        for use_dora in (False, True):
            with self.subTest(use_dora=use_dora):
                conv = not use_dora
                source_model, target_model = _toy_pair(conv=conv)
                plain_model = copy.deepcopy(target_model)
                source = _quiet(_build_network, source_model, nora="continuous", use_dora=use_dora, conv=conv)[0]
                _randomize(source)
                tokens, image = _inputs(conv)
                expected = _outputs(source_model, tokens, image)

                state_dict = source.get_state_dict(dtype=torch.float32)
                down_keys = [key for key in state_dict if key.endswith("lora_A.weight")]
                self.assertEqual(len(down_keys), 4 if conv else 2)
                for key in down_keys:
                    self.assertLess(column_norm_deviation(state_dict[key]), 1e-5)

                for model, nora in ((target_model, "continuous"), (plain_model, "off")):
                    network = _quiet(_build_network, model, nora=nora, use_dora=use_dora, conv=conv, seed=99)[0]
                    _, log = _quiet(network.load_weights, state_dict)
                    self.assertNotIn("do not have unit columns", log)
                    for actual, reference in zip(_outputs(model, tokens, image), expected):
                        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_loading_a_regular_lora_into_continuous_warns(self):
        source_model, target_model = _toy_pair()
        source = _quiet(_build_network, source_model, nora="off")[0]
        _randomize(source, raw_scale=False)
        target = _quiet(_build_network, target_model, nora="continuous")[0]
        _, log = _quiet(target.load_weights, source.get_state_dict(dtype=torch.float32))
        self.assertIn("4 loaded LoRA down weights do not have unit columns", log)

    def test_merge_in_uses_normalized_down_weight(self):
        model, _ = _toy_pair()
        network = _quiet(_build_network, model, nora="continuous")[0]
        _randomize(network)
        tokens, image = _inputs()
        expected = _outputs(model, tokens, image)
        network.merge_in(merge_weight=1.0)
        self.assertTrue(network.is_merged_in)
        for actual, reference in zip(_outputs(model, tokens, image), expected):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)

    def test_paired_training_state_restores_raw_down_weight(self):
        source_model, resumed_model = _toy_pair()
        tokens, image = _inputs()

        def train_step(network, model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(output.square().mean() for output in _outputs(model, tokens, image))
            loss.backward()
            optimizer.step()

        source = _quiet(_build_network, source_model, nora="continuous")[0]
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-2)
        for _ in range(3):
            train_step(source, source_model, optimizer)

        # safetensors may keep the file mapped on Windows; don't fail on cleanup
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            checkpoint = Path(directory) / "nora.safetensors"
            source.save_weights(str(checkpoint), dtype=torch.float32)
            save_training_state(checkpoint, optimizer, None, None, None, {"step": 3})
            raw_down = [module.lora_down.weight.detach().clone() for module in source.get_all_modules()]
            self.assertTrue(any(column_norm_deviation(weight) > 1e-3 for weight in raw_down))
            train_step(source, source_model, optimizer)
            uninterrupted = [param.detach().clone() for param in source.parameters()]

            resumed = _quiet(_build_network, resumed_model, nora="continuous", seed=99)[0]
            _quiet(resumed.load_weights, str(checkpoint))
            for module in resumed.get_all_modules():
                self.assertLess(column_norm_deviation(module.lora_down.weight), 1e-5)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
            restore_training_parameters(load_training_state(checkpoint), resumed_optimizer)

        for module, expected in zip(resumed.get_all_modules(), raw_down):
            torch.testing.assert_close(module.lora_down.weight, expected)
        train_step(resumed, resumed_model, resumed_optimizer)
        for actual, expected in zip(resumed.parameters(), uninterrupted):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_resume_warning_only_for_continuous(self):
        self.assertIsNone(nora_resume_warning(None))
        for mode in ("off", "init"):
            model, _ = _toy_pair()
            self.assertIsNone(nora_resume_warning(_quiet(_build_network, model, nora=mode)[0]))
        model, _ = _toy_pair()
        message = nora_resume_warning(_quiet(_build_network, model, nora="continuous")[0])
        self.assertIn("without its paired training state", message)


class NoraResumeModeTest(unittest.TestCase):
    """Resuming across continuous and non-continuous NoRA must stop before any raw
    weights or optimizer state are restored (init <-> off is an ordinary resume)."""

    get_training_info = staticmethod(_trainer_method("get_training_info", OrderedDict=OrderedDict))
    load_metadata = staticmethod(_trainer_method(
        "load_training_state_from_metadata",
        os=os,
        yaml=yaml,
        load_metadata_from_safetensors=load_metadata_from_safetensors,
        check_nora_resume_mode=check_nora_resume_mode,
        print_acc=lambda *args: None,
    ))
    restore_paired = staticmethod(_trainer_method(
        "restore_paired_training_state",
        load_training_state=load_training_state,
        restore_training_parameters=restore_training_parameters,
        print_acc=lambda *args: None,
    ))

    @staticmethod
    def _trainer(network, pretrained_lora_path=None, merge_network_on_save=False):
        return SimpleNamespace(
            network=network,
            step_num=3,
            epoch_num=0,
            start_step=None,
            _resume_checkpoint_path=None,
            network_config=SimpleNamespace(pretrained_lora_path=pretrained_lora_path),
            train_config=SimpleNamespace(
                start_step=None,
                merge_network_on_save=merge_network_on_save,
                ema_config=SimpleNamespace(use_ema=False),
            ),
        )

    def _checkpoint(self, directory, mode):
        """Train a few steps and save the adapter, its metadata and paired state like the trainer."""
        model, base = _toy_pair()
        tokens, image = _inputs()
        network = _quiet(_build_network, model, nora=mode)[0]
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            sum(output.square().mean() for output in _outputs(model, tokens, image)).backward()
            optimizer.step()
        checkpoint = Path(directory) / f"{mode}.safetensors"
        meta = get_meta_for_safetensors(OrderedDict({"training_info": self.get_training_info(self._trainer(network))}))
        network.save_weights(str(checkpoint), dtype=torch.float32, metadata=meta)
        save_training_state(checkpoint, optimizer, None, None, None, {"step": 3, "epoch": 0})
        return str(checkpoint), _outputs(model, tokens, image), base

    def _resume(self, checkpoint, mode, base, **trainer_kwargs):
        model = copy.deepcopy(base)
        network = _quiet(_build_network, model, nora=mode, seed=99)[0]
        _quiet(network.load_weights, checkpoint)
        trainer = self._trainer(network, **trainer_kwargs)
        self.load_metadata(trainer, checkpoint)
        return trainer, network, model

    def test_rule(self):
        for saved, current in (("continuous", "continuous"), ("off", "init"), ("init", "off"), ("off", "off")):
            check_nora_resume_mode(saved, current, "ckpt")
        for saved, current in (("continuous", "off"), ("continuous", "init"), ("off", "continuous"), ("init", "continuous")):
            with self.subTest(saved=saved, current=current), self.assertRaisesRegex(ValueError, "new job name"):
                check_nora_resume_mode(saved, current, "ckpt")

    def test_trainer_resume_blocks_continuity_switch_before_restoring_state(self):
        tokens, image = _inputs()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            for saved in ("continuous", "init", "off"):
                checkpoint, expected, base = self._checkpoint(directory, saved)
                self.assertEqual(
                    load_metadata_from_safetensors(checkpoint)["training_info"].get("nora_mode", "off"), saved
                )
                for current in ("continuous", "init", "off"):
                    with self.subTest(saved=saved, current=current):
                        if (saved == "continuous") != (current == "continuous"):
                            with self.assertRaisesRegex(ValueError, "Resuming across continuous"):
                                self._resume(checkpoint, current, base)
                            continue
                        trainer, network, model = self._resume(checkpoint, current, base)
                        self.assertEqual(trainer._resume_checkpoint_path, checkpoint)
                        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
                        self.assertIsNotNone(self.restore_paired(trainer, optimizer))
                        for actual, reference in zip(_outputs(model, tokens, image), expected):
                            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_pretrained_and_merged_loads_are_not_resumes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            checkpoint, _, base = self._checkpoint(directory, "continuous")
            trainer = self._resume(checkpoint, "off", base, pretrained_lora_path=checkpoint)[0]
            self.assertIsNone(trainer._resume_checkpoint_path)
            self._resume(checkpoint, "off", base, merge_network_on_save=True)


if __name__ == "__main__":
    unittest.main()
