"""CPU regressions for accumulation, encoder caches, resume, and Anima routing.

Load trainer methods from their source AST to avoid importing every diffusion
backend. The methods, tensors, optimizers, EMA and Accelerate are real.
"""

import ast
import base64
import copy
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate import Accelerator

from toolkit.ema import ExponentialMovingAverage
from toolkit.util.cache_identity import encoder_cache_identity, source_identity
from toolkit.util.training_state import (
    load_training_state, remove_training_state, restore_training_gradients,
    restore_training_parameters, save_training_state, training_state_path,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = "jobs/process/BaseSDTrainProcess.py"
TRAINER = "extensions_built_in/sd_trainer/SDTrainer.py"
ANIMA = "extensions_built_in/diffusion_models/anima/anima.py"


def source_object(path, class_name, method=None, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    if method is not None:
        node = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == method)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[method or class_name]


def toy_trainer():
    param = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.SGD([param], lr=0.05)
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    optimizer = accelerator.prepare(optimizer)
    trainer = SimpleNamespace(
        optimizer=optimizer, accelerator=accelerator, _accumulated_samples=0, _accumulation_microsteps=0,
        sd=SimpleNamespace(is_multistage=False), steps_this_boundary=0,
        train_config=SimpleNamespace(optimizer="sgd", max_grad_norm=1e6),
        model_config=SimpleNamespace(low_vram=False), is_grad_accumulation_step=False,
        timer=lambda _: nullcontext(), adapter=None, embedding=None,
        lr_scheduler=Mock(), ema=Mock(), end_of_training_loop=lambda: None,
    )
    trainer.hook_train_loop = MethodType(source_object(
        TRAINER, "SDTrainer", "hook_train_loop", torch=torch, OrderedDict=OrderedDict,
        CustomAdapter=type("CustomAdapter", (), {}), ClipVisionAdapter=type("ClipVisionAdapter", (), {}),
    ), trainer)

    def microbatch(batch):
        loss = ((batch.x @ param - batch.y) ** 2).mean()
        accelerator.backward(loss * len(batch.file_items))
        return loss.detach()

    trainer.train_single_accumulation = microbatch
    return trainer, param


def batches():
    x = torch.tensor([[1., 2.], [3., -1.], [-2., 0.5], [4., 1.], [0.5, 2.]])
    y = torch.tensor([2., -1., 0.5, 3., -2.])
    def batch(x, y):
        return SimpleNamespace(x=x, y=y, file_items=[None] * len(x))
    return batch(x, y), [batch(x[:2], y[:2]), batch(x[2:4], y[2:4]), batch(x[4:], y[4:])]


@pytest.mark.parametrize("legacy", [False, True])
def test_accumulation_matches_full_batch_with_uneven_microbatches(legacy):
    reference, expected = toy_trainer()
    trainer, actual = toy_trainer()
    full, split = batches()
    expected_loss = reference.hook_train_loop(full)["loss"]
    if legacy:
        for index, batch in enumerate(split):
            trainer.is_grad_accumulation_step = index < len(split) - 1
            trainer.hook_train_loop(batch)
            if index < len(split) - 1:
                trainer.lr_scheduler.step.assert_not_called()
                trainer.ema.update.assert_not_called()
                assert actual.grad is not None
    else:
        assert trainer.hook_train_loop(split)["loss"] == pytest.approx(expected_loss)
    torch.testing.assert_close(actual, expected)
    assert trainer._accumulated_samples == 0
    trainer.lr_scheduler.step.assert_called_once()
    trainer.ema.update.assert_called_once()


def test_partial_window_uses_actual_sample_count():
    trainer, actual = toy_trainer()
    reference, expected = toy_trainer()
    _, split = batches()
    trainer.is_grad_accumulation_step = True
    trainer.hook_train_loop(split[0])
    trainer.is_grad_accumulation_step = False
    trainer.hook_train_loop(split[2])
    reference.hook_train_loop([split[0], split[2]])
    torch.testing.assert_close(actual, expected)


def test_skipped_optimizer_update_does_not_advance_ema_or_schedule():
    trainer, _ = toy_trainer()
    trainer.accelerator = SimpleNamespace(
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_, optimizer_step_was_skipped=True,
    )
    trainer.hook_train_loop(batches()[0])
    trainer.lr_scheduler.step.assert_not_called()
    trainer.ema.update.assert_not_called()


@pytest.mark.parametrize("window,regularization,expected_steps", [(2, False, [2, 4, 5]), (-1, False, [3, 5]), (-1, True, [5])])
def test_real_batch_collection_steps_at_window_epoch_and_final_boundaries(window, regularization, expected_steps):
    tree = ast.parse((ROOT / BASE).read_text())
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run")
    loop = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                and ast.unparse(node.target) == "step")
    collect = next(node for node in loop.body if isinstance(node, ast.With)
                   and "batch_list = []" in ast.unparse(node))
    final_window = next(node for node in loop.body if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "step == self.train_config.steps - 1")
    code = compile(ast.Module(body=[collect, final_window], type_ignores=[]), BASE, "exec")
    trainer, _ = toy_trainer()
    trainer.grad_accumulation_step = 1
    trainer._epoch_batches_seen = 0
    trainer.epoch_num = 0
    trainer.progress_bar = None
    trainer.data_loader_reg = batches()[1] if regularization else None
    trainer.is_regularization_step = MethodType(source_object(BASE, "BaseSDTrainProcess", "is_regularization_step"), trainer)
    trainer.save_config = SimpleNamespace(save_every=0)
    trainer.sample_config = SimpleNamespace(sample_every=0)
    trainer.train_config.gradient_accumulation = 1
    trainer.train_config.gradient_accumulation_steps = window
    trainer.train_config.disable_sampling = True
    trainer.train_config.free_u = False
    trainer.train_config.steps = 5
    _, split = batches()
    trainer.data_loader = split
    namespace = dict(self=trainer, torch=torch, dataloader=split,
                     dataloader_iterator=iter(split), dataloader_reg=trainer.data_loader_reg,
                     dataloader_iterator_reg=iter(split),
                     trigger_dataloader_setup_epoch=lambda _: None)
    updates = []
    for step in range(5):
        trainer.step_num = step
        trainer.is_grad_accumulation_step = True
        namespace["step"] = step
        exec(code, namespace)
        trainer.hook_train_loop(namespace["batch_list"])
        if not trainer.is_grad_accumulation_step:
            updates.append(step + 1)
        trainer.grad_accumulation_step += 1
    assert updates == expected_steps
    assert trainer.lr_scheduler.step.call_count == len(expected_steps)
    assert source_object(BASE, "BaseSDTrainProcess", "get_scheduler_training_steps")(trainer) == len(expected_steps)


def training_objects():
    param = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    ema = ExponentialMovingAverage([param], decay=0.9, use_num_updates=True)
    return param, optimizer, scheduler, ema


def update(param, optimizer, scheduler, ema):
    (param.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    ema.update()


@pytest.mark.parametrize("pending", [False, True])
def test_ema_resume_restores_raw_weights_optimizer_scheduler_and_pending_grads(tmp_path, pending):
    param, optimizer, scheduler, ema = training_objects()
    for _ in range(3):
        update(param, optimizer, scheduler, ema)
    if pending:
        (param.sum() * 0.7).backward()
    checkpoint = tmp_path / "model.safetensors"
    ema.eval()
    exported = param.detach().clone()
    checkpoint.write_bytes(exported.numpy().tobytes())
    ema.train()
    raw = param.detach().clone()
    assert not torch.equal(exported, raw)
    progress = dict(step=3, epoch=0, accumulated_samples=int(pending), accumulation_step=int(pending), epoch_batches_seen=3)
    save_training_state(checkpoint, optimizer, ema, scheduler, None, progress)

    resumed, resumed_optimizer, resumed_scheduler, resumed_ema = training_objects()
    with torch.no_grad():
        resumed.copy_(exported)
    state = load_training_state(checkpoint)
    restore_training_parameters(state, resumed_optimizer)
    resumed_ema.load_state_dict(state["ema"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    restore_training_gradients(state, resumed_optimizer)
    torch.testing.assert_close(resumed, raw)
    assert state["progress"] == progress
    update(param, optimizer, scheduler, ema)
    update(resumed, resumed_optimizer, resumed_scheduler, resumed_ema)
    torch.testing.assert_close(resumed, param, rtol=0, atol=0)
    torch.testing.assert_close(resumed_ema.shadow_params[0], ema.shadow_params[0], rtol=0, atol=0)
    assert resumed_ema.num_updates == ema.num_updates
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_sidecar_rejects_replaced_export_and_is_removed_with_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first")
    _, optimizer, scheduler, ema = training_objects()
    save_training_state(checkpoint, optimizer, ema, scheduler, None, {})
    checkpoint.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="does not match"):
        load_training_state(checkpoint)
    remove_training_state(checkpoint)
    assert not training_state_path(checkpoint).exists()
    assert load_training_state(checkpoint) is None


def test_ema_state_restores_update_settings_and_accepts_legacy_state():
    param = torch.nn.Parameter(torch.ones(2))
    ema = ExponentialMovingAverage([param], use_feedback=True, param_multiplier=0.95)
    restored = ExponentialMovingAverage([param])
    state = ema.state_dict()
    restored.load_state_dict(state)
    assert restored.use_feedback is True
    assert restored.param_multiplier == 0.95
    state.pop("use_feedback")
    state.pop("param_multiplier")
    restored.load_state_dict(state)
    assert restored.use_feedback is True


def test_state_rejects_changed_parameter_layout_before_overwriting(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"model")
    _, optimizer, scheduler, ema = training_objects()
    save_training_state(checkpoint, optimizer, ema, scheduler, None, {})
    replacement = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match="shapes have changed"):
        restore_training_parameters(load_training_state(checkpoint), torch.optim.AdamW([replacement]))
    torch.testing.assert_close(replacement, torch.ones(3))


def save_method():
    return source_object(BASE, "BaseSDTrainProcess", "save", flush=lambda: None,
                         save_training_state=save_training_state)


def test_export_failure_restores_raw_weights_and_network_multiplier():
    param, _, _, ema = training_objects()
    with torch.no_grad():
        param.add_(1)
    raw = param.detach().clone()
    network = SimpleNamespace(multiplier=0.7)
    def fail(_):
        torch.testing.assert_close(param, ema.shadow_params[0])
        network.multiplier = 1
        raise RuntimeError("export failed")
    process = SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True),
                              ema=ema, network=network, _save_model=fail)
    with pytest.raises(RuntimeError, match="export failed"):
        save_method()(process)
    torch.testing.assert_close(param, raw)
    assert ema._is_train_mode
    assert network.multiplier == 0.7


def test_real_save_wrapper_pairs_ema_export_with_raw_state(tmp_path):
    param, optimizer, scheduler, ema = training_objects()
    update(param, optimizer, scheduler, ema)
    checkpoint = tmp_path / "model.safetensors"
    def export(path, **kwargs):
        Path(path).write_bytes(param.detach().numpy().tobytes())
    process = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True, scaler=None), ema=ema,
        network=SimpleNamespace(multiplier=0.7, save_weights=export),
        optimizer=optimizer, lr_scheduler=scheduler,
        save_root=str(tmp_path), job=SimpleNamespace(name="model"),
        is_fine_tuning=False, named_lora=False, meta={},
        train_config=SimpleNamespace(merge_network_on_save=False),
        save_config=SimpleNamespace(dtype="fp32"), embedding=None, decorator=None,
        adapter=None, snr_gos=None, update_training_metadata=lambda: None,
        _completed_steps=1, step_num=0, epoch_num=0, _accumulated_samples=0,
        grad_accumulation_step=0, _accumulation_microsteps=0, _epoch_batches_seen=1,
        clean_up_saves=Mock(), post_save_hook=Mock(),
    )
    process._save_model = MethodType(source_object(
        BASE, "BaseSDTrainProcess", "_save_model", os=os, copy=copy, torch=torch,
        get_meta_for_safetensors=lambda meta, name: meta,
        get_torch_dtype=lambda _: torch.float32, unwrap_model=lambda model: model,
        print_acc=lambda *args: None,
    ), process)
    save_method()(process)
    assert (tmp_path / "optimizer.pt").exists()
    state = load_training_state(checkpoint)
    torch.testing.assert_close(state["parameters"][0], param)
    assert state["progress"]["step"] == 1
    assert state["ema"]["collected_params"] is None
    assert checkpoint.read_bytes() == ema.shadow_params[0].numpy().tobytes()
    process.post_save_hook.assert_called_once_with(str(checkpoint))


def file_item(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"original")
    item = SimpleNamespace(
        path=str(path), caption="a cat", scale_to_width=512, scale_to_height=512,
        crop_x=0, crop_y=0, crop_width=512, crop_height=512, latent_space_version="anima",
        text_embedding_space_version="anima", latent_version=1, text_embedding_version=1,
        latent_cache_identity="vae-a", text_cache_identity="encoder-a", flip_x=False,
        flip_y=False, is_audio_model=False, encode_control_in_text_embeddings=True,
        control_path=None, _latent_path=None, _text_embedding_path=None,
        dataset_config=SimpleNamespace(auto_frame_count=False, num_frames=1),
    )
    for cls, names in (
        ("LatentCachingFileItemDTOMixin", ["get_latent_info_dict", "get_latent_path"]),
        ("TextEmbeddingFileItemDTOMixin", ["get_text_embedding_info_dict", "get_text_embedding_path"]),
    ):
        for name in names:
            method = source_object("toolkit/dataloader_mixins.py", cls, name, os=os,
                                   OrderedDict=OrderedDict, source_identity=source_identity,
                                   json=json, hashlib=hashlib, base64=base64)
            setattr(item, name, MethodType(method, item))
    return item


def test_latent_cache_invalidates_same_name_size_and_mtime_content_replacement(tmp_path):
    item = file_item(tmp_path)
    before = item.get_latent_path()
    path = Path(item.path)
    stat = path.stat()
    path.write_bytes(b"replaced")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert item.get_latent_path(recalculate=True) != before


def test_text_cache_invalidates_encoder_caption_and_control_content(tmp_path):
    item = file_item(tmp_path)
    original = item.get_text_embedding_path()
    assert item.get_text_embedding_path(recalculate=True) == original
    item.text_cache_identity = "encoder-b"
    changed_encoder = item.get_text_embedding_path(recalculate=True)
    assert changed_encoder != original
    item.caption = "a dog"
    assert item.get_text_embedding_path(recalculate=True) != changed_encoder
    control = tmp_path / "control.png"
    control.write_bytes(b"one")
    item.control_path = [str(control)]
    first_control = item.get_text_embedding_path(recalculate=True)
    control.write_bytes(b"two")
    assert item.get_text_embedding_path(recalculate=True) != first_control


def test_encoder_identity_tracks_settings_revisions_and_local_weight_replacements(tmp_path):
    weights = tmp_path / "encoder.safetensors"
    weights.write_bytes(b"one")
    sd = SimpleNamespace(
        model_config=SimpleNamespace(arch="anima", name_or_path="repo/model", model_kwargs={"max_sequence_length": 512}, te_name_or_path=str(weights)),
        text_encoder=SimpleNamespace(config={"_commit_hash": "revision-1"}), tokenizer=None,
    )
    first = encoder_cache_identity(sd, "text")
    assert encoder_cache_identity(sd, "text") == first
    sd.model_config.model_kwargs["max_sequence_length"] = 256
    second = encoder_cache_identity(sd, "text")
    assert second != first
    sd.text_encoder.config["_commit_hash"] = "revision-2"
    third = encoder_cache_identity(sd, "text")
    assert third != second
    weights.write_bytes(b"two")
    assert encoder_cache_identity(sd, "text") != third


class AnimaComponent(torch.nn.Module):
    def __init__(self, blocks_name):
        super().__init__()
        setattr(self, blocks_name, torch.nn.ModuleList([torch.nn.Linear(2, 2)]))
        self.set_attention_backend = Mock()


def test_anima_attention_reaches_both_components_and_block_paths_resolve():
    wrapper = source_object(ANIMA, "AnimaTrainableModel", torch=torch)(
        AnimaComponent("transformer_blocks"), AnimaComponent("blocks"))
    wrapper.set_attention_backend("native", _checks=False)
    for component in (wrapper.transformer, wrapper.text_conditioner):
        component.set_attention_backend.assert_called_once_with("native", _checks=False)
    names = source_object(ANIMA, "AnimaModel", "get_transformer_block_names")
    assert len(names(SimpleNamespace(train_text_conditioner=False))) == 1
    for path in names(SimpleNamespace(train_text_conditioner=True)):
        module = wrapper
        for part in path.split("."):
            module = getattr(module, part)
        assert isinstance(module, torch.nn.ModuleList)
        inputs = torch.randn(1, 2)
        expected = module[0](inputs)
        module[0] = torch.compile(module[0], backend="eager")
        torch.testing.assert_close(module[0](inputs), expected)


def test_anima_prediction_uses_prepared_model_forward():
    method = source_object(ANIMA, "AnimaModel", "get_noise_prediction", torch=torch)
    prepared_forward = Mock(side_effect=lambda **kwargs: (kwargs["hidden_states"] * 2,))
    process = SimpleNamespace(
        trainable_model=SimpleNamespace(transformer=SimpleNamespace(device=torch.device("cpu"))),
        model=prepared_forward, device_torch=torch.device("cpu"), torch_dtype=torch.float32,
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        _condition_prompt_embeds=lambda *args, **kwargs: torch.zeros(1, 2, 2),
    )
    latent = torch.ones(1, 2, 2, 2)
    result = method(process, latent, torch.tensor([500.]), None)
    torch.testing.assert_close(result, latent * 2)
    prepared_forward.assert_called_once()
