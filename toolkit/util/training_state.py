"""Checkpoint-paired training state, separate from inference/EMA exports."""

import hashlib
import os
import tempfile
from pathlib import Path

import torch


STATE_VERSION = 2
# Bytes hashed from each end of a checkpoint file to identify its contents.
SIGNATURE_SAMPLE_BYTES = 1 << 20


def training_state_path(checkpoint):
    checkpoint = Path(checkpoint)
    return checkpoint.parent / ".training_state" / (checkpoint.name + ".pt")


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _sampled_digest(path, size):
    """Hash both ends of a file: cheap for multi-GB checkpoints, unlike a full hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        digest.update(source.read(SIGNATURE_SAMPLE_BYTES))
        if size > SIGNATURE_SAMPLE_BYTES:
            source.seek(max(SIGNATURE_SAMPLE_BYTES, size - SIGNATURE_SAMPLE_BYTES))
            digest.update(source.read())
    return digest.hexdigest()


def _checkpoint_signature(checkpoint, version=STATE_VERSION):
    checkpoint = Path(checkpoint)
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file()) if checkpoint.is_dir() else [checkpoint]
    signature = []
    for path in files:
        stat = path.stat()
        if version == 1:
            # Version 1 used mtime, which changes when an output folder is copied or downloaded.
            name = str(path.relative_to(checkpoint)) if path != checkpoint else path.name
            signature.append((name, stat.st_size, stat.st_mtime_ns))
        else:
            name = path.relative_to(checkpoint).as_posix() if path != checkpoint else path.name
            signature.append((name, stat.st_size, _sampled_digest(path, stat.st_size)))
    return signature


def save_training_state(checkpoint, optimizer, ema, scheduler, scaler, progress):
    if ema is not None and not ema._is_train_mode:
        raise ValueError("Restore training parameters before saving training state")
    params = [param for group in optimizer.param_groups for param in group["params"]]
    ema_state = ema.state_dict() if ema is not None else None
    if ema_state is not None:
        # eval()/train() leaves a redundant snapshot behind; raw params are saved below.
        ema_state = dict(ema_state, collected_params=None)
    state = _cpu_copy({
        "version": STATE_VERSION,
        "checkpoint": _checkpoint_signature(checkpoint),
        "group_sizes": [len(group["params"]) for group in optimizer.param_groups],
        "parameters": params,
        "gradients": [param.grad for param in params],
        "optimizer": optimizer.state_dict(),
        "ema": ema_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress,
    })
    destination = training_state_path(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _load_state_file(path):
    try:
        from toolkit.optimizers.optimizer_utils import Auto8bitTensor
    except ImportError:
        return torch.load(path, map_location="cpu", weights_only=True)
    # Earlier Prodigy8bit states pickled Auto8bitTensor objects directly.
    with torch.serialization.safe_globals([Auto8bitTensor]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_training_state(checkpoint):
    path = training_state_path(checkpoint)
    if not path.exists():
        return None
    state = _load_state_file(path)
    version = state.get("version")
    if version not in (1, STATE_VERSION) or state.get("checkpoint") != _checkpoint_signature(checkpoint, version):
        raise ValueError(f"Training state does not match checkpoint: {checkpoint}")
    return state


def restore_training_parameters(state, optimizer):
    params = [param for group in optimizer.param_groups for param in group["params"]]
    if state["group_sizes"] != [len(group["params"]) for group in optimizer.param_groups]:
        raise ValueError("Checkpoint optimizer parameter groups have changed")
    if len(params) != len(state["parameters"]) or any(
        param.shape != saved.shape for param, saved in zip(params, state["parameters"])
    ):
        raise ValueError("Checkpoint training parameter shapes have changed")
    # Load the optimizer before touching parameters so a failure leaves both
    # untouched and the caller can fall back to a normal checkpoint resume.
    previous_optimizer = optimizer.__getstate__()
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        optimizer.__setstate__(previous_optimizer)
        raise
    with torch.no_grad():
        for param, saved in zip(params, state["parameters"]):
            param.copy_(saved)


def restore_training_gradients(state, optimizer):
    params = [param for group in optimizer.param_groups for param in group["params"]]
    if len(params) != len(state["gradients"]):
        raise ValueError("Checkpoint gradient count has changed")
    for param, gradient in zip(params, state["gradients"]):
        param.grad = None if gradient is None else gradient.to(device=param.device, dtype=param.dtype)


def remove_training_state(checkpoint):
    training_state_path(checkpoint).unlink(missing_ok=True)
