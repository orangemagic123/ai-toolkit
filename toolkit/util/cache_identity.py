"""Stable cache identities for the inputs and encoders used during preprocessing."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16384)
def _file_digest(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path):
    """Hash media contents once per file version, including control-image lists."""
    if path is None:
        return None
    if isinstance(path, (list, tuple)):
        return [source_identity(item) for item in path]
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _model_source(value):
    """Identify local model replacements without re-hashing multi-GB weights."""
    if isinstance(value, dict):
        return {key: _model_source(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_source(item) for item in value]
    if not isinstance(value, str) or not value:
        return _json_value(value)
    path = Path(value).expanduser()
    if not path.exists():
        return value  # Hub ID/revision; loaded component configs also identify revisions.
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    manifest = []
    for file in files:
        if file.is_file() and (file == path or file.suffix in {
            ".safetensors", ".bin", ".pt", ".pth", ".json", ".model", ".txt",
        }):
            stat = file.stat()
            manifest.append((str(file.relative_to(path)) if file != path else file.name,
                             stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return {"path": str(path.resolve()), "files": manifest}


def _component_identity(component):
    if isinstance(component, (list, tuple)):
        return [_component_identity(item) for item in component]
    if component is None:
        return None
    config = getattr(component, "config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    return {
        "class": type(component).__qualname__,
        "config": _json_value(config),
        "dtype": str(getattr(component, "dtype", None)),
        "source": _model_source(getattr(component, "name_or_path", None)),
        "tokenizer": _json_value(getattr(component, "init_kwargs", {})),
    }


def encoder_cache_identity(sd, kind):
    """Compute once per dataset, then share the digest with all file items."""
    if sd is None:
        return None
    config = sd.model_config
    fields = ["arch", "name_or_path", "extras_name_or_path", "model_paths", "model_kwargs"]
    if kind == "latent":
        fields += ["vae_path", "vae_dtype"]
        components = ["vae"]
    elif kind == "text":
        fields += ["te_name_or_path", "te_dtype", "text_encoder_bits", "quantize_te", "qtype_te"]
        components = ["text_encoder", "tokenizer", "t5_tokenizer"]
    else:
        raise ValueError(f"Unknown cache kind: {kind}")
    identity = {
        "version": 2,
        "model": type(sd).__qualname__,
        "settings": {field: _model_source(getattr(config, field, None)) for field in fields},
        "components": {name: _component_identity(getattr(sd, name, None)) for name in components},
        "padding_side": getattr(sd, "te_padding_side", None) if kind == "text" else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
