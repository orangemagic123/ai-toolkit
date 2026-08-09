import ast
import copy
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_update_training_metadata():
    """Load the lightweight metadata method without importing training dependencies."""
    source_path = Path(__file__).resolve().parents[1] / "jobs" / "process" / "BaseSDTrainProcess.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    process_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess"
    )
    method = next(
        node
        for node in process_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_training_metadata"
    )
    namespace = {"OrderedDict": OrderedDict, "copy": copy}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["update_training_metadata"]


class TrainingMetadataTest(unittest.TestCase):
    def test_copies_process_config_into_common_model_metadata(self):
        raw_process_config = OrderedDict({
            "type": "sd_trainer",
            "model": {"arch": "anima", "name_or_path": "test/anima"},
            "network": {"type": "lora", "linear": 16},
            "train": {"steps": 250, "lr": 1e-4},
            "datasets": [{"folder_path": "dataset", "resolution": [512, 768]}],
        })
        metadata = OrderedDict({"name": "test-model"})
        process = SimpleNamespace(
            raw_process_config=raw_process_config,
            get_training_info=lambda: OrderedDict({"step": 250, "epoch": 2}),
            sd=SimpleNamespace(get_base_model_version=lambda: "anima"),
            job=SimpleNamespace(name="test-model"),
            trigger_word=None,
            add_meta=lambda additional: metadata.update(additional),
        )

        load_update_training_metadata()(process)

        self.assertEqual(metadata["training_config"], raw_process_config)
        self.assertIsNot(metadata["training_config"], raw_process_config)
        self.assertIsNot(metadata["training_config"]["train"], raw_process_config["train"])
        self.assertEqual(metadata["training_info"], {"step": 250, "epoch": 2})
        self.assertEqual(metadata["ss_base_model_version"], "anima")
        self.assertEqual(metadata["ss_output_name"], "test-model")

        raw_process_config["train"]["steps"] = 999
        self.assertEqual(metadata["training_config"]["train"]["steps"], 250)
        self.assertEqual(json.loads(json.dumps(metadata["training_config"]))["model"]["arch"], "anima")


if __name__ == "__main__":
    unittest.main()
