import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_scheduler_module():
    torch = ModuleType('torch')
    torch.optim = SimpleNamespace(Optimizer=object)

    optimization = ModuleType('diffusers.optimization')
    cosine_with_warmup = Mock(return_value=object())
    optimization.SchedulerType = lambda name: name
    optimization.TYPE_TO_SCHEDULER_FUNCTION = {}
    optimization.get_constant_schedule_with_warmup = Mock(return_value=object())
    optimization.get_cosine_schedule_with_warmup = cosine_with_warmup

    diffusers = ModuleType('diffusers')
    diffusers.optimization = optimization

    module_path = ROOT / 'toolkit' / 'scheduler.py'
    spec = importlib.util.spec_from_file_location('scheduler_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            'torch': torch,
            'diffusers': diffusers,
            'diffusers.optimization': optimization,
        },
    ):
        spec.loader.exec_module(module)

    return module, cosine_with_warmup


class CosineWithWarmupSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.scheduler_module, self.cosine_with_warmup = load_scheduler_module()
        self.optimizer = object()

    def test_maps_total_iterations_to_diffusers_training_steps(self):
        scheduler = self.scheduler_module.get_lr_scheduler(
            'cosine_with_warmup',
            self.optimizer,
            total_iters=2000,
            num_warmup_steps=100,
        )

        self.assertIs(scheduler, self.cosine_with_warmup.return_value)
        self.cosine_with_warmup.assert_called_once_with(
            self.optimizer,
            num_training_steps=2000,
            num_warmup_steps=100,
        )

    def test_requires_a_total_training_step_count(self):
        with self.assertRaisesRegex(
            ValueError,
            'requires total_iters or num_training_steps',
        ):
            self.scheduler_module.get_lr_scheduler(
                'cosine_with_warmup',
                self.optimizer,
                num_warmup_steps=100,
            )

    def test_accepts_an_explicit_training_step_count(self):
        self.scheduler_module.get_lr_scheduler(
            'cosine_with_warmup',
            self.optimizer,
            total_iters=2000,
            num_training_steps=1500,
            num_warmup_steps=100,
        )

        self.cosine_with_warmup.assert_called_once_with(
            self.optimizer,
            num_training_steps=1500,
            num_warmup_steps=100,
        )


class CosineWithWarmupUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / 'ui' / 'src' / 'app' / 'jobs' / 'new' / 'SimpleJob.tsx'
        ).read_text(encoding='utf-8')

    def test_exposes_cosine_with_warmup_option(self):
        self.assertIn(
            "{ value: 'cosine_with_warmup', label: 'Cosine with Warmup' }",
            self.source,
        )

    def test_uses_the_warmup_input_for_both_supported_schedulers(self):
        self.assertIn(
            "new Set(['constant_with_warmup', 'cosine_with_warmup'])",
            self.source,
        )
        self.assertGreaterEqual(self.source.count('WARMUP_LR_SCHEDULERS.has('), 2)


if __name__ == '__main__':
    unittest.main()
