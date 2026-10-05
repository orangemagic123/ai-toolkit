import ast
import base64
import contextlib
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Dict, List, Union
import unittest
from unittest.mock import patch

from toolkit.caption_utils import (
    drop_caption_tags,
    expand_secondary_separator,
    join_caption_sections,
    select_mixed_caption_selection,
    shuffle_caption_tags,
    split_caption_at_separator,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_definitions(relative_path, names, namespace):
    """Load top-level classes/functions without importing the heavy training stack."""
    source_path = ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace


class FakePromptEmbeds:
    """Records the caption each cache file was encoded from."""

    def __init__(self, caption):
        self.caption = caption

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(self.caption, encoding='utf-8')

    @classmethod
    def load(cls, path):
        embeds = cls(Path(path).read_text(encoding='utf-8'))
        embeds.path = path
        return embeds


NAMESPACE = _load_definitions(
    'toolkit/prompt_utils.py',
    {'inject_trigger_into_prompt'},
    {},
)
NAMESPACE.update({
    'OrderedDict': OrderedDict,
    'PromptEmbeds': FakePromptEmbeds,
    'Union': Union,
    'List': List,
    'accelerator': SimpleNamespace(main_process_first=contextlib.nullcontext),
    'base64': base64,
    'drop_caption_tags': drop_caption_tags,
    'expand_secondary_separator': expand_secondary_separator,
    'hashlib': hashlib,
    'join_caption_sections': join_caption_sections,
    'json': json,
    'os': os,
    'print_acc': lambda *args, **kwargs: None,
    'random': random,
    'shuffle_caption_tags': shuffle_caption_tags,
    'split_caption_at_separator': split_caption_at_separator,
    'tqdm': lambda iterable, **kwargs: iterable,
})
_load_definitions(
    'toolkit/dataloader_mixins.py',
    {'CaptionProcessingDTOMixin', 'TextEmbeddingFileItemDTOMixin', 'TextEmbeddingCachingMixin'},
    NAMESPACE,
)


def _load_dataset_config_class():
    namespace = {
        'ControlTypes': str,
        'Dict': Dict,
        'GuidanceType': str,
        'List': List,
        'Union': Union,
        'math': math,
        'os': os,
    }
    return _load_definitions('toolkit/config_modules.py', {'DatasetConfig'}, namespace)['DatasetConfig']


DATASET_CONFIG = _load_dataset_config_class()


class FakeItem(NAMESPACE['CaptionProcessingDTOMixin'], NAMESPACE['TextEmbeddingFileItemDTOMixin']):
    def __init__(self, raw_caption, path='image.png', trigger_word=None, mixed_selection=None, **config):
        NAMESPACE['TextEmbeddingFileItemDTOMixin'].__init__(self)
        defaults = dict(
            caption_dropout_rate=0.0,
            protected_caption_dropout_rate=0.0,
            token_dropout_rate=0.0,
            keep_tokens=0,
            keep_tokens_separator='|||',
            secondary_separator=None,
            shuffle_caption=False,
            random_triggers=[],
            random_triggers_max=1,
            cache_text_embeddings=False,
            diff_output_preservation=False,
            diff_output_preservation_class='woman',
        )
        defaults.update(config)
        self.dataset_config = SimpleNamespace(**defaults)
        self.path = path
        self.trigger_word = trigger_word
        self.is_reg = False
        self.raw_caption = raw_caption
        self.raw_caption_short = None
        self.mixed_caption_selection = mixed_selection
        self.caption = None
        self.caption_dop = None
        self.text_cache_identity = 'fake-encoder'
        self.text_embedding_space_version = 1
        self.encode_control_in_text_embeddings = False
        self.control_path = None
        self.control_video_paths = []
        self.is_video = False

    def load_caption(self):
        # the parts of the real load_caption the text embedding cache relies on
        self.caption = self.get_caption()
        if self.dataset_config.diff_output_preservation:
            self.caption_dop = self.caption
            if self.trigger_word is not None:
                self.caption_dop = self.caption.replace(
                    self.trigger_word, self.dataset_config.diff_output_preservation_class
                )


class FakeDataset(NAMESPACE['TextEmbeddingCachingMixin']):
    def __init__(self, file_list, dataset_config):
        self.file_list = file_list
        self.dataset_config = dataset_config
        self.dataset_path = 'fake'
        self.encoded = []
        self.sd = SimpleNamespace(
            device='cpu',
            encode_prompt=self._encode_prompt,
            set_device_state_preset=lambda preset: None,
        )

    def _encode_prompt(self, caption, **kwargs):
        self.encoded.append(caption)
        return FakePromptEmbeds(caption)


CAPTION = 'year 2025, 1girl, charA, @styleX ||| red hair, smile, outdoors'
PROTECTED = 'year 2025, 1girl, charA, @styleX'


class ProtectedCaptionDropoutTest(unittest.TestCase):
    def test_protected_dropout_keeps_only_separator_prefix(self):
        item = FakeItem(
            CAPTION,
            protected_caption_dropout_rate=1.0,
            token_dropout_rate=0.5,
            shuffle_caption=True,
            random_triggers=['random tag'],
        )

        self.assertEqual(item.get_caption(), PROTECTED)

    def test_protected_dropout_keeps_first_keep_tokens_without_separator(self):
        item = FakeItem(
            'charA, @styleX, red hair, smile',
            protected_caption_dropout_rate=1.0,
            keep_tokens=2,
        )

        self.assertEqual(item.get_caption(), 'charA, @styleX')

    def test_secondary_separator_group_is_one_protected_unit(self):
        item = FakeItem(
            'charA;;;@styleX, red hair',
            protected_caption_dropout_rate=1.0,
            keep_tokens=1,
            secondary_separator=';;;',
        )

        self.assertEqual(item.get_caption(), 'charA, @styleX')

    def test_trigger_outside_protected_section_is_reinjected(self):
        item = FakeItem(
            '1girl ||| charA, red hair',
            trigger_word='charA',
            protected_caption_dropout_rate=1.0,
        )

        self.assertEqual(item.get_caption(), 'charA 1girl')

    def test_nothing_protected_falls_back_to_dropout_caption(self):
        self.assertEqual(
            FakeItem('red hair, smile', protected_caption_dropout_rate=1.0).get_caption(),
            '',
        )
        # an empty prefix before the separator counts as nothing protected
        for raw in ('red hair, smile', ' ||| red hair, smile', '|||red hair'):
            with self.subTest(raw=raw):
                item = FakeItem(raw, trigger_word='charA', protected_caption_dropout_rate=1.0)
                self.assertEqual(item.get_caption(), item.get_dropout_caption())

    def test_nothing_protected_in_mixed_caption_falls_back_to_dropout_caption(self):
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}
        for variant in weights:
            with self.subTest(variant=variant):
                with patch('toolkit.caption_utils.random.choices', return_value=[variant]):
                    selection = select_mixed_caption_selection(
                        ' ||| a, b',
                        'A woman, standing outdoors.',
                        weights,
                        '|||',
                    )
                item = FakeItem(
                    selection.render(),
                    trigger_word='charA',
                    mixed_selection=selection,
                    protected_caption_dropout_rate=1.0,
                )

                self.assertEqual(item.get_caption(), item.get_dropout_caption())

    def test_nothing_protected_shares_the_blank_cache_file(self):
        with TemporaryDirectory() as temp_dir:
            item = FakeItem(
                ' ||| red hair',
                path=os.path.join(temp_dir, 'image.png'),
                trigger_word='charA',
                caption_dropout_rate=0.2,
                protected_caption_dropout_rate=0.3,
                cache_text_embeddings=True,
            )

            self.assertEqual(item.get_protected_dropout_caption(), item.get_dropout_caption())
            # without control conditioning the two embeddings are the same file
            self.assertEqual(
                item.get_protected_text_embedding_path(),
                item.get_blank_text_embedding_path(),
            )

    def test_blank_and_protected_rates_share_one_exclusive_roll(self):
        item = FakeItem(CAPTION, caption_dropout_rate=0.2, protected_caption_dropout_rate=0.3)
        expected = {
            0.1: '',
            0.4: PROTECTED,
            0.6: 'year 2025, 1girl, charA, @styleX, red hair, smile, outdoors',
        }

        for roll, caption in expected.items():
            with self.subTest(roll=roll), patch('random.random', return_value=roll) as roll_mock:
                self.assertEqual(item.get_caption(), caption)
                self.assertEqual(roll_mock.call_count, 1)

    def test_rates_at_zero_do_not_roll(self):
        item = FakeItem(CAPTION)

        with patch('random.random') as roll_mock:
            item.get_caption()

        roll_mock.assert_not_called()

    def test_live_roll_is_skipped_when_caching_text_embeddings(self):
        item = FakeItem(CAPTION, protected_caption_dropout_rate=1.0, cache_text_embeddings=True)

        with patch('random.random') as roll_mock:
            caption = item.get_caption()

        roll_mock.assert_not_called()
        self.assertEqual(caption, 'year 2025, 1girl, charA, @styleX, red hair, smile, outdoors')

    def test_mixed_caption_drops_natural_language_in_every_variant(self):
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}
        for variant in weights:
            with self.subTest(variant=variant):
                with patch('toolkit.caption_utils.random.choices', return_value=[variant]):
                    selection = select_mixed_caption_selection(
                        'fixed ||| a, b',
                        'A woman, standing outdoors.',
                        weights,
                        '|||',
                    )
                item = FakeItem(
                    selection.render(),
                    mixed_selection=selection,
                    protected_caption_dropout_rate=1.0,
                    keep_tokens=1,
                    shuffle_caption=True,
                )

                self.assertEqual(item.get_caption(), 'fixed, a')

    def test_cached_protected_caption_matches_live_caption(self):
        live = FakeItem(CAPTION, trigger_word='charA', protected_caption_dropout_rate=1.0)
        cached = FakeItem(
            CAPTION,
            trigger_word='charA',
            protected_caption_dropout_rate=1.0,
            cache_text_embeddings=True,
        )

        self.assertEqual(cached.get_protected_dropout_caption(), live.get_caption())

    def test_cache_encodes_protected_embeddings_and_loads_them_on_the_roll(self):
        with TemporaryDirectory() as temp_dir:
            item = FakeItem(
                CAPTION,
                path=os.path.join(temp_dir, 'image.png'),
                trigger_word='charA',
                caption_dropout_rate=0.2,
                protected_caption_dropout_rate=0.3,
                cache_text_embeddings=True,
                diff_output_preservation=True,
            )
            dataset = FakeDataset([item], item.dataset_config)

            dataset.cache_text_embeddings()

            self.assertIn(PROTECTED, dataset.encoded)
            self.assertIn('year 2025, 1girl, woman, @styleX', dataset.encoded)
            self.assertTrue(item.is_text_embedding_cached)

            expected = {
                0.1: ('charA ', 'woman '),
                0.4: (PROTECTED, 'year 2025, 1girl, woman, @styleX'),
                0.6: (
                    'year 2025, 1girl, charA, @styleX, red hair, smile, outdoors',
                    'year 2025, 1girl, woman, @styleX, red hair, smile, outdoors',
                ),
            }
            for roll, (caption, dop_caption) in expected.items():
                with self.subTest(roll=roll), patch('random.random', return_value=roll):
                    item.prompt_embeds = None
                    item.dop_prompt_embeds = None
                    item.load_prompt_embedding()

                    self.assertEqual(item.prompt_embeds.caption, caption)
                    self.assertEqual(item.dop_prompt_embeds.caption, dop_caption)

    def test_dataset_config_validates_protected_rate(self):
        with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
            DATASET_CONFIG(protected_caption_dropout_rate=1.5)
        with self.assertRaisesRegex(ValueError, 'cannot exceed 1'):
            DATASET_CONFIG(caption_dropout_rate=0.5, protected_caption_dropout_rate=0.6)
        for blank in (-0.1, float('nan')):
            with self.subTest(blank=blank), self.assertRaisesRegex(ValueError, 'caption_dropout_rate must be between'):
                DATASET_CONFIG(caption_dropout_rate=blank, protected_caption_dropout_rate=0.2)

        config = DATASET_CONFIG(caption_dropout_rate=0.02, protected_caption_dropout_rate=0.08)
        self.assertEqual(config.protected_caption_dropout_rate, 0.08)
        # existing configs without the new option keep loading unchanged
        self.assertEqual(DATASET_CONFIG(caption_dropout_rate=1.5).protected_caption_dropout_rate, 0.0)
        self.assertEqual(DATASET_CONFIG(caption_dropout_rate=-0.1).caption_dropout_rate, -0.1)


if __name__ == '__main__':
    unittest.main()
