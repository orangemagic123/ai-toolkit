"""Exercise real caption/cache methods without loading GPU model dependencies."""
import ast
import copy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from toolkit.caption_utils import parse_structured_caption
from testing.test_protected_caption_dropout import (
    DATASET_CONFIG, FakeDataset, FakeItem, NAMESPACE, ROOT,
)


NAMESPACE['parse_structured_caption'] = parse_structured_caption
NAMESPACE['clean_caption'] = lambda caption: caption


class JsonItem(FakeItem):
    load_caption = NAMESPACE['CaptionProcessingDTOMixin'].load_caption

    def __init__(self, path, **config):
        super().__init__(None, path=str(path))
        self.dataset_config = DATASET_CONFIG(caption_format='moss_json', **config)
        self.structured_caption_source = None
        self.mixed_caption_sources = None
        self.caption_short = None

    def load_and_process_image(self, transform, **kwargs):
        # Use the real embedding loader at the point image processing calls it.
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()


def annotation():
    return {
        'caption': {
            'short': 'A woman outdoors.',
            'medium': 'A woman, wearing a white shirt, stands outdoors.',
            'long': 'A woman, wearing a white shirt, stands outdoors.\nThe sign reads "가든".',
        },
        'tags': ['1girl', 'white shirt', 'outdoors'],
        'keep_tags': ['mychar', 'portrait'],
        'watermark': [{'label': 'DO NOT TRAIN THIS', 'box_2d': [0, 0, 10, 10]}],
        'borders': ['none'],
    }


def write_caption(folder, value=None):
    image = Path(folder) / 'image.png'
    image.with_suffix('.json').write_text(
        json.dumps(annotation() if value is None else value, ensure_ascii=False), encoding='utf-8-sig',
    )
    return image


def load_single_item_method():
    source = ROOT / 'toolkit' / 'data_loader.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AiToolkitDataset')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_get_single_item')
    namespace = {'copy': copy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_get_single_item']


GET_SINGLE_ITEM = load_single_item_method()


class MossJsonCaptionTest(unittest.TestCase):
    def test_defaults_and_partial_length_weights(self):
        config = DATASET_CONFIG(caption_format='moss_json')
        self.assertEqual(config.caption_ext, '.json')
        self.assertEqual(config.caption_length_weights, {'short': 1, 'medium': 1, 'long': 1})
        self.assertEqual(DATASET_CONFIG().caption_ext, '.txt')
        config = DATASET_CONFIG(caption_length_weights={'long': 70})
        self.assertEqual(config.caption_length_weights, {'short': 0, 'medium': 0, 'long': 70})

    def test_invalid_length_weights_are_rejected(self):
        invalid = [[], {}, {'short': -1}, {'short': float('nan')}, {'long': float('inf')},
                   {'short': 1e308, 'long': 1e308}, {'long': 'bad'}, {'medium': None}, {'typo': 1}]
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, 'caption_length_weights'):
                DATASET_CONFIG(caption_length_weights=weights)

    def test_cache_requires_finite_caption_augmentation(self):
        for options in ({'shuffle_caption': True}, {'token_dropout_rate': 0.1}, {'random_triggers': ['tag']}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'random tag augmentation'):
                DATASET_CONFIG(caption_format='moss_json', cache_text_embeddings=True, **options)

    def test_protected_tags_are_deduplicated_in_order(self):
        value = annotation()
        value['keep_tags'] = ['mychar', 'portrait', 'mychar']
        value['tags'] += ['mychar', 'outdoors']
        parsed = parse_structured_caption(value)
        self.assertEqual(parsed.keep_tags, ('mychar', 'portrait'))
        self.assertEqual(parsed.tags, ('1girl', 'white shirt', 'outdoors'))

    def test_invalid_schema_reports_the_source(self):
        values = [[], {'caption': 'text'}, annotation(), annotation(), annotation()]
        values[2]['caption']['short'] = ''
        values[3]['tags'] = 'comma, separated'
        values[4]['keep_tags'] = [7]
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'image.json'):
                parse_structured_caption(value, 'image.json')

    def test_keep_prefix_and_exact_natural_language_in_all_variants(self):
        value = annotation()
        nl = value['caption']['long']
        expected = {
            'tags': 'mychar, portrait, outdoors, white shirt, 1girl',
            'nl': f'mychar, portrait, {nl}',
            'tags_nl': f'mychar, portrait, outdoors, white shirt, 1girl, {nl}',
            'nl_tags': f'mychar, portrait, {nl}, outdoors, white shirt, 1girl',
        }
        with TemporaryDirectory() as folder:
            image = write_caption(folder, value)
            for variant, caption in expected.items():
                item = JsonItem(image, caption_mode='mixed', mixed_weights={variant: 1},
                                caption_length_weights={'long': 1}, shuffle_caption=True)
                with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda tags: tags.reverse()):
                    item.load_caption()
                self.assertEqual(item.caption, caption)
                self.assertNotIn('DO NOT TRAIN THIS', item.caption)
                self.assertEqual(item.caption.count('mychar'), 1)

    def test_tag_dropout_never_splits_natural_language(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), caption_mode='mixed', mixed_weights={'tags_nl': 1},
                            caption_length_weights={'long': 1}, token_dropout_rate=1)
            item.load_caption()
            self.assertEqual(item.caption, 'mychar, portrait, ' + annotation()['caption']['long'])

    def test_length_is_reselected_for_each_load(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder))
            with patch('toolkit.caption_utils.random.choices', side_effect=[['short'], ['long']]):
                item.load_caption()
                first = item.caption
                # The cached annotation is reused, without pinning its length.
                item.load_caption()
            self.assertEqual(first, 'mychar, portrait, ' + annotation()['caption']['short'])
            self.assertEqual(item.caption, 'mychar, portrait, ' + annotation()['caption']['long'])

    def test_relative_probabilities_and_disabled_lengths(self):
        source = parse_structured_caption(annotation())
        counts = dict.fromkeys(('short', 'medium', 'long'), 0)
        state = random.getstate()
        try:
            random.seed(913)
            for _ in range(6000):
                selection = source.select('single', {}, {'short': 10, 'medium': 20, 'long': 70})
                length = next(key for key, text in source.captions.items() if text == selection.nl_caption)
                counts[length] += 1
        finally:
            random.setstate(state)
        for length, probability in [('short', .1), ('medium', .2), ('long', .7)]:
            self.assertAlmostEqual(counts[length] / 6000, probability, delta=.025)
        selections = list(source.selections('mixed', {'tags': 1, 'nl': 1, 'tags_nl': 1, 'nl_tags': 1},
                                            {'short': 0, 'medium': 0, 'long': 1}))
        self.assertEqual(len(selections), 4)
        self.assertTrue(all(not choice.includes_nl or choice.nl_caption == source.captions['long']
                            for choice in selections))

    def test_original_moss_without_tags_and_aggregate_metadata(self):
        value = annotation()
        del value['tags']
        del value['keep_tags']
        item = JsonItem('not-on-disk.png', caption_mode='mixed', mixed_weights={'tags': 1},
                        caption_length_weights={'medium': 1})
        item.load_caption({item.path: value})
        self.assertEqual(item.caption, value['caption']['medium'])

    def test_protected_and_full_dropout_are_distinct(self):
        with TemporaryDirectory() as folder:
            image = write_caption(folder)
            protected = JsonItem(image, protected_caption_dropout_rate=1)
            protected.load_caption()
            self.assertEqual(protected.caption, 'mychar, portrait')
            blank = JsonItem(image, caption_dropout_rate=1)
            blank.load_caption()
            self.assertEqual(blank.caption, '')

    def test_existing_trigger_is_not_duplicated(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), caption_length_weights={'short': 1})
            item.trigger_word = 'mychar'
            item.load_caption()
            self.assertEqual(item.caption, 'mychar, portrait, A woman outdoors.')

    def test_raw_json_prompt_stays_verbatim_without_format_switch(self):
        with TemporaryDirectory() as folder:
            image = write_caption(folder)
            item = JsonItem(image)
            item.dataset_config.caption_format = 'text'
            item.load_caption()
            # Text mode preserves even an annotation JSON as raw conditioning.
            self.assertIn('"keep_tags"', item.caption)
            self.assertIn('DO NOT TRAIN THIS', item.caption)

    def test_missing_or_malformed_json_fails_with_path(self):
        with TemporaryDirectory() as folder:
            image = Path(folder) / 'missing.png'
            for contents in (None, '{broken JSON'):
                if contents is not None:
                    image.with_suffix('.json').write_text(contents, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'missing.json'):
                    JsonItem(image).load_caption()

    def test_all_cached_variants_and_preservation_match_selected_caption(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), cache_text_embeddings=True, caption_mode='mixed',
                            diff_output_preservation=True, diff_output_preservation_class='woman')
            item.trigger_word = 'mychar'
            dataset = FakeDataset([item], item.dataset_config)
            dataset.caption_dict = None
            dataset.transform = None
            dataset.cache_text_embeddings()
            normal = [caption for caption in dataset.encoded if caption.startswith('mychar, portrait')]
            self.assertEqual(len(normal), 10)
            self.assertTrue(item.is_text_embedding_cached)
            self.assertEqual(len(dataset.encoded), 20)
            for variant in ('tags', 'nl', 'tags_nl', 'nl_tags'):
                for length in ('short', 'medium', 'long'):
                    # Exercise the real dataset ordering: selection must precede
                    # the cached embedding load inside image processing.
                    choices = [[variant]] + ([] if variant == 'tags' else [[length]])
                    with patch('toolkit.caption_utils.random.choices', side_effect=choices):
                        loaded = GET_SINGLE_ITEM(dataset, 0)
                    self.assertEqual(loaded.prompt_embeds.caption, loaded.caption)
                    self.assertEqual(loaded.dop_prompt_embeds.caption, loaded.caption.replace('mychar', 'woman'))
                    self.assertTrue(loaded.caption.startswith('mychar, portrait'))
            dataset.cache_text_embeddings()
            self.assertEqual(len(dataset.encoded), 20, 'reuses every existing cache file')

    def test_cached_blank_and_protected_dropout(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), cache_text_embeddings=True, caption_mode='mixed',
                            caption_dropout_rate=.1, protected_caption_dropout_rate=.2)
            dataset = FakeDataset([item], item.dataset_config)
            dataset.caption_dict = None
            dataset.transform = None
            dataset.cache_text_embeddings()
            for roll, expected in ((.05, ''), (.2, 'mychar, portrait')):
                with patch('toolkit.caption_utils.random.random', return_value=roll):
                    loaded = GET_SINGLE_ITEM(dataset, 0)
                self.assertEqual(loaded.prompt_embeds.caption, expected)


if __name__ == '__main__':
    unittest.main()
