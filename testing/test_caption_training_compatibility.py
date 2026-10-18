"""Regression tests at the caption loader / trainer boundary, without models."""
import ast
import contextlib
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List
import unittest

from testing.test_moss_json_captions import JsonItem, annotation, write_caption, GET_SINGLE_ITEM
from testing.test_protected_caption_dropout import FakeDataset, FakeItem, NAMESPACE, ROOT, _load_definitions


class PromptsPrepared(Exception):
    pass


def _stop_before_latents(_):
    # Stop the real trainer after prompt preparation, before any model work.
    raise PromptsPrepared(sys._getframe(1).f_locals['conditioned_prompts'])


def _trainer_prompts(item, *, global_trigger=None, dual=False):
    path = ROOT / 'jobs/process/BaseSDTrainProcess.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseSDTrainProcess')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_general_training_batch')
    namespace = {
        'torch': SimpleNamespace(no_grad=contextlib.nullcontext),
        'get_torch_dtype': _stop_before_latents,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    trainer = SimpleNamespace(
        timer=lambda _: contextlib.nullcontext(),
        train_config=SimpleNamespace(short_and_long_captions=dual, prompt_saturation_chance=0, dtype='float32'),
        model_config=SimpleNamespace(refiner_name_or_path=None),
        embedding=None, adapter=None, trigger_word=global_trigger,
        sd=SimpleNamespace(inject_trigger_into_prompt=NAMESPACE['inject_trigger_into_prompt']),
    )
    batch = SimpleNamespace(
        file_items=[item],
        get_caption_list=lambda: [item.caption],
        get_caption_short_list=lambda: [item.caption_short],
        get_is_reg_list=lambda: [item.is_reg],
    )
    try:
        namespace[method.name](trainer, batch)
    except PromptsPrepared as result:
        return result.args[0]
    raise AssertionError('Trainer did not reach prompt preparation')


def _validate(dataset, **train_options):
    namespace = dict(TrainConfig=object, ModelConfig=object, SaveConfig=object, DatasetConfig=object, List=List)
    validate = _load_definitions('toolkit/config_modules.py', {'validate_configs'}, namespace)['validate_configs']
    options = dict(
        bypass_guidance_embedding=False, do_guidance_loss=False, unload_text_encoder=False,
        diff_output_preservation=False, blank_prompt_preservation=False,
        short_and_long_captions=False, short_and_long_captions_encoder_split=False,
    )
    options.update(train_options)
    validate(
        SimpleNamespace(**options),
        SimpleNamespace(is_flux=False, accuracy_recovery_adapter=None, arch='sdxl'),
        SimpleNamespace(), [dataset],
    )


class CaptionTrainingCompatibilityTest(unittest.TestCase):
    def test_moss_short_caption_reaches_dual_caption_training_intact(self):
        with TemporaryDirectory() as folder:
            value = annotation()
            value['caption']['short'] = 'A woman, outdoors.\nA sign reads "||| garden".'
            item = JsonItem(write_caption(folder, value), caption_length_weights={'long': 1},
                            shuffle_caption=True, token_dropout_rate=1, keep_tokens_separator='|||')
            item.trigger_word = 'mychar'
            item.load_caption()
            expected_short = 'mychar, portrait, ' + value['caption']['short']
            self.assertEqual(item.caption_short, expected_short)
            self.assertEqual(_trainer_prompts(item, global_trigger='mychar', dual=True),
                             [item.caption, expected_short])

    def test_moss_short_caption_survives_cache_enumeration(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), caption_mode='mixed', cache_text_embeddings=True)
            item.load_caption()
            expected = item.caption_short
            dataset = FakeDataset([item], item.dataset_config)
            dataset.cache_text_embeddings()
            self.assertEqual(item.caption_short, expected)
            self.assertEqual(expected, 'mychar, portrait, A woman outdoors.')

    def test_moss_dual_caption_cache_combinations_fail_before_model_loading(self):
        for flag in ('short_and_long_captions', 'short_and_long_captions_encoder_split'):
            with self.subTest(flag=flag):
                item = JsonItem('unused.png', cache_text_embeddings=True)
                with self.assertRaisesRegex(ValueError, 'caption_length_weights'):
                    _validate(item.dataset_config, **{flag: True})
                item.dataset_config.cache_text_embeddings = False
                _validate(item.dataset_config, **{flag: True})
                item.dataset_config.cache_text_embeddings = True
                _validate(item.dataset_config)

    def test_dataset_only_trigger_dropout_matches_live_trainer_and_cache(self):
        for is_reg in (False, True):
            for trigger in (None, 'dataset_trigger'):
                with self.subTest(is_reg=is_reg, trigger=trigger), TemporaryDirectory() as folder:
                    image = write_caption(folder)
                    options = dict(caption_dropout_rate=1, diff_output_preservation=True,
                                   diff_output_preservation_class='person')
                    live = JsonItem(image, **options)
                    cached = JsonItem(image, cache_text_embeddings=True, **options)
                    for item in (live, cached):
                        item.trigger_word, item.is_reg = trigger, is_reg
                    live.load_caption()
                    dataset = FakeDataset([cached], cached.dataset_config)
                    dataset.caption_dict, dataset.transform = None, None
                    dataset.cache_text_embeddings()
                    loaded = GET_SINGLE_ITEM(dataset, 0)
                    self.assertEqual(_trainer_prompts(live)[0], loaded.prompt_embeds.caption)
                    self.assertEqual(live.caption_dop, loaded.dop_prompt_embeds.caption)
                    self.assertEqual(live.caption, 'dataset_trigger ' if trigger and not is_reg else '')

    def test_text_dropout_keeps_same_trigger_policy_as_cached_dropout(self):
        for is_reg in (False, True):
            item = FakeItem('tag1, tag2', trigger_word='dataset_trigger', caption_dropout_rate=1)
            item.is_reg = is_reg
            self.assertEqual(item.get_caption(), item.get_dropout_caption())

    def test_dataset_trigger_override_is_not_followed_by_global_trigger(self):
        with TemporaryDirectory() as folder:
            item = JsonItem(write_caption(folder), caption_dropout_rate=1)
            item.trigger_word = 'dataset_trigger'
            item.load_caption()
            self.assertEqual(_trainer_prompts(item, global_trigger='global_trigger')[0], item.get_dropout_caption())
            item.dataset_config.caption_dropout_rate = 0
            item.load_caption()
            prompts = _trainer_prompts(item, global_trigger='global_trigger', dual=True)
            self.assertTrue(all('dataset_trigger' in prompt for prompt in prompts))
            self.assertTrue(all('global_trigger' not in prompt for prompt in prompts))


if __name__ == '__main__':
    unittest.main()
