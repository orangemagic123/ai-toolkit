"""Style/character balanced sampling: tag classification, multipliers, sampler, dataloader wiring."""
from collections import Counter
from types import SimpleNamespace
import unittest

from toolkit.style_character_balance import (
    NO_STYLE,
    BalancedSampler,
    StyleCharacterBalanceConfig,
    build_balanced_sampler,
    classify_keep_tags,
    compute_multipliers,
)


def _records(rows, cfg):
    """rows: (count, keep_tags, exposure) -> exposure and record dicts with generated paths."""
    exposure, records = {}, {}
    for r, (count, keep_tags, e) in enumerate(rows):
        for i in range(count):
            path = f'img_{r}_{i}.png'
            exposure[path] = e
            records[path] = classify_keep_tags(keep_tags, cfg)
    return exposure, records


def _mix(result, exposure, records, concept):
    by_style = Counter()
    for p, rec in records.items():
        if concept in rec.concepts:
            by_style[rec.style] += exposure[p] * result.multipliers[p]
    total = sum(by_style.values())
    return {s: v / total for s, v in by_style.items()}, total


def _style_share(result, exposure, records):
    by_style = Counter()
    for p, rec in records.items():
        by_style[rec.style] += exposure[p] * result.multipliers[p]
    total = sum(by_style.values())
    return {s: v / total for s, v in by_style.items()}


class ConfigTest(unittest.TestCase):
    def test_off_values(self):
        self.assertIsNone(StyleCharacterBalanceConfig.from_value(None))
        self.assertIsNone(StyleCharacterBalanceConfig.from_value(False))
        self.assertIsNone(StyleCharacterBalanceConfig.from_value({'enabled': False}))

    def test_on_values(self):
        cfg = StyleCharacterBalanceConfig.from_value(True)
        self.assertEqual(cfg.max_multiplier, 3.0)
        self.assertAlmostEqual(cfg.min_multiplier, 1 / 3)
        self.assertIs(StyleCharacterBalanceConfig.from_value(cfg), cfg)

    def test_rejects_bad_options(self):
        with self.assertRaisesRegex(ValueError, 'unknown option'):
            StyleCharacterBalanceConfig.from_value({'max_multipler': 2})
        with self.assertRaises(ValueError):
            StyleCharacterBalanceConfig.from_value({'max_multiplier': 0.5})
        with self.assertRaises(ValueError):
            StyleCharacterBalanceConfig.from_value({'style_multipliers': {'@a': 0}})
        with self.assertRaises(ValueError):
            StyleCharacterBalanceConfig.from_value('yes')


class ClassifyTest(unittest.TestCase):
    def test_splits_style_counts_ignored_and_concepts(self):
        cfg = StyleCharacterBalanceConfig(ignore_tags=['Very Aesthetic', 'location'])
        rec = classify_keep_tags(
            ['@dmftk', 'very aesthetic', 'cecilia (series)', 'lawrence (series)', '1boy', '1girl', 'location',
             'tptlf costume'], cfg)
        self.assertEqual(rec.style, '@dmftk')
        self.assertEqual(rec.concepts, ('cecilia (series)', 'lawrence (series)', 'tptlf costume'))

    def test_no_style_and_other_counts(self):
        cfg = StyleCharacterBalanceConfig()
        rec = classify_keep_tags(['2girls', '6+boys', 'no humans', 'multiple others', 'chaos'], cfg)
        self.assertEqual(rec.style, NO_STYLE)
        self.assertEqual(rec.concepts, ('chaos',))


class MultiplierTest(unittest.TestCase):
    # concept A: 8 images in @x, 2 in @y; plain images: 10 in @x, 10 in @y -> global @x 60%, @y 40%
    ROWS = [(8, ['@x', 'a'], 1), (2, ['@y', 'a'], 1), (10, ['@x'], 1), (10, ['@y'], 1)]

    def test_concept_mix_matches_global_and_keeps_exposure(self):
        cfg = StyleCharacterBalanceConfig()
        exposure, records = _records(self.ROWS, cfg)
        result = compute_multipliers(exposure, records, cfg)
        mix, total = _mix(result, exposure, records, 'a')
        self.assertAlmostEqual(mix['@x'], 0.6, places=3)
        self.assertAlmostEqual(mix['@y'], 0.4, places=3)
        self.assertAlmostEqual(total, 10, places=3)

    def test_keep_style_share_holds_global_shares(self):
        cfg = StyleCharacterBalanceConfig(keep_style_share=True)
        exposure, records = _records(self.ROWS, cfg)
        result = compute_multipliers(exposure, records, cfg)
        share = _style_share(result, exposure, records)
        self.assertAlmostEqual(share['@x'], 0.6, places=3)
        mix, _ = _mix(result, exposure, records, 'a')
        self.assertAlmostEqual(mix['@x'], 0.6, places=2)

    def test_style_multipliers_set_global_shares(self):
        cfg = StyleCharacterBalanceConfig(style_multipliers={'@y': 1.5})
        exposure, records = _records(self.ROWS, cfg)
        result = compute_multipliers(exposure, records, cfg)
        share = _style_share(result, exposure, records)
        # 0.4 * 1.5 / (0.6 + 0.4 * 1.5) = 0.5
        self.assertAlmostEqual(share['@y'], 0.5, places=3)

    def test_existing_repeats_count_as_exposure(self):
        # same as ROWS, but the 2 @y images of A already have 3x exposure (e.g. num_repeats)
        rows = [(8, ['@x', 'a'], 1), (2, ['@y', 'a'], 3), (10, ['@x'], 1), (10, ['@y'], 1)]
        cfg = StyleCharacterBalanceConfig()
        exposure, records = _records(rows, cfg)
        result = compute_multipliers(exposure, records, cfg)
        mix, _ = _mix(result, exposure, records, 'a')
        self.assertAlmostEqual(mix['@x'], 18 / 34, places=3)

    def test_multipliers_are_clipped(self):
        rows = [(20, ['@x', 'a'], 1), (1, ['@y', 'a'], 1), (20, ['@y'], 1)]
        cfg = StyleCharacterBalanceConfig(max_multiplier=2.0)
        exposure, records = _records(rows, cfg)
        result = compute_multipliers(exposure, records, cfg)
        values = result.multipliers.values()
        self.assertLessEqual(max(values), 2.0 + 1e-9)
        self.assertGreaterEqual(min(values), 0.5 - 1e-9)
        self.assertTrue(any('at the 2x cap' in line for line in result.report))

    def test_single_style_concept_is_reported_and_left_alone(self):
        rows = [(5, ['@k', 'deez'], 1), (5, ['@x'], 1)]
        cfg = StyleCharacterBalanceConfig()
        exposure, records = _records(rows, cfg)
        result = compute_multipliers(exposure, records, cfg)
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in result.multipliers.values()))
        self.assertTrue(any('deez (@k)' in line for line in result.report))


class SamplerTest(unittest.TestCase):
    def test_epoch_length_and_expected_counts(self):
        weights = [1.0, 1.0, 2.0, 4.0]
        sampler = BalancedSampler(weights, seed=0)
        totals = Counter()
        epochs = 4000
        for _ in range(epochs):
            order = list(sampler)
            self.assertEqual(len(order), len(weights))
            totals.update(order)
        for i, w in enumerate(weights):
            expected = w * len(weights) / sum(weights)
            self.assertAlmostEqual(totals[i] / epochs, expected, delta=0.05)

    def test_every_epoch_fills_its_length(self):
        sampler = BalancedSampler([1, 1, 2, 0.000001], seed=1)
        for _ in range(50):
            counts = Counter(sampler)
            self.assertEqual(sum(counts.values()), 4)
            self.assertGreaterEqual(counts[2], 1)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            BalancedSampler([1, 0])


def _item(path, keep_tags):
    return SimpleNamespace(path=path, structured_caption_source=SimpleNamespace(keep_tags=tuple(keep_tags)))


def _dataset(items, buckets=True, batches=None):
    ds = SimpleNamespace(dataset_config=SimpleNamespace(buckets=buckets), file_list=items)
    if buckets:
        ds.batch_indices = batches if batches is not None else [[i] for i in range(len(items))]
    return ds


class BuildSamplerTest(unittest.TestCase):
    def _datasets(self):
        a = [_item(f'a{i}.png', ['@x', 'a']) for i in range(8)] + [_item(f'b{i}.png', ['@y', 'a']) for i in range(2)]
        plain = [_item(f'x{i}.png', ['@x']) for i in range(10)] + [_item(f'y{i}.png', ['@y']) for i in range(10)]
        return [_dataset(a), _dataset(plain, buckets=False)]

    def test_off_returns_none(self):
        self.assertIsNone(build_balanced_sampler(self._datasets(), 30, None, log=lambda _: None))

    def test_builds_weights_per_unit(self):
        lines = []
        sampler = build_balanced_sampler(self._datasets(), 30, {'enabled': True}, log=lines.append)
        self.assertEqual(len(sampler), 30)
        # A's two @y images are drawn 2x, its @x images 0.75x
        self.assertAlmostEqual(sampler.weights[8], 2.0, places=3)
        self.assertAlmostEqual(sampler.weights[0], 0.75, places=3)
        self.assertTrue(lines[0].startswith('Style/character balance'))

    def test_bucket_batches_average_member_weights(self):
        items = [_item('p.png', ['@x', 'a']), _item('q.png', ['@y', 'a']), _item('r.png', ['@x'])]
        ds = _dataset(items, batches=[[0, 1], [2]])
        sampler = build_balanced_sampler([ds], 2, True, log=lambda _: None)
        self.assertEqual(len(sampler), 2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            build_balanced_sampler(self._datasets(), 31, True, log=lambda _: None)

    def test_requires_structured_captions(self):
        ds = _dataset([SimpleNamespace(path='a.png', structured_caption_source=None)])
        with self.assertRaisesRegex(ValueError, 'moss_json'):
            build_balanced_sampler([ds], 1, True, log=lambda _: None)

    def test_works_as_torch_dataloader_sampler(self):
        try:
            from torch.utils.data import ConcatDataset, DataLoader
        except ImportError:
            self.skipTest('torch not installed')

        class Fake:
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds.batch_indices) if self.ds.dataset_config.buckets else len(self.ds.file_list)

            def __getitem__(self, i):
                if self.ds.dataset_config.buckets:
                    return [self.ds.file_list[j].path for j in self.ds.batch_indices[i]]
                return [self.ds.file_list[i].path]

        datasets = self._datasets()
        concat = ConcatDataset([Fake(d) for d in datasets])
        sampler = build_balanced_sampler(datasets, len(concat), True, log=lambda _: None)
        loader = DataLoader(concat, batch_size=None, shuffle=False, sampler=sampler, collate_fn=lambda b: b)
        seen = [path for batch in loader for path in batch]
        self.assertEqual(len(seen), 30)
        self.assertEqual(len(loader), 30)


if __name__ == '__main__':
    unittest.main()
