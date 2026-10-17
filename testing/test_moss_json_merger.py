import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.moss_json_merger import merge_core as core


def annotation(**extra):
    return {
        'caption': {'short': 'A woman.', 'medium': 'A woman holding an apple.',
                    'long': 'A woman holds an apple.\nThe sign reads "사과".'},
        'watermark': [], 'borders': ['none'], 'custom': {'text': 'visible text', 'values': [1, 2]},
        **extra,
    }


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.js, self.ts = self.root / 'json', self.root / 'tags'
        self.js.mkdir()
        self.ts.mkdir()

    def pair(self, name='001', data=None, tags='1girl, char, artist ||| apple, solo, red eyes'):
        js, ts = self.js / f'{name}.json', self.ts / f'{name}.txt'
        js.parent.mkdir(parents=True, exist_ok=True)
        ts.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(annotation() if data is None else data, ensure_ascii=False), encoding='utf-8-sig')
        ts.write_text(tags, encoding='utf-8-sig')
        return js, ts

    def plan(self, **kwargs):
        return core.build_plan(core.MergeOptions(self.js, self.ts, **kwargs))

    def run_plan(self, plan, **kwargs):
        return core.execute_plan(plan, {item.key for item in plan.items if item.status == 'ready'}, **kwargs)

    def test_exact_user_example(self):
        self.assertEqual(core.parse_tags('1girl, char, artist ||| apple, solo, red eyes'),
                         (['1girl', 'char', 'artist'], ['apple', 'solo', 'red eyes']))

    def test_tag_spelling_order_duplicates_and_no_separator(self):
        self.assertEqual(core.parse_tags(' char, Char, hair_color, (tag:1.2), char '),
                         ([], ['char', 'Char', 'hair_color', '(tag:1.2)']))
        self.assertEqual(core.parse_tags('char, char ||| char, apple, apple'), (['char'], ['apple']))

    def test_empty_repeated_separator_and_multiline_are_explicit(self):
        for text in (' ', '|||', 'a ||| b ||| c', 'a, b\nc, d'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                core.parse_tags(text)
        self.assertEqual(core.parse_tags('a, b\r\nc, d', multiline=True), ([], ['a', 'b', 'c', 'd']))
        self.assertEqual(core.parse_tags('char |||', multiline=False), (['char'], []))

    def test_metadata_and_natural_language_are_unchanged_in_all_policies(self):
        before = annotation(keep_tags=['old', 'char'], tags=['old', 'smile'])
        snapshot = copy.deepcopy(before)
        for policy, keep, tags in (
            ('merge', ['old', 'char', 'common', 'artist'], ['smile', 'apple']),
            ('replace', ['common', 'char', 'artist'], ['apple', 'old']),
            ('fill', ['old', 'char'], ['smile']),
        ):
            with self.subTest(policy=policy):
                merged = core.merge_annotation(before, ['char', 'artist'], ['apple', 'old'], policy, ['common'])
                self.assertEqual(merged['keep_tags'], keep)
                self.assertEqual(merged['tags'], tags)
                for key in ('caption', 'watermark', 'borders', 'custom'):
                    self.assertEqual(before[key], merged[key])
                self.assertEqual(core.merge_annotation(merged, ['char', 'artist'], ['apple', 'old'], policy, ['common']), merged)
        self.assertEqual(before, snapshot)

    def test_fill_missing_fields_and_empty_lists(self):
        merged = core.merge_annotation(annotation(tags=['old']), ['char'], ['apple'], 'fill')
        self.assertEqual(merged['keep_tags'], ['char'])
        self.assertEqual(merged['tags'], ['old'])
        merged = core.merge_annotation(annotation(keep_tags=[], tags=[]), ['char'], ['apple'], 'fill')
        self.assertEqual(merged['tags'], ['apple'])

    def test_schema_validation_rejects_bad_fields_duplicate_keys_and_nan(self):
        bad = [annotation(tags='tag'), annotation(keep_tags=[3]), {'caption': {'short': '', 'medium': 'm', 'long': 'l'}}]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                core.load_annotation(json.dumps(data).encode())
        for text in ('{"caption": {}, "caption": {}}', '{"number": NaN}'):
            with self.assertRaises(ValueError):
                core.load_annotation(text.encode())

    def test_relative_paths_prevent_cross_folder_matches(self):
        self.pair('a/001', tags='a ||| apple')
        self.pair('b/001', tags='b ||| banana')
        self.pair('c/002')
        (self.ts / 'c/002.txt').unlink()
        (self.ts / 'orphan.txt').write_text('tag')
        plan = self.plan()
        self.assertEqual(plan.counts, {'ready': 2, 'missing_txt': 1, 'missing_json': 1})
        self.assertEqual(plan.items[0].merged['keep_tags'], ['a'])
        self.assertEqual(plan.items[1].merged['keep_tags'], ['b'])
        self.assertEqual(self.plan(recursive=False).counts, {'missing_json': 1})

    def test_same_input_folder_and_unrelated_natural_language_txt(self):
        self.pair()
        (self.js / '001.txt').write_text('char ||| apple', encoding='utf-8')
        (self.js / '001.long.txt').write_text('A natural caption.', encoding='utf-8')
        plan = core.build_plan(core.MergeOptions(self.js, self.js))
        self.assertEqual(plan.counts, {'ready': 1, 'missing_json': 1})
        self.assertEqual(plan.items[0].merged['tags'], ['apple'])

    def test_conflicts_are_not_arbitrarily_matched(self):
        js, ts = self.pair()
        with patch.object(core, '_index_files', side_effect=[{'001': [js, js]}, {'001': [ts]}]):
            self.assertEqual(self.plan().counts, {'conflict': 1})

    def test_individual_errors_do_not_stop_good_pairs(self):
        self.pair()
        js, _ = self.pair('002')
        js.write_text('{bad json', encoding='utf-8')
        self.pair('003', tags='')
        plan = self.plan()
        self.assertEqual(plan.counts, {'ready': 1, 'invalid_json': 1, 'invalid_txt': 1})
        result = self.run_plan(plan)
        self.assertEqual(result.counts['written'], 1)

    def test_utf8_bom_cp949_and_wrong_encoding(self):
        _, ts = self.pair(tags='캐릭터 ||| 사과, 빨간 눈')
        self.assertEqual(self.plan().items[0].merged['keep_tags'], ['캐릭터'])
        ts.write_bytes('캐릭터 ||| 사과'.encode('cp949'))
        self.assertEqual(self.plan().counts, {'invalid_txt': 1})
        self.assertEqual(self.plan(encoding='cp949').items[0].merged['tags'], ['사과'])

    def test_real_files_backup_report_and_repeat_no_write(self):
        js, _ = self.pair()
        original = js.read_bytes()
        result = self.run_plan(self.plan())
        self.assertEqual(result.counts, {'written': 1})
        backup = Path(result.rows[0]['backup'])
        self.assertEqual(backup.read_bytes(), original)
        merged = core.load_annotation(js.read_bytes())
        self.assertEqual(merged['keep_tags'], ['1girl', 'char', 'artist'])
        self.assertEqual(merged['tags'], ['apple', 'solo', 'red eyes'])
        self.assertEqual(merged['caption'], annotation()['caption'])
        before_second_run = js.read_bytes()
        self.assertEqual(self.plan().counts, {'unchanged': 1}, 'run/backup/report folders must be excluded')
        self.assertEqual(js.read_bytes(), before_second_run)
        with (result.run_dir / 'report.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['status'], 'written')
        self.assertEqual(json.loads((result.run_dir / 'summary.json').read_text(encoding='utf-8'))['counts'], {'written': 1})

    def test_only_selected_files_change(self):
        first, _ = self.pair('001')
        second, _ = self.pair('002')
        original_second = second.read_bytes()
        result = core.execute_plan(self.plan(), {'001'})
        self.assertEqual(result.counts, {'written': 1, 'not_selected': 1})
        self.assertIn('keep_tags', json.loads(first.read_text(encoding='utf-8')))
        self.assertEqual(second.read_bytes(), original_second)

    def test_stale_json_and_txt_never_overwrite_new_edits(self):
        for changed in ('json', 'txt'):
            with self.subTest(changed=changed):
                js, ts = self.pair()
                plan = self.plan()
                target = js if changed == 'json' else ts
                target.write_bytes(target.read_bytes() + b' ')
                source_now = js.read_bytes()
                result = self.run_plan(plan)
                self.assertEqual(result.counts, {'stale': 1})
                self.assertEqual(js.read_bytes(), source_now)
                self.assertFalse((result.run_dir / 'backups').exists())

    def test_edit_during_backup_is_detected_before_write(self):
        js, _ = self.pair()
        plan = self.plan()
        atomic_write = core._atomic_write
        edited = b'new external edit'
        def edit_after_backup(path, content, overwrite=True):
            atomic_write(path, content, overwrite)
            if 'backups' in path.parts:
                js.write_bytes(edited)
        with patch.object(core, '_atomic_write', side_effect=edit_after_backup):
            result = self.run_plan(plan)
        self.assertEqual(result.counts, {'stale': 1})
        self.assertEqual(js.read_bytes(), edited)

    def test_backup_failure_leaves_original_intact(self):
        js, _ = self.pair()
        original = js.read_bytes()
        plan = self.plan()
        atomic_write = core._atomic_write
        def fail_backup(path, content, overwrite=True):
            if 'backups' in path.parts:
                raise PermissionError('backup unavailable')
            return atomic_write(path, content, overwrite)
        with patch.object(core, '_atomic_write', side_effect=fail_backup):
            result = self.run_plan(plan)
        self.assertEqual(result.counts, {'error': 1})
        self.assertEqual(js.read_bytes(), original)

    def test_atomic_replacement_failure_retains_original_and_backup(self):
        js, _ = self.pair()
        original = js.read_bytes()
        plan = self.plan()
        with patch.object(core.os, 'replace', side_effect=PermissionError('locked')):
            result = self.run_plan(plan)
        self.assertEqual(result.counts, {'error': 1})
        self.assertEqual(js.read_bytes(), original)
        self.assertEqual(Path(result.rows[0]['backup']).read_bytes(), original)
        self.assertFalse(list(self.js.glob('.moss-merge-*')))

    def test_output_mode_preserves_input_and_skips_existing_output(self):
        js, _ = self.pair('sub/001')
        original = js.read_bytes()
        output = self.js / 'merged'
        plan = self.plan(output_dir=output)
        self.assertEqual(self.run_plan(plan).counts, {'written': 1})
        destination = output / 'sub/001.json'
        self.assertTrue(destination.is_file())
        self.assertEqual(js.read_bytes(), original)
        self.assertEqual(self.plan(output_dir=output).counts, {'unchanged': 1})
        destination.write_text('already here', encoding='utf-8')
        self.assertEqual(self.plan(output_dir=output).counts, {'output_exists': 1})
        self.assertEqual(destination.read_text(), 'already here')

    def test_output_created_after_preview_is_not_overwritten(self):
        self.pair()
        output = self.root / 'output'
        plan = self.plan(output_dir=output)
        output.mkdir()
        destination = output / '001.json'
        destination.write_text('other content')
        self.assertEqual(self.run_plan(plan).counts, {'stale': 1})
        self.assertEqual(destination.read_text(), 'other content')

    def test_output_can_be_the_tags_images_folder(self):
        self.pair('sub/001')
        plan = self.plan(output_dir=self.ts)
        self.assertEqual(plan.counts, {'ready': 1})
        self.assertEqual(self.run_plan(plan).counts, {'written': 1})
        self.assertTrue((self.ts / 'sub/001.json').is_file())
        self.assertTrue((self.ts / 'sub/001.txt').is_file())

    def test_cancel_finishes_current_then_skips_remaining(self):
        self.pair('001')
        self.pair('002')
        stop = [False]
        def progress(*args):
            stop[0] = True
        result = self.run_plan(self.plan(), progress=progress, cancelled=lambda: stop[0])
        self.assertEqual(result.counts, {'written': 1, 'cancelled': 1})
        self.assertEqual(len(result.rows), 2)
        with self.assertRaises(core.ScanCancelled):
            core.build_plan(core.MergeOptions(self.js, self.ts), cancelled=lambda: True)

    def test_invalid_selections_and_output_roots(self):
        self.pair()
        for selected in (set(), {'not_a_real_key'}):
            with self.assertRaises(ValueError):
                core.execute_plan(self.plan(), selected)
        file_path = self.root / 'not-a-folder'
        file_path.write_text('file')
        for output in (self.js, file_path):
            with self.assertRaises(ValueError):
                self.plan(output_dir=output)


if __name__ == '__main__':
    unittest.main()
