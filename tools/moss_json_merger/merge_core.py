"""Standalone, standard-library-only MOSS caption/tag batch merging."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable
import uuid


RUNS_FOLDER = '_moss_merge_runs'
STATUS_LABELS = {
    'ready': '변경 예정', 'unchanged': '변경 없음', 'missing_txt': 'TXT 없음',
    'missing_json': 'JSON 없음', 'invalid_json': 'JSON 오류', 'invalid_txt': 'TXT 오류',
    'conflict': '매칭 충돌', 'output_exists': '출력 파일 존재', 'written': '저장 완료',
    'stale': '입력/출력 변경됨', 'error': '저장 실패', 'cancelled': '취소됨',
    'not_selected': '선택 제외',
}


class ScanCancelled(Exception):
    pass


def unique(tags):
    return list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


def _tag_list(text, multiline=False):
    text = text.strip()
    if '\n' in text or '\r' in text:
        if not multiline:
            raise ValueError('여러 줄 TXT입니다. 줄바꿈을 태그 구분자로 사용할지 확인해 주세요.')
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', ',')
    return unique(text.split(','))


def parse_tags(text, separator='|||', multiline=False):
    """Return (protected, general) with exact tag spelling and stable order."""
    text = text.strip()
    if not text:
        raise ValueError('태그 TXT가 비어 있습니다.')
    if separator and text.count(separator) > 1:
        raise ValueError('보호 구분자가 두 번 이상 있습니다. TXT 내용을 확인해 주세요.')
    if separator and separator in text:
        before, after = text.split(separator, 1)
        keep = _tag_list(before, multiline)
        tags = _tag_list(after, multiline)
    else:
        keep, tags = [], _tag_list(text, multiline)
    tags = [tag for tag in tags if tag not in set(keep)]
    if not keep and not tags:
        raise ValueError('TXT에서 태그를 찾지 못했습니다.')
    return keep, tags


def _reject_constant(value):
    raise ValueError(f'JSON에 사용할 수 없는 숫자: {value}')


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON에 중복 키가 있습니다: {key}')
        result[key] = value
    return result


def load_annotation(raw):
    data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get('caption'), dict):
        raise ValueError('caption.short / medium / long이 있는 MOSS JSON이 아닙니다.')
    for length in ('short', 'medium', 'long'):
        value = data['caption'].get(length)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'caption.{length}는 비어 있지 않은 문자열이어야 합니다.')
    for key in ('tags', 'keep_tags'):
        tags = data.get(key, [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f'{key}는 비어 있지 않은 문자열 배열이어야 합니다.')
    return data


def merge_annotation(original, keep, tags, policy='merge', common_keep=()):
    old_keep, old_tags = original.get('keep_tags', []), original.get('tags', [])
    imported_keep = unique([*common_keep, *keep])
    if policy == 'merge':
        new_keep, new_tags = unique([*old_keep, *imported_keep]), unique([*old_tags, *tags])
    elif policy == 'replace':
        new_keep, new_tags = imported_keep, unique(tags)
    elif policy == 'fill':
        new_keep = unique(old_keep) if old_keep else imported_keep
        new_tags = unique(old_tags) if old_tags else unique(tags)
    else:
        raise ValueError(f'지원하지 않는 정책: {policy}')
    protected = set(new_keep)
    result = dict(original)
    result['keep_tags'] = new_keep
    result['tags'] = [tag for tag in new_tags if tag not in protected]
    return result


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _within(path, root):
    return path.resolve().is_relative_to(root.resolve())


@dataclass(frozen=True)
class MergeOptions:
    json_dir: Path
    txt_dir: Path
    recursive: bool = True
    separator: str = '|||'
    common_keep: str = ''
    policy: str = 'merge'
    encoding: str = 'utf-8-sig'
    multiline: bool = False
    output_dir: Path | None = None

    def validate(self):
        for label, path in [('JSON', self.json_dir), ('TXT', self.txt_dir)]:
            if not path.is_dir():
                raise ValueError(f'{label} 폴더를 확인해 주세요: {path}')
        if self.output_dir is not None:
            if self.output_dir.resolve() == self.json_dir.resolve():
                raise ValueError('JSON 폴더에 저장하려면 원본 갱신 방식을 선택해 주세요.')
            if self.output_dir.exists() and not self.output_dir.is_dir():
                raise ValueError('출력 경로가 폴더가 아닙니다.')
        if self.policy not in ('merge', 'replace', 'fill'):
            raise ValueError('기존 태그 처리 정책을 확인해 주세요.')
        if self.encoding not in ('utf-8-sig', 'cp949'):
            raise ValueError('TXT 인코딩을 확인해 주세요.')
        if any(char in self.separator for char in '\r\n,'):
            raise ValueError('보호 구분자에 쉼표나 줄바꿈을 사용할 수 없습니다.')
        _tag_list(self.common_keep)


@dataclass
class PlanItem:
    key: str
    relative: str
    status: str
    message: str = ''
    json_path: Path | None = None
    txt_path: Path | None = None
    destination: Path | None = None
    original: dict | None = None
    merged: dict | None = None
    json_hash: str = ''
    txt_hash: str = ''


@dataclass
class MergePlan:
    options: MergeOptions
    items: list[PlanItem]

    @property
    def counts(self):
        return Counter(item.status for item in self.items)


def _index_files(root, suffix, options, cancelled):
    index = defaultdict(list)
    exclude_output = (suffix == '.json' and options.output_dir is not None
                      and _within(options.output_dir, root) and options.output_dir.resolve() != root)
    def walk_error(error):
        raise error
    for current, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        if cancelled():
            raise ScanCancelled()
        dirs[:] = sorted(name for name in dirs if not name.startswith('.') and name != RUNS_FOLDER
                         and not Path(current, name).is_symlink()
                         and not getattr(os.path, 'isjunction', lambda p: False)(Path(current, name))
                         and not (exclude_output and _within(Path(current, name), options.output_dir)))
        for name in sorted(files):
            path = Path(current, name)
            if path.suffix.lower() != suffix or path.is_symlink():
                continue
            relative = path.relative_to(root)
            index[relative.with_suffix('').as_posix().casefold()].append(path)
        if not options.recursive:
            break
    return index


def build_plan(options: MergeOptions, progress=None, cancelled=lambda: False):
    options.validate()
    json_root, txt_root = options.json_dir.resolve(), options.txt_dir.resolve()
    json_files = _index_files(json_root, '.json', options, cancelled)
    txt_files = _index_files(txt_root, '.txt', options, cancelled)
    keys = sorted(json_files.keys() | txt_files.keys())
    common_keep = _tag_list(options.common_keep)
    items = []
    for number, key in enumerate(keys, 1):
        if cancelled():
            raise ScanCancelled()
        js, ts = json_files.get(key, []), txt_files.get(key, [])
        relative = (js[0].relative_to(json_root) if js else ts[0].relative_to(txt_root)).as_posix()
        item = PlanItem(key, relative, 'ready', json_path=js[0] if js else None,
                        txt_path=ts[0] if ts else None)
        if len(js) > 1 or len(ts) > 1:
            item.status, item.message = 'conflict', '같은 상대 경로로 해석되는 파일이 여러 개입니다.'
        elif not js:
            item.status, item.message = 'missing_json', '같은 이름의 MOSS JSON이 없습니다.'
        elif not ts:
            item.status, item.message = 'missing_txt', '같은 이름의 태그 TXT가 없습니다.'
        else:
            try:
                raw_json = js[0].read_bytes()
                item.original = load_annotation(raw_json)
                item.json_hash = digest(raw_json)
            except (OSError, ValueError) as exc:
                item.status, item.message = 'invalid_json', str(exc)
            if item.status == 'ready':
                try:
                    raw_txt = ts[0].read_bytes()
                    keep, tags = parse_tags(raw_txt.decode(options.encoding), options.separator, options.multiline)
                    item.txt_hash = digest(raw_txt)
                except (OSError, ValueError) as exc:
                    item.status, item.message = 'invalid_txt', str(exc)
            if item.status == 'ready':
                item.merged = merge_annotation(item.original, keep, tags, options.policy, common_keep)
                item.destination = ((options.output_dir.resolve() / relative) if options.output_dir else js[0])
                if options.output_dir and item.destination.exists():
                    try:
                        same = load_annotation(item.destination.read_bytes()) == item.merged
                    except (OSError, ValueError):
                        same = False
                    item.status = 'unchanged' if same else 'output_exists'
                    item.message = '출력 파일이 이미 같습니다.' if same else '출력 파일이 이미 있습니다. 빈 출력 폴더를 선택해 주세요.'
                elif not options.output_dir and item.merged == item.original:
                    item.status = 'unchanged'
        items.append(item)
        if progress:
            progress(number, len(keys), relative)
    return MergePlan(options, items)


def _serialize(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _atomic_write(path, content, overwrite=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.moss-merge-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        elif os.name == 'nt':
            os.rename(temporary, path)  # On Windows rename fails if a destination exists.
        else:
            os.link(temporary, path)  # Atomic no-clobber creation on other platforms.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class RunResult:
    run_dir: Path
    rows: list[dict] = field(default_factory=list)

    @property
    def counts(self):
        return Counter(row['status'] for row in self.rows)


def execute_plan(plan: MergePlan, selected: set[str], progress=None, cancelled=lambda: False):
    eligible = {item.key for item in plan.items if item.status == 'ready'}
    if not selected or not selected <= eligible:
        raise ValueError('미리보기에서 변경 예정 항목을 선택해 주세요.')
    options = plan.options
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_KST')
    run_dir = options.json_dir.resolve() / RUNS_FOLDER / f'{stamp}_{uuid.uuid4().hex[:8]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    result = RunResult(run_dir)
    fields = ['relative', 'status', 'message', 'json', 'txt', 'destination', 'backup']
    with (run_dir / 'report.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, item in enumerate(plan.items, 1):
            row = dict(relative=item.relative, status=item.status, message=item.message,
                       json=str(item.json_path or ''), txt=str(item.txt_path or ''),
                       destination=str(item.destination or ''), backup='')
            if item.status == 'ready' and item.key not in selected:
                row['status'] = 'not_selected'
            elif item.status == 'ready' and cancelled():
                row['status'] = 'cancelled'
            elif item.status == 'ready':
                try:
                    _execute_item(item, options, run_dir, row)
                except (OSError, ValueError) as exc:
                    row['status'], row['message'] = 'error', str(exc)
            result.rows.append(row)
            writer.writerow(row)
            stream.flush()
            if progress:
                progress(number, len(plan.items), row)
    summary = {
        'json_dir': str(options.json_dir), 'txt_dir': str(options.txt_dir),
        'output_dir': str(options.output_dir) if options.output_dir else None,
        'policy': options.policy, 'separator': options.separator,
        'common_keep': options.common_keep, 'encoding': options.encoding,
        'multiline': options.multiline, 'recursive': options.recursive,
        'counts': dict(result.counts), 'rows': result.rows,
    }
    _atomic_write(run_dir / 'summary.json', _serialize(summary), overwrite=False)
    return result


def _execute_item(item, options, run_dir, row):
    json_root, txt_root = options.json_dir.resolve(), options.txt_dir.resolve()
    if not _within(item.json_path, json_root) or not _within(item.txt_path, txt_root):
        raise ValueError('입력 파일이 선택한 폴더 밖을 가리킵니다.')
    if item.json_path.is_symlink() or item.txt_path.is_symlink():
        raise ValueError('심볼릭 링크로 변경된 입력 파일은 처리하지 않습니다.')
    raw_json = item.json_path.read_bytes()
    if digest(raw_json) != item.json_hash or digest(item.txt_path.read_bytes()) != item.txt_hash:
        row['status'], row['message'] = 'stale', '미리보기 이후 입력 파일이 바뀌었습니다. 다시 검사해 주세요.'
        return
    destination_root = options.output_dir.resolve() if options.output_dir else json_root
    if not _within(item.destination, destination_root):
        raise ValueError('출력 경로가 선택한 폴더 밖을 가리킵니다.')
    if options.output_dir and item.destination.exists():
        row['status'], row['message'] = 'stale', '미리보기 이후 출력 파일이 생겼습니다. 다시 검사해 주세요.'
        return
    if not options.output_dir:
        backup = run_dir / 'backups' / item.relative
        _atomic_write(backup, raw_json, overwrite=False)
        row['backup'] = str(backup)
    # Recheck after creating directories/backups, immediately before replacing.
    if digest(item.json_path.read_bytes()) != item.json_hash or digest(item.txt_path.read_bytes()) != item.txt_hash:
        row['status'], row['message'] = 'stale', '저장 직전에 입력이 바뀌어 건너뛰었습니다.'
        return
    _atomic_write(item.destination, _serialize(item.merged), overwrite=options.output_dir is None)
    row['status'], row['message'] = 'written', '결합한 JSON을 저장했습니다.'
