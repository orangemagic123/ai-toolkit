"""Windows GUI for previewing and merging MOSS JSON caption annotations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from merge_core import (
    MergeOptions, STATUS_LABELS, ScanCancelled, build_plan, execute_plan,
)


POLICIES = {'합치기 (기존 태그 유지)': 'merge', 'TXT 기준 교체': 'replace', '빈 필드만 채우기': 'fill'}
ENCODINGS = {'UTF-8 / BOM': 'utf-8-sig', '한국어 CP949': 'cp949'}
FILTERS = {'전체': None, '변경 예정': {'ready'}, '변경 없음': {'unchanged'},
           '누락 / 오류': {'missing_txt', 'missing_json', 'invalid_json', 'invalid_txt', 'conflict', 'output_exists'},
           '저장 완료': {'written'}}


class MergerApp:
    def __init__(self, root, args):
        self.root = root
        root.title('MOSS JSON 태그 결합기')
        width, height = min(1800, root.winfo_screenwidth() - 80), min(1400, root.winfo_screenheight() - 90)
        root.geometry(f'{width}x{height}')
        root.minsize(980, 690)
        root.configure(background='#f3f6fa')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.close_pending = False
        self.plan = None
        self.fresh = False
        self.selected = set()
        self.results = {}
        self.run_dir = None
        self.inputs = []
        self.visible_items = {}
        self.json_dir = tk.StringVar(value=str(args.json_dir or ''))
        self.txt_dir = tk.StringVar(value=str(args.txt_dir or ''))
        self.output_dir = tk.StringVar(value=str(args.output_dir or ''))
        self.output_mode = tk.StringVar(value='folder' if args.output_dir else 'inplace')
        self.recursive = tk.BooleanVar(value=True)
        self.separator = tk.StringVar(value='|||')
        self.common_keep = tk.StringVar()
        self.policy = tk.StringVar(value=next(iter(POLICIES)))
        self.encoding = tk.StringVar(value=next(iter(ENCODINGS)))
        self.multiline = tk.BooleanVar(value=False)
        self.filter = tk.StringVar(value='전체')
        self.status = tk.StringVar(value='JSON 폴더와 태그 TXT 폴더를 선택한 뒤 검사해 주세요.')
        self.summary = tk.StringVar(value='검사 전')
        self.selection_label = tk.StringVar(value='선택 0개')
        self._style()
        self._layout()
        for variable in (self.json_dir, self.txt_dir, self.output_dir, self.output_mode, self.recursive,
                         self.separator, self.common_keep, self.policy, self.encoding, self.multiline):
            variable.trace_add('write', self.invalidate)
        self._sync_output()
        root.after(100, self.poll)

    def _style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('맑은 고딕', 10), background='#f3f6fa')
        style.configure('TLabel', foreground='#273449')
        style.configure('Title.TLabel', font=('맑은 고딕', 19, 'bold'), foreground='#17263c')
        style.configure('Hint.TLabel', foreground='#64748b', font=('맑은 고딕', 9))
        style.configure('TLabelframe', padding=10, bordercolor='#d8e0ec')
        style.configure('TLabelframe.Label', font=('맑은 고딕', 10, 'bold'), foreground='#263b59')
        style.configure('TButton', padding=(11, 6))
        style.configure('Accent.TButton', background='#2466c2', foreground='white')
        style.map('Accent.TButton', background=[('active', '#1954a5'), ('disabled', '#b8c7dd')])
        style.configure('Treeview', rowheight=29, background='white', fieldbackground='white', borderwidth=0)
        style.configure('Treeview.Heading', font=('맑은 고딕', 9, 'bold'), padding=5)

    def _input(self, widget, state='normal'):
        self.inputs.append((widget, state))
        return widget

    def _folder(self, parent, row, label, variable):
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky='w', pady=4)
        entry = self._input(ttk.Entry(parent, textvariable=variable))
        entry.grid(row=row, column=1, sticky='ew', padx=(0, 8), pady=4)
        button = self._input(ttk.Button(parent, text='폴더 선택', command=lambda: self.choose_folder(variable)))
        button.grid(row=row, column=2, pady=4)
        return entry, button

    def _layout(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='MOSS JSON 태그 결합기', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='같은 이름의 JSON + TXT를 연결하고, 보호 태그와 일반 태그를 분리해 저장합니다.',
                  style='Hint.TLabel').pack(anchor='w', pady=(3, 12))

        paths = ttk.LabelFrame(outer, text='1. 입력 / 저장 폴더')
        paths.pack(fill='x', pady=(0, 10))
        paths.columnconfigure(1, weight=1)
        self._folder(paths, 0, 'MOSS JSON 폴더', self.json_dir)
        self._folder(paths, 1, '태그 TXT 폴더', self.txt_dir)
        mode = ttk.Frame(paths)
        mode.grid(row=2, column=0, columnspan=3, sticky='w', pady=3)
        self._input(ttk.Checkbutton(mode, text='하위 폴더 포함', variable=self.recursive)).pack(side='left', padx=(0, 22))
        self._input(ttk.Radiobutton(mode, text='원본 JSON 갱신 + 자동 백업', variable=self.output_mode,
                                    value='inplace')).pack(side='left', padx=(0, 22))
        self._input(ttk.Radiobutton(mode, text='별도 폴더에 JSON만 출력', variable=self.output_mode,
                                    value='folder')).pack(side='left')
        self.output_entry, self.output_button = self._folder(paths, 3, '출력 폴더', self.output_dir)

        options = ttk.LabelFrame(outer, text='2. 태그 가져오기 규칙')
        options.pack(fill='x', pady=(0, 10))
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text='보호 구분자').grid(row=0, column=0, sticky='w', padx=(0, 8))
        self._input(ttk.Entry(options, textvariable=self.separator, width=9)).grid(row=0, column=1, sticky='w', padx=(0, 18))
        ttk.Label(options, text='공통 보호 태그').grid(row=0, column=2, padx=(0, 8))
        self._input(ttk.Entry(options, textvariable=self.common_keep)).grid(row=0, column=3, sticky='ew', padx=(0, 12))
        self._input(ttk.Combobox(options, textvariable=self.policy, values=list(POLICIES), width=22,
                                 state='readonly'), 'readonly').grid(row=0, column=4, sticky='e')
        ttk.Label(options, text='TXT 인코딩').grid(row=1, column=0, sticky='w', pady=(8, 0))
        self._input(ttk.Combobox(options, textvariable=self.encoding, values=list(ENCODINGS), width=14,
                                 state='readonly'), 'readonly').grid(row=1, column=1, sticky='w', pady=(8, 0))
        self._input(ttk.Checkbutton(options, text='줄바꿈도 태그 구분자로 사용', variable=self.multiline)).grid(
            row=1, column=2, columnspan=2, sticky='w', pady=(8, 0))
        ttk.Label(options, text='예: 1girl, char ||| apple, solo  →  keep_tags: 1girl, char  /  tags: apple, solo',
                  style='Hint.TLabel').grid(row=2, column=0, columnspan=5, sticky='w', pady=(7, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill='x', pady=(0, 8))
        self.scan_button = ttk.Button(toolbar, text='검사 / 미리보기', command=self.scan)
        self.scan_button.pack(side='left')
        self.apply_button = ttk.Button(toolbar, text='선택한 항목 결합', style='Accent.TButton', command=self.apply,
                                       state='disabled')
        self.apply_button.pack(side='left', padx=8)
        self.cancel_button = ttk.Button(toolbar, text='취소', command=self.cancel_event.set, state='disabled')
        self.cancel_button.pack(side='left')
        ttk.Label(toolbar, textvariable=self.selection_label).pack(side='right')
        self.clear_button = ttk.Button(toolbar, text='선택 해제', command=lambda: self.select_visible(False))
        self.clear_button.pack(side='right', padx=(5, 10))
        self.all_button = ttk.Button(toolbar, text='표시된 항목 선택', command=lambda: self.select_visible(True))
        self.all_button.pack(side='right')

        info = ttk.Frame(outer)
        info.pack(fill='x', pady=(0, 6))
        ttk.Label(info, textvariable=self.summary).pack(side='left')
        ttk.Combobox(info, textvariable=self.filter, values=list(FILTERS), state='readonly', width=14).pack(side='right')
        self.filter.trace_add('write', lambda *args: self.render_table())

        bottom = ttk.Frame(outer)
        bottom.pack(side='bottom', fill='x')
        self.progress = ttk.Progressbar(bottom)
        self.progress.pack(fill='x', pady=(10, 5))
        footer = ttk.Frame(bottom)
        footer.pack(fill='x')
        ttk.Label(footer, textvariable=self.status, style='Hint.TLabel', wraplength=870).pack(side='left', fill='x', expand=True)
        self.report_button = ttk.Button(footer, text='보고서 폴더 열기', command=self.open_report, state='disabled')
        self.report_button.pack(side='right', padx=(10, 0))

        panes = ttk.Panedwindow(outer, orient='horizontal')
        panes.pack(fill='both', expand=True)
        table_frame = ttk.Frame(panes)
        panes.add(table_frame, weight=3)
        columns = ('check', 'file', 'state', 'keep', 'tags')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='browse')
        for name, title, width in [('check', '선택', 45), ('file', '상대 경로', 265), ('state', '상태', 105),
                                   ('keep', '보호 태그', 75), ('tags', '일반 태그', 75)]:
            self.table.heading(name, text=title)
            self.table.column(name, width=width, minwidth=width if name != 'file' else 140,
                              stretch=name == 'file', anchor='w' if name == 'file' else 'center')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.table.pack(side='left', fill='both', expand=True)
        self.table.tag_configure('error', foreground='#a63732')
        self.table.tag_configure('written', foreground='#16724d')
        self.table.bind('<<TreeviewSelect>>', self.show_details)
        self.table.bind('<Button-1>', self.toggle_click)
        self.table.bind('<space>', self.toggle_focus)

        details = ttk.Notebook(panes)
        panes.add(details, weight=2)
        self.details = {}
        for name in ('변경 내용', '원본 JSON', '결합 결과'):
            text = scrolledtext.ScrolledText(details, wrap='word', font=('맑은 고딕', 10), width=36,
                                             padx=10, pady=10, borderwidth=0, background='white', state='disabled')
            details.add(text, text=name)
            self.details[name] = text
        self._set_text('변경 내용', '검사 후 목록에서 파일을 선택하면 변경될 태그와 상세 내용을 확인할 수 있습니다.')

    def choose_folder(self, variable):
        path = filedialog.askdirectory(parent=self.root, initialdir=variable.get() or None)
        if path:
            variable.set(path)
            if variable is self.json_dir and not self.txt_dir.get():
                self.txt_dir.set(path)

    def _sync_output(self):
        enabled = not self.busy and self.output_mode.get() == 'folder'
        for widget in (self.output_entry, self.output_button):
            widget.configure(state='normal' if enabled else 'disabled')

    def invalidate(self, *args):
        self.fresh = False
        self.apply_button.configure(state='disabled')
        if self.plan:
            self.status.set('설정이 변경되었습니다. 다시 검사한 뒤 결합해 주세요.')
        self._sync_output()

    def options(self):
        if not self.json_dir.get().strip() or not self.txt_dir.get().strip():
            raise ValueError('JSON 폴더와 태그 TXT 폴더를 모두 선택해 주세요.')
        if self.output_mode.get() == 'folder' and not self.output_dir.get().strip():
            raise ValueError('출력 폴더를 선택해 주세요.')
        return MergeOptions(
            json_dir=Path(self.json_dir.get().strip()).resolve(), txt_dir=Path(self.txt_dir.get().strip()).resolve(),
            recursive=self.recursive.get(), separator=self.separator.get(), common_keep=self.common_keep.get(),
            policy=POLICIES[self.policy.get()], encoding=ENCODINGS[self.encoding.get()], multiline=self.multiline.get(),
            output_dir=Path(self.output_dir.get().strip()).resolve() if self.output_mode.get() == 'folder' else None,
        )

    def set_busy(self, busy):
        self.busy = busy
        for widget, state in self.inputs:
            widget.configure(state='disabled' if busy else state)
        for widget in (self.scan_button, self.all_button, self.clear_button):
            widget.configure(state='disabled' if busy else 'normal')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self.apply_button.configure(state='normal' if not busy and self.fresh and self.selected else 'disabled')
        self._sync_output()

    def worker(self, kind, function):
        def run():
            try:
                self.events.put((kind, function()))
            except ScanCancelled:
                self.events.put(('cancelled', None))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        self.cancel_event.clear()
        self.set_busy(True)
        self.progress.configure(value=0, mode='indeterminate')
        self.progress.start(12)
        threading.Thread(target=run, daemon=False).start()

    def progress_callback(self):
        last = [0.0]
        def callback(done, total, value):
            now = time.monotonic()
            if done == total or now - last[0] > .12:
                self.events.put(('progress', (done, total, value)))
                last[0] = now
        return callback

    def scan(self):
        try:
            options = self.options()
            options.validate()
        except (ValueError, OSError) as exc:
            messagebox.showerror('입력 확인', str(exc), parent=self.root)
            return
        self.fresh = False
        self.status.set('폴더를 검사하고 JSON / TXT를 매칭하는 중입니다…')
        self.worker('plan', lambda: build_plan(options, self.progress_callback(), self.cancel_event.is_set))

    def apply(self):
        if not self.plan or not self.fresh or not self.selected:
            return
        if self.options() != self.plan.options:
            self.invalidate()
            return
        selected = set(self.selected)
        self.status.set(f'{len(selected):,}개 파일을 결합하는 중입니다. 변경 전 원본은 자동 백업합니다.')
        self.worker('result', lambda: execute_plan(self.plan, selected, self.progress_callback(), self.cancel_event.is_set))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    done, total, detail = value
                    self.progress.stop()
                    self.progress.configure(mode='determinate', maximum=max(total, 1), value=done)
                    relative = detail['relative'] if isinstance(detail, dict) else detail
                    self.status.set(f'{done:,} / {total:,}  {relative}')
                    continue
                self.progress.stop()
                if kind == 'plan':
                    self.plan, self.results = value, {}
                    self.selected = {item.key for item in value.items if item.status == 'ready'}
                    self.fresh = True
                    self.render_table()
                    self.status.set('검사 완료. 선택된 항목의 태그를 확인하고 결합 버튼을 눌러 주세요.')
                elif kind == 'result':
                    self.run_dir = value.run_dir
                    self.report_button.configure(state='normal')
                    self.results = {row['relative']: row for row in value.rows}
                    self.selected.clear()
                    self.fresh = False
                    self.render_table()
                    counts = value.counts
                    self.status.set(f"완료: 저장 {counts['written']:,}개 / 실패 {counts['error']:,}개 / "
                                    f"변경 감지 {counts['stale']:,}개 / 취소 {counts['cancelled']:,}개. 보고서에서 결과를 확인하세요.")
                elif kind == 'cancelled':
                    self.fresh = False
                    self.status.set('검사를 취소했습니다. 파일은 변경되지 않았습니다.')
                else:
                    self.fresh = False
                    self.status.set(f'작업 오류: {value}')
                    if not self.close_pending:
                        messagebox.showerror('작업 오류', value, parent=self.root)
                self.set_busy(False)
                self.update_selection()
                if self.close_pending:
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def _status(self, item):
        return self.results.get(item.relative, {}).get('status', item.status)

    def render_table(self):
        self.table.delete(*self.table.get_children())
        self.visible_items.clear()
        if not self.plan:
            return
        allowed = FILTERS[self.filter.get()]
        for item in self.plan.items:
            state = self._status(item)
            if allowed is not None and state not in allowed:
                continue
            before, after = item.original or {}, item.merged or {}
            counts = [f"{len(before.get(key, []))} → {len(after.get(key, []))}" if item.merged is not None else '—'
                      for key in ('keep_tags', 'tags')]
            check = '☑' if item.key in self.selected else ('☐' if state == 'ready' else '')
            tags = ('written',) if state == 'written' else (() if state in ('ready', 'unchanged', 'not_selected') else ('error',))
            iid = self.table.insert('', 'end', values=(check, item.relative, STATUS_LABELS[state], *counts), tags=tags)
            self.visible_items[iid] = item
        counts = Counter(self._status(item) for item in self.plan.items)
        self.summary.set(f"전체 {len(self.plan.items):,}개 · 변경 예정 {counts['ready']:,} · 변경 없음 {counts['unchanged']:,} "
                         f"· 저장 완료 {counts['written']:,}")
        first = next(iter(self.visible_items), None)
        if first:
            self.table.selection_set(first)
            self.table.focus(first)
        else:
            self._set_text('변경 내용', '현재 필터에 해당하는 항목이 없습니다.')
            self._set_text('원본 JSON', '')
            self._set_text('결합 결과', '')
        self.update_selection()

    def update_selection(self):
        self.selection_label.set(f'선택 {len(self.selected):,}개')
        self.apply_button.configure(state='normal' if self.selected and self.fresh and not self.busy else 'disabled')

    def select_visible(self, selected):
        if self.busy:
            return
        for item in self.visible_items.values():
            if self._status(item) == 'ready':
                (self.selected.add if selected else self.selected.discard)(item.key)
        self._refresh_checks()

    def _refresh_checks(self):
        for iid, item in self.visible_items.items():
            state = self._status(item)
            self.table.set(iid, 'check', '☑' if item.key in self.selected else ('☐' if state == 'ready' else ''))
        self.update_selection()

    def toggle_click(self, event):
        if self.table.identify_column(event.x) == '#1':
            self.toggle(self.table.identify_row(event.y))

    def toggle_focus(self, event):
        self.toggle(self.table.focus())
        return 'break'

    def toggle(self, iid):
        item = self.visible_items.get(iid)
        if self.busy or item is None or self._status(item) != 'ready':
            return
        (self.selected.discard if item.key in self.selected else self.selected.add)(item.key)
        self._refresh_checks()

    def _set_text(self, name, content):
        widget = self.details[name]
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', content)
        widget.configure(state='disabled')

    def show_details(self, event=None):
        selected = self.table.selection()
        if not selected or selected[0] not in self.visible_items:
            return
        item = self.visible_items[selected[0]]
        row = self.results.get(item.relative, {})
        lines = [item.relative, f"상태: {STATUS_LABELS[self._status(item)]}", row.get('message', item.message), '']
        if item.json_path:
            lines.extend([f'JSON: {item.json_path}', f'TXT: {item.txt_path or "없음"}', ''])
        if item.merged is not None:
            for key in ('keep_tags', 'tags'):
                lines.extend([key, f"기존: {', '.join(item.original.get(key, [])) or '(없음)'}",
                              f"결과: {', '.join(item.merged[key]) or '(없음)'}", ''])
            lines.append('자연어 캡션과 기타 JSON 필드는 보존됩니다.')
        if row.get('backup'):
            lines.extend(['', f"백업: {row['backup']}"])
        self._set_text('변경 내용', '\n'.join(lines))
        self._set_text('원본 JSON', json.dumps(item.original, ensure_ascii=False, indent=2) if item.original is not None else '읽을 수 있는 원본 JSON이 없습니다.')
        self._set_text('결합 결과', json.dumps(item.merged, ensure_ascii=False, indent=2) if item.merged is not None else '결합할 수 없는 항목입니다. 변경 내용의 이유를 확인해 주세요.')

    def open_report(self):
        if self.run_dir and self.run_dir.is_dir():
            os.startfile(self.run_dir)

    def close(self):
        if self.busy:
            self.close_pending = True
            self.cancel_event.set()
            self.status.set('현재 파일 처리를 마친 뒤 종료합니다…')
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='MOSS JSON + 태그 TXT 일괄 결합 GUI')
    parser.add_argument('--json-dir', type=Path)
    parser.add_argument('--txt-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if os.name == 'nt':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    MergerApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
