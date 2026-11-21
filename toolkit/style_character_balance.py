"""Style/character balanced sampling.

When most images of a character come from one style, the character tag also learns that
style: the model can explain the style with either tag. This reweights how often each image
is drawn so that, for every concept tag (character, costume, ...), the mix of styles it is
trained with matches the global style mix, restricted to the styles that concept actually has.
Each concept keeps its total exposure. Global style shares shift as a side effect unless
``keep_style_share`` is on; keeping them conflicts with decorrelation when a style is made up
mostly of one concept (that style's images are that concept's images).

Tags come from MOSS JSON ``keep_tags``: the tag starting with ``style_prefix`` is the style,
count tags (1girl, 2boys, ...) and ``ignore_tags`` are skipped, every other keep tag is a concept.
Concepts drawn in only one style cannot be separated by reweighting and are reported as such.

Config (train section)::

    style_character_balance:
      enabled: true
      ignore_tags: ['very aesthetic', 'location', 'chibi', 'sketch']
      max_multiplier: 3.0          # cap on how much more often an image is drawn
      keep_style_share: false      # true: hold each style's overall share fixed
      style_multipliers: {}        # e.g. {'@dmftk': 1.5}: set overall style shares (implies holding them)
"""
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_COUNT_PATTERN = r'(\d+\+?(boy|girl|other)s?|no humans|multiple (boys|girls|others))'
NO_STYLE = '(no style)'


@dataclass
class StyleCharacterBalanceConfig:
    enabled: bool = True
    style_prefix: str = '@'
    ignore_tags: List[str] = field(default_factory=list)
    count_tag_pattern: str = DEFAULT_COUNT_PATTERN
    max_multiplier: float = 3.0
    # defaults to 1 / max_multiplier
    min_multiplier: Optional[float] = None
    keep_style_share: bool = False
    style_multipliers: Dict[str, float] = field(default_factory=dict)
    iterations: int = 50
    # how many concepts to list in the startup report
    report_top: int = 12

    def __post_init__(self):
        if not isinstance(self.style_prefix, str) or not self.style_prefix:
            raise ValueError('style_character_balance.style_prefix must be a non-empty string')
        if not isinstance(self.ignore_tags, (list, tuple)) or not all(isinstance(t, str) for t in self.ignore_tags):
            raise ValueError('style_character_balance.ignore_tags must be a list of strings')
        self.max_multiplier = float(self.max_multiplier)
        if self.max_multiplier < 1:
            raise ValueError('style_character_balance.max_multiplier must be >= 1')
        if self.min_multiplier is None:
            self.min_multiplier = 1.0 / self.max_multiplier
        self.min_multiplier = float(self.min_multiplier)
        if not 0 < self.min_multiplier <= 1:
            raise ValueError('style_character_balance.min_multiplier must be in (0, 1]')
        if not isinstance(self.style_multipliers, dict) or any(
                not isinstance(v, (int, float)) or v <= 0 for v in self.style_multipliers.values()):
            raise ValueError('style_character_balance.style_multipliers values must be positive numbers')
        if int(self.iterations) < 1:
            raise ValueError('style_character_balance.iterations must be >= 1')
        self.iterations = int(self.iterations)
        self.count_re = re.compile(self.count_tag_pattern, re.IGNORECASE)
        self.ignore_set = {t.strip().casefold() for t in self.ignore_tags}

    @classmethod
    def from_value(cls, value) -> Optional['StyleCharacterBalanceConfig']:
        """None/False -> disabled, True -> defaults, dict -> options. Returns None when disabled."""
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            cfg = value
        elif value is True:
            cfg = cls()
        elif isinstance(value, dict):
            known = set(cls.__dataclass_fields__)
            unknown = sorted(set(value) - known)
            if unknown:
                raise ValueError(f'style_character_balance: unknown option(s) {unknown}; known: {sorted(known)}')
            cfg = cls(**value)
        else:
            raise ValueError('style_character_balance must be true/false or a mapping of options')
        return cfg if cfg.enabled else None


@dataclass(frozen=True)
class BalanceRecord:
    style: str
    concepts: Tuple[str, ...]


def classify_keep_tags(keep_tags: Iterable[str], cfg: StyleCharacterBalanceConfig) -> BalanceRecord:
    styles, concepts = [], []
    for tag in keep_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag.startswith(cfg.style_prefix):
            styles.append(tag)
        elif cfg.count_re.fullmatch(tag) or tag.casefold() in cfg.ignore_set:
            continue
        else:
            concepts.append(tag)
    style = ' + '.join(styles) if styles else NO_STYLE
    return BalanceRecord(style, tuple(dict.fromkeys(concepts)))


@dataclass
class BalanceResult:
    multipliers: Dict[str, float]
    report: List[str]


def compute_multipliers(
        exposure: Dict[str, float],
        records: Dict[str, BalanceRecord],
        cfg: StyleCharacterBalanceConfig,
) -> BalanceResult:
    """Per-path sampling multipliers.

    exposure: how many sampling slots each path already has (repeats, resolutions, flips).
    Alternates two scalings until they agree (iterative proportional fitting):
      1. for every concept seen in 2+ styles, scale its images so its style mix matches the
         global target mix over its styles while its total exposure stays the same;
      2. scale each style so the global style shares match the target.
    Multipliers are clipped to [min_multiplier, max_multiplier] after every pass.
    """
    paths = [p for p in exposure if exposure[p] > 0]
    e = {p: float(exposure[p]) for p in paths}
    total = sum(e.values())
    style_of = {p: records[p].style for p in paths}

    base_style = Counter()
    for p in paths:
        base_style[style_of[p]] += e[p]
    unknown_styles = sorted(set(cfg.style_multipliers) - set(base_style))
    raw_target = {s: base_style[s] / total * cfg.style_multipliers.get(s, 1.0) for s in base_style}
    z = sum(raw_target.values())
    target = {s: v / z for s, v in raw_target.items()}

    cells: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for c in records[p].concepts:
            cells[c][style_of[p]].append(p)
    balanced = {c for c, by_style in cells.items() if len(by_style) > 1}
    single = {c: next(iter(by_style)) for c, by_style in cells.items() if len(by_style) == 1}
    concept_exposure = {c: sum(e[p] for ps in cells[c].values() for p in ps) for c in cells}

    lo, hi = cfg.min_multiplier, cfg.max_multiplier
    rescale_styles = cfg.keep_style_share or bool(cfg.style_multipliers)
    m = {p: 1.0 for p in paths}
    for _ in range(cfg.iterations):
        log_sum, log_n = defaultdict(float), Counter()
        for c in balanced:
            support = sum(target[s] for s in cells[c])
            for s, ps in cells[c].items():
                current = sum(e[p] * m[p] for p in ps)
                want = concept_exposure[c] * target[s] / support
                f = math.log(want / current)
                for p in ps:
                    log_sum[p] += f
                    log_n[p] += 1
        for p, n in log_n.items():
            m[p] *= math.exp(log_sum[p] / n)
        if rescale_styles:
            cur = Counter()
            for p in paths:
                cur[style_of[p]] += e[p] * m[p]
            cur_total = sum(cur.values())
            for p in paths:
                m[p] *= target[style_of[p]] * cur_total / cur[style_of[p]]
        # keep the mean multiplier at 1 so the clip bounds mean what they say
        scale = total / sum(e[p] * m[p] for p in paths)
        for p in paths:
            m[p] = min(hi, max(lo, m[p] * scale))

    return BalanceResult(m, _report(paths, e, m, style_of, base_style, cells, balanced, single,
                                    concept_exposure, unknown_styles, cfg))


def _pct(x):
    return f'{x * 100:4.0f}%'


def _report(paths, e, m, style_of, base_style, cells, balanced, single, concept_exposure, unknown_styles, cfg):
    total = sum(e.values())
    after_total = sum(e[p] * m[p] for p in paths)
    after_style = Counter()
    for p in paths:
        after_style[style_of[p]] += e[p] * m[p]
    lines = [
        f'Style/character balance: {len(paths)} images, {len(base_style)} styles, '
        f'{len(cells)} concept tags ({len(balanced)} balanced across styles)',
        '  style share (before -> after):',
    ]
    for s in sorted(base_style, key=lambda s: -base_style[s]):
        lines.append(f'    {s:<16} {_pct(base_style[s] / total)} -> {_pct(after_style[s] / after_total)}')
    lines.append('  concept style mix (before -> after):')
    for c in sorted(balanced, key=lambda c: -concept_exposure[c])[:cfg.report_top]:
        before = {s: sum(e[p] for p in ps) for s, ps in cells[c].items()}
        after = {s: sum(e[p] * m[p] for p in ps) for s, ps in cells[c].items()}
        bt, at = sum(before.values()), sum(after.values())
        n_images = len({p for ps in cells[c].values() for p in ps})
        mix = ', '.join(f'{s} {before[s] / bt:.0%}->{after[s] / at:.0%}'
                        for s in sorted(before, key=lambda s: -before[s]))
        lines.append(f'    {c} ({n_images}): {mix}')
    if len(balanced) > cfg.report_top:
        lines.append(f'    ... and {len(balanced) - cfg.report_top} more')
    if single:
        shown = ', '.join(f'{c} ({s})' for c, s in sorted(single.items(), key=lambda kv: -concept_exposure[kv[0]]))
        lines.append(f'  single-style concepts (cannot be separated by reweighting): {shown}')
    values = [m[p] for p in paths]
    at_hi = sum(1 for v in values if v >= cfg.max_multiplier - 1e-9)
    at_lo = sum(1 for v in values if v <= cfg.min_multiplier + 1e-9)
    lines.append(f'  per-image multiplier: min {min(values):.2f}x, max {max(values):.2f}x '
                 f'({at_hi} at the {cfg.max_multiplier:g}x cap, {at_lo} at the {cfg.min_multiplier:.2f}x floor)')
    # draws relative to an image with the smallest base exposure (e.g. num_repeats 1), unweighted
    unit = min(e.values())
    top = sorted(paths, key=lambda p: -e[p] * m[p])[:3]
    lines.append('  most drawn images (x an unweighted repeat-1 image): ' + ', '.join(
        f'{os.path.basename(p)} {e[p] * m[p] / unit:.1f}x' for p in top))
    if unknown_styles:
        lines.append(f'  WARNING: style_multipliers names styles not in the dataset: {unknown_styles}')
    return lines


class BalancedSampler:
    """Yields one epoch of dataset indices drawn in proportion to the weights.

    Residual sampling: each index appears floor(expected) times, and the remaining slots go to
    indices picked without replacement by their fractional part, so per-epoch counts stay as
    close to the expected values as possible. The epoch length equals the dataset length.
    """

    def __init__(self, weights: Sequence[float], seed: Optional[int] = None):
        if not weights or any(w <= 0 for w in weights):
            raise ValueError('BalancedSampler needs positive weights')
        self.weights = [float(w) for w in weights]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        n = len(self.weights)
        total = sum(self.weights)
        expected = [w * n / total for w in self.weights]
        counts = [int(x) for x in expected]
        remaining = n - sum(counts)
        if remaining > 0:
            frac = [x - c for x, c in zip(expected, counts)]
            # weighted sampling without replacement (Efraimidis-Spirakis keys)
            keys = sorted(range(n), reverse=True,
                          key=lambda i: self.rng.random() ** (1.0 / frac[i]) if frac[i] > 1e-12 else -1.0)
            for i in keys[:remaining]:
                counts[i] += 1
        order = [i for i, c in enumerate(counts) for _ in range(c)]
        self.rng.shuffle(order)
        return iter(order)


def build_balanced_sampler(
        datasets: Sequence,
        dataset_length: int,
        value,
        log: Callable[[str], None] = print,
) -> Optional[BalancedSampler]:
    """Sampler over a ConcatDataset of AiToolkitDatasets, or None when the option is off.

    A sampling unit is one dataset index: a bucket batch when buckets are on, else one file item.
    """
    cfg = StyleCharacterBalanceConfig.from_value(value)
    if cfg is None:
        return None
    units = []
    for ds in datasets:
        if ds.dataset_config.buckets:
            units.extend([ds.file_list[i] for i in batch] for batch in ds.batch_indices)
        else:
            units.extend([item] for item in ds.file_list)
    if len(units) != dataset_length:
        raise RuntimeError(f'style_character_balance: found {len(units)} sampling units '
                           f'but the dataset has {dataset_length}')
    records, exposure = {}, Counter()
    for unit in units:
        for item in unit:
            if item.path not in records:
                source = getattr(item, 'structured_caption_source', None)
                if source is None:
                    raise ValueError(
                        f'style_character_balance needs caption_format: moss_json (keep_tags); '
                        f'no structured caption for {item.path}')
                records[item.path] = classify_keep_tags(source.keep_tags, cfg)
            exposure[item.path] += 1
    result = compute_multipliers(exposure, records, cfg)
    for line in result.report:
        log(line)
    weights = [sum(result.multipliers[item.path] for item in unit) / len(unit) for unit in units]
    return BalancedSampler(weights)
