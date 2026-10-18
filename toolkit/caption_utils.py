import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


MIXED_CAPTION_VARIANTS = ('tags', 'nl', 'tags_nl', 'nl_tags')
CAPTION_LENGTHS = ('short', 'medium', 'long')


def _join_with_keep_separator(protected: str, processable: str, separator: str) -> str:
    """Keep a protected prefix marked until later caption processing."""
    return f'{protected.strip()} {separator} {processable.strip()}'.strip()


def _join_caption_parts(*parts: str) -> str:
    """Join ordered caption sections while normalizing their comma boundaries."""
    caption = ''
    for part in parts:
        caption = join_caption_sections(caption, part)
    return caption


@dataclass(frozen=True)
class MixedCaptionSelection:
    """A mixed-caption choice whose tag and natural-language sections stay distinct."""

    variant: str
    protected_tags: str
    processable_tags: str
    nl_caption: str
    keep_tokens_separator: Optional[str] = None
    is_structured: bool = False

    @property
    def includes_tags(self) -> bool:
        return self.variant in ('tags', 'tags_nl', 'nl_tags')

    @property
    def includes_nl(self) -> bool:
        return self.variant in ('nl', 'tags_nl', 'nl_tags')

    def render(self) -> str:
        """Render the selected raw caption, retaining its keep-token boundary."""
        tags = self.processable_tags if self.includes_tags else ''
        nl_caption = self.nl_caption if self.includes_nl else ''
        if self.variant == 'tags':
            processable = tags
        elif self.variant == 'nl':
            processable = nl_caption
        elif self.variant == 'tags_nl':
            processable = _join_caption_parts(tags, nl_caption)
        else:
            processable = _join_caption_parts(nl_caption, tags)

        if self.keep_tokens_separator is not None:
            return _join_with_keep_separator(
                self.protected_tags,
                processable,
                self.keep_tokens_separator,
            )
        return _join_caption_parts(self.protected_tags, processable)

    def render_processed(
        self,
        protected_tags: str,
        processable_tags: str,
        nl_caption: str,
        extra_tags: str = '',
    ) -> str:
        """Render processed tags around an untouched natural-language section."""
        if self.variant == 'tags':
            parts = (protected_tags, processable_tags, extra_tags)
        elif self.variant == 'nl':
            parts = (protected_tags, nl_caption, extra_tags)
        elif self.variant == 'tags_nl':
            parts = (protected_tags, processable_tags, extra_tags, nl_caption)
        else:
            parts = (protected_tags, nl_caption, processable_tags, extra_tags)
        return _join_caption_parts(*parts)


@dataclass(frozen=True)
class StructuredCaption:
    """MOSS captions plus distinct ordered protected and processable tag lists."""

    keep_tags: Tuple[str, ...]
    tags: Tuple[str, ...]
    captions: Dict[str, str]

    def _variants(self, mode, mixed_weights):
        if mode != 'mixed' or not self.tags:
            return ['nl']
        return [name for name in MIXED_CAPTION_VARIANTS if mixed_weights[name] > 0]

    def _selection(self, variant, length=None):
        return MixedCaptionSelection(
            variant=variant,
            protected_tags=', '.join(self.keep_tags),
            processable_tags=', '.join(self.tags),
            nl_caption=self.captions[length] if length is not None else '',
            is_structured=True,
        )

    def short_selection(self):
        """The explicit short caption for legacy dual-caption training."""
        return self._selection('nl', 'short')

    def select(self, mode, mixed_weights, length_weights):
        variants = self._variants(mode, mixed_weights)
        variant = variants[0] if len(variants) == 1 else random.choices(
            variants, weights=[mixed_weights[name] for name in variants], k=1,
        )[0]
        length = None
        if variant != 'tags':
            lengths = [name for name in CAPTION_LENGTHS if length_weights[name] > 0]
            length = lengths[0] if len(lengths) == 1 else random.choices(
                lengths, weights=[length_weights[name] for name in lengths], k=1,
            )[0]
        return self._selection(variant, length)

    def selections(self, mode, mixed_weights, length_weights):
        """Enumerate every enabled variant for text embedding precaching."""
        for variant in self._variants(mode, mixed_weights):
            if variant == 'tags':
                yield self._selection(variant)
            else:
                for length in CAPTION_LENGTHS:
                    if length_weights[length] > 0:
                        yield self._selection(variant, length)


def parse_structured_caption(value, source='caption JSON') -> StructuredCaption:
    """Parse annotation JSON explicitly; unrelated raw JSON prompts stay untouched."""
    if not isinstance(value, dict) or not isinstance(value.get('caption'), dict):
        raise ValueError(f'{source}: expected a caption object with short, medium, long')
    captions = {}
    for length in CAPTION_LENGTHS:
        text = value['caption'].get(length)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{source}: caption.{length} must be a non-empty string')
        # Preserve visible text and line breaks exactly as stored in the JSON.
        captions[length] = text

    def read_tags(key):
        tags = value.get(key, [])
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise ValueError(f'{source}: {key} must be an array of non-empty strings')
        return tuple(dict.fromkeys(tag.strip() for tag in tags))

    keep_tags = read_tags('keep_tags')
    tags = tuple(tag for tag in read_tags('tags') if tag not in keep_tags)
    return StructuredCaption(keep_tags, tags, captions)


def select_mixed_caption_selection(
    tags_caption: str,
    nl_caption: str,
    weights: Dict[str, float],
    keep_tokens_separator: Optional[str] = None,
) -> MixedCaptionSelection:
    """Select a weighted variant without flattening its tags and natural language."""
    tags_caption = tags_caption.strip()
    nl_caption = nl_caption.strip()
    protected, processable_tags, has_keep_separator = split_caption_at_separator(
        tags_caption, keep_tokens_separator
    )
    has_tags = bool(protected.strip() or processable_tags.strip())

    if not has_tags:
        selected = 'nl'
    elif not nl_caption:
        selected = 'tags'
    else:
        variant_names = list(MIXED_CAPTION_VARIANTS)
        selected = random.choices(
            variant_names,
            weights=[weights[name] for name in variant_names],
            k=1,
        )[0]

    return MixedCaptionSelection(
        variant=selected,
        protected_tags=protected.strip(),
        processable_tags=processable_tags.strip(),
        nl_caption=nl_caption,
        keep_tokens_separator=(
            keep_tokens_separator if has_keep_separator and has_tags else None
        ),
    )


def select_mixed_caption(
    tags_caption: str,
    nl_caption: str,
    weights: Dict[str, float],
    keep_tokens_separator: Optional[str] = None,
) -> str:
    """Select one weighted caption variant, falling back when one side is absent."""
    return select_mixed_caption_selection(
        tags_caption,
        nl_caption,
        weights,
        keep_tokens_separator,
    ).render()


def _get_caption_tag_groups(caption: str, secondary_separator: Optional[str] = None):
    """Return comma-separated units while retaining secondary tag groups."""
    groups = [group.strip() for group in caption.split(',') if group.strip()]
    if secondary_separator:
        normalized_groups = []
        for group in groups:
            tags = [tag.strip() for tag in group.split(secondary_separator) if tag.strip()]
            if tags:
                normalized_groups.append(secondary_separator.join(tags))
        groups = normalized_groups
    return groups


def shuffle_caption_tags(caption: str, secondary_separator: Optional[str] = None) -> str:
    """Shuffle comma-separated tags or secondary-separated tag groups."""
    groups = _get_caption_tag_groups(caption, secondary_separator)
    random.shuffle(groups)
    return ', '.join(groups)


def drop_caption_tags(
    caption: str,
    dropout_rate: float,
    keep_tokens: int = 0,
    secondary_separator: Optional[str] = None,
) -> str:
    """Randomly drop tags or secondary-separated tag groups."""
    groups = _get_caption_tag_groups(caption, secondary_separator)
    kept_groups = []
    for index, group in enumerate(groups):
        if index < keep_tokens:
            kept_groups.append(group)
        elif dropout_rate < 1.0 and random.random() > dropout_rate:
            kept_groups.append(group)
    return ', '.join(kept_groups)


def expand_secondary_separator(caption: str, secondary_separator: Optional[str]) -> str:
    """Expand grouped tags into a normal comma-separated training caption."""
    if not secondary_separator:
        return caption
    groups = _get_caption_tag_groups(caption, secondary_separator)
    tags = []
    for group in groups:
        tags.extend(tag.strip() for tag in group.split(secondary_separator) if tag.strip())
    return ', '.join(tags)


def split_caption_at_separator(caption: str, separator: Optional[str]) -> Tuple[str, str, bool]:
    """Split a caption into protected and processable sections."""
    if separator and separator in caption:
        protected, processable = caption.split(separator, 1)
        return protected, processable, True
    return '', caption, False


def join_caption_sections(protected: str, processable: str) -> str:
    """Join caption sections after removing their separator boundary."""
    protected = protected.strip().rstrip(',').rstrip()
    processable = processable.strip().lstrip(',').lstrip()
    return ', '.join(part for part in (protected, processable) if part)
