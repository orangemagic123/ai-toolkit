# MOSS JSON captions

Use one file per image (`001.png` and `001.json`):

```json
{
  "caption": {
    "short": "A woman standing outdoors.",
    "medium": "A black-haired woman in a white shirt stands in a garden.",
    "long": "A black-haired woman in a white shirt stands in a garden, facing the viewer. Soft daylight illuminates her face, with green foliage in the background."
  },
  "tags": ["1girl", "black hair", "white shirt", "outdoors", "garden"],
  "keep_tags": ["mychar"],
  "watermark": [],
  "borders": ["none"]
}
```

In the web UI started with `npm run build_and_start`, open **New Training Job → Datasets → Caption Format → MOSS JSON**. Select **Natural Language**, **Tags Only**, or **Mixed** under **Caption Mode**, then set the **MOSS Caption Lengths** and, for Mixed, **Caption Mix Weights**. The displayed percentages show the relative probabilities. Selecting MOSS JSON sets Caption Extension to `json`. Settings are preserved when saving, editing, or cloning a job. **Dataset Trigger Word** overrides the job trigger for that dataset; leave it empty to inherit the job trigger.

Equivalent dataset YAML:

```yaml
datasets:
  - folder_path: C:/dataset/images
    caption_format: moss_json
    caption_ext: json
    caption_mode: mixed
    mixed_weights:
      tags: 40
      nl: 30
      tags_nl: 20
      nl_tags: 10
    caption_length_weights:
      short: 10
      medium: 20
      long: 70
    keep_tokens: 0
    caption_dropout_rate: 0.05
    protected_caption_dropout_rate: 0.08
    token_dropout_rate: 0.1
    shuffle_caption: true
    cache_text_embeddings: false
```

Weights are relative, need not total 100, must be finite and nonnegative, and must include at least one positive value. An omitted weight within an explicitly provided mapping is zero. If `caption_length_weights` is omitted, all three lengths have equal probability. A zero weight disables that choice. The length draw is independent of the mixed-variant draw and only occurs for variants containing natural language. In the example, 60% of samples include natural language; among those, short/medium/long have probabilities 10%/20%/70%.

`caption_mode: single` selects only natural language using the configured length weights. To train only tags, use `caption_mode: mixed` and `mixed_weights: {tags: 1}`. Missing or empty `tags` automatically falls back to natural language. All three caption strings are required and must be nonempty; the original MOSS JSON works without adding tags.

`keep_tags` and `tags` are arrays of strings. Protected tags stay at the front in their written order for tags-only, natural-language-only, tags-then-natural-language, and natural-language-then-tags. Duplicate tags are removed while preserving order; a tag listed in both arrays is kept only in `keep_tags`. General tag shuffle/dropout never changes the natural-language sentence or `keep_tags`. Newlines and visible text within the natural-language field are retained. `keep_tokens` can additionally protect the first general tags from tag dropout; `keep_tokens_separator` is not needed in MOSS JSON. A configured trigger already present in keep_tags is not injected a second time.

Whole-caption dropout removes the caption and retains the configured dataset trigger for non-regularization images, consistently with or without text embedding caching. Without a trigger, or for regularization images, it produces an empty caption. Protected-caption dropout keeps `keep_tags` plus any general tags protected by `keep_tokens`, and removes natural language. Watermark/border metadata and other annotation fields are never appended to the training prompt.

This is a LoRA caption-loading policy, not a reproduction of every MOSS pretraining condition. The [official captioning guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#image-captioning) places watermark descriptions outside the three caption fields, while the [technical report](https://openmoss.ai/MOSS-Image/paper/MOSS_Image_Technical_Report.pdf) describes artifact annotations and combined caption/structure/metadata conditioning during pretraining. This loader does not remove watermark descriptions already embedded in a caption, does not remove watermark-related tags, and does not mask watermark pixels in an image.

Text embedding caching supports all enabled MOSS JSON variants and lengths (up to ten normal captions per image), plus configured dropout and preservation variants. With caching enabled, set `shuffle_caption: false`, `token_dropout_rate: 0`, and `random_triggers: []`; these random tag augmentations require live encoding. Caption length and mixed-variant selection still change on each sample load when caching is enabled.

The legacy `train.short_and_long_captions` option adds a second training row using `keep_tags` plus `caption.short`; the primary row still uses the configured MOSS mode and length weights. With SDXL, `short_and_long_captions_encoder_split` uses that short caption for the first text encoder. Both legacy options require text embedding caching to be disabled; incompatible combinations are rejected before model loading and shown as errors in the web UI. For random short/medium/long sampling with caching, leave both legacy options off and use `caption_length_weights`.

Existing TXT captions and raw JSON prompts keep their behavior under the default `caption_format: text`. A `.json` extension alone continues to pass the entire file as a prompt. This explicit format switch avoids changing structured prompts used by other models.

An aggregate metadata JSON is also supported with `dataset_path: C:/dataset/captions.json` and `caption_format: moss_json`. Map each absolute image path to the same object shown above.
