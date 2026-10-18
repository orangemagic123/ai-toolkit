import type { DatasetConfig, TrainConfig } from '@/types';

export const mixedCaptionDefaults = { tags: 40, nl: 30, tags_nl: 20, nl_tags: 10 };
export const captionLengthDefaults = { short: 1, medium: 1, long: 1 };

export function captionLengthWeights(dataset: DatasetConfig) {
  const weights = dataset.caption_length_weights;
  return weights
    ? { short: weights.short ?? 0, medium: weights.medium ?? 0, long: weights.long ?? 0 }
    : { ...captionLengthDefaults };
}

export function mixedCaptionWeights(dataset: DatasetConfig) {
  const weights = dataset.mixed_weights;
  return weights
    ? { tags: weights.tags ?? 0, nl: weights.nl ?? 0, tags_nl: weights.tags_nl ?? 0, nl_tags: weights.nl_tags ?? 0 }
    : { ...mixedCaptionDefaults };
}

export function setCaptionFormat(dataset: DatasetConfig, format: 'text' | 'moss_json'): DatasetConfig {
  return {
    ...dataset,
    caption_format: format,
    caption_ext: format === 'moss_json' ? 'json' : dataset.caption_ext,
    caption_length_weights: captionLengthWeights(dataset),
    mixed_weights: mixedCaptionWeights(dataset),
  };
}

export function captionContentMode(dataset: DatasetConfig): 'single' | 'mixed' | 'tags_only' {
  if (dataset.caption_mode !== 'mixed') return 'single';
  const weights = mixedCaptionWeights(dataset);
  if (
    dataset.caption_format === 'moss_json' &&
    weights.tags > 0 &&
    weights.nl === 0 &&
    weights.tags_nl === 0 &&
    weights.nl_tags === 0
  )
    return 'tags_only';
  return 'mixed';
}

export function setCaptionContentMode(dataset: DatasetConfig, mode: 'single' | 'mixed' | 'tags_only'): DatasetConfig {
  return {
    ...dataset,
    caption_mode: mode === 'tags_only' ? 'mixed' : mode,
    mixed_weights:
      mode === 'tags_only'
        ? { tags: 1, nl: 0, tags_nl: 0, nl_tags: 0 }
        : mode === 'mixed' && captionContentMode(dataset) === 'tags_only'
          ? { ...mixedCaptionDefaults }
          : mixedCaptionWeights(dataset),
  };
}

function validWeights(weights: Record<string, number>) {
  const values = Object.values(weights);
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.every(value => Number.isFinite(value) && value >= 0) && Number.isFinite(total) && total > 0;
}

export function captionSettingsErrors(dataset: DatasetConfig, train: TrainConfig): string[] {
  const errors: string[] = [];
  if (dataset.caption_mode === 'mixed' && !validWeights(mixedCaptionWeights(dataset))) {
    errors.push('Caption mix weights must be finite, nonnegative, and include at least one positive weight.');
  }
  if (dataset.caption_format !== 'moss_json') return errors;
  if (!validWeights(captionLengthWeights(dataset))) {
    errors.push('Caption length weights must be finite, nonnegative, and include at least one positive weight.');
  }
  const cached = train.cache_text_embeddings || dataset.cache_text_embeddings;
  if (
    cached &&
    ((dataset.shuffle_caption ?? dataset.shuffle_tokens ?? false) ||
      (dataset.token_dropout_rate ?? 0) > 0 ||
      (dataset.random_triggers?.length ?? 0) > 0)
  ) {
    errors.push('Turn off Cache Text Embeddings to shuffle or drop tags or use random triggers.');
  }
  if (cached && (train.short_and_long_captions || train.short_and_long_captions_encoder_split)) {
    errors.push(
      'Short/long dual-caption training requires Cache Text Embeddings off. Alternatively, disable the legacy short/long options and use the MOSS length weights below.',
    );
  }
  return errors;
}
