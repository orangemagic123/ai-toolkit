'use client';

import type { DatasetConfig, TrainConfig } from '@/types';
import { NumberInput, SelectInput, TextInput } from '@/components/formInputs';
import {
  captionContentMode,
  captionLengthWeights,
  captionSettingsErrors,
  mixedCaptionWeights,
  setCaptionContentMode,
  setCaptionFormat,
} from './captionConfig';

type Props = {
  dataset: DatasetConfig;
  train: TrainConfig;
  onChange: (dataset: DatasetConfig) => void;
};

const lengthNames = ['short', 'medium', 'long'] as const;
const mixLabels = {
  tags: 'Tags',
  nl: 'Natural Language',
  tags_nl: 'Tags + Natural Language',
  nl_tags: 'Natural Language + Tags',
};

function percent(weight: number, total: number) {
  return Number.isFinite(total) && total > 0 ? `${Number(((100 * weight) / total).toFixed(1))}%` : '—';
}

export default function CaptionSettings({ dataset, train, onChange }: Props) {
  const isMoss = dataset.caption_format === 'moss_json';
  const mode = captionContentMode(dataset);
  const lengths = captionLengthWeights(dataset);
  const mix = mixedCaptionWeights(dataset);
  const lengthTotal = Object.values(lengths).reduce((a, b) => a + b, 0);
  const mixTotal = Object.values(mix).reduce((a, b) => a + b, 0);
  const errors = captionSettingsErrors(dataset, train);

  return (
    <div className="space-y-3" aria-label="Caption settings">
      <SelectInput
        label="Caption Format"
        value={dataset.caption_format ?? 'text'}
        onChange={value => onChange(setCaptionFormat(dataset, value as 'text' | 'moss_json'))}
        options={[
          { value: 'text', label: 'Text / Raw JSON Prompt' },
          { value: 'moss_json', label: 'MOSS JSON' },
        ]}
        docKey="datasets.caption_format"
      />
      {isMoss ? (
        <div className="rounded border border-blue-500/40 bg-blue-500/10 p-3 text-sm">
          <h3 className="font-semibold">MOSS JSON captions</h3>
          <p className="mt-1">Place matching files together, for example image.png and image.json.</p>
          <p className="mt-1">Reads caption.short / medium / long, tags, and keep_tags. Protected tags stay first.</p>
        </div>
      ) : (
        <TextInput
          label="Default Caption"
          value={dataset.default_caption}
          onChange={value => onChange({ ...dataset, default_caption: value })}
          placeholder="eg. A photo of a cat"
        />
      )}
      <TextInput
        label="Dataset Trigger Word"
        value={dataset.trigger_word ?? ''}
        onChange={value => onChange({ ...dataset, trigger_word: value.trim() ? value : null })}
        placeholder="Uses the job trigger when empty"
      />
      <SelectInput
        label="Caption Mode"
        value={mode}
        onChange={value => onChange(setCaptionContentMode(dataset, value as typeof mode))}
        options={[
          { value: 'single', label: isMoss ? 'Natural Language' : 'Single Caption' },
          ...(isMoss ? [{ value: 'tags_only', label: 'Tags Only' }] : []),
          { value: 'mixed', label: 'Mixed (Tags / Natural Language)' },
        ]}
        docKey="datasets.caption_mode"
      />
      {isMoss && (
        <fieldset className="rounded border border-gray-500/30 p-3">
          <legend className="px-1 text-sm font-semibold">MOSS Caption Lengths</legend>
          <div className="grid grid-cols-3 gap-2">
            {lengthNames.map(length => (
              <NumberInput
                key={length}
                label={`${length[0].toUpperCase() + length.slice(1)} Weight`}
                value={lengths[length]}
                min={0}
                required
                docKey="datasets.caption_length_weights"
                onChange={value =>
                  onChange({ ...dataset, caption_length_weights: { ...lengths, [length]: value ?? 0 } })
                }
              />
            ))}
          </div>
          <p className="mt-2 text-xs text-gray-400">
            {lengthNames.map(length => `${length}: ${percent(lengths[length], lengthTotal)}`).join(' · ')}
          </p>
          <p className="mt-1 text-xs text-gray-400">
            Relative weights when natural language is used. Set a weight to 0 to exclude that length.
          </p>
          {mode === 'tags_only' && (
            <p className="mt-1 text-xs text-gray-400">
              Images without general tags fall back to natural language using these weights.
            </p>
          )}
        </fieldset>
      )}
      {mode === 'mixed' && (
        <fieldset className="rounded border border-gray-500/30 p-3">
          <legend className="px-1 text-sm font-semibold">Caption Mix Weights</legend>
          <div className="grid grid-cols-2 gap-2">
            {(Object.keys(mixLabels) as (keyof typeof mixLabels)[]).map(key => (
              <NumberInput
                key={key}
                label={`${mixLabels[key]} Weight`}
                value={mix[key]}
                min={0}
                required
                docKey="datasets.mixed_weights"
                onChange={value => onChange({ ...dataset, mixed_weights: { ...mix, [key]: value ?? 0 } })}
              />
            ))}
          </div>
          <p className="mt-2 text-xs text-gray-400">
            {Object.entries(mixLabels)
              .map(([key, label]) => `${label}: ${percent(mix[key as keyof typeof mix], mixTotal)}`)
              .join(' · ')}
          </p>
        </fieldset>
      )}
      {errors.length > 0 && (
        <div role="alert" className="space-y-1 text-sm text-red-500">
          {errors.map(error => (
            <p key={error}>{error}</p>
          ))}
        </div>
      )}
    </div>
  );
}
