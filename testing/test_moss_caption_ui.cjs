// Run: node testing/test_moss_caption_ui.cjs (after installing UI dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../ui/node_modules/typescript');

const file = path.join(__dirname, '../ui/src/app/jobs/new/jobConfig.ts');
const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const moduleExports = {};
vm.runInNewContext(compiled, {
  exports: moduleExports,
  require: name => {
    if (name === '@/helpers/basic') return { isMac: () => false };
    if (name === '@/helpers/defaultSamples') return { defaultSampleConfig: {} };
    throw new Error(`Unexpected dependency: ${name}`);
  },
});
const { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } = moduleExports;
const clone = value => JSON.parse(JSON.stringify(value));
const migrate = dataset => {
  const job = clone(defaultJobConfig);
  job.config.process[0].datasets = [dataset];
  return clone(migrateJobConfig(job).config.process[0].datasets[0]);
};

assert.deepEqual(clone(defaultDatasetConfig.caption_length_weights), { short: 1, medium: 1, long: 1 });
const tagsOnly = migrate({
  folder_path: 'C:/images',
  caption_format: 'moss_json',
  caption_mode: 'mixed',
  mixed_weights: { tags: 1 },
  caption_length_weights: { long: 70 },
});
assert.deepEqual(tagsOnly.mixed_weights, { tags: 1, nl: 0, tags_nl: 0, nl_tags: 0 });
assert.deepEqual(tagsOnly.caption_length_weights, { short: 0, medium: 0, long: 70 });
const custom = migrate({ ...tagsOnly, caption_length_weights: { short: 10, medium: 20, long: 70 } });
assert.deepEqual(custom.caption_length_weights, { short: 10, medium: 20, long: 70 });
assert.deepEqual(migrate(clone(custom)), custom, 'saving/loading twice must not reactivate disabled choices');
const legacy = migrate({ folder_path: 'C:/images', caption_mode: 'mixed' });
assert.deepEqual(legacy.mixed_weights, { tags: 40, nl: 30, tags_nl: 20, nl_tags: 10 });
assert.deepEqual(legacy.caption_length_weights, { short: 1, medium: 1, long: 1 });
assert.equal(tagsOnly.caption_ext, 'json');

// Exercise the real form handlers, so a displayed option must update the
// dataset config consumed by training (rather than only render a label).
const loadModule = (relative, dependencies = {}) => {
  const source = fs.readFileSync(path.join(__dirname, relative), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {
    exports,
    require: name => {
      if (Object.hasOwn(dependencies, name)) return dependencies[name];
      if (name === 'react/jsx-runtime') return require('../ui/node_modules/react/jsx-runtime');
      throw new Error(`Unexpected dependency: ${name}`);
    },
  });
  return exports;
};
const captionConfig = loadModule('../ui/src/app/jobs/new/captionConfig.ts');
const fields = { SelectInput: 'select-field', NumberInput: 'number-field', TextInput: 'text-field' };
const CaptionSettings = loadModule('../ui/src/app/jobs/new/CaptionSettings.tsx', {
  './captionConfig': captionConfig,
  '@/components/formInputs': fields,
}).default;
const elements = tree => {
  if (Array.isArray(tree)) return tree.flatMap(elements);
  if (!tree || typeof tree !== 'object') return [];
  return [tree, ...elements(tree.props?.children)];
};
let formDataset = clone(defaultDatasetConfig);
const train = clone(defaultJobConfig.config.process[0].train);
const render = () => CaptionSettings({ dataset: formDataset, train, onChange: value => { formDataset = clone(value); } });
const control = label => {
  const field = elements(render()).find(node => node.props?.label === label);
  assert.ok(field, `Missing GUI field: ${label}`);
  return field.props;
};
control('Caption Format').onChange('moss_json');
assert.equal(formDataset.caption_ext, 'json');
assert.equal(control('Caption Mode').options.some(option => option.value === 'tags_only'), true);
control('Dataset Trigger Word').onChange('my_character');
control('Short Weight').onChange(10);
control('Medium Weight').onChange(20);
control('Long Weight').onChange(70);
control('Caption Mode').onChange('tags_only');
assert.equal(formDataset.caption_mode, 'mixed');
assert.deepEqual(formDataset.mixed_weights, { tags: 1, nl: 0, tags_nl: 0, nl_tags: 0 });
control('Caption Mode').onChange('mixed');
control('Tags Weight').onChange(25);
control('Natural Language Weight').onChange(25);
control('Tags + Natural Language Weight').onChange(25);
control('Natural Language + Tags Weight').onChange(25);
const savedForm = migrate(formDataset);
assert.equal(savedForm.caption_format, 'moss_json');
assert.equal(savedForm.trigger_word, 'my_character');
assert.deepEqual(savedForm.caption_length_weights, { short: 10, medium: 20, long: 70 });
assert.deepEqual(savedForm.mixed_weights, { tags: 25, nl: 25, tags_nl: 25, nl_tags: 25 });
assert.deepEqual(migrate(clone(savedForm)), savedForm);
assert.equal(captionConfig.captionSettingsErrors(formDataset, train).length, 0);
for (const conflict of [
  { shuffle_caption: true }, { shuffle_tokens: true, shuffle_caption: undefined },
  { token_dropout_rate: .1 }, { random_triggers: ['tag'] },
]) {
  for (const datasetCache of [false, true]) {
    assert.ok(captionConfig.captionSettingsErrors(
      { ...formDataset, ...conflict, cache_text_embeddings: datasetCache },
      { ...train, cache_text_embeddings: !datasetCache },
    ).some(error => error.includes('Cache Text Embeddings')));
  }
}
for (const flag of ['short_and_long_captions', 'short_and_long_captions_encoder_split']) {
  assert.equal(captionConfig.captionSettingsErrors(formDataset, { ...train, [flag]: true }).length, 0);
  assert.ok(captionConfig.captionSettingsErrors(formDataset, {
    ...train, cache_text_embeddings: true, [flag]: true,
  }).some(error => error.includes('Short/long')));
}
for (const weights of [{ short: 0, medium: 0, long: 0 }, { short: -1 }, { short: Infinity }]) {
  assert.ok(captionConfig.captionSettingsErrors({ ...formDataset, caption_length_weights: weights }, train).length);
}
formDataset.caption_length_weights = { short: 0, medium: 0, long: 0 };
assert.ok(elements(render()).some(node => node.props?.role === 'alert'), 'Invalid settings must show a visible error');
console.log('MOSS JSON UI config: defaults, partial weights, and save/load round trips passed.');
console.log('MOSS JSON GUI: format, content modes, weights, trigger, and validation handlers passed.');

// Build-generated Next route validators expose unrelated existing API signature
// errors. Validate application sources independently of those generated files.
const uiRoot = path.resolve(__dirname, '../ui');
const config = ts.readConfigFile(path.join(uiRoot, 'tsconfig.json'), ts.sys.readFile);
if (config.error) throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, '\n'));
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, uiRoot);
const sources = parsed.fileNames.filter(name => !name.replaceAll('\\', '/').includes('/.next/'));
const program = ts.createProgram(sources, { ...parsed.options, incremental: false });
const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
if (diagnostics.length) {
  console.error(ts.formatDiagnosticsWithColorAndContext(diagnostics, {
    getCanonicalFileName: name => name,
    getCurrentDirectory: () => uiRoot,
    getNewLine: () => '\n',
  }));
  process.exit(1);
}
console.log('UI source TypeScript check passed (generated Next route validators excluded).');
