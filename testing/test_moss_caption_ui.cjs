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
console.log('MOSS JSON UI config: defaults, partial weights, and save/load round trips passed.');

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
