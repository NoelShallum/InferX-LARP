/**
 * Architecture snapshot from Hugging Face: for each catalog model, read config.json at a pinned commit and work out
 * KV-cache bytes per token, recurrent state per sequence, mixture-of-experts shape and parameter counts.
 *
 *   npm run data:hf
 *
 * Plain HTTP is enough. Environment:
 *   MODELS=qwen38_27b,gptoss_120b   limit to some catalog models
 */
import { sleep, writeJson } from './lib/snapshot.mjs';
import { MODEL_REPOS, archFromConfig } from './lib/arch-core.mjs';

const HF = 'https://huggingface.co';
const HEADERS = { 'user-agent': 'Mozilla/5.0 (compatible; inferx-data/0.1)', accept: 'application/json' };

async function getJson(url) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    const res = await fetch(url, { headers: HEADERS });
    if (res.ok) return res.json();
    if (res.status === 429 || res.status >= 500) await sleep(1500 * attempt);
    else throw new Error(`${res.status} for ${url}`);
  }
  throw new Error(`Kept failing: ${url}`);
}

const wanted = process.env.MODELS ? process.env.MODELS.split(',') : Object.keys(MODEL_REPOS);
const models = {};
for (const id of wanted) {
  const spec = MODEL_REPOS[id];
  if (!spec) {
    console.warn(`${id}: no repository mapped, skipped`);
    continue;
  }
  try {
    const info = await getJson(`${HF}/api/models/${spec.repo}`);
    const sha = info.sha;
    const config = await getJson(`${HF}/${spec.repo}/resolve/${sha}/config.json`);
    const paramsRepo = spec.paramsRepo ?? spec.repo;
    const meta = await getJson(`${HF}/api/models/${paramsRepo}?expand%5B%5D=safetensors`);
    const total = meta.safetensors?.total ?? null;
    const arch = archFromConfig(config, { totalParams: total, towersInTotal: paramsRepo === spec.repo, card: spec.card ?? null });
    models[id] = {
      repo: spec.repo,
      sha,
      configUrl: `${HF}/${spec.repo}/blob/${sha}/config.json`,
      paramsRepo,
      note: spec.note ?? null,
      ...arch,
    };
    const kv = (arch.fullTokBytes / 1000).toFixed(1);
    console.log(`${id.padEnd(15)} ${arch.family.padEnd(18)} KV ${kv} KB/token, state ${(arch.stateBytes / 1e6).toFixed(0)} MB, active ${arch.params ? (arch.params.active / 1e9).toFixed(1) + 'B' : 'n/a'} of ${arch.params ? (arch.params.text / 1e9).toFixed(1) + 'B' : 'n/a'} text, routed ${arch.params ? Math.round(arch.params.routedShare * 100) : 0}%${arch.kvSource === 'model-card' ? ' (KV from the model card)' : ''}`);
  } catch (e) {
    console.error(`${id}: ${e.message}`);
  }
  await sleep(300);
}

if (Object.keys(models).length === 0) {
  console.error('No model was read, nothing written.');
  process.exit(1);
}
writeJson('data/hf/archs.json', { source: 'huggingface.co', generatedAt: new Date().toISOString(), models });
console.log(`Wrote data/hf/archs.json with ${Object.keys(models).length} models`);
