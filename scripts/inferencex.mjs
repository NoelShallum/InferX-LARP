/**
 * Measured benchmark results from InferenceX (SemiAnalysis): throughput, speed per user and time to first token for
 * real serving stacks on H100, H200, B200 and the RTX PRO 6000, single GPU and tensor-parallel.
 *
 *   npm run data:inferencex
 *
 * Plain HTTP is enough. Environment:
 *   MODELS="gpt-oss-120b,Llama-3.3-70B-Instruct-FP8"   limit to some InferenceX display models
 */
import { sleep, writeJson } from './lib/snapshot.mjs';
import { IX_HARDWARE, IX_MODELS, compactRows, toCurves } from './lib/inferencex-core.mjs';

const API = 'https://inferencex.semianalysis.com/api/v1';
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

const wanted = process.env.MODELS ? process.env.MODELS.split(',') : Object.keys(IX_MODELS);
const models = {};
for (const name of wanted) {
  const spec = IX_MODELS[name];
  if (!spec) {
    console.warn(`${name}: not mapped, skipped`);
    continue;
  }
  try {
    const rows = await getJson(`${API}/benchmarks?model=${encodeURIComponent(name)}`);
    // Only the cards the catalog has; AMD and TPU rows can come back when the catalog does.
    const compact = compactRows(rows).filter((r) => r.hardware in IX_HARDWARE);
    models[name] = { id: spec.id, calibrationOnly: Boolean(spec.calibrationOnly), curves: toCurves(compact) };
    const hw = [...new Set(compact.map((r) => r.hardware))].sort().join(', ');
    console.log(`${name.padEnd(28)} ${String(rows.length).padStart(4)} rows → ${String(compact.length).padStart(4)} kept in ${models[name].curves.length} curves · ${hw || 'none'}`);
  } catch (e) {
    console.error(`${name}: ${e.message}`);
  }
  await sleep(300);
}
if (Object.keys(models).length === 0) {
  console.error('No model was read, nothing written.');
  process.exit(1);
}
writeJson('data/inferencex/bench.json', { source: 'inferencex.semianalysis.com/api/v1/benchmarks', generatedAt: new Date().toISOString(), models }, 0);
console.log(`Wrote data/inferencex/bench.json`);
