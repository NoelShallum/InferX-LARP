/**
 * OpenRouter market snapshot: per-model endpoint prices, cache-read prices, typical stream speed, uptime,
 * traffic and the request shape that follows from it.
 *
 * Everything comes from OpenRouter's public JSON, so plain HTTP is enough and no browser is needed.
 *
 *   npm run data:openrouter
 *
 * Environment:
 *   MODELS=qwen38_27b,gptoss_120b   limit to some catalog models
 */
import { writeSnapshot, sleep } from './lib/snapshot.mjs';
import { MODEL_SLUGS, batchRatio, compactOrForHistory, normaliseEndpoint, statsByProviderName, summariseEndpoints, trafficFromActivity } from './lib/openrouter-core.mjs';

const API = 'https://openrouter.ai/api/v1';
const FRONT = 'https://openrouter.ai/api/frontend/v1';
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

const wanted = process.env.MODELS ? process.env.MODELS.split(',') : Object.keys(MODEL_SLUGS);
const catalog = (await getJson(`${API}/models`)).data;
const bySlug = new Map(catalog.map((m) => [m.id, m]));
const now = new Date();

const models = {};
for (const id of wanted) {
  const slug = MODEL_SLUGS[id];
  const entry = slug ? bySlug.get(slug) : null;
  if (!entry) {
    console.warn(`${id}: ${slug ?? 'no slug'} is not listed on OpenRouter, skipped`);
    continue;
  }
  const permaslug = entry.canonical_slug ?? slug;
  const q = `permaslug=${encodeURIComponent(permaslug)}`;
  const [eps, stats, activity] = await Promise.all([
    getJson(`${API}/models/${slug}/endpoints`),
    getJson(`${FRONT}/stats/endpoint?latencyMetric=latency&perfWorkload=text_generation&${q}&variant=standard`).catch(() => ({ data: [] })),
    getJson(`${FRONT}/stats/model-activity?${q}&variant=standard`).catch(() => ({ data: { analytics: [] } })),
  ]);
  const byProvider = statsByProviderName(stats.data);
  const endpoints = (eps.data?.endpoints ?? []).map((e) => normaliseEndpoint(e, byProvider));
  const batch = batchRatio(entry, bySlug.get(`${slug}:batch`));
  models[id] = {
    slug,
    permaslug,
    name: entry.name,
    contextLength: entry.context_length ?? null,
    summary: summariseEndpoints(endpoints),
    traffic: trafficFromActivity(activity.data?.analytics, 7, now),
    batch,
    endpoints: endpoints.sort((a, b) => (a.outPerM ?? 0) - (b.outPerM ?? 0)),
  };
  const s = models[id].summary;
  const t = models[id].traffic;
  console.log(
    `${id.padEnd(15)} ${String(s.endpointsListed).padStart(2)} endpoints, in ${s.inMin}/${s.inMedian}, out ${s.outMin}/${s.outMedian}/${s.outMax}, ` +
      `${t ? (t.tokensPerDay / 1e9).toFixed(1) + 'B tokens/day, ' + (t.outShare * 100).toFixed(1) + '% output, ' + (t.cacheHit * 100).toFixed(0) + '% cached' : 'no traffic data'}` +
      `, median stream ${s.p50TpsMedian ?? '—'} tok/s`,
  );
  await sleep(300);
}

const payload = {
  source: 'openrouter.ai',
  generatedAt: now.toISOString(),
  notes: 'Prices are dollars per million tokens. Traffic is the mean of the last full UTC days of standard (paid) traffic. Speed is the median of each endpoint\'s typical (p50) stream speed.',
  models,
};
console.log(writeSnapshot('openrouter', payload, now, compactOrForHistory));
