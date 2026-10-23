import { median, round, summarise } from './stats.mjs';

const PER_M = 1e6;
const num = (x) => (x === null || x === undefined || x === '' ? null : Number(x));

/** Our catalog model id -> OpenRouter slug. */
export const MODEL_SLUGS = {
  qwen38_27b: 'qwen/qwen3.8-27b',
  qwen36_35b: 'qwen/qwen3.6-35b-a3b',
  gemma4_26b: 'google/gemma-4-26b-a4b-it',
  gemma4_31b: 'google/gemma-4-31b-it',
  gptoss_120b: 'openai/gpt-oss-120b',
  gptoss_20b: 'openai/gpt-oss-20b',
  llama33_70b: 'meta-llama/llama-3.3-70b-instruct',
  mistral_nemo: 'mistralai/mistral-nemo',
  flashnext: 'qwen/qwen3.8-flash',
  nemotron35: 'nvidia/nemotron-3.5-super-vl',
  qwen35_122b: 'qwen/qwen3.5-122b-a10b',
  mistral_small4: 'mistralai/mistral-small-2603',
  devstral2: 'mistralai/devstral-2512',
};

/** One endpoint from /api/v1/models/{slug}/endpoints, merged with its frontend stats where the names match. */
export function normaliseEndpoint(e, statsByProvider) {
  const inP = num(e.pricing?.prompt);
  const outP = num(e.pricing?.completion);
  const cacheP = num(e.pricing?.input_cache_read);
  const st = statsByProvider.get(e.provider_name) ?? null;
  return {
    provider: e.provider_name,
    quantization: e.quantization ?? 'unknown',
    inPerM: inP === null ? null : round(inP * PER_M, 5),
    outPerM: outP === null ? null : round(outP * PER_M, 5),
    cachePerM: cacheP === null ? null : round(cacheP * PER_M, 5),
    status: e.status ?? null,
    uptime1d: round(e.uptime_last_1d, 2),
    uptime30m: round(e.uptime_last_30m, 2),
    // OpenRouter reports speed percentiles where p75 and p90 are the slower tail, so p50 is the typical stream.
    p50Tps: st?.p50_throughput ?? null,
    p90Tps: st?.p90_throughput ?? null,
    p50LatencyMs: st?.p50_latency ?? null,
    requests30m: st?.request_count ?? null,
    zdr: st?.is_zdr ?? null,
  };
}

export function statsByProviderName(statEndpoints) {
  const m = new Map();
  for (const s of statEndpoints ?? []) {
    const provider = String(s.name ?? '').split(' | ')[0];
    if (provider && s.stats) m.set(provider, { ...s.stats, is_zdr: s.is_zdr });
  }
  return m;
}

export function summariseEndpoints(endpoints) {
  const live = endpoints.filter((e) => e.inPerM !== null && e.outPerM !== null);
  const inS = summarise(live.map((e) => e.inPerM));
  const outS = summarise(live.map((e) => e.outPerM));
  const fractions = live.filter((e) => e.cachePerM !== null && e.inPerM > 0).map((e) => e.cachePerM / e.inPerM);
  const tps = endpoints.map((e) => e.p50Tps).filter((x) => Number.isFinite(x) && x > 0);
  const lat = endpoints.map((e) => e.p50LatencyMs).filter((x) => Number.isFinite(x) && x > 0);
  const up = endpoints.map((e) => e.uptime1d).filter(Number.isFinite);
  return {
    endpointsListed: endpoints.length,
    endpointsUp: endpoints.filter((e) => e.status === 0).length,
    inMin: round(inS.min, 5),
    inMedian: round(inS.median, 5),
    outMin: round(outS.min, 5),
    outMedian: round(outS.median, 5),
    outMax: round(outS.max, 5),
    cacheFractionMedian: fractions.length ? round(median(fractions), 4) : null,
    cacheEndpoints: fractions.length,
    speedEndpoints: tps.length,
    p50TpsMedian: tps.length ? round(median(tps), 1) : null,
    p50TpsP25: tps.length ? round(summarise(tps).p25, 1) : null,
    p50LatencyMsMedian: lat.length ? round(median(lat), 0) : null,
    uptime1dMedian: up.length ? round(median(up), 2) : null,
  };
}

/**
 * Traffic from /stats/model-activity, which lists one row per UTC day. The newest day is partial, so it is dropped.
 * The request shape follows from the totals: average tokens a request, output share, and cache hit.
 */
export function trafficFromActivity(analytics, days = 7, today = new Date()) {
  const todayKey = today.toISOString().slice(0, 10);
  const rows = [...(analytics ?? [])]
    .map((r) => ({
      date: String(r.date).slice(0, 10),
      prompt: num(r.total_prompt_tokens) ?? 0,
      completion: num(r.total_completion_tokens) ?? 0,
      cached: num(r.total_native_tokens_cached) ?? 0,
      requests: num(r.count) ?? 0,
    }))
    .filter((r) => r.date < todayKey)
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, days);
  if (rows.length === 0) return null;
  const sum = (k) => rows.reduce((a, r) => a + r[k], 0);
  const prompt = sum('prompt');
  const completion = sum('completion');
  const requests = sum('requests');
  const d = rows.length;
  return {
    days: d,
    from: rows[rows.length - 1].date,
    to: rows[0].date,
    tokensPerDay: Math.round((prompt + completion) / d),
    promptPerDay: Math.round(prompt / d),
    completionPerDay: Math.round(completion / d),
    requestsPerDay: Math.round(requests / d),
    avgTokensPerRequest: requests > 0 ? Math.round((prompt + completion) / requests) : null,
    outShare: prompt + completion > 0 ? round(completion / (prompt + completion), 4) : null,
    cacheHit: prompt > 0 ? round(sum('cached') / prompt, 4) : null,
    daily: rows.map((r) => ({ date: r.date, tokens: r.prompt + r.completion, requests: r.requests })).reverse(),
  };
}

/** History copy: keep the summary and traffic, drop the endpoint lists. */
export function compactOrForHistory(payload) {
  const models = {};
  for (const [id, m] of Object.entries(payload.models)) {
    const { endpoints, ...rest } = m;
    models[id] = rest;
  }
  return { ...payload, models };
}

/** Batch variants (slug:batch) show how much providers discount offline work. */
export function batchRatio(main, batch) {
  const a = num(main?.pricing?.prompt);
  const b = num(batch?.pricing?.prompt);
  const c = num(main?.pricing?.completion);
  const d = num(batch?.pricing?.completion);
  if (![a, b, c, d].every((x) => x !== null && x > 0)) return null;
  return { inRatio: round(b / a, 3), outRatio: round(d / c, 3) };
}
