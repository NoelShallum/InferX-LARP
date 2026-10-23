import { describe, expect, it } from 'vitest';
import { percentile, summarise } from './stats.mjs';
import { GPU_TARGETS, bidStats, buildQuery, compactOffer, compactVastForHistory, viewStats } from './vast-core.mjs';
import { batchRatio, compactOrForHistory, normaliseEndpoint, statsByProviderName, summariseEndpoints, trafficFromActivity } from './openrouter-core.mjs';

const offer = (over) => ({ id: 1, machine_id: 9, gpu_name: 'RTX 5090', num_gpus: 1, dph_total: 1, reliability2: 0.95, verification: 'verified', hosting_type: 0, geolocation: 'Germany, DE', gpu_ram: 32768, ...over });

describe('stats', () => {
  it('interpolates percentiles', () => {
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(percentile([10], 0.9)).toBe(10);
    expect(percentile([], 0.5)).toBeNull();
  });
  it('summarises and ignores non-numbers', () => {
    expect(summarise([3, 1, NaN, 2])).toEqual({ n: 3, min: 1, p25: 1.5, median: 2, max: 3 });
  });
});

describe('vast.ai offers', () => {
  it('divides a node price across its GPUs and reports memory in GB', () => {
    const c = compactOffer(offer({ num_gpus: 4, dph_total: 3.2, gpu_ram: 24576 }));
    expect(c.perGpu).toBe(0.8);
    expect(c.vramGb).toBe(24);
    expect(c.datacenter).toBe(false);
  });
  it('flags data-centre hosts', () => {
    expect(compactOffer(offer({ hosting_type: 1 })).datacenter).toBe(true);
  });
  it('applies each reliability view to the same offers', () => {
    const offers = [
      offer({ id: 1, dph_total: 0.5, reliability2: 0.85 }),
      offer({ id: 2, dph_total: 0.7, reliability2: 0.92 }),
      offer({ id: 3, dph_total: 0.9, reliability2: 0.99, hosting_type: 1 }),
      offer({ id: 4, dph_total: 0.4, reliability2: 0.99, verification: 'unverified' }),
    ].map(compactOffer);
    const v = viewStats(offers);
    expect(v.any.count).toBe(4);
    expect(v.any.minPerGpu).toBe(0.4);
    expect(v.r90.count).toBe(2);
    expect(v.r90.minPerGpu).toBe(0.7);
    expect(v.r98.count).toBe(1);
    expect(v.dc99.count).toBe(1);
    expect(v.dc99.cheapest[0].id).toBe(3);
  });
  it('builds a query that keeps the 16 GB Tesla V100 out', () => {
    const t = GPU_TARGETS.v100;
    const q = buildQuery({ names: t.names, numGpus: 2, minVramMb: t.minVramMb, verified: true });
    expect(q.gpu_ram).toEqual({ gte: 30000 });
    expect(q.num_gpus).toEqual({ eq: 2 });
    expect(q.verified).toEqual({ eq: true });
    expect(q.type).toBe('on-demand');
    expect(buildQuery({ names: ['x'], numGpus: 1, verified: false }).verified).toBeUndefined();
  });
  it('summarises interruptible bids per GPU', () => {
    expect(bidStats([{ min_bid: 0.4, num_gpus: 2 }, { min_bid: 0.5, num_gpus: 1 }])).toEqual({ count: 2, minPerGpu: 0.2, medianPerGpu: 0.35 });
  });
});

describe('openrouter', () => {
  const day = (date, prompt, completion, cached, count) => ({ date: `${date} 00:00:00`, total_prompt_tokens: prompt, total_completion_tokens: completion, total_native_tokens_cached: cached, count });

  it('drops the partial current day and derives the request shape from the totals', () => {
    const t = trafficFromActivity([day('2026-10-23', 1, 1, 0, 1), day('2026-10-22', 900, 100, 600, 10), day('2026-10-21', 900, 100, 300, 10)], 7, new Date('2026-10-23T12:00:00Z'));
    expect(t.days).toBe(2);
    expect(t.to).toBe('2026-10-22');
    expect(t.tokensPerDay).toBe(1000);
    expect(t.avgTokensPerRequest).toBe(100);
    expect(t.outShare).toBe(0.1);
    expect(t.cacheHit).toBe(0.5);
  });
  it('uses at most the requested number of days', () => {
    const rows = Array.from({ length: 10 }, (_, i) => day(`2026-10-${String(10 + i).padStart(2, '0')}`, 100, 10, 0, 1));
    expect(trafficFromActivity(rows, 7, new Date('2026-10-30T00:00:00Z')).days).toBe(7);
  });
  it('merges endpoint stats by provider and converts prices to dollars per million tokens', () => {
    const stats = statsByProviderName([{ name: 'Modal | qwen/qwen3.8-27b', is_zdr: true, stats: { p50_throughput: 93, p50_latency: 664, request_count: 6612 } }]);
    const e = normaliseEndpoint({ provider_name: 'Modal', pricing: { prompt: '0.00000004', completion: '0.0000023', input_cache_read: '0.00000002' }, quantization: 'unknown', status: 0, uptime_last_1d: 99.9 }, stats);
    expect(e.inPerM).toBe(0.04);
    expect(e.outPerM).toBe(2.3);
    expect(e.cachePerM).toBe(0.02);
    expect(e.p50Tps).toBe(93);
    expect(e.zdr).toBe(true);
  });
  it('summarises endpoints: medians, cache fraction and typical stream speed', () => {
    const eps = [
      { inPerM: 0.1, outPerM: 1, cachePerM: 0.02, status: 0, p50Tps: 40, p50LatencyMs: 500, uptime1d: 99 },
      { inPerM: 0.2, outPerM: 2, cachePerM: 0.1, status: 0, p50Tps: 80, p50LatencyMs: 700, uptime1d: 98 },
      { inPerM: 0.3, outPerM: 3, cachePerM: null, status: -2, p50Tps: null, p50LatencyMs: null, uptime1d: 90 },
    ];
    const s = summariseEndpoints(eps);
    expect(s.endpointsListed).toBe(3);
    expect(s.endpointsUp).toBe(2);
    expect(s.inMedian).toBe(0.2);
    expect(s.outMin).toBe(1);
    expect(s.outMax).toBe(3);
    expect(s.cacheFractionMedian).toBe(0.35);
    expect(s.p50TpsMedian).toBe(60);
    expect(s.speedEndpoints).toBe(2);
  });
  it('reads how much a batch variant discounts', () => {
    expect(batchRatio({ pricing: { prompt: '0.0000002', completion: '0.000001' } }, { pricing: { prompt: '0.0000001', completion: '0.0000005' } })).toEqual({ inRatio: 0.5, outRatio: 0.5 });
    expect(batchRatio({ pricing: { prompt: '0.1', completion: '1' } }, undefined)).toBeNull();
  });
});

describe('history copies', () => {
  it('keeps every statistic but only the cheapest offer per view', () => {
    const offers = [1, 2, 3].map((i) => compactOffer(offer({ id: i, dph_total: i, reliability2: 0.95 })));
    const full = { gpus: { rtx5090: { nodes: { 1: { views: viewStats(offers) } } } } };
    expect(full.gpus.rtx5090.nodes[1].views.r90.cheapest).toHaveLength(3);
    const small = compactVastForHistory(full);
    const v = small.gpus.rtx5090.nodes[1].views.r90;
    expect(v.cheapest).toHaveLength(1);
    expect(v.cheapest[0].id).toBe(1);
    expect(v.count).toBe(3);
    expect(v.medianPerGpu).toBe(2);
    expect(full.gpus.rtx5090.nodes[1].views.r90.cheapest).toHaveLength(3);
  });
  it('drops endpoint lists but keeps the summary and traffic', () => {
    const small = compactOrForHistory({ models: { m: { summary: { endpointsListed: 2 }, traffic: { tokensPerDay: 5 }, endpoints: [{ provider: 'a' }, { provider: 'b' }] } } });
    expect(small.models.m.endpoints).toBeUndefined();
    expect(small.models.m.summary.endpointsListed).toBe(2);
    expect(small.models.m.traffic.tokensPerDay).toBe(5);
  });
});
