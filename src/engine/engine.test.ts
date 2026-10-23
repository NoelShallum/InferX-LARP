import { describe, expect, it } from 'vitest';
import { GPU_BY_ID } from '../data/gpus';
import { MODEL_BY_ID } from '../data/models';
import { capacityFor, chooseQuant, curveFor, mtpGain, mtpInfo, tpsAt } from './capacity';
import { calibrationFor } from './calibration';
import { DEFAULT_SETTINGS, makeSlot } from './defaults';
import { portfolioSteady, programCosts } from './portfolio';
import { elasticityFactor } from './pricing';
import { reportSettings } from './crosscheck';
import { evaluateSlot } from './slot';
import { tornado } from './sensitivity';
import { liveMarket, liveRent, OPENROUTER, orTrend, VAST, vastTrend } from '../data/live';
import { floorFor, marketFor, rentFor } from './sources';

const pro = GPU_BY_ID['pro6000']!;

describe('fit rules', () => {
  it('rejects Llama 3.3 70B on the 48 GB Ada, as the report does (1.5 GB of KV left)', () => {
    const f = chooseQuant(MODEL_BY_ID['llama33_70b']!, GPU_BY_ID['ada6000']!, 1, 'auto');
    expect(f.fits).toBe(false);
  });
  it('fits Qwen3.8-27B NVFP4 on a 32 GB RTX 5090', () => {
    const f = chooseQuant(MODEL_BY_ID['qwen38_27b']!, GPU_BY_ID['rtx5090']!, 1, 'auto');
    expect(f.fits).toBe(true);
    expect(f.quant).toBe('nvfp4');
  });
  it('falls back to a 4-bit integer build on a V100, which has no FP8 or FP4', () => {
    const f = chooseQuant(MODEL_BY_ID['qwen38_27b']!, GPU_BY_ID['v100']!, 1, 'auto');
    expect(f.fits).toBe(true);
    expect(f.quant).toBe('awq4');
  });
  it('does not fit gpt-oss-120b on a single 32 GB card', () => {
    expect(chooseQuant(MODEL_BY_ID['gptoss_120b']!, GPU_BY_ID['rtx5090']!, 1, 'auto').fits).toBe(false);
  });
  it('needs four 5090s to hold gpt-oss-120b, not two', () => {
    expect(chooseQuant(MODEL_BY_ID['gptoss_120b']!, GPU_BY_ID['rtx5090']!, 2, 'auto').fits).toBe(false);
    expect(chooseQuant(MODEL_BY_ID['gptoss_120b']!, GPU_BY_ID['rtx5090']!, 4, 'auto').fits).toBe(true);
  });
});

describe('throughput curve and speculative decoding', () => {
  const model = MODEL_BY_ID['qwen38_27b']!;
  const curve = curveFor(model, pro, 'nvfp4', pro)!;
  it('passes through the measured 862 tok/s at 16 concurrent', () => {
    expect(tpsAt(curve, 16)).toBeCloseTo(862, 0);
  });
  it('MTP gain is calibrated to the measured 1,752 tok/s at 32 concurrent', () => {
    const info = mtpInfo(model, curve);
    expect(info).not.toBeNull();
    expect(tpsAt(curve, 32) * mtpGain(info, 32)).toBeCloseTo(1752, -1);
  });
  it('MTP stops helping at high concurrency', () => {
    const info = mtpInfo(model, curve);
    expect(mtpGain(info, 200)).toBeCloseTo(1, 5);
    expect(mtpGain(info, 1)).toBeGreaterThan(mtpGain(info, 16));
  });
  it('a stricter per-user speed floor lowers concurrency and total throughput', () => {
    const base = { model, gpu: pro, proGpu: pro, count: 1, quant: 'nvfp4' as const, mtp: false, extraMult: 1, shape: { avgTokens: 15300, outShare: 0.041, cacheHit: 0.41 } };
    const loose = capacityFor({ ...base, qos: { mode: 'floor', floorTps: 20 } });
    const strict = capacityFor({ ...base, qos: { mode: 'floor', floorTps: 60 } });
    expect(strict.cPerGpu).toBeLessThan(loose.cPerGpu);
    expect(strict.tpsNode).toBeLessThan(loose.tpsNode);
    expect(strict.userTps).toBeGreaterThanOrEqual(59.9);
  });
});

describe('demand and price', () => {
  it('cutting the price to 0.36× the median buys 1.90× the traffic (elasticity -0.63)', () => {
    expect(elasticityFactor(0.36).factor).toBeCloseTo(1.9, 1);
  });
  it('penalises prices above the 1.2× ceiling the report has no data for', () => {
    expect(elasticityFactor(1.6).aboveData).toBe(true);
    expect(elasticityFactor(1.6).factor).toBeLessThan(elasticityFactor(1.2).factor);
  });
  it('calibrates Qwen3.8-27B output share within half a point of the report\'s 4.1%', () => {
    const cal = calibrationFor(MODEL_BY_ID['qwen38_27b']!)!;
    expect(Math.abs(cal.outShare - 0.041)).toBeLessThan(0.005);
  });
  it('calibrates gpt-oss-120b output share within a point of the report\'s 11.8%', () => {
    const cal = calibrationFor(MODEL_BY_ID['gptoss_120b']!)!;
    expect(Math.abs(cal.outShare - 0.118)).toBeLessThan(0.01);
  });
});

describe('economics', () => {
  const s = reportSettings('base');
  it('a rented PRO 6000 on Qwen3.8-27B breaks even at 12% busy, as the report says', () => {
    const slot = makeSlot({ id: 'r', ownership: 'rent', gpuId: 'pro6000', modelId: 'qwen38_27b' });
    const r = evaluateSlot(slot, s);
    expect(r.breakEvenUtil).not.toBeNull();
    expect(r.breakEvenUtil!).toBeCloseTo(0.12, 2);
  });
  it('a rented RTX 5090 breaks even at 4% busy', () => {
    const slot = makeSlot({ id: 'r', ownership: 'rent', gpuId: 'rtx5090', modelId: 'qwen38_27b' });
    expect(evaluateSlot(slot, s).breakEvenUtil!).toBeCloseTo(0.04, 2);
  });
  it('an owned card left on Lium earns exactly its bar', () => {
    const slot = makeSlot({ id: 'o', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'lium' });
    const r = evaluateSlot(slot, s);
    expect(r.marginUsd).toBe(0);
    expect(r.revenueUsd).toBeCloseTo(r.barUsd, 6);
  });
  it('overflow picks up only the traffic the main card cannot serve', () => {
    const main = makeSlot({ id: 'main', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' });
    const mirror = makeSlot({ id: 'mirror', ownership: 'rent', gpuId: 'rtx5090', modelId: 'qwen38_27b', demand: { mode: 'overflow', utilisation: 0, shareScale: 1, of: 'main' } });
    const base = portfolioSteady([main, mirror], reportSettings('base'));
    expect(base.results[1]!.revenueUsd).toBe(0);
    const high = portfolioSteady([main, mirror], reportSettings('high'));
    expect(high.results[1]!.revenueUsd).toBeGreaterThan(0);
  });
  it('after-tax margin on a fiat route is the pre-tax margin less 25.17%', () => {
    const slot = makeSlot({ id: 'o', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' });
    const r = evaluateSlot(slot, s);
    expect(r.afterTaxMarginUsd).toBeCloseTo(r.marginUsd * (1 - 0.2517), 6);
  });
  it('program costs for the base case: build, two queue months, one review month, one bridge month', () => {
    const slot = makeSlot({ id: 'o', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' });
    const { costs } = programCosts([slot], reportSettings('base'), 'base');
    const fx = DEFAULT_SETTINGS.fx;
    expect(costs.buildUsd * fx).toBeCloseTo(10_000, 0);
    expect(costs.queueUsd * fx).toBeCloseTo(2000, 0);
    expect(costs.rejectedUsd * fx).toBeCloseTo(65_274, -2);
  });
  it('the sensitivity tornado ranks share of traffic above the availability swing', () => {
    const slot = makeSlot({ id: 'o', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' });
    const t = tornado([slot], DEFAULT_SETTINGS, 'steady');
    const share = t.rows.find((r) => r.id === 'share')!;
    const avail = t.rows.find((r) => r.id === 'availability')!;
    expect(share.span).toBeGreaterThan(avail.span);
  });
});

describe('live data sources', () => {
  const live = { ...DEFAULT_SETTINGS, rentSource: 'live' as const, marketSource: 'live' as const };
  const rpt = { ...DEFAULT_SETTINGS, rentSource: 'report' as const, marketSource: 'report' as const };

  it('uses a live Vast.ai price for a rented RTX 5090 and tags it live', () => {
    const r = rentFor(GPU_BY_ID['rtx5090']!, 1, live)!;
    expect(r.p.kind).toBe('live');
    expect(r.v).toBeGreaterThan(0.2);
    expect(r.v).toBeLessThan(3);
  });
  it('uses the report price when the report source is chosen', () => {
    const r = rentFor(GPU_BY_ID['pro6000']!, 1, rpt)!;
    expect(r.v).toBe(1.64);
    expect(r.p.kind).toBe('report');
  });
  const noOffer = Object.entries(VAST.gpus).find(([id, g]) => (g.nodes['1']?.views.r90.count ?? 0) === 0 && GPU_BY_ID[id]?.rentUsdHr)?.[0];
  it('falls back to the catalog price, and says so, when no offer passes the filter', () => {
    expect(noOffer).toBeDefined();
    const gpu = GPU_BY_ID[noOffer!]!;
    expect(liveRent(gpu.id, 1, 'r90', 'cheapest')).toBeNull();
    const r = rentFor(gpu, 1, live)!;
    expect(r.p.kind).not.toBe('live');
    expect(r.v).toBe(gpu.rentUsdHr!.v);
    expect(r.p.note).toContain('No Vast.ai offer passes');
  });
  it('a stricter reliability filter never lowers the cheapest price', () => {
    for (const id of Object.keys(VAST.gpus)) {
      const a = liveRent(id, 1, 'r90', 'cheapest');
      const b = liveRent(id, 1, 'r98', 'cheapest');
      if (a && b && !a.borrowedFromSingle && !b.borrowedFromSingle) expect(b.perGpu).toBeGreaterThanOrEqual(a.perGpu);
    }
  });
  it('borrows the single-card per-GPU price when a larger node has no offer', () => {
    const entry = Object.entries(VAST.gpus).find(([, g]) => (g.nodes['1']?.views.r90.count ?? 0) > 0 && (g.nodes['8']?.views.r90.count ?? 0) === 0);
    expect(entry).toBeDefined();
    const l = liveRent(entry![0], 8, 'r90', 'cheapest')!;
    expect(l.borrowedFromSingle).toBe(true);
    expect(l.nodeSize).toBe(1);
  });
  it('lays live OpenRouter numbers over the report row', () => {
    const resolved = marketFor('qwen38_27b', live)!;
    expect(resolved.live).not.toBeNull();
    expect(resolved.row.endpoints.v).toBe(OPENROUTER.models['qwen38_27b']!.summary.endpointsListed);
    expect(resolved.row.endpoints.p.kind).toBe('live');
    const report = marketFor('qwen38_27b', rpt)!;
    expect(report.live).toBeNull();
    expect(report.row.endpoints.v).toBe(20);
  });
  it('observed request shape comes from live traffic totals', () => {
    const lm = liveMarket('gptoss_120b', false)!;
    expect(lm.shape).not.toBeNull();
    expect(lm.shape!.outShare).toBeGreaterThan(0.05);
    expect(lm.shape!.outShare).toBeLessThan(0.2);
  });
  it('the OpenRouter-competitive quality bar is the median live stream speed for the model', () => {
    const router = { ...live, qos: { mode: 'router' as const, floorTps: 40, ttftP90Sec: 10 } };
    const f = floorFor('qwen38_27b', router);
    expect(f.v).not.toBeNull();
    expect(f.floor).toBe(OPENROUTER.models['qwen38_27b']!.summary.p50TpsMedian);
    const none = floorFor('qwen38_27b', { ...router, qos: { mode: 'floor' as const, floorTps: 40, ttftP90Sec: 10 } });
    expect(none.floor).toBe(40);
  });
  it('a rented slot is charged the live rent', () => {
    const slot = makeSlot({ id: 'r', ownership: 'rent', gpuId: 'rtx5090', modelId: 'qwen38_27b' });
    const r = evaluateSlot(slot, { ...live, availabilityCustom: 1 });
    const expected = rentFor(GPU_BY_ID['rtx5090']!, 1, live)!.v * 730;
    expect(r.costUsd).toBeCloseTo(expected, 4);
    expect(r.rentFallback).toBe(false);
  });
  it('flags a rented slot whose live price fell back', () => {
    const slot = makeSlot({ id: 'r', ownership: 'rent', gpuId: noOffer!, modelId: 'gptoss_20b' });
    const r = evaluateSlot(slot, live);
    expect(r.rentFallback).toBe(true);
  });
});

describe('trends from saved snapshots', () => {
  const vsnap = (date: string, min: number | null) =>
    ({ generatedAt: `${date}T10:00:00Z`, gpus: { rtx5090: { nodes: { '1': { views: { r90: { count: min === null ? 0 : 5, minPerGpu: min, medianPerGpu: min === null ? null : min + 0.3, p25PerGpu: null, cheapest: [] } } } } } } }) as unknown as Parameters<typeof vastTrend>[0][number];
  it('lists one point per snapshot that had an offer, oldest first', () => {
    const t = vastTrend([vsnap('2026-10-23', 0.7), vsnap('2026-10-24', null), vsnap('2026-10-25', 0.65)], 'rtx5090', 1, 'r90', 'cheapest');
    expect(t).toEqual([{ date: '2026-10-23', value: 0.7 }, { date: '2026-10-25', value: 0.65 }]);
  });
  it('can follow the median instead of the cheapest', () => {
    const t = vastTrend([vsnap('2026-10-23', 0.7)], 'rtx5090', 1, 'r90', 'median');
    expect(t[0]!.value).toBeCloseTo(1.0, 6);
  });
  it('reads one OpenRouter measure per model, converting traffic to billions', () => {
    const osnap = (date: string, tokens: number, med: number) =>
      ({ generatedAt: `${date}T10:00:00Z`, models: { m: { summary: { endpointsListed: 5, outMedian: med, inMedian: 0.1, outMin: 0.5, p50TpsMedian: 40 }, traffic: { tokensPerDay: tokens } } } }) as unknown as Parameters<typeof orTrend>[0][number];
    const h = [osnap('2026-10-23', 60e9, 2.3), osnap('2026-10-24', 66e9, 2.2)];
    expect(orTrend(h, 'm', 'tokensPerDay').map((p) => p.value)).toEqual([60, 66]);
    expect(orTrend(h, 'm', 'outMedian').map((p) => p.value)).toEqual([2.3, 2.2]);
    expect(orTrend(h, 'missing', 'outMedian')).toEqual([]);
  });
});
