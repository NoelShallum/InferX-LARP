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
    const base = { model, gpu: pro, proGpu: pro, count: 1, quant: 'nvfp4' as const, mtp: false, extraMult: 1, avgContextTokens: 15300 };
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
    const { costs } = programCosts([slot], DEFAULT_SETTINGS, 'base');
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
