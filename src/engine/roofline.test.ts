import { describe, expect, it } from 'vitest';
import { kvBytesPerSeq, kvReadBytesPerSeq } from '../data/arch';
import { GPUS, GPU_BY_ID } from '../data/gpus';
import { bestIxCurve } from '../data/ix';
import { CAL_MODEL_BY_ID, MODELS, MODEL_BY_ID } from '../data/models';
import { capacityFor, fitOn, seqKvBytes, weightsGB } from './capacity';
import { bandwidthCeilingTps, computeFormat, decodeStep, decodeTps, expertCoverage, prefillRate, requestParts, servingPoint } from './roofline';

const pro = GPU_BY_ID['pro6000']!;
const CTX = 8700;

const inputs = (modelId: string, gpuId: string, quant: Parameters<typeof weightsGB>[1], count = 1, ctx = CTX) => {
  const model = MODEL_BY_ID[modelId]!;
  return { model, gpu: GPU_BY_ID[gpuId]!, quant, count, weightsGB: weightsGB(model, quant)!.gb, ctx };
};

describe('architecture from the Hugging Face configs', () => {
  it('Qwen3.8-27B: 16 of 64 layers hold a KV cache, 4 KV heads of 256 → 32 KB a token, plus a recurrent state', () => {
    const net = MODEL_BY_ID['qwen38_27b']!.net!;
    expect(net.fullTokBytes).toBe(16 * 2 * 4 * 256);
    expect(net.stateBytes / 1e6).toBeGreaterThan(140);
    expect(net.stateBytes / 1e6).toBeLessThan(170);
  });
  it('the report derived about 30 KB a token for Qwen3.8-27B from its 1.7M-token pool; the config agrees within 10%', () => {
    expect(Math.abs(MODEL_BY_ID['qwen38_27b']!.net!.fullTokBytes / 30_300 - 1)).toBeLessThan(0.1);
  });
  it('Llama 3.3 70B costs 160 KB a token, five times Qwen3.8-27B', () => {
    expect(MODEL_BY_ID['llama33_70b']!.net!.fullTokBytes).toBe(80 * 2 * 8 * 128);
  });
  it('DeepSeek V4.1 Flash: the model card\'s 890 B a token, with the 196 GB Engram table counted out of the decode read', () => {
    const m = MODEL_BY_ID['dsv41flash']!;
    expect(m.net!.fullTokBytes).toBe(890);
    expect(m.net!.kvSource).toBe('model-card');
    expect(m.net!.lookupGB).toBe(196);
    expect(m.net!.params!.activePrefill!).toBeLessThan(m.net!.params!.active);
    const w = weightsGB(m, 'nvfp4')!.gb;
    expect(w).toBeGreaterThan(420);
  });
  it('sparse-attention models read a bounded slice: GLM-5.3 reads the top 2048 tokens, MiniMax-M3 17 blocks of 128', () => {
    const glm = MODEL_BY_ID['glm53']!.net!;
    const mm = MODEL_BY_ID['minimaxm3']!.net!;
    expect(glm.readCapTokens).toBe(2048);
    expect(mm.readCapTokens).toBe(17 * 128);
    // Past the cap a longer context adds only the index scan, not attention reads.
    const a = kvReadBytesPerSeq(mm, 8192);
    const b = kvReadBytesPerSeq(mm, 65536);
    expect(b / a).toBeLessThan(65536 / 8192);
  });
  it('sliding-window layers stop growing at the window, so Gemma 4 31B costs the same at 8K and 64K for them', () => {
    const net = MODEL_BY_ID['gemma4_31b']!.net!;
    const at = (ctx: number) => kvBytesPerSeq(net, ctx) - net.fullTokBytes * ctx;
    expect(at(8192)).toBeCloseTo(at(65536), 0);
  });
  it('active parameters from the config land near the published figures', () => {
    const active = (id: string) => MODEL_BY_ID[id]!.net!.params!.active / 1e9;
    expect(active('gptoss_120b')).toBeGreaterThan(5.0);
    expect(active('gptoss_120b')).toBeLessThan(5.4);
    expect(active('gptoss_20b')).toBeGreaterThan(3.4);
    expect(active('gptoss_20b')).toBeLessThan(3.8);
    expect(active('qwen36_35b')).toBeGreaterThan(2.7);
    expect(active('qwen36_35b')).toBeLessThan(3.3);
  });
  it('gives every model a config-derived KV figure, none left as a placeholder', () => {
    for (const m of MODELS) {
      expect(m.net).not.toBeNull();
      expect(['sourced', 'derived']).toContain(m.kvBytesPerTok.p.kind);
      // A derived figure is one the model's config alone doesn't settle; it carries the assumption in its note.
      if (m.kvBytesPerTok.p.kind === 'derived') expect(m.kvBytesPerTok.p.note).toBeTruthy();
    }
  });
  it('the original catalog models take their KV figure straight from the config', () => {
    for (const m of MODELS.filter((m) => !m.liveOnly)) expect(m.kvBytesPerTok.p.kind, m.id).toBe('sourced');
  });
  it('a model with a sliding window and recurrent state needs more than tokens × KV bytes per token', () => {
    const m = MODEL_BY_ID['qwen38_27b']!;
    expect(seqKvBytes(m, 8192)).toBeGreaterThan(m.kvBytesPerTok.v * 8192);
  });
});

describe('roofline decode', () => {
  it('is memory-bound at small batch and spends about weights ÷ bandwidth per step', () => {
    const i = inputs('qwen38_27b', 'pro6000', 'nvfp4');
    const s = decodeStep(i, 1);
    expect(s.bound).toBe('memory');
    const ideal = (weightsGB(i.model, 'nvfp4')!.gb * 1e9) / (pro.bwGBs.v * 1e9);
    expect(s.tMem).toBeGreaterThan(ideal * 0.7);
    expect(s.tMem).toBeLessThan(ideal * 1.6);
  });
  it("sits below the report's measured 862 tok/s for Qwen3.8-27B at 16 concurrent, because it is fitted to InferenceX's runs, not to that point", () => {
    const r = decodeTps(inputs('qwen38_27b', 'pro6000', 'nvfp4'), 16) / 862;
    expect(r).toBeGreaterThan(0.5);
    expect(r).toBeLessThan(0.9);
  });
  it("puts the report's gpt-oss estimates for one PRO 6000 well above what the fitted roofline allows", () => {
    expect(decodeTps(inputs('gptoss_120b', 'pro6000', 'mxfp4'), 50) / 1525).toBeLessThan(0.8);
    expect(decodeTps(inputs('gptoss_20b', 'pro6000', 'mxfp4'), 50) / 3753).toBeLessThan(1.0);
  });
  it("shows the report's Llama 3.3 70B figure of 700 tok/s at 16 concurrent is above what the memory bus allows", () => {
    const i = inputs('llama33_70b', 'pro6000', 'nvfp4');
    const ceiling = 16 / ((i.weightsGB * 1e9) / (pro.bwGBs.v * 1e9));
    expect(ceiling).toBeLessThan(700 * 1.05);
    expect(decodeTps(i, 16)).toBeLessThan(700);
  });
  it('a longer context slows decode because more cache is read each step', () => {
    const short = decodeTps(inputs('llama33_70b', 'pro6000', 'nvfp4', 1, 2000), 16);
    const long = decodeTps(inputs('llama33_70b', 'pro6000', 'nvfp4', 1, 30000), 16);
    expect(long).toBeLessThan(short);
  });
  it('a mixture-of-experts model reads more experts as the batch grows', () => {
    const net = MODEL_BY_ID['qwen36_35b']!.net;
    expect(expertCoverage(net, 1)).toBeLessThan(expertCoverage(net, 16));
    expect(expertCoverage(net, 1000)).toBeCloseTo(1, 2);
    expect(expertCoverage(MODEL_BY_ID['qwen38_27b']!.net, 8)).toBe(1);
  });
  it('splitting a model over two GPUs speeds each step but pays the all-reduce, more on PCIe than NVLink', () => {
    const one = decodeStep(inputs('llama33_70b', 'pro6000', 'nvfp4', 1), 16);
    const two = decodeStep(inputs('llama33_70b', 'pro6000', 'nvfp4', 2), 32);
    expect(two.tComm).toBeGreaterThan(0);
    expect(two.t).toBeLessThan(one.t * 1.5);
    const nv = decodeStep(inputs('llama33_70b', 'h100', 'fp8', 2), 32);
    const pcie = decodeStep({ ...inputs('llama33_70b', 'h100', 'fp8', 2), gpu: { ...GPU_BY_ID['h100']!, link: 'pcie' } }, 32);
    expect(pcie.tComm).toBeGreaterThan(nv.tComm);
  });
  it('weight-only 4-bit formats run their maths at 16 bits on cards without FP4', () => {
    expect(computeFormat(GPU_BY_ID['h100']!, 'mxfp4').format).toBe('fp16');
    expect(computeFormat(GPU_BY_ID['pro6000']!, 'nvfp4').format).toBe('fp4');
    expect(computeFormat(GPU_BY_ID['pro6000']!, 'fp8').format).toBe('fp8');
  });
});

describe('prefill and time to first token', () => {
  it('prefill is compute-bound and slower for a bigger model', () => {
    const small = prefillRate(inputs('qwen36_35b', 'pro6000', 'fp8'), 8192);
    const big = prefillRate(inputs('llama33_70b', 'pro6000', 'nvfp4'), 8192);
    expect(small.bound).toBe('compute');
    expect(small.tokPerSec).toBeGreaterThan(big.tokPerSec);
  });
  it('a PRO 6000 prefills a 27B model at some 10-25 thousand tokens a second', () => {
    const r = prefillRate(inputs('qwen38_27b', 'pro6000', 'nvfp4'), 8192);
    expect(r.tokPerSec).toBeGreaterThan(10_000);
    expect(r.tokPerSec).toBeLessThan(25_000);
  });
  it('splits a request into output, prompt, uncached prompt and decode context', () => {
    const p = requestParts({ avgTokens: 10_000, outShare: 0.1, cacheHit: 0.5 });
    expect(p.out).toBe(1000);
    expect(p.input).toBe(9000);
    expect(p.fresh).toBe(4500);
    expect(p.ctx).toBe(9500);
  });
  it('prefill takes a larger share of the node when prompts are long and little is cached', () => {
    const i = inputs('qwen38_27b', 'pro6000', 'nvfp4');
    const light = servingPoint(i, { avgTokens: 4000, outShare: 0.25, cacheHit: 0.8 }, 16);
    const heavy = servingPoint(i, { avgTokens: 60_000, outShare: 0.02, cacheHit: 0.2 }, 16);
    expect(heavy.share).toBeGreaterThan(light.share);
    expect(heavy.tps).toBeLessThan(light.tps);
    expect(heavy.ttftP90).toBeGreaterThan(light.ttftP90);
  });
  it('the share is a/(1+a) and the effective rate is what decode leaves', () => {
    const sp = servingPoint(inputs('qwen38_27b', 'pro6000', 'nvfp4'), { avgTokens: 16000, outShare: 0.04, cacheHit: 0.5 }, 16);
    expect(sp.share).toBeCloseTo(sp.a / (1 + sp.a), 10);
    expect(sp.tps).toBeCloseTo(sp.decodeTps * (1 - sp.share), 6);
  });
});

describe('throughput bases', () => {
  const model = MODEL_BY_ID['qwen38_27b']!;
  const observed = { avgTokens: 16118, outShare: 0.041, cacheHit: 0.73 };
  const base = { model, gpu: pro, proGpu: pro, count: 1, quant: 'nvfp4' as const, mtp: false, extraMult: 1, shape: observed, qos: { mode: 'report' as const, floorTps: 40 } };
  it("'adjusted' equals the report's measured number when the request is the one the point describes", () => {
    const rep = capacityFor({ ...base, basis: 'report' });
    const adj = capacityFor({ ...base, basis: 'adjusted', baseShape: observed });
    expect(adj.evidence.kind).toBe('report-measured');
    expect(adj.tpsNode).toBeCloseTo(rep.tpsNode, 6);
  });
  it("'adjusted' lowers throughput for a prompt-heavy agentic request and raises it for a short chat", () => {
    const rep = capacityFor({ ...base, basis: 'report' }).tpsNode;
    const agent = capacityFor({ ...base, basis: 'adjusted', baseShape: observed, shape: { avgTokens: 60_000, outShare: 0.02, cacheHit: 0.2 } });
    const chat = capacityFor({ ...base, basis: 'adjusted', baseShape: observed, shape: { avgTokens: 3000, outShare: 0.3, cacheHit: 0.1 } });
    expect(agent.tpsNode).toBeLessThan(rep);
    expect(chat.tpsNode).toBeGreaterThan(rep);
  });
  it('off the report basis, no operating point holds more users than the KV pool does, for any model, card and node', () => {
    for (const m of MODELS.filter((m) => m.economics)) {
      for (const g of GPUS.filter((g) => g.role === 'serve')) {
        for (const count of [1, 2, 4, 8]) {
          for (const basis of ['adjusted', 'roofline'] as const) {
            const c = capacityFor({ ...base, model: m, gpu: g, count, quant: 'auto', basis, qos: { mode: 'report' as const, floorTps: 40 } });
            if (!c.ok) continue;
            expect(c.cPerGpu, `${m.id} on ${count}× ${g.id} (${basis})`).toBeLessThanOrEqual(Math.max(1, c.kvConcPerGpu) * 1.0001);
          }
        }
      }
    }
  });
  it('a card the cache cannot fill at 16 users stops at what it holds and says the cache is the limit (Qwen3.8-27B on a 5090)', () => {
    const c = capacityFor({ ...base, gpu: GPU_BY_ID['rtx5090']!, basis: 'adjusted', baseShape: observed });
    expect(c.ok).toBe(true);
    expect(c.kvLimited).toBe(true);
    expect(c.cPerGpu).toBeCloseTo(c.kvConcPerGpu, 6);
    expect(c.cPerGpu).toBeLessThan(16);
    // The report's own points are left as printed.
    const rep = capacityFor({ ...base, gpu: GPU_BY_ID['rtx5090']!, basis: 'report' });
    expect(rep.cPerGpu).toBe(16);
  });
  it("'roofline' needs no published points and works for a model the report has none for", () => {
    const m = MODEL_BY_ID['qwen35_122b']!;
    const c = capacityFor({ ...base, model: m, quant: 'nvfp4', basis: 'roofline', qos: { mode: 'floor', floorTps: 10 } });
    expect(c.curve).toBeNull();
    expect(c.basis).toBe('roofline');
  });
  it('without published points an asked-for report basis falls back to the roofline and says so', () => {
    const m = MODEL_BY_ID['qwen35_122b']!;
    const c = capacityFor({ ...base, model: m, quant: 'nvfp4', basis: 'report', qos: { mode: 'floor', floorTps: 10 } });
    if (c.ok) {
      expect(c.basis).toBe('roofline');
      expect(c.basisNote).toMatch(/roofline/);
    }
  });
  it('a time-to-first-token limit lowers the operating point when prompts are long', () => {
    const long = { avgTokens: 30_000, outShare: 0.02, cacheHit: 0 };
    const b = { ...base, shape: long, basis: 'roofline' as const, qos: { mode: 'floor' as const, floorTps: 5 } };
    const loose = capacityFor({ ...b, qos: { ...b.qos, ttftP90Sec: 1000 } });
    const tight = capacityFor({ ...b, qos: { ...b.qos, ttftP90Sec: 4 } });
    expect(tight.cPerGpu).toBeLessThan(loose.cPerGpu);
    expect(tight.ttftLimited).toBe(true);
    expect(tight.serving!.ttftP90).toBeLessThanOrEqual(4.01);
  });
  it('when even one user cannot get a first token in time the card is flagged as below the bar', () => {
    const long = { avgTokens: 120_000, outShare: 0.01, cacheHit: 0 };
    const c = capacityFor({ ...base, shape: long, basis: 'roofline', qos: { mode: 'floor', floorTps: 5, ttftP90Sec: 5 } });
    expect(c.belowFloor).toBe(true);
    expect(c.cPerGpu).toBe(1);
  });
  it('batch work is not held to a time-to-first-token limit', () => {
    const long = { avgTokens: 120_000, outShare: 0.01, cacheHit: 0 };
    const b = { ...base, shape: long, basis: 'roofline' as const, qos: { mode: 'floor' as const, floorTps: 5, ttftP90Sec: 0.5 } };
    expect(capacityFor({ ...b, batch: true }).cPerGpu).toBeGreaterThanOrEqual(capacityFor(b).cPerGpu);
  });
  it('the PRO 6000 Max-Q loses prefill speed, so prompt-heavy traffic gets less out of it', () => {
    const b = { ...base, basis: 'roofline' as const, shape: { avgTokens: 60_000, outShare: 0.02, cacheHit: 0.2 } };
    expect(capacityFor({ ...b, prefillMult: 0.62 }).tpsNode).toBeLessThan(capacityFor(b).tpsNode);
  });
  it('a KV pool bounds concurrency by what one request costs, recurrent state included', () => {
    const c = capacityFor({ ...base, basis: 'roofline', qos: { mode: 'floor', floorTps: 1 } });
    const f = fitOn(model, pro, 1, 'nvfp4');
    expect(c.kvConcPerGpu).toBeCloseTo((f.kvPoolGB * 1e9) / seqKvBytes(model, requestParts(observed).ctx), 3);
    expect(c.kvLimited || c.cPerGpu <= c.kvConcPerGpu).toBe(true);
  });
  it("'adjusted' is the fitted roofline scaled by how far the measurement sits from it", () => {
    const adj = capacityFor({ ...base, basis: 'adjusted', baseShape: observed });
    const physics = capacityFor({ ...base, basis: 'roofline' });
    expect(adj.tpsNode / physics.tpsNode).toBeCloseTo(adj.evidence.k, 6);
    expect(adj.evidence.k).toBeGreaterThan(1);
  });
  it('without a measurement for the card, adjusted is the roofline alone and says so', () => {
    const gemma = MODEL_BY_ID['gemma4_26b']!;
    const c = capacityFor({ ...base, model: gemma, quant: 'fp8', basis: 'adjusted', shape: { avgTokens: 3000, outShare: 0.06, cacheHit: 0.25 } });
    const r = capacityFor({ ...base, model: gemma, quant: 'fp8', basis: 'roofline', shape: { avgTokens: 3000, outShare: 0.06, cacheHit: 0.25 } });
    expect(c.evidence.kind).toBe('none');
    expect(c.tpsNode).toBeCloseTo(r.tpsNode, 6);
  });
  it('the roofline never delivers more than the memory bus could', () => {
    const llama = MODEL_BY_ID['llama33_70b']!;
    const chat = { avgTokens: 3000, outShare: 0.15, cacheHit: 0.3 };
    const c = capacityFor({ ...base, model: llama, quant: 'nvfp4', basis: 'roofline', shape: chat, qos: { mode: 'floor', floorTps: 20 } });
    const w = weightsGB(llama, 'nvfp4')!.gb;
    const ceiling = bandwidthCeilingTps({ model: llama, gpu: pro, quant: 'nvfp4', count: 1, weightsGB: w, ctx: requestParts(chat).ctx }, c.cNode, true);
    expect(c.tpsNode / c.prefillFactor).toBeLessThanOrEqual(ceiling);
  });
  it("reproduces an InferenceX curve exactly at the concurrencies it measured, and follows the roofline between them", () => {
    const gpt = MODEL_BY_ID['gptoss_120b']!;
    const h200 = GPU_BY_ID['h200']!;
    const curve = bestIxCurve('gptoss_120b', 'h200', 'mxfp4', 8, 8700)!;
    expect(curve).not.toBeNull();
    const shape = { avgTokens: curve.isl + curve.osl, outShare: curve.osl / (curve.isl + curve.osl), cacheHit: 0 };
    const pt = curve.pts[Math.floor(curve.pts.length / 2)]!;
    const c = capacityFor({ ...base, model: gpt, gpu: h200, count: 8, quant: 'mxfp4', basis: 'adjusted', shape, sweepAt: [pt.conc / 8] });
    expect(c.evidence.kind).toBe('inferencex');
    expect(c.sweep[0]!.tps / (pt.outPerGpu * 8)).toBeCloseTo(1, 2);
  });
  it('four PRO 6000s over PCIe: the measured Qwen3.5-397B curve is reproduced and PCIe all-reduce costs real time', () => {
    const q = CAL_MODEL_BY_ID['qwen35_397b']!;
    const curve = bestIxCurve('qwen35_397b', 'pro6000', 'nvfp4', 4, 8700)!;
    expect(curve).not.toBeNull();
    const shape = { avgTokens: curve.isl + curve.osl, outShare: curve.osl / (curve.isl + curve.osl), cacheHit: 0 };
    const pt = curve.pts.find((p) => p.conc === 16)!;
    const c = capacityFor({ ...base, model: q, count: 4, quant: 'nvfp4', basis: 'adjusted', shape, sweepAt: [pt.conc / 4] });
    expect(c.sweep[0]!.tps / (pt.outPerGpu * 4)).toBeCloseTo(1, 2);
    const step = decodeStep({ model: q, gpu: pro, quant: 'nvfp4', count: 4, weightsGB: 251.2, ctx: 8700 }, 16);
    expect(step.tComm).toBeGreaterThan(step.tMem * 0.2);
  });
  it('the bus ceiling drops when the cache each user reads is counted', () => {
    const i = inputs('llama33_70b', 'pro6000', 'nvfp4');
    expect(bandwidthCeilingTps(i, 16, true)).toBeLessThan(bandwidthCeilingTps(i, 16, false));
  });
});
