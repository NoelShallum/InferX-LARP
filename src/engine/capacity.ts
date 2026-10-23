import { kvBytesPerSeq } from '../data/arch';
import type { Gpu, Quant } from '../data/gpus';
import { BYTES_PER_PARAM, type Model } from '../data/models';
import { MARKET_BY_MODEL, RENT_REFS } from '../data/market';
import { type Prov, ph, drv } from '../data/provenance';
import { calibrationFor } from './calibration';
import { bestIxCurve, ixProv, type IxCurve } from '../data/ix';
import { decodeTps, requestParts, roofClassOf, roofParamsFor, servingPoint, type Bound, type ServingPoint, type ShapeIn } from './roofline';

export const GPU_MEM_UTIL = ph(0.9, 'vLLM default gpu_memory_utilization', 'Share of VRAM the serving engine may use');
export const OVERHEAD_GB_PER_GPU = ph(4, 'No source yet', 'CUDA context, activations and graph capture per GPU; the report does not give one');
export const MIN_KV_TOKENS = ph(32768, 'No source yet', 'A model counts as fitting only if the KV pool holds at least one 32K-token user');
export const DRAFT_COST = ph(0.08, 'No source yet', 'Cost of one draft step relative to a decode step');
export const MTP_DEFAULT_CSAT = ph(64, 'No source yet', 'Concurrency at which speculative decoding stops helping when nothing was measured');
export const VOLTA_KERNEL_PENALTY = ph(0.6, 'No source yet', 'Older attention and GEMM kernels on Volta (no FlashAttention-2, no FP8)');
export const MAX_EXTRAPOLATION = ph(1.6, 'No source yet', 'How far past the last measured concurrency the report-anchored curve is trusted');
export const ROOFLINE_MAX_CONC = ph(512, 'No source yet', 'Largest concurrency the physics-only curve is searched to');

/** Where throughput comes from: the report's points as printed, the report's points adjusted to the request, or the roofline alone. */
export type ThroughputBasis = 'report' | 'adjusted' | 'roofline';

const FALLBACK_ORDER: Quant[] = ['fp8', 'nvfp4', 'mxfp4', 'awq4', 'fp16'];

export interface WeightsInfo {
  gb: number;
  prov: Prov;
}

export function weightsGB(model: Model, quant: Quant): WeightsInfo | null {
  const listed = model.weights[quant];
  if (listed) return { gb: listed.v, prov: listed.p };
  // 4-bit float formats need a published checkpoint, so they are never derived.
  if (quant === 'nvfp4' || quant === 'mxfp4') return null;
  const bpp = BYTES_PER_PARAM[quant];
  return {
    gb: model.paramsB.v * bpp.v,
    prov: { kind: 'derived', src: 'Parameters × bytes per parameter', note: `${model.paramsB.v}B × ${bpp.v} bytes at ${quant}; the report lists no ${quant} build` },
  };
}

/** Bytes one sequence of `ctx` tokens holds in the cache, recurrent state and sliding windows included. */
export function seqKvBytes(model: Model, ctx: number, kvElt = 1): number {
  return model.net ? kvBytesPerSeq(model.net, ctx, kvElt) : model.kvBytesPerTok.v * ctx * kvElt;
}

export interface Fit {
  fits: boolean;
  reason: string;
  quant: Quant;
  weightsGB: number;
  weightsProv: Prov;
  usableGB: number;
  overheadGB: number;
  kvPoolGB: number;
  /** Context tokens the pool holds if every one costs the full-attention rate. */
  kvTokens: number;
  forced: boolean;
}

/** Sequences of `ctx` tokens a fit's KV pool holds, across all its GPUs. */
export function kvUsersAt(model: Model, fit: Fit, ctx: number, kvElt = 1): number {
  return (fit.kvPoolGB * 1e9) / Math.max(1, seqKvBytes(model, ctx, kvElt));
}

export function fitOn(model: Model, gpu: Gpu, count: number, quant: Quant): Fit {
  const w = weightsGB(model, quant);
  const usable = GPU_MEM_UTIL.v * gpu.vramGB.v * count;
  const overhead = OVERHEAD_GB_PER_GPU.v * count;
  const base = { quant, usableGB: usable, overheadGB: overhead, forced: false };
  if (!w) {
    return { ...base, fits: false, reason: `No ${quant} checkpoint is listed for ${model.name}`, weightsGB: 0, weightsProv: model.paramsB.p, kvPoolGB: 0, kvTokens: 0 };
  }
  const kvPool = usable - overhead - w.gb;
  const kvTokens = Math.max(0, (kvPool * 1e9) / model.kvBytesPerTok.v);
  const common = { ...base, weightsGB: w.gb, weightsProv: w.prov, kvPoolGB: Math.max(0, kvPool), kvTokens };
  if (!gpu.quants.includes(quant)) {
    return { ...common, fits: false, reason: `${gpu.short} has no native ${quant} support` };
  }
  const forced = model.forceFit?.gpuIds.includes(gpu.id) && count === 1;
  if (forced) return { ...common, fits: true, reason: model.forceFit!.why, forced: true, kvTokens: Math.max(kvTokens, MIN_KV_TOKENS.v) };
  if (kvPool <= 0) return { ...common, fits: false, reason: `Weights (${w.gb.toFixed(1)} GB) exceed the usable ${(usable - overhead).toFixed(0)} GB` };
  const need = seqKvBytes(model, MIN_KV_TOKENS.v) / 1e9;
  if (kvPool < need) {
    return { ...common, fits: false, reason: `Only ${kvPool.toFixed(1)} GB left for KV, under the ${need.toFixed(1)} GB one 32K user needs` };
  }
  return { ...common, fits: true, reason: 'Fits' };
}

export function quantOrder(model: Model, wanted: Quant | 'auto'): Quant[] {
  if (wanted !== 'auto') return [wanted];
  return [model.defaultQuant, ...FALLBACK_ORDER].filter((q, i, a) => a.indexOf(q) === i);
}

export function chooseQuant(model: Model, gpu: Gpu, count: number, wanted: Quant | 'auto'): Fit {
  const order = quantOrder(model, wanted);
  let firstSupported: Fit | null = null;
  for (const q of order) {
    const f = fitOn(model, gpu, count, q);
    if (f.fits) return f;
    if (!firstSupported && (gpu.quants.includes(q) || wanted !== 'auto')) firstSupported = f;
  }
  return firstSupported ?? fitOn(model, gpu, count, order[0] ?? model.defaultQuant);
}

export interface Curve {
  pts: { c: number; tps: number }[];
  k: number;
  beta: number;
  cRef: number;
  mult: number;
  source: 'pro6000' | 'ada' | 'scaled';
  multNote: string;
  multProv: Prov;
  baseProvs: Prov[];
  singlePoint: boolean;
}

/** The request the report's throughput points are taken to correspond to when the caller names none: the model's average request in the report. */
export function reportShape(model: Model): ShapeIn | null {
  const cal = calibrationFor(model);
  const mr = MARKET_BY_MODEL[model.id];
  return cal && mr ? { avgTokens: cal.avgTokens, outShare: cal.outShare, cacheHit: mr.cacheHit.v } : null;
}

/** Context length the report's throughput points were measured at, used when moving them between cards and weight formats. */
export const REPORT_TEST_CTX = ph(8700, 'Report §5.1', 'The report measures at 8K prompts; half an average output is added');

/** Speed of one config relative to another on the same model, from the roofline, at concurrency `c` per GPU. */
function rooflineRatio(model: Model, target: { gpu: Gpu; quant: Quant }, anchor: { gpu: Gpu; quant: Quant }, c: number): number | null {
  const wt = weightsGB(model, target.quant);
  const wa = weightsGB(model, anchor.quant);
  if (!wt || !wa) return null;
  const ctx = REPORT_TEST_CTX.v;
  const t = decodeTps({ model, gpu: target.gpu, quant: target.quant, count: 1, weightsGB: wt.gb, ctx }, c);
  const b = decodeTps({ model, gpu: anchor.gpu, quant: anchor.quant, count: 1, weightsGB: wa.gb, ctx }, c);
  return t / b;
}

function fitPower(pts: { c: number; tps: number }[]): { k: number; beta: number } {
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  if (pts.length === 1) return { k: first.tps / first.c, beta: 1 };
  const beta = Math.log(last.tps / first.tps) / Math.log(last.c / first.c);
  return { k: first.tps / Math.pow(first.c, beta), beta };
}

export function curveFor(model: Model, gpu: Gpu, quant: Quant, proGpu: Gpu): Curve | null {
  const proPts = model.pro6000;
  if (proPts.length === 0) return null;
  let pts: { c: number; tps: number }[];
  let mult = 1;
  let source: Curve['source'];
  let multNote = 'Measured or estimated points used as published';
  let multProv: Prov = { kind: 'report', src: 'Report §5.2' };
  let baseProvs: Prov[];
  let refIdx = model.refIdx;
  const refC = proPts[Math.min(refIdx, proPts.length - 1)]!.c;

  const quantRatio = (from: Quant) => {
    if (from === quant) return null;
    const r = rooflineRatio(model, { gpu, quant }, { gpu, quant: from }, refC);
    return r === null ? null : Math.min(3, Math.max(0.3, r));
  };
  const quantNote = (from: Quant) => `Roofline: the points were taken at ${from} and this runs at ${quant}; the weights and cache a step reads set the speed`;
  const quantProv: Prov = { kind: 'derived', src: 'Roofline', note: 'Decode is memory-bound, so speed follows the bytes a step reads (weights plus cache)' };

  if (gpu.id === 'ada6000' && model.ada.length > 0) {
    pts = model.ada.map((p) => ({ c: p.c, tps: p.tps.v }));
    baseProvs = model.ada.map((p) => p.tps.p);
    source = 'ada';
    const r = quantRatio(model.pointsQuant.ada);
    refIdx = Math.min(refIdx, pts.length - 1);
    if (r !== null && r !== 1) {
      mult = r;
      multNote = quantNote(model.pointsQuant.ada);
      multProv = quantProv;
    }
  } else if (gpu.id === 'pro6000') {
    pts = proPts.map((p) => ({ c: p.c, tps: p.tps.v }));
    baseProvs = proPts.map((p) => p.tps.p);
    source = 'pro6000';
    const r = quantRatio(model.pointsQuant.pro6000);
    if (r !== null && r !== 1) {
      mult = r;
      multNote = quantNote(model.pointsQuant.pro6000);
      multProv = quantProv;
    }
  } else {
    pts = proPts.map((p) => ({ c: p.c, tps: p.tps.v }));
    baseProvs = proPts.map((p) => p.tps.p);
    source = 'scaled';
    const refSelf = RENT_REFS.find((r) => r.gpuId === gpu.id && r.modelId === model.id);
    const refPro = RENT_REFS.find((r) => r.gpuId === 'pro6000' && r.modelId === model.id);
    if (refSelf && refPro) {
      mult = refSelf.busyHourMedianUsd / refPro.busyHourMedianUsd;
      multNote = `Report §7.1 revenue per busy hour: ${refSelf.busyHourMedianUsd} vs ${refPro.busyHourMedianUsd} dollars on a PRO 6000`;
      multProv = { kind: 'derived', src: 'Report §7.1', note: 'Ratio of revenue per busy hour at median prices' };
    } else {
      const r = rooflineRatio(model, { gpu, quant }, { gpu: proGpu, quant: model.pointsQuant.pro6000 }, refC);
      mult = r ?? 1;
      if (gpu.arch === 'Volta') mult *= VOLTA_KERNEL_PENALTY.v;
      multNote = `Roofline ratio to a PRO 6000 (${gpu.bwGBs.v} against ${proGpu.bwGBs.v} GB/s, weights and cache read per step)${gpu.arch === 'Volta' ? ' × Volta kernel penalty' : ''}`;
      multProv =
        gpu.arch === 'Volta'
          ? { kind: 'placeholder', src: 'Roofline and kernel penalty', note: 'Roofline ratio times a placeholder penalty for Volta kernels' }
          : { kind: 'derived', src: 'Roofline', note: 'Decode speed follows the bytes a step reads. The kernel efficiency fitted on the PRO 6000 is assumed to carry over to this card; no measurement checks that.' };
    }
  }
  const cal = calibrationFor(model);
  if (cal && Math.abs(cal.tpsScale - 1) > 0.005) {
    mult *= cal.tpsScale;
    multNote += `; ×${cal.tpsScale.toFixed(2)} to reconcile §5.2 with the report's §6.1 busy-hour revenue`;
    multProv = { kind: 'derived', src: 'Report §6.1 and §6.3', note: 'The §5.2 throughput point and the report\'s revenue figures disagree for this model; revenue figures win' };
  }
  pts = pts.map((p) => ({ c: p.c, tps: p.tps * mult })).sort((a, b) => a.c - b.c);
  const { k, beta } = fitPower(pts);
  return { pts, k, beta, cRef: pts[Math.min(refIdx, pts.length - 1)]!.c, mult, source, multNote, multProv, baseProvs, singlePoint: pts.length === 1 };
}

export function tpsAt(curve: Curve, c: number): number {
  const cLast = curve.pts[curve.pts.length - 1]!.c;
  const cap = cLast * MAX_EXTRAPOLATION.v;
  if (curve.singlePoint) {
    const p = curve.pts[0]!;
    if (c <= p.c) return p.tps * (c / p.c);
    return p.tps * Math.pow(c / p.c, 0.3);
  }
  const cc = Math.min(c, cap);
  return curve.k * Math.pow(cc, curve.beta);
}

export interface MtpInfo {
  s1: number;
  cSat: number;
  calibrated: boolean;
}

/** Plain (no speculative decoding) aggregate tokens per second on one PRO 6000 at concurrency c. */
export type PlainTps = (c: number) => number;

export function mtpInfo(model: Model, proCurve: Curve | PlainTps | null): MtpInfo | null {
  const m = model.mtp;
  if (!m) return null;
  const s1 = m.al.v / (1 + m.k * DRAFT_COST.v);
  if (s1 <= 1) return { s1: 1, cSat: Infinity, calibrated: false };
  if (m.measured && proCurve) {
    const plain = typeof proCurve === 'function' ? proCurve(m.measured.c) : tpsAt(proCurve, m.measured.c);
    const gain = m.measured.tps.v / plain;
    if (gain <= 1) return { s1, cSat: 0, calibrated: true };
    if (gain >= s1) return { s1, cSat: Infinity, calibrated: true };
    return { s1, cSat: m.measured.c / (1 - (gain - 1) / (s1 - 1)), calibrated: true };
  }
  return { s1, cSat: MTP_DEFAULT_CSAT.v, calibrated: false };
}

export function mtpGain(info: MtpInfo | null, cPerGpu: number): number {
  if (!info) return 1;
  if (!Number.isFinite(info.cSat)) return info.s1;
  return 1 + (info.s1 - 1) * Math.max(0, 1 - cPerGpu / info.cSat);
}

export interface QosInput {
  mode: 'report' | 'floor';
  floorTps: number;
  /** 90th-percentile time to first token the operating point must stay under, in seconds. Not applied on the report's own numbers. */
  ttftP90Sec?: number;
}

/** What anchors the speed: a measurement, or nothing but the fitted roofline. */
export interface CapacityEvidence {
  kind: 'inferencex' | 'report-measured' | 'report-points' | 'none';
  label: string;
  prov: Prov | null;
  /** How far the measurement sits above (>1) or below (<1) the fitted roofline at the operating point. */
  k: number;
  curve: IxCurve | null;
}

export interface Capacity {
  ok: boolean;
  reason: string;
  fit: Fit;
  quant: Quant;
  curve: Curve | null;
  basis: ThroughputBasis;
  /** Why the basis differs from the one asked for, if it does. */
  basisNote: string;
  cPerGpu: number;
  cNode: number;
  tpsNode: number;
  /** Tokens per second one user sees. */
  userTps: number;
  mtp: { on: boolean; gain: number; info: MtpInfo | null };
  /** Throughput kept per GPU when the model is split over several, from the roofline. */
  eff: number;
  extrapolated: boolean;
  belowFloor: boolean;
  /** Concurrent users the KV pool holds at the request's context length, per GPU. */
  kvConcPerGpu: number;
  kvLimited: boolean;
  /** True when the time-to-first-token limit, not the speed floor, set the operating point. */
  ttftLimited: boolean;
  serving: ServingPoint | null;
  /** Throughput kept after prefill takes its time (against the report's request when adjusting). */
  prefillFactor: number;
  /** Throughput kept after this request's context costs more or less cache than the report's request. */
  contextFactor: number;
  bound: Bound | null;
  multipliers: { name: string; value: number; prov: Prov }[];
  /** Curve through the operating range, when `sweepSteps` was asked for. */
  sweep: SweepPoint[];
  evidence: CapacityEvidence;
}

export interface CapacityArgs {
  model: Model;
  gpu: Gpu;
  proGpu: Gpu;
  count: number;
  quant: Quant | 'auto';
  mtp: boolean;
  qos: QosInput;
  /** Product of what-if, edition and heat factors. */
  extraMult: number;
  extraNotes?: { name: string; value: number; prov: Prov }[];
  /** The request this slot serves. */
  shape: ShapeIn;
  /** The request the report's throughput points correspond to. Defaults to the model's average request in the report. */
  baseShape?: ShapeIn;
  /** Defaults to the report's own numbers. */
  basis?: ThroughputBasis;
  /** Multiplies prefill speed (the PRO 6000 Max-Q loses prefill speed far more than decode). */
  prefillMult?: number;
  /** Batch work is not latency-bound, so time to first token is not enforced. */
  batch?: boolean;
  /** Also evaluate this many log-spaced concurrencies from one user to the most the node can hold, for plotting. */
  sweepSteps?: number;
  /** Or evaluate exactly these per-GPU concurrencies. */
  sweepAt?: number[];
}

export interface SweepPoint {
  /** Concurrent requests per GPU. */
  c: number;
  /** Aggregate output tokens per second, all GPUs. */
  tps: number;
  /** Output tokens per second one user sees. */
  user: number;
  ttftP90: number;
  /** Share of GPU time spent on prefill. */
  share: number;
}

export const TTFT_DEFAULT_SEC = 10;

export function capacityFor(a: CapacityArgs): Capacity {
  const fit = chooseQuant(a.model, a.gpu, a.count, a.quant);
  const requested: ThroughputBasis = a.basis ?? 'report';
  const empty = (reason: string): Capacity => ({
    ok: false,
    reason,
    fit,
    quant: fit.quant,
    curve: null,
    basis: requested,
    basisNote: '',
    cPerGpu: 0,
    cNode: 0,
    tpsNode: 0,
    userTps: 0,
    mtp: { on: false, gain: 1, info: null },
    eff: 1,
    extrapolated: false,
    belowFloor: false,
    kvConcPerGpu: 0,
    kvLimited: false,
    ttftLimited: false,
    serving: null,
    prefillFactor: 1,
    contextFactor: 1,
    bound: null,
    multipliers: [],
    sweep: [],
    evidence: { kind: 'none', label: '', prov: null, k: 1, curve: null },
  });
  if (!fit.fits) return empty(fit.reason);
  const curve = curveFor(a.model, a.gpu, fit.quant, a.proGpu);
  let basis = requested;
  let basisNote = '';
  if (!curve && requested === 'report') {
    if (!a.model.net) return empty(`The report has no throughput data for ${a.model.name}`);
    basis = 'roofline';
    basisNote = `The report has no throughput points for ${a.model.name}, so speed comes from the roofline alone`;
  }
  if (!curve && !a.model.net) return empty(`No throughput data for ${a.model.name}`);

  const n = a.count;
  const shape = a.shape;
  const base0 = a.baseShape ?? reportShape(a.model) ?? shape;
  const parts = requestParts(shape);
  const parts0 = requestParts(base0);
  const roofIn = { model: a.model, gpu: a.gpu, quant: fit.quant, count: n, weightsGB: fit.weightsGB };
  const prefillMult = a.prefillMult ?? 1;
  const pBase = roofParamsFor(a.gpu);
  // The PRO 6000 Max-Q loses prefill speed far more than decode.
  const pSlot = prefillMult === 1 ? pBase : { ...pBase, etaPrefill: pBase.etaPrefill * prefillMult, etaPrefillMoe: pBase.etaPrefillMoe * prefillMult };

  // What anchors the roofline: an InferenceX run of exactly this config, else the report's measured point for this card, else nothing.
  const refC = curve ? curve.cRef : 16;
  const ix = basis === 'adjusted' ? bestIxCurve(a.model.id, a.gpu.id, fit.quant, n, parts.ctx) : null;
  const reportMeasured = basis === 'adjusted' && !ix && curve && curve.source !== 'scaled' && curve.baseProvs[Math.min(a.model.refIdx, curve.baseProvs.length - 1)]?.kind === 'measured';
  const ixRatios = ix
    ? ix.pts.map((pt) => {
        const sh = { avgTokens: ix.isl + ix.osl, outShare: ix.osl / (ix.isl + ix.osl), cacheHit: 0 };
        return { x: Math.log(pt.conc), y: Math.log((pt.outPerGpu * ix.gpus) / servingPoint(roofIn, sh, pt.conc, pBase).tps) };
      })
    : [];
  const kReport = reportMeasured && curve ? tpsAt(curve, curve.cRef) / servingPoint({ ...roofIn, count: 1 }, base0, refC, pBase).tps : 1;
  const kAt = (B: number): number => {
    if (ixRatios.length) {
      const x = Math.log(Math.max(1, B));
      const first = ixRatios[0]!;
      const last = ixRatios[ixRatios.length - 1]!;
      if (x <= first.x) return Math.exp(first.y);
      if (x >= last.x) return Math.exp(last.y);
      const j = ixRatios.findIndex((r) => r.x >= x);
      const lo = ixRatios[j - 1]!;
      const hi = ixRatios[j]!;
      return Math.exp(lo.y + ((hi.y - lo.y) * (x - lo.x)) / (hi.x - lo.x));
    }
    return reportMeasured ? kReport : 1;
  };

  // Speculative-decoding gain is calibrated against the report's measured point, so its plain speed must come from the same basis.
  const proFit = chooseQuant(a.model, a.proGpu, 1, a.model.pointsQuant.pro6000);
  const proCurve = curveFor(a.model, a.proGpu, proFit.quant, a.proGpu);
  const proIn = { model: a.model, gpu: a.proGpu, quant: proFit.quant, count: 1, weightsGB: proFit.weightsGB };
  const proPlain = (c: number) => servingPoint(proIn, base0, c).tps;
  const proMeasured = proCurve && proCurve.source === 'pro6000' && proCurve.baseProvs[Math.min(a.model.refIdx, proCurve.baseProvs.length - 1)]?.kind === 'measured';
  const kPro = proMeasured && proCurve ? tpsAt(proCurve, proCurve.cRef) / proPlain(proCurve.cRef) : 1;
  const plainPro: PlainTps | null = basis === 'report' ? (proCurve ? (c) => tpsAt(proCurve, c) : null) : (c) => kPro * proPlain(c);
  const info = a.mtp ? mtpInfo(a.model, plainPro) : null;
  const gain = (c: number) => (a.mtp ? mtpGain(info, c) : 1);

  // Tensor-parallel scaling from the roofline at the same per-GPU concurrency, for display and for the report's single-card points.
  const tpEff = (c: number): number => {
    if (n <= 1) return 1;
    const ctx = basis === 'report' ? parts0.ctx : parts.ctx;
    const node = decodeTps({ ...roofIn, ctx }, c * n);
    const one = decodeTps({ ...roofIn, count: 1, ctx }, c);
    return node / (n * one);
  };

  /** Everything at per-GPU concurrency c: speed per user, time to first token, prefill's share. */
  const evalAt = (c: number) => {
    const B = c * n;
    if (basis === 'report') {
      const tps = tpsAt(curve!, c) * n * tpEff(c) * gain(c) * a.extraMult;
      return { tps, user: tps / B, ttftP90: 0, serving: null as ServingPoint | null, prefillFactor: 1, k: 1 };
    }
    const sp = servingPoint(roofIn, shape, B, pSlot);
    const k = kAt(B);
    const decode = k * sp.decodeTps * gain(c) * a.extraMult;
    const tps = decode * (1 - sp.share);
    return { tps, user: tps / B, ttftP90: sp.ttftP90, serving: { ...sp, tps }, prefillFactor: 1 - sp.share, k };
  };

  const cLast = curve ? curve.pts[curve.pts.length - 1]!.c : ROOFLINE_MAX_CONC.v;
  const cMaxCurve = basis === 'report' ? cLast * MAX_EXTRAPOLATION.v : ROOFLINE_MAX_CONC.v;
  const cMaxKv = Math.max(1, (fit.kvPoolGB * 1e9) / Math.max(1, seqKvBytes(a.model, parts.ctx)) / n);
  const cMax = Math.max(1, Math.min(cMaxCurve, cMaxKv));

  const ttftLimit = a.batch || basis === 'report' ? Infinity : a.qos.ttftP90Sec ?? TTFT_DEFAULT_SEC;
  const ok = (c: number) => {
    const e = evalAt(c);
    return e.user >= a.qos.floorTps && e.ttftP90 <= ttftLimit;
  };

  let c: number;
  let belowFloor = false;
  let ttftLimited = false;
  if (a.qos.mode === 'report') {
    c = curve ? curve.cRef : 16;
  } else if (!ok(1)) {
    c = 1;
    belowFloor = true;
    ttftLimited = evalAt(1).user >= a.qos.floorTps;
  } else if (ok(cMax)) {
    c = cMax;
  } else {
    let lo = 1;
    let hi = cMax;
    for (let i = 0; i < 50; i++) {
      const mid = (lo + hi) / 2;
      if (ok(mid)) lo = mid;
      else hi = mid;
    }
    c = lo;
    ttftLimited = evalAt(Math.min(cMax, c * 1.02)).user >= a.qos.floorTps;
  }

  const at = evalAt(c);
  const eff = tpEff(c);
  // Dense prefill on the workstation family has no measurement behind it.
  const prefillIsGuess = roofClassOf(a.gpu) === 'blackwell-ws' && a.model.arch === 'dense';
  const evidence: CapacityEvidence =
    basis === 'report'
      ? { kind: 'report-points', label: "The report's published points", prov: curve?.multProv ?? null, k: 1, curve: null }
      : ix
        ? { kind: 'inferencex', label: `InferenceX measured ${ix.gpus}× ${a.gpu.short}, ${ix.framework}, ${ix.isl / 1024}K in / ${ix.osl / 1024}K out`, prov: ixProv(ix), k: at.k, curve: ix }
        : reportMeasured && curve
          ? { kind: 'report-measured', label: "The report's measured point for this card", prov: curve.baseProvs[Math.min(a.model.refIdx, curve.baseProvs.length - 1)] ?? null, k: at.k, curve: null }
          : { kind: 'none', label: 'No measurement for this card and model: the fitted roofline alone', prov: null, k: 1, curve: null };
  const multipliers = [
    ...(basis === 'report' && curve ? [{ name: curve.source === 'scaled' ? 'GPU scaling' : 'Quant scaling', value: curve.mult, prov: curve.multProv }] : []),
    ...(basis !== 'report' && evidence.kind !== 'none' && evidence.prov ? [{ name: evidence.kind === 'inferencex' ? 'Anchored to InferenceX measurement' : 'Anchored to the report\'s measured point', value: at.k, prov: evidence.prov }] : []),
    ...(a.mtp ? [{ name: 'MTP gain', value: gain(c), prov: { kind: (info?.calibrated ? 'derived' : 'placeholder') as Prov['kind'], src: 'Acceptance length', note: info ? `AL/(1+k·ρ) = ${info.s1.toFixed(2)} at one user, fading to 1 by about ${Number.isFinite(info.cSat) ? Math.round(info.cSat) : '∞'} concurrent` : '' } }] : []),
    ...(n > 1
      ? [{ name: `${n}-GPU scaling`, value: eff, prov: ix ? ixProv(ix) : { kind: 'derived' as const, src: 'Roofline', note: `Weights split over ${n} GPUs; the all-reduce latency and effective ${a.gpu.link === 'nvlink' ? 'NVLink' : 'PCIe'} bandwidth are fitted to InferenceX multi-GPU runs${a.gpu.link === 'pcie' ? ' (four PRO 6000 points only)' : ''}.` } }]
      : []),
    ...(basis !== 'report' && at.serving && Math.abs(at.prefillFactor - 1) > 0.005
      ? [{ name: 'Prefill share', value: at.prefillFactor, prov: { kind: (prefillIsGuess ? 'placeholder' : 'derived') as Prov['kind'], src: 'Roofline', note: `Prefill takes ${(at.serving.share * 100).toFixed(0)}% of GPU time at this load (${Math.round(at.serving.freshTokens)} fresh prompt tokens per request at ${Math.round(at.serving.prefill.tokPerSec)} tok/s). Prefill efficiency is fitted to InferenceX time-to-first-token runs on H100, H200 and B200.` } }]
      : []),
    ...(a.extraNotes ?? []),
  ];
  return {
    ok: true,
    reason: fit.reason,
    fit,
    quant: fit.quant,
    curve: curve ?? null,
    basis,
    basisNote,
    cPerGpu: c,
    cNode: c * n,
    tpsNode: at.tps,
    userTps: at.user,
    mtp: { on: a.mtp && !!info, gain: gain(c), info },
    eff,
    extrapolated: basis === 'report' && c > cLast * 1.0001,
    belowFloor,
    kvConcPerGpu: cMaxKv,
    kvLimited: c > cMaxKv * 1.0001,
    ttftLimited,
    serving: at.serving,
    prefillFactor: at.prefillFactor,
    contextFactor: 1,
    bound: at.serving?.step.bound ?? null,
    multipliers,
    sweep: (a.sweepAt ?? (a.sweepSteps ? Array.from({ length: a.sweepSteps + 1 }, (_, i) => Math.exp((Math.log(Math.max(1.0001, cMax)) * i) / a.sweepSteps!)) : [])).map((cc) => {
      const e = evalAt(cc);
      return { c: cc, tps: e.tps, user: e.user, ttftP90: e.ttftP90, share: e.serving?.share ?? 0 };
    }),
    evidence,
  };
}

export const drvNote = (note: string): Prov => drv(0, 'Engine', note).p;
