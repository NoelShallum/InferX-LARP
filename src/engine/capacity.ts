import type { Gpu, Quant } from '../data/gpus';
import { BYTES_PER_PARAM, type Model } from '../data/models';
import { RENT_REFS } from '../data/market';
import { type Prov, ph, drv } from '../data/provenance';
import { calibrationFor } from './calibration';

export const GPU_MEM_UTIL = ph(0.9, 'vLLM default gpu_memory_utilization', 'Share of VRAM the serving engine may use');
export const OVERHEAD_GB_PER_GPU = ph(4, 'No source yet', 'CUDA context, activations and graph capture per GPU; the report does not give one');
export const MIN_KV_TOKENS = ph(32768, 'No source yet', 'A model counts as fitting only if the KV pool holds at least one 32K-token user');
export const MULTI_GPU_EFF = {
  nvlink: ph(0.92, 'No source yet', 'Throughput kept per doubling of NVLink GPUs; replace with InferenceX multi-GPU curves'),
  pcie: ph(0.78, 'No source yet', 'Throughput kept per doubling of PCIe-only GPUs; replace with InferenceX multi-GPU curves'),
};
export const DRAFT_COST = ph(0.08, 'No source yet', 'Cost of one draft step relative to a decode step');
export const MTP_DEFAULT_CSAT = ph(64, 'No source yet', 'Concurrency at which speculative decoding stops helping when nothing was measured');
export const VOLTA_KERNEL_PENALTY = ph(0.6, 'No source yet', 'Older attention and GEMM kernels on Volta (no FlashAttention-2, no FP8)');
export const MAX_EXTRAPOLATION = ph(1.6, 'No source yet', 'How far past the last measured concurrency the curve is trusted');

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

export interface Fit {
  fits: boolean;
  reason: string;
  quant: Quant;
  weightsGB: number;
  weightsProv: Prov;
  usableGB: number;
  overheadGB: number;
  kvPoolGB: number;
  kvTokens: number;
  forced: boolean;
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
  if (kvTokens < MIN_KV_TOKENS.v) {
    return { ...common, fits: false, reason: `Only ${kvPool.toFixed(1)} GB left for KV (${Math.round(kvTokens / 1000)}K tokens), under one 32K user` };
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

const bwRatio = (gpu: Gpu, ref: Gpu) => gpu.bwGBs.v / ref.bwGBs.v;

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
  const ratioW = (from: Quant) => {
    const a = weightsGB(model, from);
    const b = weightsGB(model, quant);
    if (!a || !b) return 1;
    return Math.min(2.5, Math.max(0.4, a.gb / b.gb));
  };
  let pts: { c: number; tps: number }[];
  let mult = 1;
  let source: Curve['source'];
  let multNote = 'Measured or estimated points used as published';
  let multProv: Prov = { kind: 'report', src: 'Report §5.2' };
  let baseProvs: Prov[];
  let refIdx = model.refIdx;

  if (gpu.id === 'ada6000' && model.ada.length > 0) {
    pts = model.ada.map((p) => ({ c: p.c, tps: p.tps.v }));
    baseProvs = model.ada.map((p) => p.tps.p);
    source = 'ada';
    mult = ratioW(model.pointsQuant.ada);
    refIdx = Math.min(refIdx, pts.length - 1);
    if (mult !== 1) {
      multNote = `Scaled by weight size, ${model.pointsQuant.ada} points run at ${quant}`;
      multProv = { kind: 'derived', src: 'Weights ratio', note: 'Decode is memory-bound, so speed scales with bytes read per token' };
    }
  } else if (gpu.id === 'pro6000') {
    pts = proPts.map((p) => ({ c: p.c, tps: p.tps.v }));
    baseProvs = proPts.map((p) => p.tps.p);
    source = 'pro6000';
    mult = ratioW(model.pointsQuant.pro6000);
    if (mult !== 1) {
      multNote = `Scaled by weight size, ${model.pointsQuant.pro6000} points run at ${quant}`;
      multProv = { kind: 'derived', src: 'Weights ratio', note: 'Decode is memory-bound, so speed scales with bytes read per token' };
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
      mult = bwRatio(gpu, proGpu) * ratioW(model.pointsQuant.pro6000);
      if (gpu.arch === 'Volta') mult *= VOLTA_KERNEL_PENALTY.v;
      multNote = `Memory-bandwidth ratio ${gpu.bwGBs.v}/${proGpu.bwGBs.v} × weight ratio${gpu.arch === 'Volta' ? ' × Volta kernel penalty' : ''}`;
      multProv = { kind: 'placeholder', src: 'Bandwidth scaling', note: 'Placeholder until the roofline engine and InferenceX multi-GPU data replace it' };
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

export function mtpInfo(model: Model, proCurve: Curve | null): MtpInfo | null {
  const m = model.mtp;
  if (!m) return null;
  const s1 = m.al.v / (1 + m.k * DRAFT_COST.v);
  if (s1 <= 1) return { s1: 1, cSat: Infinity, calibrated: false };
  if (m.measured && proCurve) {
    const plain = tpsAt(proCurve, m.measured.c);
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

export function multiGpuEff(gpu: Gpu, count: number): number {
  if (count <= 1) return 1;
  const e = (gpu.link === 'nvlink' ? MULTI_GPU_EFF.nvlink : MULTI_GPU_EFF.pcie).v;
  return Math.pow(e, Math.log2(count));
}

export interface QosInput {
  mode: 'report' | 'floor';
  floorTps: number;
}

export interface Capacity {
  ok: boolean;
  reason: string;
  fit: Fit;
  quant: Quant;
  curve: Curve | null;
  cPerGpu: number;
  cNode: number;
  tpsNode: number;
  userTps: number;
  mtp: { on: boolean; gain: number; info: MtpInfo | null };
  eff: number;
  extrapolated: boolean;
  belowFloor: boolean;
  /** Concurrent users the KV pool holds at the average context length, per GPU. */
  kvConcPerGpu: number;
  kvLimited: boolean;
  multipliers: { name: string; value: number; prov: Prov }[];
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
  avgContextTokens: number;
}

export function capacityFor(a: CapacityArgs): Capacity {
  const fit = chooseQuant(a.model, a.gpu, a.count, a.quant);
  const empty = (reason: string): Capacity => ({
    ok: false,
    reason,
    fit,
    quant: fit.quant,
    curve: null,
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
    multipliers: [],
  });
  if (!fit.fits) return empty(fit.reason);
  const curve = curveFor(a.model, a.gpu, fit.quant, a.proGpu);
  if (!curve) return empty(`The report has no throughput data for ${a.model.name}`);

  const proFit = chooseQuant(a.model, a.proGpu, 1, a.model.pointsQuant.pro6000);
  const proCurve = curveFor(a.model, a.proGpu, proFit.quant, a.proGpu);
  const info = a.mtp ? mtpInfo(a.model, proCurve) : null;
  const eff = multiGpuEff(a.gpu, a.count);
  const gain = (c: number) => (a.mtp ? mtpGain(info, c) : 1);
  const userAt = (c: number) => (tpsAt(curve, c) / c) * gain(c) * eff * a.extraMult;

  const cLast = curve.pts[curve.pts.length - 1]!.c;
  const cMaxCurve = cLast * MAX_EXTRAPOLATION.v;
  const cMaxKv = Math.max(1, fit.kvTokens / Math.max(1, a.avgContextTokens) / a.count);
  const cMax = Math.max(1, Math.min(cMaxCurve, cMaxKv));

  let c: number;
  let belowFloor = false;
  if (a.qos.mode === 'report') {
    c = curve.cRef;
  } else if (userAt(1) < a.qos.floorTps) {
    c = 1;
    belowFloor = true;
  } else if (userAt(cMax) >= a.qos.floorTps) {
    c = cMax;
  } else {
    let lo = 1;
    let hi = cMax;
    for (let i = 0; i < 50; i++) {
      const mid = (lo + hi) / 2;
      if (userAt(mid) >= a.qos.floorTps) lo = mid;
      else hi = mid;
    }
    c = lo;
  }

  const tpsSingle = tpsAt(curve, c) * gain(c) * a.extraMult;
  const multipliers = [
    { name: `${curve.source === 'scaled' ? 'GPU scaling' : 'Quant scaling'}`, value: curve.mult, prov: curve.multProv },
    ...(a.mtp ? [{ name: 'MTP gain', value: gain(c), prov: { kind: (info?.calibrated ? 'derived' : 'placeholder') as Prov['kind'], src: 'Acceptance length', note: info ? `AL/(1+k·ρ) = ${info.s1.toFixed(2)} at one user, fading to 1 by about ${Number.isFinite(info.cSat) ? Math.round(info.cSat) : '∞'} concurrent` : '' } }] : []),
    ...(a.count > 1 ? [{ name: `${a.count}-GPU scaling`, value: eff, prov: (a.gpu.link === 'nvlink' ? MULTI_GPU_EFF.nvlink : MULTI_GPU_EFF.pcie).p }] : []),
    ...(a.extraNotes ?? []),
  ];
  return {
    ok: true,
    reason: fit.reason,
    fit,
    quant: fit.quant,
    curve,
    cPerGpu: c,
    cNode: c * a.count,
    tpsNode: tpsSingle * a.count * eff,
    userTps: userAt(c),
    mtp: { on: a.mtp && !!info, gain: gain(c), info },
    eff,
    extrapolated: c > cLast * 1.0001,
    belowFloor,
    kvConcPerGpu: cMaxKv,
    kvLimited: c > cMaxKv * 1.0001,
    multipliers,
  };
}

export const drvNote = (note: string): Prov => drv(0, 'Engine', note).p;
