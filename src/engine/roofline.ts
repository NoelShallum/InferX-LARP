import { kvReadBytesPerSeq, type ArchInfo } from '../data/arch';
import type { Gpu, Quant } from '../data/gpus';
import { BYTES_PER_PARAM, type Model } from '../data/models';
import { type V, drv, ph } from '../data/provenance';

/**
 * Roofline model of one serving step.
 *
 * Decode reads the weights the batch needs plus every sequence's cache and recurrent state once per step, so it is
 * bound by memory bandwidth until the batch is large enough for the tensor cores to matter. Prefill pushes many
 * tokens through the same weights at once and is bound by tensor-core throughput.
 *
 * The efficiencies are what real serving stacks achieve against the datasheet peaks. They are fitted per card family to
 * InferenceX's measured runs (see `ROOF_FIT`), and carried to cards InferenceX has not measured by family.
 */

export interface RoofParams {
  /** Share of peak memory bandwidth decode kernels reach. */
  etaMem: number;
  /** Share of peak tensor throughput small-batch decode GEMMs reach. */
  etaDecodeCompute: number;
  /** Share of peak tensor throughput prefill reaches on a dense model. */
  etaPrefill: number;
  /** The same for mixture-of-experts models, whose grouped GEMMs are less efficient. */
  etaPrefillMoe: number;
  /** Time every step spends outside the kernels the roofline counts (sampling, scheduling, launch gaps), in ms. */
  stepOverheadMs: number;
  /** Tokens one prefill chunk carries; weights are read once per chunk. */
  prefillChunkTokens: number;
  /** Tensor-parallel all-reduce latency and usable link bandwidth, per link kind. */
  allReduceUs: { nvlink: number; pcie: number };
  linkGBs: { nvlink: number; pcie: number };
}

export const ROOF_DEFAULT: RoofParams = {
  etaMem: 0.85,
  etaDecodeCompute: 0.35,
  etaPrefill: 0.5,
  etaPrefillMoe: 0.35,
  stepOverheadMs: 0.8,
  prefillChunkTokens: 8192,
  allReduceUs: { nvlink: 12, pcie: 40 },
  linkGBs: { nvlink: 300, pcie: 25 },
};

/** Card families whose serving efficiency is fitted separately. */
export type RoofClass = 'hopper' | 'blackwell-dc' | 'blackwell-ws';

/**
 * Hopper and the older cards share one family (no native FP4, HBM or GDDR6/6X); data-centre Blackwell has its own; the
 * RTX PRO 6000 and the GeForce Blackwell cards (same GB202 silicon, GDDR7, PCIe only) share the workstation family.
 */
export function roofClassOf(gpu: Gpu): RoofClass {
  if (gpu.id === 'b200') return 'blackwell-dc';
  if (gpu.id === 'pro6000' || gpu.id === 'rtx5090') return 'blackwell-ws';
  return 'hopper';
}

/**
 * Fitted to 393 InferenceX measurement points (vLLM and SGLang runs of gpt-oss-120b and Llama 3.3 70B on H100, H200 and
 * B200 at 1 to 8 GPUs, Qwen3.5-397B on B200 and on four RTX PRO 6000s), by minimising the squared log error of output
 * speed, speed per user and time to first token. See the Data audit for the residuals.
 *
 * The workstation family has only four measured points (four PRO 6000s over PCIe), so it inherits the data-centre
 * Blackwell family's memory efficiency and step overhead and fits only its prefill efficiency and the PCIe link.
 */
export const ROOF_CLASSES: Record<RoofClass, RoofParams> = {
  hopper: { etaMem: 0.637, etaDecodeCompute: 0.35, etaPrefill: 0.508, etaPrefillMoe: 0.132, stepOverheadMs: 1.93, prefillChunkTokens: 8192, allReduceUs: { nvlink: 22, pcie: 74 }, linkGBs: { nvlink: 91, pcie: 9 } },
  'blackwell-dc': { etaMem: 0.617, etaDecodeCompute: 0.35, etaPrefill: 0.374, etaPrefillMoe: 0.05, stepOverheadMs: 1.17, prefillChunkTokens: 8192, allReduceUs: { nvlink: 22, pcie: 74 }, linkGBs: { nvlink: 91, pcie: 9 } },
  'blackwell-ws': { etaMem: 0.617, etaDecodeCompute: 0.35, etaPrefill: 0.5, etaPrefillMoe: 0.181, stepOverheadMs: 1.17, prefillChunkTokens: 8192, allReduceUs: { nvlink: 22, pcie: 74 }, linkGBs: { nvlink: 91, pcie: 9 } },
};

export const roofParamsFor = (gpu: Gpu): RoofParams => ROOF_CLASSES[roofClassOf(gpu)];

export interface RoofInputs {
  model: Model;
  gpu: Gpu;
  quant: Quant;
  count: number;
  /** Checkpoint size at this quant, GB. */
  weightsGB: number;
  /** Average context per sequence during decode, tokens. */
  ctx: number;
  /** Bytes per element of the KV cache. */
  kvElt?: number;
}

export interface ComputeFormat {
  format: 'fp4' | 'fp8' | 'fp16';
  tflops: number;
  /** False where the weights are stored small but the maths runs at a wider precision. */
  native: boolean;
  tflopsV: V;
}

/** Which tensor-core rate a weight format runs at on this card. */
export function computeFormat(gpu: Gpu, quant: Quant): ComputeFormat {
  const t = gpu.tflops;
  const four = quant === 'nvfp4' || quant === 'mxfp4';
  if (four && t.fp4) return { format: 'fp4', tflops: t.fp4.v, native: true, tflopsV: t.fp4 };
  if (four || quant === 'awq4') {
    // Weight-only: weights are dequantised and the maths runs at 16 bits, or at FP8 where the card has it and the format allows.
    return { format: 'fp16', tflops: t.fp16.v, native: false, tflopsV: t.fp16 };
  }
  if (quant === 'fp8' && t.fp8) return { format: 'fp8', tflops: t.fp8.v, native: true, tflopsV: t.fp8 };
  return { format: 'fp16', tflops: t.fp16.v, native: quant === 'fp16', tflopsV: t.fp16 };
}

export interface WeightSplit {
  /** Read on every step: attention, shared experts, dense layers, output head. */
  denseGB: number;
  /** Routed experts. A step reads only the experts its batch touches. */
  routedGB: number;
  /** Embedding table, a lookup that is never read in full. N-gram and Engram tables are lookups too and are left out of both other parts. */
  embedGB: number;
}

export function weightSplit(model: Model, quant: Quant, weightsGB: number): WeightSplit {
  const net = model.net;
  const routedShare = net?.params?.routedShare ?? 0;
  const embedGB = net && !net.params?.embedTied ? (net.vocab * net.hidden * 2) / 1e9 : 0;
  // N-gram and Engram tables are lookups too: a step reads a few rows, not the table.
  const readableGB = Math.max(0, weightsGB - (net?.lookupGB ?? 0));
  const routedGB = readableGB * routedShare;
  const denseGB = Math.max(0, readableGB - routedGB - embedGB);
  void quant;
  void BYTES_PER_PARAM;
  return { denseGB, routedGB, embedGB };
}

/** Share of a model's routed experts a batch of `tokens` touches, if routing were uniform. */
export function expertCoverage(net: ArchInfo | null, tokens: number): number {
  if (!net?.moe) return 1;
  const { experts, topK } = net.moe;
  return 1 - Math.pow(1 - topK / experts, Math.max(1, tokens));
}

const activeParams = (model: Model) => (model.net?.params ? model.net.params.active : model.activeB.v * 1e9);

/** Attention FLOPs for one token attending over `ctx` cached tokens. */
export function attnFlopsPerToken(net: ArchInfo | null, ctx: number): number {
  if (!net) return 0;
  const qk = net.mla ? net.heads * (net.mla.latent + net.mla.rope) : net.heads * net.headDim;
  const full = net.readCapTokens ? Math.min(ctx, net.readCapTokens) : ctx;
  const sliding = net.window ? Math.min(ctx, net.window) : ctx;
  return 4 * qk * (net.layerMix.full * full + net.layerMix.sliding * sliding);
}

export type Bound = 'memory' | 'compute' | 'comm';

export interface StepResult {
  /** Seconds for one step. */
  t: number;
  tMem: number;
  tCompute: number;
  tComm: number;
  bound: Bound;
  bytes: { weights: number; cache: number; state: number };
}

/**
 * One decode step for `batch` sequences, each producing `rows` token rows (1 normally; draft length + 1 when
 * verifying speculative tokens).
 */
export function decodeStep(i: RoofInputs, batch: number, rows = 1, p: RoofParams = roofParamsFor(i.gpu)): StepResult {
  const { model, gpu, count } = i;
  const net = model.net;
  const split = weightSplit(model, i.quant, i.weightsGB);
  const cover = expertCoverage(net, batch * rows);
  const weights = (split.denseGB + split.routedGB * cover) * 1e9;
  const kvElt = i.kvElt ?? 1;
  const cache = net ? batch * (kvReadBytesPerSeq(net, i.ctx, kvElt) - net.stateBytes) : batch * i.ctx * model.kvBytesPerTok.v * kvElt;
  const state = net ? batch * net.stateBytes * 2 : 0;
  const bytes = { weights, cache, state };
  const tMem = (weights + cache + state) / count / (gpu.bwGBs.v * 1e9 * p.etaMem);

  const cf = computeFormat(gpu, i.quant);
  const flops = batch * rows * (2 * activeParams(model) + attnFlopsPerToken(net, i.ctx));
  const tCompute = flops / count / (cf.tflops * 1e12 * p.etaDecodeCompute);

  let tComm = 0;
  if (count > 1) {
    const layers = net?.layers ?? 60;
    const hidden = net?.hidden ?? 5120;
    const msg = batch * rows * hidden * 2;
    tComm = layers * 2 * (p.allReduceUs[gpu.link] * 1e-6 + msg / (p.linkGBs[gpu.link] * 1e9));
  }
  const core = Math.max(tMem, tCompute);
  const bound: Bound = tComm > core ? 'comm' : tMem >= tCompute ? 'memory' : 'compute';
  return { t: core + tComm + p.stepOverheadMs * 1e-3, tMem, tCompute, tComm, bound, bytes };
}

/**
 * The most output tokens per second the memory bus could ever deliver at this batch: every step reads what it must at
 * full bandwidth and takes no other time. No kernel gets there; a published figure above it cannot be right. With
 * `includeCache` the cache and recurrent state each sequence needs are read too; without it only the weights are.
 */
export function bandwidthCeilingTps(i: RoofInputs, batch: number, includeCache = false): number {
  const split = weightSplit(i.model, i.quant, i.weightsGB);
  const cover = expertCoverage(i.model.net, batch);
  let bytes = (split.denseGB + split.routedGB * cover) * 1e9;
  if (includeCache) {
    const net = i.model.net;
    const kvElt = i.kvElt ?? 1;
    bytes += net ? batch * (kvReadBytesPerSeq(net, i.ctx, kvElt) + net.stateBytes) : batch * i.ctx * i.model.kvBytesPerTok.v * kvElt;
  }
  return batch / (bytes / i.count / (i.gpu.bwGBs.v * 1e9));
}

/** Aggregate output tokens per second for `batch` concurrent sequences (all GPUs together). */
export function decodeTps(i: RoofInputs, batch: number, p: RoofParams = roofParamsFor(i.gpu)): number {
  return batch / decodeStep(i, batch, 1, p).t;
}

export interface PrefillResult {
  /** Prompt tokens processed per second, all GPUs together. */
  tokPerSec: number;
  /** Seconds to prefill `promptTokens` on an otherwise idle node. */
  seconds: number;
  bound: 'compute' | 'memory';
}

/** Prefill speed for prompts of `promptTokens` fresh tokens. */
export function prefillRate(i: RoofInputs, promptTokens: number, p: RoofParams = roofParamsFor(i.gpu)): PrefillResult {
  const { model, gpu, count } = i;
  const net = model.net;
  const cf = computeFormat(gpu, i.quant);
  const eta = model.arch === 'moe' ? p.etaPrefillMoe : p.etaPrefill;
  const flopsPerTok = 2 * (net?.params?.activePrefill ?? activeParams(model)) + attnFlopsPerToken(net, promptTokens / 2);
  const computeRate = (eta * cf.tflops * 1e12 * count) / flopsPerTok;
  // Each chunk also streams the whole weight file, which only matters for small chunks on big models.
  const chunkCompute = p.prefillChunkTokens / computeRate;
  const chunkRead = (i.weightsGB * 1e9) / count / (gpu.bwGBs.v * 1e9 * p.etaMem);
  // Tensor-parallel nodes all-reduce the whole chunk's activations twice per layer, which is slow over PCIe.
  const chunkComm = count > 1 ? (net?.layers ?? 60) * 2 * (p.allReduceUs[gpu.link] * 1e-6 + (p.prefillChunkTokens * (net?.hidden ?? 5120) * 2) / (p.linkGBs[gpu.link] * 1e9)) : 0;
  const bound = chunkRead > chunkCompute ? 'memory' : 'compute';
  const tokPerSec = p.prefillChunkTokens / (Math.max(chunkCompute, chunkRead) + chunkComm);
  return { tokPerSec, seconds: promptTokens / tokPerSec, bound };
}

/** What one request costs the server: its output, its prompt, the part of the prompt that is not cached, and the context during decode. */
export interface ShapeIn {
  avgTokens: number;
  outShare: number;
  cacheHit: number;
}

export function requestParts(s: ShapeIn) {
  const out = s.avgTokens * s.outShare;
  const input = s.avgTokens - out;
  return { out, input, fresh: input * (1 - s.cacheHit), ctx: input + out / 2 };
}

export interface ServingPoint {
  /** Concurrent sequences on the node. */
  batch: number;
  /** Aggregate output tokens per second while decoding, before prefill takes its share of time. */
  decodeTps: number;
  step: StepResult;
  prefill: PrefillResult;
  /** Fresh prompt tokens per request after the prefix cache. */
  freshTokens: number;
  outTokens: number;
  /** Seconds of prefill the node owes for every second of decode at this load. */
  a: number;
  /** Fraction of the node's time spent on prefill. */
  share: number;
  /** Aggregate output tokens per second after prefill's share. */
  tps: number;
  /** Seconds to first token: median and 90th percentile, queueing included. */
  ttftP50: number;
  ttftP90: number;
}

/**
 * The node at `batch` concurrent sequences serving requests of this shape. Prefill and decode share the GPUs, so
 * the decode rate is cut by the share of time prefill needs; the queue in front of prefill sets time to first token
 * (an M/D/1 queue: Poisson arrivals, one fixed prefill time per request).
 */
export function servingPoint(i: Omit<RoofInputs, 'ctx'>, shape: ShapeIn, batch: number, p: RoofParams = roofParamsFor(i.gpu)): ServingPoint {
  const parts = requestParts(shape);
  const full: RoofInputs = { ...i, ctx: parts.ctx };
  const step = decodeStep(full, batch, 1, p);
  const decodeTps = batch / step.t;
  const prefill = prefillRate(full, parts.input, p);
  const sp = parts.fresh / prefill.tokPerSec;
  const a = parts.out > 0 ? (decodeTps * sp) / parts.out : Infinity;
  const share = a / (1 + a);
  const wq = share < 1 ? (share * sp) / (2 * (1 - share)) : Infinity;
  const tail = share > 0.1 && share < 1 ? (Math.log(10 * share) * sp) / (2 * (1 - share)) : 0;
  return {
    batch,
    decodeTps,
    step,
    prefill,
    freshTokens: parts.fresh,
    outTokens: parts.out,
    a,
    share,
    tps: decodeTps * (1 - share),
    ttftP50: sp + wq + step.t,
    ttftP90: sp + Math.max(0, tail) + step.t,
  };
}

const FIT_SRC = 'Fit to InferenceX measured runs';
const fitNote = (what: string) => `${what}. Fitted to 393 InferenceX points (vLLM and SGLang, H100, H200, B200, four RTX PRO 6000s).`;
const carried = (what: string, from: string) => `${what}. Not measured for this family; carried over from ${from}.`;

export const ROOF_PROV: Record<string, V> = {
  'hopper.etaMem': drv(ROOF_CLASSES.hopper.etaMem, FIT_SRC, fitNote('Share of peak memory bandwidth decode reaches on H100 and H200. Also used for Ada, Volta and other cards InferenceX has not measured')),
  'hopper.stepOverheadMs': drv(ROOF_CLASSES.hopper.stepOverheadMs, FIT_SRC, fitNote('Time per decode step outside the counted kernels')),
  'hopper.etaPrefill': drv(ROOF_CLASSES.hopper.etaPrefill, FIT_SRC, fitNote('Share of peak tensor throughput dense prefill reaches (Llama 3.3 70B at FP8)')),
  'hopper.etaPrefillMoe': drv(ROOF_CLASSES.hopper.etaPrefillMoe, FIT_SRC, fitNote('The same for a mixture of experts (gpt-oss-120b); its weights run at 16 bits')),
  'blackwell-dc.etaMem': drv(ROOF_CLASSES['blackwell-dc'].etaMem, FIT_SRC, fitNote('Share of peak memory bandwidth decode reaches on B200')),
  'blackwell-dc.stepOverheadMs': drv(ROOF_CLASSES['blackwell-dc'].stepOverheadMs, FIT_SRC, fitNote('Time per decode step outside the counted kernels')),
  'blackwell-dc.etaPrefill': drv(ROOF_CLASSES['blackwell-dc'].etaPrefill, FIT_SRC, fitNote('Share of peak tensor throughput dense prefill reaches')),
  'blackwell-dc.etaPrefillMoe': drv(ROOF_CLASSES['blackwell-dc'].etaPrefillMoe, FIT_SRC, fitNote('The same for a mixture of experts; low because FP4 peak is not what vLLM reaches in prefill')),
  'blackwell-ws.etaMem': drv(ROOF_CLASSES['blackwell-ws'].etaMem, FIT_SRC, carried('Memory bandwidth share on the RTX PRO 6000 and RTX 5090', 'B200; four measured PRO 6000 points are consistent with it (output speed within 2%)')),
  'blackwell-ws.stepOverheadMs': drv(ROOF_CLASSES['blackwell-ws'].stepOverheadMs, FIT_SRC, carried('Time per decode step', 'B200')),
  'blackwell-ws.etaPrefill': ph(ROOF_CLASSES['blackwell-ws'].etaPrefill, 'No source yet', 'Dense prefill share on the PRO 6000: no dense model was measured on it'),
  'blackwell-ws.etaPrefillMoe': drv(ROOF_CLASSES['blackwell-ws'].etaPrefillMoe, FIT_SRC, 'Mixture-of-experts prefill share on the PRO 6000, from four measured points of Qwen3.5-397B on four cards'),
  etaDecodeCompute: ph(ROOF_CLASSES.hopper.etaDecodeCompute, 'No source yet', 'Share of peak tensor throughput small-batch decode reaches; decode is memory-bound in every case checked, so it rarely matters'),
  prefillChunkTokens: ph(8192, 'vLLM default', 'Tokens per chunked-prefill step; weights are read once per chunk'),
  allReduceNvlinkUs: drv(ROOF_CLASSES.hopper.allReduceUs.nvlink, FIT_SRC, fitNote('Effective all-reduce latency per layer and direction over NVLink, including the launch and synchronisation of the stack')),
  allReducePcieUs: drv(ROOF_CLASSES.hopper.allReduceUs.pcie, FIT_SRC, 'Effective all-reduce latency over PCIe. Fitted to four measured PRO 6000 points only.'),
  linkNvlinkGBs: drv(ROOF_CLASSES.hopper.linkGBs.nvlink, FIT_SRC, fitNote('Effective NVLink bandwidth an all-reduce sees, far below the link rating')),
  linkPcieGBs: drv(ROOF_CLASSES.hopper.linkGBs.pcie, FIT_SRC, 'Effective PCIe bandwidth an all-reduce sees across four PRO 6000s. Fitted to four measured points only.'),
};
