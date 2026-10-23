import archs from '../../data/hf/archs.json';
import { type Prov, type V, drv, srcd } from './provenance';

/** What a model's Hugging Face config says about memory and compute. Produced by `npm run data:hf`. */
export interface ArchInfo {
  repo: string;
  sha: string;
  configUrl: string;
  note: string | null;
  family: string;
  layers: number;
  layerMix: { full: number; sliding: number; linear: number; mamba: number; other: number };
  hidden: number;
  vocab: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  /** Multi-head latent attention: cached latent width and rotary width per token per layer. */
  mla: { latent: number; rope: number } | null;
  window: number | null;
  /** KV bytes per context token across all full-attention layers, at one byte per element. */
  fullTokBytes: number;
  /** The part of `fullTokBytes` a decode step reads only up to `readCapTokens` tokens of (sparse attention). */
  cappedTokBytes: number;
  /** Per-token bytes of a sparse-attention index, kept for every token and scanned in full each step. Not in `fullTokBytes`. */
  scanTokBytes: number;
  /** The same for sliding-window layers, which stop growing at `window` tokens. */
  slidingTokBytes: number;
  /** Recurrent state of linear-attention and Mamba layers, bytes per sequence. */
  stateBytes: number;
  stateDetail: string;
  /** Weights that are lookup tables (n-gram or Engram), never read in full, in GB at the checkpoint's own format. */
  lookupGB: number;
  /** Where the KV figures come from: the config, or the model card where the config cannot say. */
  kvSource: 'config' | 'model-card';
  kvNote: string;
  /** Sparse-attention models read only this many cached tokens per step. */
  readCapTokens: number | null;
  moe: { experts: number; topK: number; moeLayers: number; expertInter: number; matrices: number; dimIn: number; routedParams: number } | null;
  params: {
    total: number;
    text: number;
    active: number;
    /** Activated parameters during prefill, where the model activates fewer then than in decode. */
    activePrefill: number | null;
    embed: number;
    embedTied: boolean;
    routedShare: number;
    excluded: { towers: number; mtpExperts: number; ngram: number };
    /** Model card the parameter counts come from, when they do. */
    source: string | null;
  } | null;
  mtpLayers: number;
  maxPositions: number | null;
}

export const ARCH_SNAPSHOT_AT: string = archs.generatedAt;
export const ARCH: Record<string, ArchInfo> = archs.models as unknown as Record<string, ArchInfo>;

/** Bytes per element of the KV cache. The report's own KV arithmetic assumes an FP8 cache. */
export const KV_BYTES_PER_ELT = 1;

/** Bytes a sequence of `ctx` tokens holds in the cache, recurrent state and sparse-attention index included. */
export function kvBytesPerSeq(a: ArchInfo, ctx: number, bytesPerElt = KV_BYTES_PER_ELT): number {
  const sliding = a.window ? Math.min(ctx, a.window) : ctx;
  return ((a.fullTokBytes + a.scanTokBytes) * ctx + a.slidingTokBytes * sliding) * bytesPerElt + a.stateBytes;
}

/**
 * Bytes a decode step reads from the cache for one sequence. Sparse attention reads a capped number of tokens of the main
 * cache but scans its whole index; the recurrent state is read in full.
 */
export function kvReadBytesPerSeq(a: ArchInfo, ctx: number, bytesPerElt = KV_BYTES_PER_ELT): number {
  const capped = a.readCapTokens ? Math.min(ctx, a.readCapTokens) : ctx;
  const sliding = a.window ? Math.min(ctx, a.window) : ctx;
  const full = (a.fullTokBytes - a.cappedTokBytes) * ctx + a.cappedTokBytes * capped + a.scanTokBytes * ctx;
  return (full + a.slidingTokBytes * sliding) * bytesPerElt + a.stateBytes;
}

export function archProv(a: ArchInfo, note?: string): Prov {
  return {
    kind: 'sourced',
    src: a.configUrl,
    note: note ?? `Hugging Face config.json at commit ${a.sha.slice(0, 8)}, read ${ARCH_SNAPSHOT_AT.slice(0, 10)}`,
    asOf: ARCH_SNAPSHOT_AT,
  };
}

/** Marginal KV bytes per token as a tagged value. A figure resting on an assumption about the config is tagged derived. */
export function kvPerTokV(a: ArchInfo): V {
  const parts = [
    a.layerMix.full ? `${a.layerMix.full} full-attention layers` : null,
    a.layerMix.sliding && a.window ? `${a.layerMix.sliding} sliding-window layers (${a.window} tokens, ${((a.slidingTokBytes * a.window) / 1e6).toFixed(1)} MB per sequence)` : null,
    a.layerMix.linear + a.layerMix.mamba ? `${a.stateDetail} of recurrent state per sequence` : null,
    a.scanTokBytes ? `${(a.scanTokBytes / 1000).toFixed(1)} KB of sparse-attention index per token` : null,
  ].filter(Boolean);
  const total = a.fullTokBytes + a.scanTokBytes;
  const cardOrAssumed = a.kvSource === 'model-card' ? a.kvNote : a.kvNote ? `Assumptions: ${a.kvNote}.` : '';
  const note = `Hugging Face config at commit ${a.sha.slice(0, 8)}: ${parts.join('; ')}. One byte per cache element (FP8 KV). ${cardOrAssumed}`.trim();
  const kind = a.kvSource === 'model-card' || !a.kvNote ? srcd : drv;
  return kind(total, a.params?.source ?? a.configUrl, note, ARCH_SNAPSHOT_AT);
}

/** Total parameters in billions, from the Hugging Face file listing or the model card, for models that gave only a placeholder. */
export function totalFromConfigV(a: ArchInfo): V | null {
  if (!a.params) return null;
  const v = Math.round((a.params.total / 1e9) * 10) / 10;
  if (a.params.source) return srcd(v, a.params.source, 'Model card: total parameters', ARCH_SNAPSHOT_AT);
  return drv(v, a.configUrl, 'Parameter count from the Hugging Face API for the checkpoint');
}

export function activeFromConfigV(a: ArchInfo): V | null {
  if (!a.params) return null;
  const v = Math.round((a.params.active / 1e9) * 10) / 10;
  if (a.params.source) return srcd(v, a.params.source, 'Model card: activated parameters per token', ARCH_SNAPSHOT_AT);
  return drv(v, a.configUrl, 'Total parameters (Hugging Face API) minus the embedding lookup table, the speculative-decoding head, vision towers and the inactive experts, using the config\'s expert counts');
}
