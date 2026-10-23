import vastJson from '../../data/live/vast.latest.json';
import orJson from '../../data/live/openrouter.latest.json';
import type { Gpu } from './gpus';
import { MARKET_BY_MODEL, type MarketRow } from './market';
import { type Prov, type V } from './provenance';

export type RentView = 'r90' | 'r98' | 'dc99' | 'any';
export type RentStat = 'cheapest' | 'median';

export interface VastOffer {
  id: number;
  machineId: number;
  gpuName: string;
  numGpus: number;
  dph: number;
  perGpu: number;
  reliability: number | null;
  verified: boolean;
  datacenter: boolean;
  geo: string | null;
  inetDown: number | null;
  vramGb: number | null;
}

export interface VastViewStats {
  count: number;
  minPerGpu: number | null;
  p25PerGpu: number | null;
  medianPerGpu: number | null;
  cheapest: VastOffer[];
}

export interface VastSnapshot {
  source: string;
  generatedAt: string;
  views: Record<RentView, string>;
  gpus: Record<
    string,
    {
      vastNames: string[];
      note: string | null;
      nodes: Record<string, { views: Record<RentView, VastViewStats>; interruptible?: { count: number; minPerGpu: number | null; medianPerGpu: number | null } }>;
    }
  >;
}

export interface OrEndpoint {
  provider: string;
  quantization: string;
  inPerM: number | null;
  outPerM: number | null;
  cachePerM: number | null;
  status: number | null;
  uptime1d: number | null;
  p50Tps: number | null;
  p50LatencyMs: number | null;
  zdr: boolean | null;
}

export interface OrModel {
  slug: string;
  name: string;
  summary: {
    endpointsListed: number;
    endpointsUp: number;
    inMin: number | null;
    inMedian: number | null;
    outMin: number | null;
    outMedian: number | null;
    outMax: number | null;
    cacheFractionMedian: number | null;
    speedEndpoints: number;
    p50TpsMedian: number | null;
    p50LatencyMsMedian: number | null;
    uptime1dMedian: number | null;
  };
  traffic: {
    days: number;
    from: string;
    to: string;
    tokensPerDay: number;
    avgTokensPerRequest: number | null;
    outShare: number | null;
    cacheHit: number | null;
    daily: { date: string; tokens: number; requests: number }[];
  } | null;
  batch: { inRatio: number; outRatio: number } | null;
  /** Present in the latest snapshot only; history keeps statistics, not lists. */
  endpoints: OrEndpoint[];
}

export interface OrSnapshot {
  source: string;
  generatedAt: string;
  models: Record<string, OrModel>;
}

export const VAST = vastJson as unknown as VastSnapshot;
export const OPENROUTER = orJson as unknown as OrSnapshot;

const vastHistory = import.meta.glob('../../data/live/history/vast-*.json', { eager: true, import: 'default' }) as Record<string, VastSnapshot>;
const orHistory = import.meta.glob('../../data/live/history/openrouter-*.json', { eager: true, import: 'default' }) as Record<string, OrSnapshot>;

export const VAST_HISTORY: VastSnapshot[] = Object.values(vastHistory).sort((a, b) => a.generatedAt.localeCompare(b.generatedAt));
export const OPENROUTER_HISTORY: OrSnapshot[] = Object.values(orHistory).sort((a, b) => a.generatedAt.localeCompare(b.generatedAt));

const day = (iso: string) => iso.slice(0, 10);
export const VAST_DAY = day(VAST.generatedAt);
export const OR_DAY = day(OPENROUTER.generatedAt);

export const RENT_VIEW_LABEL: Record<RentView, string> = {
  r90: 'Verified, reliability ≥ 0.90',
  r98: 'Verified, reliability ≥ 0.98',
  dc99: 'Data-centre host, ≥ 0.99',
  any: 'Any offer',
};

export interface LiveRent {
  perGpu: number;
  nodeSize: number;
  offers: number;
  geo: string | null;
  reliability: number | null;
  view: RentView;
  stat: RentStat;
  /** True when the node size asked for had no offer and a smaller node's per-GPU price stands in. */
  borrowedFromSingle: boolean;
}

/** Live Vast.ai rent per GPU-hour, or null when no offer passes the filter at this node size or at one card. */
export function liveRent(gpuId: string, nodeSize: number, view: RentView, stat: RentStat): LiveRent | null {
  const g = VAST.gpus[gpuId];
  if (!g) return null;
  const pick = (size: number) => {
    const v = g.nodes[String(size)]?.views[view];
    if (!v || v.count === 0) return null;
    const value = stat === 'cheapest' ? v.minPerGpu : v.medianPerGpu;
    if (value === null) return null;
    const top = v.cheapest[0];
    return { perGpu: value, nodeSize: size, offers: v.count, geo: top?.geo ?? null, reliability: top?.reliability ?? null };
  };
  const direct = pick(nodeSize);
  if (direct) return { ...direct, view, stat, borrowedFromSingle: false };
  if (nodeSize > 1) {
    const single = pick(1);
    if (single) return { ...single, view, stat, borrowedFromSingle: true };
  }
  return null;
}

export interface RentChoice {
  source: 'live' | 'report';
  view: RentView;
  stat: RentStat;
}

/** Rent per GPU-hour with its provenance. Live where an offer exists, otherwise the report or catalog price. */
export function effectiveRent(gpu: Gpu, nodeSize: number, choice: RentChoice): V | null {
  if (choice.source === 'live') {
    const l = liveRent(gpu.id, nodeSize, choice.view, choice.stat);
    if (l) {
      const where = l.geo ? `, cheapest host in ${l.geo}${l.reliability ? ` at reliability ${l.reliability}` : ''}` : '';
      const prov: Prov = {
        kind: 'live',
        src: 'Vast.ai (browser scrape)',
        asOf: VAST_DAY,
        note: `${choice.stat === 'cheapest' ? 'Cheapest' : 'Median'} of ${l.offers} on-demand offer${l.offers === 1 ? '' : 's'}, ${RENT_VIEW_LABEL[choice.view].toLowerCase()}${where}${l.borrowedFromSingle ? `. No ${nodeSize}-GPU offer passes this filter, so the single-card per-GPU price stands in.` : ''}`,
      };
      return { v: l.perGpu, p: prov };
    }
  }
  if (!gpu.rentUsdHr) return null;
  if (choice.source === 'live') {
    return { v: gpu.rentUsdHr.v, p: { ...gpu.rentUsdHr.p, note: `No Vast.ai offer passes the current filter for ${gpu.short}, so this is the ${gpu.rentUsdHr.p.kind === 'report' || gpu.rentUsdHr.p.kind === 'report-est' ? "report's" : 'catalog'} price. ${gpu.rentUsdHr.p.note ?? ''}`.trim() } };
  }
  return gpu.rentUsdHr;
}

/** True when the live choice fell back to a static price. */
export function rentFellBack(gpu: Gpu, nodeSize: number, choice: RentChoice): boolean {
  return choice.source === 'live' && liveRent(gpu.id, nodeSize, choice.view, choice.stat) === null;
}

const liveP = (src: string, note: string, asOf: string): Prov => ({ kind: 'live', src, note, asOf });

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
};

export interface LiveMarket {
  row: MarketRow;
  /** Observed request shape, when OpenRouter's traffic totals are available. */
  shape: { avgTokens: number; outShare: number; cacheHit: number } | null;
  /** Median cache-read price over input price across endpoints, when listed. */
  cacheFraction: number | null;
  /** Median typical stream speed of listed endpoints, tokens per second per user. */
  p50Tps: number | null;
}

/** A model the report has no row for: the market row is built from live OpenRouter numbers alone. */
function liveOnlyBase(modelId: string, m: OrModel): MarketRow | null {
  const s = m.summary;
  const t = m.traffic;
  if (!t || s.inMedian === null || s.outMedian === null) return null;
  const src = 'OpenRouter public API';
  const v = (x: number, note: string) => ({ v: x, p: liveP(src, note, OR_DAY) });
  return {
    modelId,
    paidTokensPerDayB: v(t.tokensPerDay / 1e9, 'Paid tokens a day'),
    endpoints: v(s.endpointsListed, 'Endpoints listed'),
    inMin: v(s.inMin ?? s.inMedian, 'Cheapest listed endpoint'),
    inMed: v(s.inMedian, 'Median across listed endpoints'),
    outMin: v(s.outMin ?? s.outMedian, 'Cheapest listed endpoint'),
    outMed: v(s.outMedian, 'Median across listed endpoints'),
    outMax: v(s.outMax ?? s.outMedian, 'Dearest listed endpoint'),
    cacheHit: t.cacheHit != null ? v(t.cacheHit, 'Cached share of prompt tokens') : { v: 0.5, p: { kind: 'placeholder', src: 'No source yet', note: 'OpenRouter gave no cache figure for this model' } },
    busyHourMedianUsd: null,
    busyHourLowestUsd: null,
    typicalMedianUsd: null,
  };
}

/** The static report row with live OpenRouter numbers laid over it. Null when the model is not in the live snapshot. */
export function liveMarket(modelId: string, useMedianVolume: boolean): LiveMarket | null {
  const m = OPENROUTER.models[modelId];
  if (!m || m.summary.endpointsListed === 0 || m.summary.inMedian === null || m.summary.outMedian === null) return null;
  const base = MARKET_BY_MODEL[modelId] ?? liveOnlyBase(modelId, m);
  if (!base) return null;
  const s = m.summary;
  const src = 'OpenRouter public API';
  const asOf = OR_DAY;
  const t = m.traffic;
  const perDayB = t ? (useMedianVolume ? median(t.daily.map((d) => d.tokens)) : t.tokensPerDay) / 1e9 : base.paidTokensPerDayB.v;
  const row: MarketRow = {
    ...base,
    paidTokensPerDayB: t ? { v: perDayB, p: liveP(src, `Paid tokens a day, ${useMedianVolume ? 'median' : 'mean'} of ${t.days} full days to ${t.to}`, asOf) } : base.paidTokensPerDayB,
    endpoints: { v: s.endpointsListed, p: liveP(src, `${s.endpointsListed} listed, ${s.endpointsUp} up`, asOf) },
    inMin: { v: s.inMin ?? base.inMin.v, p: liveP(src, 'Cheapest listed endpoint', asOf) },
    inMed: { v: s.inMedian as number, p: liveP(src, 'Median across listed endpoints', asOf) },
    outMin: { v: s.outMin ?? base.outMin.v, p: liveP(src, 'Cheapest listed endpoint', asOf) },
    outMed: { v: s.outMedian as number, p: liveP(src, 'Median across listed endpoints', asOf) },
    outMax: { v: s.outMax ?? base.outMax.v, p: liveP(src, 'Dearest listed endpoint', asOf) },
    cacheHit: t?.cacheHit != null ? { v: t.cacheHit, p: liveP(src, `Cached share of prompt tokens over ${t.days} days`, asOf) } : base.cacheHit,
  };
  return {
    row,
    shape: t && t.outShare != null && t.cacheHit != null && t.avgTokensPerRequest ? { avgTokens: t.avgTokensPerRequest, outShare: t.outShare, cacheHit: t.cacheHit } : null,
    cacheFraction: s.cacheFractionMedian,
    p50Tps: s.p50TpsMedian,
  };
}

/** Median typical stream speed on OpenRouter, the "OpenRouter-competitive" quality bar for one model. */
export function routerSpeed(modelId: string): V | null {
  const m = OPENROUTER.models[modelId];
  if (!m || m.summary.p50TpsMedian === null || m.summary.speedEndpoints < 2) return null;
  return {
    v: m.summary.p50TpsMedian,
    p: liveP('OpenRouter endpoint stats', `Median of the typical (p50) stream speed of ${m.summary.speedEndpoints} listed endpoints`, OR_DAY),
  };
}

/** Every live number as a V, for the data audit register. */
export function liveRegister(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [id, m] of Object.entries(OPENROUTER.models)) {
    out[`openrouter.${id}`] = {
      endpoints: { v: m.summary.endpointsListed, p: liveP('OpenRouter public API', 'Endpoints listed', OR_DAY) },
      outMedian: { v: m.summary.outMedian, p: liveP('OpenRouter public API', 'Median output price per million tokens', OR_DAY) },
      tokensPerDay: m.traffic ? { v: m.traffic.tokensPerDay, p: liveP('OpenRouter public API', 'Paid tokens a day', OR_DAY) } : null,
      streamTps: m.summary.p50TpsMedian !== null ? { v: m.summary.p50TpsMedian, p: liveP('OpenRouter endpoint stats', 'Median typical stream speed', OR_DAY) } : null,
    };
  }
  for (const [id, g] of Object.entries(VAST.gpus)) {
    const one = g.nodes['1']?.views.r90;
    out[`vast.${id}`] = {
      cheapestR90: one && one.minPerGpu !== null ? { v: one.minPerGpu, p: liveP('Vast.ai (browser scrape)', `Cheapest of ${one.count} offers at reliability ≥ 0.90`, VAST_DAY) } : null,
    };
  }
  return out;
}

export interface TrendPoint {
  date: string;
  value: number;
}

/** Vast.ai rent per GPU-hour over the saved snapshots for one GPU, node size, filter and statistic. */
export function vastTrend(history: VastSnapshot[], gpuId: string, nodeSize: number, view: RentView, stat: RentStat): TrendPoint[] {
  const out: TrendPoint[] = [];
  for (const snap of history) {
    const v = snap.gpus[gpuId]?.nodes[String(nodeSize)]?.views[view];
    const value = v ? (stat === 'cheapest' ? v.minPerGpu : v.medianPerGpu) : null;
    if (v && v.count > 0 && value !== null) out.push({ date: snap.generatedAt.slice(0, 10), value });
  }
  return out;
}

export type OrField = 'outMedian' | 'inMedian' | 'outMin' | 'tokensPerDay' | 'endpoints' | 'p50Tps';

/** One OpenRouter number for one model over the saved snapshots. */
export function orTrend(history: OrSnapshot[], modelId: string, field: OrField): TrendPoint[] {
  const out: TrendPoint[] = [];
  for (const snap of history) {
    const m = snap.models[modelId];
    if (!m) continue;
    const value =
      field === 'tokensPerDay' ? (m.traffic ? m.traffic.tokensPerDay / 1e9 : null) : field === 'endpoints' ? m.summary.endpointsListed : field === 'p50Tps' ? m.summary.p50TpsMedian : m.summary[field];
    if (value !== null && value !== undefined) out.push({ date: snap.generatedAt.slice(0, 10), value });
  }
  return out;
}
