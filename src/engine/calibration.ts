import { CACHE_PRICE_FRACTION, MARKET_BY_MODEL } from '../data/market';
import type { Model } from '../data/models';
import { type Prov, ph } from '../data/provenance';

export const REPORT_SHARE_FACTOR = 0.3;
export const DAYS_PER_MONTH = 30;
export const DEFAULT_AVG_TOKENS = ph(15300, 'Report §5.3', "Qwen3.8-27B's average request, used for models whose own average is not stated");

export interface Calibration {
  /** Output tokens as a share of all tokens in a request. */
  outShare: number;
  /** Cache-read price as a fraction of the input price. */
  cacheFraction: number;
  /** Multiplies the §5.2 throughput curve so a busy hour earns the report's §6.1 figure. */
  tpsScale: number;
  avgTokens: number;
  basis: 'busy-hour + typical entrant' | 'busy-hour only';
  note: string;
  provs: Prov[];
}

const cache = new Map<string, Calibration | null>();

const clampShare = (s: number) => Math.min(0.5, Math.max(0.005, s));

/**
 * The report publishes, per model, what a fully busy hour earns (§6.1) and what a typical new endpoint
 * grosses (§6.3). Together with the §5.2 throughput point, those two figures pin the request's output
 * share and the cache-read price. For Qwen3.8-27B and gpt-oss-120b the output share lands within half a
 * point of the 4.1% and 11.8% the report states, which is the independent check on the method.
 */
export function calibrationFor(model: Model): Calibration | null {
  if (cache.has(model.id)) return cache.get(model.id)!;
  const out = compute(model);
  cache.set(model.id, out);
  return out;
}

function compute(model: Model): Calibration | null {
  const mr = MARKET_BY_MODEL[model.id];
  const ref = model.pro6000[model.refIdx];
  if (!mr || !ref || !mr.busyHourMedianUsd) return null;
  const busy = mr.busyHourMedianUsd.v;
  const outPerHourM = (ref.tps.v * 3600) / 1e6;
  const r = busy / outPerHourM; // dollars per million output tokens, input included
  const inP = mr.inMed.v;
  const outP = mr.outMed.v;
  const c = mr.cacheHit.v;
  const stated = mr.statedShape;
  const avgTokens = stated ? stated.avgTokens.v : DEFAULT_AVG_TOKENS.v;
  const provs: Prov[] = [mr.busyHourMedianUsd.p, ref.tps.p, mr.cacheHit.p, stated ? stated.avgTokens.p : DEFAULT_AVG_TOKENS.p];

  if (!mr.typicalMedianUsd) {
    const cf = CACHE_PRICE_FRACTION.v;
    const q = (1 - c) * inP + c * cf * inP;
    const s = clampShare(1 / (1 + (r - outP) / q));
    return {
      outShare: s,
      cacheFraction: cf,
      tpsScale: 1,
      avgTokens,
      basis: 'busy-hour only',
      note: `Output share ${(s * 100).toFixed(1)}% back-solved from the busy-hour figure alone; the report gives no uncapped typical-entrant figure for this model, so cache-read price stays at ${(cf * 100).toFixed(0)}% of input.`,
      provs,
    };
  }

  provs.push(mr.typicalMedianUsd.p);
  const tokens = (REPORT_SHARE_FACTOR / Math.max(1, mr.endpoints.v)) * mr.paidTokensPerDayB.v * 1e9 * DAYS_PER_MONTH;
  const blended = (mr.typicalMedianUsd.v * 1e6) / tokens; // dollars per million average tokens
  let s = clampShare(blended / r);
  let q = (blended - s * outP) / (1 - s);
  let cacheP = c > 0.01 ? (q - (1 - c) * inP) / c : CACHE_PRICE_FRACTION.v * inP;
  let tpsScale = 1;
  let clamped = false;
  if (c > 0.01 && (cacheP < 0 || cacheP > inP)) {
    clamped = true;
    cacheP = Math.min(inP, Math.max(0, cacheP));
    q = (1 - c) * inP + c * cacheP;
    s = clampShare((blended - q) / (outP - q));
    const rImpl = blended / s;
    tpsScale = busy / (rImpl * (3600 / 1e6)) / ref.tps.v;
  }
  const cf = inP > 0 ? cacheP / inP : CACHE_PRICE_FRACTION.v;
  return {
    outShare: s,
    cacheFraction: cf,
    tpsScale,
    avgTokens,
    basis: 'busy-hour + typical entrant',
    note: `Output share ${(s * 100).toFixed(1)}% and cache-read price ${(cf * 100).toFixed(0)}% of input, back-solved from the report's busy-hour and typical-entrant figures${stated ? `; the report states ${(stated.outShare.v * 100).toFixed(1)}% output` : ''}${clamped ? `; the throughput point was rescaled by ${tpsScale.toFixed(2)} to stay consistent` : ''}.`,
    provs,
  };
}
