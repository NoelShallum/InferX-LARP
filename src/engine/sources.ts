import { type Gpu } from '../data/gpus';
import { effectiveRent, liveMarket, rentFellBack, routerSpeed, type LiveMarket } from '../data/live';
import { MARKET_BY_MODEL, type MarketRow } from '../data/market';
import type { V } from '../data/provenance';
import type { Settings } from './types';

export interface ResolvedMarket {
  row: MarketRow;
  live: LiveMarket | null;
}

/**
 * The market row for a model: live OpenRouter numbers over the report's row when the live source is chosen and
 * available. A model the report has no row for is priced from live data whichever source is chosen.
 */
export function marketFor(modelId: string, settings: Settings): ResolvedMarket | null {
  const base = MARKET_BY_MODEL[modelId];
  if (!base) {
    const live = liveMarket(modelId, settings.medianVolume);
    return live ? { row: live.row, live } : null;
  }
  if (settings.marketSource === 'live') {
    const live = liveMarket(modelId, settings.medianVolume);
    if (live) return { row: live.row, live };
  }
  return { row: base, live: null };
}

/** Rent per GPU-hour for a node of this size, with provenance. */
export function rentFor(gpu: Gpu, nodeSize: number, settings: Settings): V | null {
  return effectiveRent(gpu, nodeSize, { source: settings.rentSource, view: settings.rentView, stat: settings.rentStat });
}

export function rentIsFallback(gpu: Gpu, nodeSize: number, settings: Settings): boolean {
  return rentFellBack(gpu, nodeSize, { source: settings.rentSource, view: settings.rentView, stat: settings.rentStat });
}

/** The tok/s/user floor for a model under the current quality bar. */
export function floorFor(modelId: string, settings: Settings): { floor: number; v: V | null } {
  if (settings.qos.mode === 'router') {
    const s = routerSpeed(modelId);
    if (s) return { floor: s.v, v: s };
  }
  return { floor: settings.qos.floorTps, v: null };
}
