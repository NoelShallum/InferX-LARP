import { CACHE_PRICE_FRACTION, type MarketRow } from '../data/market';
import type { Model } from '../data/models';
import { type Prov, ph, drv } from '../data/provenance';
import { ELASTICITY_PRICE } from '../data/scenarios';
import { calibrationFor, DAYS_PER_MONTH, DEFAULT_AVG_TOKENS } from './calibration';
import type { Pricing } from './types';

export { DAYS_PER_MONTH, DEFAULT_AVG_TOKENS };
export const ELASTICITY_CEILING = 1.2;
export const ABOVE_CEILING_PENALTY_POWER = ph(2, 'No source yet', 'Report: no data above 1.2× the median and a dear endpoint falls to the bottom of the routing order; the steepness is a guess', undefined, 'exponent on the price ratio');

export interface Shape {
  avgTokens: number;
  outShare: number;
  cacheHit: number;
  /** Cache-read price as a fraction of the input price. */
  cacheFraction: number;
  provs: Prov[];
  calibrated: boolean;
  note: string;
}

export interface Prices {
  inP: number;
  outP: number;
  cacheP: number;
}

/** Dollars per million average tokens (input and output together). */
export function blendedPerM(p: Prices, outShare: number, cacheHit: number): number {
  const q = (1 - cacheHit) * p.inP + cacheHit * p.cacheP;
  return outShare * p.outP + (1 - outShare) * q;
}

export function medianPrices(mr: MarketRow, cf: number): Prices {
  return { inP: mr.inMed.v, outP: mr.outMed.v, cacheP: mr.inMed.v * cf };
}

export function floorPrices(mr: MarketRow, cf: number): Prices {
  return { inP: mr.inMin.v, outP: mr.outMin.v, cacheP: mr.inMin.v * cf };
}

export function calibratedShape(model: Model, mr: MarketRow): Shape {
  const cal = calibrationFor(model);
  if (!cal) {
    return {
      avgTokens: DEFAULT_AVG_TOKENS.v,
      outShare: 0.05,
      cacheHit: mr.cacheHit.v,
      cacheFraction: CACHE_PRICE_FRACTION.v,
      provs: [DEFAULT_AVG_TOKENS.p, mr.cacheHit.p, CACHE_PRICE_FRACTION.p],
      calibrated: false,
      note: 'No calibration available; placeholder shape',
    };
  }
  return {
    avgTokens: cal.avgTokens,
    outShare: cal.outShare,
    cacheHit: mr.cacheHit.v,
    cacheFraction: cal.cacheFraction,
    provs: [drv(0, 'Report §6.1, §6.3 and §5.2', cal.note).p, ...cal.provs],
    calibrated: true,
    note: cal.note,
  };
}

export function pricesFor(mr: MarketRow, pricing: Pricing, cf: number): Prices {
  const med = medianPrices(mr, cf);
  switch (pricing.mode) {
    case 'median':
      return med;
    case 'floor':
      return floorPrices(mr, cf);
    case 'multiplier':
      return { inP: med.inP * pricing.mult, outP: med.outP * pricing.mult, cacheP: med.cacheP * pricing.mult };
    case 'batch': {
      const f = 1 - pricing.batchDiscount;
      return { inP: med.inP * f, outP: med.outP * f, cacheP: med.cacheP * f };
    }
    case 'custom':
      return { inP: pricing.customIn, outP: pricing.customOut, cacheP: pricing.customIn * cf };
  }
}

export function scalePrices(p: Prices, f: number): Prices {
  return { inP: p.inP * f, outP: p.outP * f, cacheP: p.cacheP * f };
}

/** How far the chosen price sits from the median, at one request shape. */
export function priceRatio(mr: MarketRow, pricing: Pricing, outShare: number, cacheHit: number, cf: number): number {
  const chosen = blendedPerM(pricesFor(mr, pricing, cf), outShare, cacheHit);
  const med = blendedPerM(medianPrices(mr, cf), outShare, cacheHit);
  return chosen / med;
}

export function elasticityFactor(m: number): { factor: number; aboveData: boolean } {
  if (m <= ELASTICITY_CEILING) return { factor: Math.pow(m, ELASTICITY_PRICE.v), aboveData: false };
  const atCeiling = Math.pow(ELASTICITY_CEILING, ELASTICITY_PRICE.v);
  return { factor: atCeiling * Math.pow(ELASTICITY_CEILING / m, ABOVE_CEILING_PENALTY_POWER.v), aboveData: true };
}
