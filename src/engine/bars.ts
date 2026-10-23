import { LIUM, ROUTE_COSTS, TAX, type CostLevel } from '../data/scenarios';
import type { Route } from './types';

export interface TieParts {
  /** Revenue that leaves the same after-tax result as staying on Lium, before route costs. */
  liumEquivalentInr: number;
  extraPowerInr: number;
  monitoringInr: number;
  upsInr: number;
  failoverInr: number;
  totalInr: number;
}

/**
 * Monthly revenue an inference route needs for the same after-tax result as Lium (report §2.2).
 * Fiat: after-tax Lium ÷ (1 − 25.17%) plus power, monitoring, UPS write-off and failover cap.
 * Crypto: taxed on gross at 31.2% with no deduction; the failover cap is not charged.
 */
export function tieInr(boxKey: string, route: Route, level: CostLevel): TieParts {
  const l = LIUM[boxKey];
  if (!l) return { liumEquivalentInr: 0, extraPowerInr: 0, monitoringInr: 0, upsInr: 0, failoverInr: 0, totalInr: 0 };
  const monitoringInr = ROUTE_COSTS.monitoringInr[level].v;
  const upsInr = ROUTE_COSTS.upsInr[level].v / ROUTE_COSTS.upsAmortMonths.v;
  const extraPowerInr = l.extraPowerInr.v;
  if (route === 'fiat') {
    const failoverInr = ROUTE_COSTS.failoverInr[level].v;
    const liumEquivalentInr = l.afterTaxInr.v / (1 - TAX.fiat.v);
    return { liumEquivalentInr, extraPowerInr, monitoringInr, upsInr, failoverInr, totalInr: liumEquivalentInr + extraPowerInr + monitoringInr + upsInr + failoverInr };
  }
  const costs = extraPowerInr + monitoringInr + upsInr;
  const liumEquivalentInr = l.afterTaxInr.v / (1 - TAX.crypto.v);
  const total = (l.afterTaxInr.v + costs) / (1 - TAX.crypto.v);
  return { liumEquivalentInr, extraPowerInr, monitoringInr, upsInr, failoverInr: 0, totalInr: total };
}

export function taxRate(route: Route): number {
  return route === 'fiat' ? TAX.fiat.v : TAX.crypto.v;
}
