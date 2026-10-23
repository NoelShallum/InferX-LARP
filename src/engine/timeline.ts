import { RAMP, FLOOR_MONTHS, MONTHS, RENT_RULES_EXIT } from './timeline-constants';
import { evaluateSlot, resolveCase, type SlotResult } from './slot';
import type { CaseId } from '../data/scenarios';
import type { Settings, Slot } from './types';

export type MonthState = 'queue' | 'floor' | 'ramp' | 'steady' | 'reverted' | 'handed-back' | 'lium';

export interface MonthRow {
  m: number;
  state: MonthState;
  revenueUsd: number;
  /** Rent for rented and bought slots. */
  costUsd: number;
  barUsd: number;
  /** Revenue minus rent or bar. Zero while the box stays on Lium. */
  gainUsd: number;
  utilisation: number;
}

export interface SlotTimeline {
  slotId: string;
  label: string;
  firstListed: number;
  months: MonthRow[];
  totalRevenueUsd: number;
  totalGainUsd: number;
  endedMonth: number | null;
  endedBy: 'reverted' | 'handed-back' | null;
  /** Tokens a month by month this slot could not serve, for overflow slots to pick up. */
  unservedTokens: number[];
}

export function listingMonthFor(slot: Slot, settings: Settings, caseId?: CaseId): number {
  return Math.max(1, Math.round(resolveCase(settings, caseId).listingMonth + slot.listingOffset));
}

export function slotTimeline(slot: Slot, settings: Settings, caseId?: CaseId, overflowByMonth?: number[]): SlotTimeline {
  const caseR = resolveCase(settings, caseId);
  const L = listingMonthFor(slot, settings, caseId);
  const months: MonthRow[] = [];
  const unserved: number[] = [];
  let ended: SlotTimeline['endedBy'] = null;
  let endedMonth: number | null = null;
  let totalRevenue = 0;
  let totalGain = 0;
  const lium = slot.modelId === 'lium';
  const rentStart = L + (slot.ownership === 'rent' ? slot.rentStartOffset : 0);

  for (let t = 1; t <= MONTHS.v; t++) {
    unserved.push(0);
    const blank = (state: MonthState): MonthRow => ({ m: t, state, revenueUsd: 0, costUsd: 0, barUsd: 0, gainUsd: 0, utilisation: 0 });
    if (lium) {
      const r = evaluateSlot(slot, settings, { caseId });
      months.push({ ...blank('lium'), barUsd: r.barUsd, revenueUsd: r.barUsd });
      continue;
    }
    if (ended) {
      months.push(blank(ended));
      continue;
    }
    if (t < L || (slot.ownership === 'rent' && t < rentStart)) {
      months.push(blank('queue'));
      continue;
    }
    const j = t - L + 1;
    const ramp = j === 1 ? RAMP[0]!.v : j === 2 ? RAMP[1]!.v : 1;
    const floorNow = settings.launchFloor && slot.pricing.mode === 'median' && j <= FLOOR_MONTHS.v;
    const pricingOverride = floorNow ? { ...slot.pricing, mode: 'floor' as const } : undefined;
    const trendFactor = Math.pow(1 + caseR.priceTrend, t - 1);
    const res: SlotResult = evaluateSlot(slot, settings, { caseId, rampFactor: ramp, trendFactor, pricingOverride, overflowTokens: overflowByMonth?.[t - 1] ?? 0 });
    if (!res.ok) {
      months.push(blank('queue'));
      continue;
    }
    unserved[t - 1] = Math.max(0, res.demandTokens - res.capTokens);
    const comparator = slot.ownership === 'owned' ? res.barUsd : res.costUsd;
    const gain = res.revenueUsd - comparator;
    totalRevenue += res.revenueUsd;
    totalGain += gain;
    months.push({
      m: t,
      state: floorNow ? 'floor' : j <= 2 ? 'ramp' : 'steady',
      revenueUsd: res.revenueUsd,
      costUsd: slot.ownership === 'owned' ? 0 : res.costUsd,
      barUsd: slot.ownership === 'owned' ? res.barUsd : 0,
      gainUsd: gain,
      utilisation: res.utilisation,
    });
    if (settings.killRules && j > FLOOR_MONTHS.v) {
      if (slot.ownership === 'owned' && res.revenueUsd < res.barUsd) {
        ended = 'reverted';
        endedMonth = t;
      } else if (slot.ownership === 'rent' && res.revenueUsd < RENT_RULES_EXIT.v * res.costUsd) {
        ended = 'handed-back';
        endedMonth = t;
      }
    }
  }
  return { slotId: slot.id, label: slot.label, firstListed: L, months, totalRevenueUsd: totalRevenue, totalGainUsd: totalGain, endedMonth, endedBy: ended, unservedTokens: unserved };
}
