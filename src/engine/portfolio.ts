import { GPU_BY_ID, HOURS_PER_MONTH } from '../data/gpus';
import { PROGRAM, type CaseId, CASES } from '../data/scenarios';
import { taxRate } from './bars';
import { evaluateSlot, resolveCase, type SlotResult } from './slot';
import { rentFor } from './sources';
import { listingMonthFor, slotTimeline, type SlotTimeline } from './timeline';
import { MONTHS } from './timeline-constants';
import type { Settings, Slot } from './types';

export interface PortfolioSteady {
  results: SlotResult[];
  revenueUsd: number;
  rentAndCostUsd: number;
  barUsd: number;
  toolsUsd: number;
  marginUsd: number;
  afterTaxMarginUsd: number;
  /** Capacity-weighted share of serving capacity that is sold. */
  utilisation: number;
  servingCount: number;
  failures: SlotResult[];
}

const isServing = (r: SlotResult) => r.kind === 'serve' && r.ok;

export function portfolioSteady(slots: Slot[], settings: Settings, caseId?: CaseId): PortfolioSteady {
  const byId = new Map<string, SlotResult>();
  const results: SlotResult[] = [];
  for (const slot of slots.filter((s) => s.demand.mode !== 'overflow')) {
    const r = evaluateSlot(slot, settings, { caseId });
    byId.set(slot.id, r);
  }
  for (const slot of slots) {
    if (slot.demand.mode === 'overflow') {
      const base = byId.get(slot.demand.of);
      const overflow = base && isServing(base) ? Math.max(0, base.demandTokens - base.capTokens) : 0;
      byId.set(slot.id, evaluateSlot(slot, settings, { caseId, overflowTokens: overflow }));
    }
    results.push(byId.get(slot.id)!);
  }
  const serving = results.filter(isServing);
  const ownedServing = serving.some((r) => r.slot.ownership === 'owned');
  const rentServing = serving.some((r) => r.slot.ownership === 'rent');
  const toolsUsd = rentServing && !ownedServing ? PROGRAM.toolsInrPerMonth.v / settings.fx : 0;
  const revenueUsd = results.reduce((a, r) => a + r.revenueUsd, 0);
  const rentAndCostUsd = results.reduce((a, r) => a + r.costUsd, 0) + toolsUsd;
  const barUsd = results.reduce((a, r) => a + r.barUsd, 0);
  const marginUsd = revenueUsd - rentAndCostUsd - barUsd;
  const tf = taxRate(settings.route);
  const afterTaxMarginUsd = results.reduce((a, r) => a + r.afterTaxMarginUsd, 0) - toolsUsd * (1 - tf);
  const capSum = serving.reduce((a, r) => a + r.capTokens, 0);
  const servedSum = serving.reduce((a, r) => a + r.servedTokens, 0);
  return {
    results,
    revenueUsd,
    rentAndCostUsd,
    barUsd,
    toolsUsd,
    marginUsd,
    afterTaxMarginUsd,
    utilisation: capSum > 0 ? servedSum / capSum : 0,
    servingCount: serving.length,
    failures: results.filter((r) => !r.ok),
  };
}

export interface PortfolioMonth {
  m: number;
  revenueUsd: number;
  rentUsd: number;
  programUsd: number;
  toolsUsd: number;
  gainUsd: number;
  cumGainUsd: number;
  /** Cumulative net spend counted against the stake cap. */
  stakeUsedUsd: number;
}

export interface ProgramCosts {
  buildUsd: number;
  queueUsd: number;
  reviewUsd: number;
  bridgeUsd: number;
  /** Cost if the endpoint is reviewed and turned down. */
  rejectedUsd: number;
  /** Cost if it is never reviewed. */
  neverUsd: number;
}

export interface PortfolioTimeline {
  caseId: CaseId;
  months: PortfolioMonth[];
  slots: SlotTimeline[];
  netUsd: number;
  revenueUsd: number;
  program: ProgramCosts;
  dependsOnListing: boolean;
  primaryListing: number;
  peakStakeUsd: number;
}

export function programCosts(slots: Slot[], settings: Settings, caseId?: CaseId): { costs: ProgramCosts; monthly: number[]; dependsOnListing: boolean } {
  const c = resolveCase(settings, caseId);
  const fx = settings.fx;
  const review = GPU_BY_ID[PROGRAM.reviewGpuId]!;
  const reviewRent = (rentFor(review, 1, settings)?.v ?? 0) * HOURS_PER_MONTH.v * settings.whatIf.rent;
  const tools = PROGRAM.toolsInrPerMonth.v / fx;
  const dependsOnListing = slots.some((s) => s.modelId !== 'lium');
  const buildUsd = PROGRAM.buildInr.v / fx;
  const queueUsd = (PROGRAM.queueInrPerMonth.v * c.queueMonths) / fx;
  const reviewUsd = c.reviewMonths * (reviewRent + tools);
  const ownedMoves = slots.some((s) => s.modelId !== 'lium' && s.ownership === 'owned');
  const bridgeUsd = ownedMoves ? reviewRent : 0;
  const monthly = Array.from({ length: MONTHS.v }, () => 0);
  if (dependsOnListing) {
    const L = Math.max(1, Math.round(c.listingMonth));
    monthly[0]! += buildUsd;
    for (let t = 1; t <= Math.min(c.queueMonths, MONTHS.v); t++) monthly[t - 1]! += PROGRAM.queueInrPerMonth.v / fx;
    const first = Math.max(1, L - c.reviewMonths);
    for (let t = first; t <= L - 1 && t <= MONTHS.v; t++) monthly[t - 1]! += reviewRent + tools;
    if (L <= MONTHS.v) monthly[L - 1]! += bridgeUsd;
  }
  return {
    costs: { buildUsd, queueUsd, reviewUsd, bridgeUsd, rejectedUsd: buildUsd + queueUsd + reviewUsd, neverUsd: buildUsd + queueUsd },
    monthly,
    dependsOnListing,
  };
}

export function portfolioTimeline(slots: Slot[], settings: Settings, caseId: CaseId = settings.caseId): PortfolioTimeline {
  const tls = new Map<string, SlotTimeline>();
  for (const s of slots.filter((x) => x.demand.mode !== 'overflow')) tls.set(s.id, slotTimeline(s, settings, caseId));
  for (const s of slots.filter((x) => x.demand.mode === 'overflow')) {
    const base = tls.get(s.demand.of);
    tls.set(s.id, slotTimeline(s, settings, caseId, base?.unservedTokens));
  }
  const ordered = slots.map((s) => tls.get(s.id)!);
  const { costs, monthly, dependsOnListing } = programCosts(slots, settings, caseId);
  const tools = PROGRAM.toolsInrPerMonth.v / settings.fx;
  const ownedServing = slots.some((s) => s.ownership === 'owned' && s.modelId !== 'lium');
  const months: PortfolioMonth[] = [];
  let cum = 0;
  let stake = 0;
  let peakStake = 0;
  let totalNet = 0;
  let totalRev = 0;
  for (let t = 1; t <= MONTHS.v; t++) {
    let revenue = 0;
    let rent = 0;
    let gain = 0;
    let rentedRevenue = 0;
    let rentedActive = false;
    ordered.forEach((tl, i) => {
      const row = tl.months[t - 1]!;
      const slot = slots[i]!;
      if (slot.modelId === 'lium') return;
      revenue += row.revenueUsd;
      rent += row.costUsd;
      gain += row.gainUsd;
      if (slot.ownership === 'rent') {
        rentedRevenue += row.revenueUsd;
        if (row.state !== 'queue' && row.state !== 'handed-back') rentedActive = true;
      }
    });
    const toolsM = rentedActive && !ownedServing ? tools : 0;
    const program = monthly[t - 1]!;
    const net = gain - program - toolsM;
    cum += net;
    stake += program + toolsM + rent - rentedRevenue;
    peakStake = Math.max(peakStake, stake);
    totalNet += net;
    totalRev += revenue;
    months.push({ m: t, revenueUsd: revenue, rentUsd: rent, programUsd: program, toolsUsd: toolsM, gainUsd: net, cumGainUsd: cum, stakeUsedUsd: stake });
  }
  const primary = Math.max(1, Math.round(resolveCase(settings, caseId).listingMonth));
  return { caseId, months, slots: ordered, netUsd: totalNet, revenueUsd: totalRev, program: costs, dependsOnListing, primaryListing: primary, peakStakeUsd: peakStake };
}

export interface EvRow {
  caseId: CaseId;
  listedNetUsd: number;
  rejectedUsd: number;
  neverUsd: number;
  evUsd: number;
  evAtOdds: { odds: number; usd: number }[];
  breakEvenOdds: number | null;
  firstListed: number;
}

/** Report §6.4: as many applications are reviewed and turned down as are listed; the rest wait. */
export function expectedValue(slots: Slot[], settings: Settings, oddsList: number[]): EvRow[] {
  return CASES.map((caseId) => {
    const tl = portfolioTimeline(slots, settings, caseId);
    const L = tl.netUsd;
    const R = tl.program.rejectedUsd;
    const N = tl.program.neverUsd;
    const ev = (p: number) => p * L - p * R - (1 - 2 * p) * N;
    const denom = L - R + 2 * N;
    return {
      caseId,
      listedNetUsd: L,
      rejectedUsd: -R,
      neverUsd: -N,
      evUsd: ev(settings.odds),
      evAtOdds: oddsList.map((odds) => ({ odds, usd: ev(odds) })),
      breakEvenOdds: denom > 0 && N > 0 ? N / denom : denom > 0 ? 0 : null,
      firstListed: tl.primaryListing,
    };
  });
}

export function afterTaxFactor(settings: Settings): number {
  return settings.afterTax ? 1 - taxRate(settings.route) : 1;
}

export { listingMonthFor };
