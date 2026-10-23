import { GPU_BY_ID, HOURS_PER_MONTH, PRO6000_EDITION, SUMMER_DERATE, type Gpu } from '../data/gpus';
import { MARKET_BY_MODEL } from '../data/market';
import { floorFor, marketFor, rentFor, rentIsFallback } from './sources';
import { MODEL_BY_ID } from '../data/models';
import { type Prov, ph } from '../data/provenance';
import {
  AVAILABILITY_PRESETS,
  BUY_HORIZON_MONTHS,
  LIUM,
  PROGRAM,
  SCENARIOS,
  UPTIME_TIERS,
  WORKLOAD_BY_ID,
  ZDR_SHARE_BONUS,
  ELASTICITY_LATENCY,
  type CaseId,
} from '../data/scenarios';
import { tieInr, taxRate } from './bars';
import { capacityFor, type Capacity } from './capacity';
import {
  DAYS_PER_MONTH,
  blendedPerM,
  calibratedShape,
  elasticityFactor,
  priceRatio,
  pricesFor,
  scalePrices,
  type Prices,
  type Shape,
} from './pricing';
import type { Pricing, ResolvedCase, Settings, Slot } from './types';

export const POWER_TARIFF_INR_PER_KWH = ph(8, 'No source yet', 'Electricity price used for buy-slot power cost; the report folds power into the Lium bar instead');
export const AVG_POWER_FRACTION = ph(0.8, 'No source yet', 'Average draw as a fraction of TDP while serving');

export function resolveCase(settings: Settings, caseId: CaseId = settings.caseId): ResolvedCase {
  const sc = SCENARIOS[caseId];
  const o = settings.override;
  return {
    shareFactor: o.shareFactor ?? sc.shareFactor.v,
    latencyLoss: o.latencyLoss ?? sc.latencyLoss.v,
    priceTrend: o.priceTrend ?? sc.priceTrend.v,
    listingMonth: o.listingMonth ?? sc.listingMonth.v,
    queueMonths: sc.queueMonths.v,
    reviewMonths: sc.reviewMonths.v,
  };
}

export function availabilityFor(slot: Slot, settings: Settings): { value: number; prov: Prov; label: string } {
  const id = slot.availabilityId === 'global' ? settings.availabilityId : slot.availabilityId;
  let value: number;
  let prov: Prov;
  let label: string;
  if (slot.availabilityId === 'global' && settings.availabilityCustom !== null) {
    value = settings.availabilityCustom;
    prov = { kind: 'user', src: 'Your input' };
    label = 'Custom';
  } else {
    const p = AVAILABILITY_PRESETS.find((x) => x.id === id) ?? AVAILABILITY_PRESETS[0]!;
    value = p.availability.v;
    prov = p.availability.p;
    label = p.label;
  }
  value = Math.min(1, Math.max(0.3, value + settings.whatIf.availabilityDelta));
  return { value, prov, label };
}

export function uptimeTier(avail: number) {
  return UPTIME_TIERS.find((t) => avail >= t.min) ?? UPTIME_TIERS[UPTIME_TIERS.length - 1]!;
}

export interface SlotEvalOpts {
  /** Multiplies the share won, used for the first listed months. */
  rampFactor?: number;
  /** Market-wide price trend factor. */
  trendFactor?: number;
  pricingOverride?: Pricing;
  /** Tokens a month this slot picks up from another slot, for overflow demand. */
  overflowTokens?: number;
  caseId?: CaseId;
}

export interface EvidenceItem {
  label: string;
  prov: Prov;
}

export interface SlotResult {
  slot: Slot;
  kind: 'lium' | 'serve';
  ok: boolean;
  reason: string;
  gpu: Gpu;
  capacity: Capacity | null;
  shape: Shape | null;
  availability: number;
  availabilityLabel: string;
  uptimeTier: string;
  // tokens a month
  capTokens: number;
  demandTokens: number;
  servedTokens: number;
  /** Share of capacity sold while the card is up. */
  utilisation: number;
  // dollars
  blendedUsdPerM: number;
  priceRatio: number;
  /** The cheapest endpoint's blended price over the median's, at this request shape. */
  floorRatio: number;
  aboveElasticityData: boolean;
  revenueUsd: number;
  capacityRevenueUsd: number;
  costUsd: number;
  /** For owned slots: the Lium tie. For rented and bought: 0. */
  barUsd: number;
  marginUsd: number;
  afterTaxMarginUsd: number;
  /** Utilisation at which revenue equals cost or bar; null if even a full card falls short. */
  breakEvenUtil: number | null;
  /** Price scale (1 = current price) at which revenue equals cost or bar; null if never. */
  breakEvenPriceScale: number | null;
  prefillTpsAtCapacity: number;
  paybackServingMonths: number | null;
  paybackLiumMonths: number | null;
  shareOfTraffic: number;
  /** A rented card whose live price was unavailable under the current filter. */
  rentFallback: boolean;
  evidence: EvidenceItem[];
  hasPlaceholder: boolean;
}

const emptyResult = (slot: Slot, gpu: Gpu, reason: string, extra: Partial<SlotResult> = {}): SlotResult => ({
  slot,
  kind: 'serve',
  ok: false,
  reason,
  gpu,
  capacity: null,
  shape: null,
  availability: 1,
  availabilityLabel: '',
  uptimeTier: '',
  capTokens: 0,
  demandTokens: 0,
  servedTokens: 0,
  utilisation: 0,
  blendedUsdPerM: 0,
  priceRatio: 1,
  floorRatio: 1,
  aboveElasticityData: false,
  revenueUsd: 0,
  capacityRevenueUsd: 0,
  costUsd: 0,
  barUsd: 0,
  marginUsd: 0,
  afterTaxMarginUsd: 0,
  breakEvenUtil: null,
  breakEvenPriceScale: null,
  prefillTpsAtCapacity: 0,
  paybackServingMonths: null,
  paybackLiumMonths: null,
  shareOfTraffic: 0,
  rentFallback: false,
  evidence: [],
  hasPlaceholder: false,
  ...extra,
});

export function slotShape(slot: Slot, settings: Settings): Shape | null {
  const model = MODEL_BY_ID[slot.modelId];
  const resolved = marketFor(slot.modelId, settings);
  const base = MARKET_BY_MODEL[slot.modelId];
  const wl = WORKLOAD_BY_ID[slot.workloadId];
  if (!model || !resolved || !base || !wl) return null;
  // The report's calibration stays the anchor for capacity; live data only changes what a token sells for.
  const cal = calibratedShape(model, base);
  const live = resolved.live;
  const cacheFraction = live?.cacheFraction ?? cal.cacheFraction;
  const cacheProv: Prov[] = live?.cacheFraction != null ? [{ kind: 'live', src: 'OpenRouter public API', asOf: resolved.row.endpoints.p.asOf, note: `Median cache-read price is ${(live.cacheFraction * 100).toFixed(0)}% of the input price across endpoints that list one` }] : [];
  if (wl.shape === null) {
    if (live?.shape) {
      return {
        avgTokens: live.shape.avgTokens,
        outShare: live.shape.outShare,
        cacheHit: live.shape.cacheHit,
        cacheFraction,
        provs: [resolved.row.paidTokensPerDayB.p, resolved.row.cacheHit.p, ...cacheProv],
        calibrated: false,
        note: 'Request shape observed on OpenRouter over the last full days',
      };
    }
    return { ...cal, cacheFraction };
  }
  const cache = wl.shape.cacheHit ? wl.shape.cacheHit.v : resolved.row.cacheHit.v;
  return {
    avgTokens: wl.shape.avgTokens.v,
    outShare: wl.shape.outShare.v,
    cacheHit: cache,
    cacheFraction,
    provs: [wl.shape.avgTokens.p, wl.shape.outShare.p, wl.shape.cacheHit ? wl.shape.cacheHit.p : resolved.row.cacheHit.p, ...cacheProv],
    calibrated: false,
    note: `${wl.label} preset`,
  };
}

export function rentUsdPerHour(slot: Slot, gpu: Gpu, settings: Settings): number {
  if (slot.rentUsdHrOverride !== undefined) return slot.rentUsdHrOverride * settings.whatIf.rent;
  return (rentFor(gpu, slot.count, settings)?.v ?? 0) * settings.whatIf.rent;
}

export function evaluateSlot(slot: Slot, settings: Settings, opts: SlotEvalOpts = {}): SlotResult {
  const gpu = GPU_BY_ID[slot.gpuId]!;
  const proGpu = GPU_BY_ID['pro6000']!;
  const fx = settings.fx;
  const tools = PROGRAM.toolsInrPerMonth.v / fx;
  const caseR = resolveCase(settings, opts.caseId);
  const avail = availabilityFor(slot, settings);

  // Owned box left on Lium: earns exactly its bar.
  if (slot.modelId === 'lium') {
    const key = slot.boxKey ?? slot.gpuId;
    const l = LIUM[key];
    const bar = ((l ? tieInr(key, settings.route, settings.costLevel).totalInr : 0) * settings.whatIf.bar) / fx;
    return emptyResult(slot, gpu, 'On Lium', {
      kind: 'lium',
      ok: true,
      barUsd: bar,
      revenueUsd: bar,
      evidence: l ? [{ label: 'Lium net, 30 days to 22 Oct 2026', prov: l.netInr.p }] : [],
      availability: 1,
    });
  }

  const model = MODEL_BY_ID[slot.modelId];
  const resolved = marketFor(slot.modelId, settings);
  const mr = resolved?.row;
  if (!model || !mr || !resolved || !model.economics) {
    return emptyResult(slot, gpu, model ? `The report does not model revenue for ${model.name}` : 'Unknown model');
  }
  if (gpu.role === 'control-plane') return emptyResult(slot, gpu, `${gpu.short} is the control plane, not an LLM serving card`);

  const wl = WORKLOAD_BY_ID[slot.workloadId]!;
  const shape = slotShape(slot, settings)!;
  const owned = slot.ownership === 'owned';
  const evidence: EvidenceItem[] = [];

  const edition = owned && gpu.id === 'pro6000' ? PRO6000_EDITION[settings.proEdition] : null;
  const derate = owned && settings.summerDerate;
  const extraNotes: { name: string; value: number; prov: Prov }[] = [];
  let extraMult = settings.whatIf.throughput;
  if (settings.whatIf.throughput !== 1) extraNotes.push({ name: 'What-if throughput', value: settings.whatIf.throughput, prov: { kind: 'user', src: 'Your input' } });
  if (edition && edition.decode !== 1) {
    extraMult *= edition.decode;
    extraNotes.push({ name: `${edition.label} edition`, value: edition.decode, prov: edition.prov });
  }
  if (derate) {
    extraMult *= SUMMER_DERATE.v;
    extraNotes.push({ name: 'Summer power cap', value: SUMMER_DERATE.v, prov: SUMMER_DERATE.p });
  }

  const routerFloor = settings.qos.mode === 'router' ? floorFor(slot.modelId, settings) : null;
  const qos = wl.batch
    ? { mode: 'floor' as const, floorTps: wl.floorTps.v }
    : settings.qos.mode === 'report'
      ? { mode: 'report' as const, floorTps: settings.qos.floorTps }
      : { mode: 'floor' as const, floorTps: routerFloor ? routerFloor.floor : settings.qos.floorTps };
  // The report's throughput points are taken to describe the model's traffic as observed; other workloads are adjusted from there.
  const observed = slotShape({ ...slot, workloadId: 'observed' }, settings);
  const capacity = capacityFor({
    model,
    gpu,
    proGpu,
    count: slot.count,
    quant: slot.quant,
    mtp: slot.mtp,
    qos: { ...qos, ttftP90Sec: settings.qos.ttftP90Sec },
    extraMult,
    extraNotes,
    shape: { avgTokens: shape.avgTokens, outShare: shape.outShare, cacheHit: shape.cacheHit },
    baseShape: observed ? { avgTokens: observed.avgTokens, outShare: observed.outShare, cacheHit: observed.cacheHit } : undefined,
    basis: settings.throughput,
    prefillMult: edition ? edition.prefill : 1,
    batch: wl.batch,
  });
  if (!capacity.ok) return emptyResult(slot, gpu, capacity.reason, { capacity, shape });

  const pricing = opts.pricingOverride ?? slot.pricing;
  const trend = opts.trendFactor ?? 1;
  const basePrices = pricesFor(mr, pricing, shape.cacheFraction);
  const prices: Prices = scalePrices(basePrices, settings.whatIf.price * trend);
  const blended = blendedPerM(prices, shape.outShare, shape.cacheHit);
  const ratio = pricing.mode === 'batch' ? 1 : priceRatio(mr, pricing, shape.outShare, shape.cacheHit, shape.cacheFraction);

  const hours = HOURS_PER_MONTH.v;
  const capTokens = (capacity.tpsNode * 3600 * hours) / shape.outShare;
  const capacityRevenueUsd = (capTokens * blended) / 1e6;

  // Demand
  let demandTokens: number;
  let shareOfTraffic = 0;
  let aboveData = false;
  const tier = uptimeTier(avail.value);
  if (slot.demand.mode === 'utilisation') {
    demandTokens = slot.demand.utilisation * capTokens;
  } else if (slot.demand.mode === 'overflow') {
    demandTokens = opts.overflowTokens ?? 0;
  } else {
    const volB = !resolved.live && settings.medianVolume && mr.paidTokens14dMedianB ? mr.paidTokens14dMedianB.v : mr.paidTokensPerDayB.v;
    const el = elasticityFactor(ratio);
    aboveData = el.aboveData;
    const share = Math.min(
      1,
      caseR.shareFactor *
        slot.demand.shareScale *
        settings.whatIf.demand *
        (1 / Math.max(1, mr.endpoints.v)) *
        el.factor *
        (1 - caseR.latencyLoss) *
        (settings.zdr ? 1 + ZDR_SHARE_BONUS.v : 1) *
        tier.mult.v *
        (opts.rampFactor ?? 1),
    );
    shareOfTraffic = share;
    demandTokens = volB * 1e9 * DAYS_PER_MONTH * share;
  }
  const servedTokens = Math.min(demandTokens, capTokens);
  const utilisation = capTokens > 0 ? servedTokens / capTokens : 0;
  const revenueUsd = (servedTokens * avail.value * blended) / 1e6;

  // Costs
  let costUsd = 0;
  let barUsd = 0;
  let rentFallback = false;
  let paybackServing: number | null = null;
  let paybackLium: number | null = null;
  if (slot.ownership === 'owned') {
    const key = slot.boxKey ?? slot.gpuId;
    barUsd = ((LIUM[key] ? tieInr(key, settings.route, settings.costLevel).totalInr : 0) * settings.whatIf.bar) / fx;
    if (LIUM[key]) evidence.push({ label: 'Lium bar (fiat tie)', prov: { kind: 'derived', src: 'Report §2.2', note: 'After-tax Lium ÷ (1 − tax) plus power, monitoring, UPS write-off, failover cap' } });
  } else if (slot.ownership === 'rent') {
    costUsd = rentUsdPerHour(slot, gpu, settings) * slot.count * hours;
    const rentProv = slot.rentUsdHrOverride !== undefined ? ({ kind: 'user', src: 'Your input' } as Prov) : rentFor(gpu, slot.count, settings)?.p;
    if (rentProv) evidence.push({ label: 'Rent per GPU-hour', prov: rentProv });
    rentFallback = slot.rentUsdHrOverride === undefined && rentIsFallback(gpu, slot.count, settings);
  } else {
    const buy = slot.buyPriceInr ?? gpu.buyInr?.v ?? 0;
    const capexUsd = (buy * slot.count) / fx;
    const powerUsd = ((gpu.tdpW.v * AVG_POWER_FRACTION.v * slot.count) / 1000 * hours * POWER_TARIFF_INR_PER_KWH.v) / fx;
    costUsd = capexUsd / BUY_HORIZON_MONTHS.v + powerUsd;
    const net = revenueUsd - powerUsd - tools;
    paybackServing = net > 0 ? capexUsd / net : null;
    const l = LIUM[gpu.id];
    paybackLium = l ? capexUsd / (l.netInr.v / fx) : null;
    evidence.push({ label: 'Purchase price', prov: (slot.buyPriceInr ? { kind: 'user', src: 'Your input' } : (gpu.buyInr?.p ?? ph(0, 'No source yet').p)) as Prov });
    evidence.push({ label: 'Power tariff', prov: POWER_TARIFF_INR_PER_KWH.p });
  }
  if (slot.ownership !== 'owned') costUsd += slot.ownership === 'rent' ? 0 : tools;

  const comparator = slot.ownership === 'owned' ? barUsd : costUsd;
  const marginUsd = revenueUsd - comparator;
  const tf = taxRate(settings.route);
  let afterTaxMarginUsd: number;
  if (slot.ownership === 'owned') afterTaxMarginUsd = marginUsd * (1 - tf);
  else if (settings.route === 'fiat') afterTaxMarginUsd = marginUsd * (1 - tf);
  else afterTaxMarginUsd = revenueUsd * (1 - tf) - comparator;

  const fullCapacityRevenue = capacityRevenueUsd * avail.value;
  const breakEvenUtil = fullCapacityRevenue > 0 && comparator / fullCapacityRevenue <= 1 ? comparator / fullCapacityRevenue : null;

  // Price scale that equalises revenue and comparator, honouring elasticity in share mode.
  let breakEvenPriceScale: number | null = null;
  if (comparator > 0 && revenueUsd > 0) {
    if (slot.demand.mode !== 'share') {
      breakEvenPriceScale = comparator / revenueUsd;
    } else {
      const baseEl = elasticityFactor(ratio).factor;
      const revAt = (p: number) => {
        const m = ratio * p;
        const dTokens = demandTokens * (elasticityFactor(m).factor / baseEl);
        return (Math.min(dTokens, capTokens) * avail.value * blended * p) / 1e6;
      };
      let prev = 0.05;
      for (let i = 1; i <= 400; i++) {
        const p = 0.05 + (i * (4 - 0.05)) / 400;
        if (revAt(p) >= comparator) {
          let lo = prev;
          let hi = p;
          for (let j = 0; j < 40; j++) {
            const mid = (lo + hi) / 2;
            if (revAt(mid) >= comparator) hi = mid;
            else lo = mid;
          }
          breakEvenPriceScale = hi;
          break;
        }
        prev = p;
      }
    }
  }

  const reqPerSec = capacity.tpsNode / (shape.outShare * shape.avgTokens);
  const prefillTpsAtCapacity = reqPerSec * shape.avgTokens * (1 - shape.outShare) * (1 - shape.cacheHit);

  evidence.push(
    ...capacity.curve!.baseProvs.map((p, i) => ({ label: `Throughput point ${i + 1}`, prov: p })),
    { label: capacity.curve!.multNote, prov: capacity.curve!.multProv },
    { label: 'Weights', prov: capacity.fit.weightsProv },
    { label: 'KV bytes per token', prov: model.kvBytesPerTok.p },
    ...shape.provs.map((p, i) => ({ label: `Request shape ${i + 1}`, prov: p })),
    { label: 'Paid tokens per day', prov: mr.paidTokensPerDayB.p },
    { label: 'Endpoints', prov: mr.endpoints.p },
    { label: 'Median price', prov: mr.outMed.p },
    ...(routerFloor?.v ? [{ label: `Quality bar: median OpenRouter stream, ${routerFloor.floor} tok/s`, prov: routerFloor.v.p }] : []),
    { label: 'Share factor', prov: SCENARIOS[opts.caseId ?? settings.caseId].shareFactor.p },
    { label: 'Latency loss', prov: SCENARIOS[opts.caseId ?? settings.caseId].latencyLoss.p },
    { label: 'Latency elasticity', prov: ELASTICITY_LATENCY.p },
    { label: `Availability: ${avail.label}`, prov: avail.prov },
  );
  const hasPlaceholder = evidence.some((e) => e.prov.kind === 'placeholder') || capacity.multipliers.some((m) => m.prov.kind === 'placeholder');

  return {
    slot,
    kind: 'serve',
    ok: true,
    reason: capacity.reason,
    gpu,
    capacity,
    shape,
    availability: avail.value,
    availabilityLabel: avail.label,
    uptimeTier: tier.label,
    capTokens,
    demandTokens,
    servedTokens,
    utilisation,
    blendedUsdPerM: blended,
    priceRatio: ratio,
    floorRatio: priceRatio(mr, { ...pricing, mode: 'floor' }, shape.outShare, shape.cacheHit, shape.cacheFraction),
    aboveElasticityData: aboveData,
    revenueUsd,
    capacityRevenueUsd,
    costUsd,
    barUsd,
    marginUsd,
    afterTaxMarginUsd,
    breakEvenUtil,
    breakEvenPriceScale,
    prefillTpsAtCapacity,
    paybackServingMonths: paybackServing,
    paybackLiumMonths: paybackLium,
    shareOfTraffic,
    rentFallback,
    evidence,
    hasPlaceholder,
  };
}
