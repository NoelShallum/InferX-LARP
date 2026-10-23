import { MARKET_BY_MODEL } from '../data/market';
import { LIUM, type CaseId } from '../data/scenarios';
import { tieInr } from './bars';
import { DEFAULT_SETTINGS, makeSlot } from './defaults';
import { expectedValue, portfolioSteady, portfolioTimeline } from './portfolio';
import { evaluateSlot } from './slot';
import type { Pricing, Settings, Slot } from './types';

export interface CheckRow {
  id: string;
  group: string;
  label: string;
  /** The report's figure, in INR. */
  report: number;
  /** The dashboard's figure under the report's own conditions, in INR. */
  dashboard: number;
  unit: 'inr' | 'pct' | 'x' | 'months';
  /** Tolerance as a fraction of the report figure, or absolute points for pct. */
  tolerance: number;
  note?: string;
}

const LAKH = 1e5;
const L = (x: number) => x * LAKH;

/** Settings that match the conditions the report states: no outage haircut, no what-ifs. */
export function reportSettings(caseId: CaseId = 'base', over: Partial<Settings> = {}): Settings {
  return { ...DEFAULT_SETTINGS, caseId, availabilityCustom: 1, killRules: false, ...over };
}

const MODELS_61 = ['qwen38_27b', 'qwen36_35b', 'gemma4_26b', 'gptoss_120b', 'gemma4_31b', 'llama33_70b', 'mistral_nemo', 'gptoss_20b'];
const LABEL: Record<string, string> = {
  qwen38_27b: 'Qwen3.8-27B',
  qwen36_35b: 'Qwen3.6-35B-A3B',
  gemma4_26b: 'Gemma 4 26B-A4B',
  gptoss_120b: 'gpt-oss-120b',
  gemma4_31b: 'Gemma 4 31B',
  llama33_70b: 'Llama 3.3 70B',
  mistral_nemo: 'Mistral Nemo',
  gptoss_20b: 'gpt-oss-20b',
};

// Report §6.3 (no latency penalty, 0.3 share): used % and revenue at median and lowest price, rupees.
const T63: Record<string, { usedMed: number; revMed: number; usedLow: number; revLow: number }> = {
  qwen38_27b: { usedMed: 0.51, revMed: L(4.83), usedLow: 0.97, revLow: L(3.32) },
  qwen36_35b: { usedMed: 0.94, revMed: L(6.38), usedLow: 1, revLow: L(3.71) },
  gemma4_26b: { usedMed: 0.67, revMed: L(3.36), usedLow: 1, revLow: L(2.36) },
  gptoss_120b: { usedMed: 0.91, revMed: L(3.41), usedLow: 1, revLow: L(1.12) },
  gemma4_31b: { usedMed: 1, revMed: L(2.59), usedLow: 1, revLow: L(1.81) },
  llama33_70b: { usedMed: 0.4, revMed: L(1.68), usedLow: 0.97, revLow: L(1.01) },
  mistral_nemo: { usedMed: 1, revMed: L(1.51), usedLow: 1, revLow: 87124 },
  gptoss_20b: { usedMed: 0.44, revMed: L(1.16), usedLow: 0.61, revLow: 96026 },
};

// Report §6.6: steady month at the median price with the case's share and latency factor, rupees.
const T66: Record<string, Record<CaseId, { used: number; rev: number }>> = {
  qwen38_27b: { low: { used: 0.21, rev: L(2.0) }, base: { used: 0.46, rev: L(4.3) }, high: { used: 1, rev: L(9.44) } },
  qwen36_35b: { low: { used: 0.39, rev: L(2.65) }, base: { used: 0.84, rev: L(5.68) }, high: { used: 1, rev: L(6.79) } },
  gemma4_26b: { low: { used: 0.28, rev: L(1.4) }, base: { used: 0.6, rev: L(2.99) }, high: { used: 1, rev: L(5.0) } },
  gptoss_120b: { low: { used: 0.38, rev: L(1.42) }, base: { used: 0.81, rev: L(3.04) }, high: { used: 1, rev: L(3.74) } },
  gemma4_31b: { low: { used: 1, rev: L(2.59) }, base: { used: 1, rev: L(2.59) }, high: { used: 1, rev: L(2.59) } },
  llama33_70b: { low: { used: 0.17, rev: 69889 }, base: { used: 0.36, rev: L(1.5) }, high: { used: 1, rev: L(4.17) } },
  mistral_nemo: { low: { used: 0.52, rev: 77654 }, base: { used: 1, rev: L(1.51) }, high: { used: 1, rev: L(1.51) } },
  gptoss_20b: { low: { used: 0.18, rev: 48108 }, base: { used: 0.39, rev: L(1.03) }, high: { used: 1, rev: L(2.64) } },
};

const pro = (modelId: string, over: Partial<Slot> = {}) =>
  makeSlot({ id: 'p', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId, ...over });

export function runCrossChecks(): CheckRow[] {
  const rows: CheckRow[] = [];
  const add = (r: Omit<CheckRow, 'unit'> & { unit?: CheckRow['unit'] }) => rows.push({ unit: 'inr', ...r });
  const fx = DEFAULT_SETTINGS.fx;

  // §2.2 bars
  const bars: [string, 'low' | 'base' | 'high', number, number][] = [
    ['pro6000', 'low', 87_000, 0.01],
    ['pro6000', 'base', 92_460, 0.001],
    ['pro6000', 'high', 98_000, 0.01],
    ['ada6000', 'low', 59_000, 0.03],
    ['ada6000', 'base', 63_226, 0.001],
    ['ada6000', 'high', 68_000, 0.01],
  ];
  for (const [box, lvl, rv, tol] of bars) {
    add({ id: `bar-${box}-${lvl}`, group: '§2.2 Bar to beat (fiat route)', label: `${box === 'pro6000' ? 'PRO 6000' : 'Ada'} · ${lvl} cost level`, report: rv, dashboard: tieInr(box, 'fiat', lvl).totalInr, tolerance: tol });
  }
  add({ id: 'bar-crypto-pro', group: '§2.2 Bar to beat (crypto route)', label: 'PRO 6000 · base', report: 94_000, dashboard: tieInr('pro6000', 'crypto', 'base').totalInr, tolerance: 0.05, note: 'The report does not show how it priced the crypto route; this uses after-tax Lium plus power, monitoring and UPS over (1 − 31.2%).' });
  add({ id: 'bar-crypto-ada', group: '§2.2 Bar to beat (crypto route)', label: 'Ada · base', report: 61_000, dashboard: tieInr('ada6000', 'crypto', 'base').totalInr, tolerance: 0.06 });

  // §6.1 per busy hour at the lowest price, and saturated month at median
  for (const id of MODELS_61) {
    const mr = MARKET_BY_MODEL[id]!;
    const s = reportSettings('base');
    const med = evaluateSlot(pro(id), s);
    const low = evaluateSlot(pro(id, { pricing: { ...pro(id).pricing, mode: 'floor' } as Pricing }), s);
    add({ id: `busy-med-${id}`, group: '§6.1 Revenue per busy hour, median price', label: LABEL[id]!, report: mr.busyHourMedianUsd!.v * fx, dashboard: (med.capacityRevenueUsd / 730) * fx, tolerance: 0.01, note: 'Calibrated to the report by construction; shown so a change to the engine cannot drift unnoticed.' });
    add({ id: `busy-low-${id}`, group: '§6.1 Revenue per busy hour, lowest price', label: LABEL[id]!, report: mr.busyHourLowestUsd!.v * fx, dashboard: (low.capacityRevenueUsd / 730) * fx, tolerance: 0.2, note: 'An independent check: the lowest listed price is not used in the calibration.' });
  }

  // §6.3 typical entrant, no latency penalty
  for (const id of MODELS_61) {
    const t = T63[id]!;
    const s = reportSettings('base', { override: { shareFactor: null, latencyLoss: 0, priceTrend: null, listingMonth: null } });
    const med = evaluateSlot(pro(id), s);
    const low = evaluateSlot(pro(id, { pricing: { ...pro(id).pricing, mode: 'floor' } as Pricing }), s);
    add({ id: `t63-rev-med-${id}`, group: '§6.3 Typical entrant revenue, median price', label: LABEL[id]!, report: t.revMed, dashboard: med.revenueUsd * fx, tolerance: 0.1 });
    add({ id: `t63-used-med-${id}`, group: '§6.3 Card used, median price', label: LABEL[id]!, report: t.usedMed, dashboard: med.utilisation, unit: 'pct', tolerance: 0.08 });
    add({ id: `t63-rev-low-${id}`, group: '§6.3 Typical entrant revenue, lowest price', label: LABEL[id]!, report: t.revLow, dashboard: low.revenueUsd * fx, tolerance: 0.2 });
  }

  // §6.6 steady month by case
  for (const id of MODELS_61) {
    for (const caseId of ['low', 'base', 'high'] as CaseId[]) {
      const t = T66[id]![caseId];
      const r = evaluateSlot(pro(id), reportSettings(caseId));
      add({ id: `t66-${id}-${caseId}`, group: `§6.6 Steady month, ${caseId} case`, label: LABEL[id]!, report: t.rev, dashboard: r.revenueUsd * fx, tolerance: 0.12 });
    }
  }

  // §6.5 PRO 6000 on Qwen3.8-27B: twelve months
  const t65: Record<CaseId, { steady: number; m12: number; rev12: number; gain: number }> = {
    low: { steady: L(1.33), m12: L(1.14), rev12: L(5.86), gain: L(1.24) },
    base: { steady: L(3.88), m12: L(3.44), rev12: L(28.2), gain: L(20.8) },
    high: { steady: L(10.0), m12: L(11.7), rev12: L(104.5), gain: L(95.3) },
  };
  for (const caseId of ['low', 'base', 'high'] as CaseId[]) {
    const s = reportSettings(caseId);
    const tl = portfolioTimeline([pro('qwen38_27b')], s, caseId);
    const first = tl.slots[0]!.firstListed;
    const steady = tl.slots[0]!.months[first + 1]!;
    add({ id: `t65-steady-${caseId}`, group: '§6.5 Twelve months, Qwen3.8-27B on the PRO 6000', label: `Steady month (third listed), ${caseId}`, report: t65[caseId].steady, dashboard: steady.revenueUsd * fx, tolerance: 0.1 });
    add({ id: `t65-m12-${caseId}`, group: '§6.5 Twelve months, Qwen3.8-27B on the PRO 6000', label: `Month 12, ${caseId}`, report: t65[caseId].m12, dashboard: tl.slots[0]!.months[11]!.revenueUsd * fx, tolerance: 0.1 });
    add({ id: `t65-rev12-${caseId}`, group: '§6.5 Twelve months, Qwen3.8-27B on the PRO 6000', label: `Revenue over 12 months, ${caseId}`, report: t65[caseId].rev12, dashboard: tl.slots[0]!.totalRevenueUsd * fx, tolerance: 0.1 });
    const lium = LIUM['pro6000']!.netInr.v;
    const monthsOn = 12 - first + 1;
    add({ id: `t65-gain-${caseId}`, group: '§6.5 Twelve months, Qwen3.8-27B on the PRO 6000', label: `Gain over Lium net, ${caseId}`, report: t65[caseId].gain, dashboard: tl.slots[0]!.totalRevenueUsd * fx - lium * monthsOn, tolerance: 0.15, note: 'Revenue minus Lium net (₹80k) for each month on the route, the basis that reproduces the report.' });
  }

  // §6.4 expected value
  const ev = expectedValue([pro('qwen38_27b')], reportSettings('base', { odds: 0.1 }), [0.05, 0.1, 0.15]);
  const evBase = ev.find((e) => e.caseId === 'base')!;
  add({ id: 'ev-base-10', group: '§6.4 Expected value of applying', label: 'Base case at 10% odds', report: L(1.75), dashboard: evBase.evUsd * fx, tolerance: 0.2 });
  add({ id: 'ev-base-listed', group: '§6.4 Expected value of applying', label: 'Base case, listed: net over 12 months', report: L(19.1), dashboard: evBase.listedNetUsd * fx, tolerance: 0.12, note: 'Uses the tie as the bar, which includes UPS write-off and monitoring, instead of the report\'s Lium net plus stage costs.' });
  add({ id: 'ev-base-reject', group: '§6.4 Expected value of applying', label: 'Base case, reviewed and turned down', report: -65_274, dashboard: evBase.rejectedUsd * fx, tolerance: 0.02 });
  if (evBase.breakEvenOdds !== null) add({ id: 'ev-base-be', group: '§6.4 Expected value of applying', label: 'Base case break-even odds', report: 0.0064, dashboard: evBase.breakEvenOdds, unit: 'pct', tolerance: 0.004 });

  // §7.4 four-card case, steady month at median prices
  const four: Slot[] = [
    pro('qwen38_27b', { id: 'box-pro6000' }),
    makeSlot({ id: 'box-ada6000', ownership: 'owned', boxKey: 'ada6000', gpuId: 'ada6000', modelId: 'gemma4_26b' }),
    makeSlot({ id: 'rent-pro6000', ownership: 'rent', gpuId: 'pro6000', modelId: 'qwen36_35b', quant: 'fp8' }),
    makeSlot({ id: 'rent-5090', ownership: 'rent', gpuId: 'rtx5090', modelId: 'qwen38_27b', demand: { mode: 'overflow', utilisation: 0, shareScale: 1, of: 'box-pro6000' } as Slot['demand'] }),
  ];
  const t74: Record<CaseId, { rev: number; margin: number }> = {
    low: { rev: 604_762, margin: 285_927 },
    base: { rev: 1_296_960, margin: 978_125 },
    high: { rev: 2_624_364, margin: 2_305_529 },
  };
  for (const caseId of ['low', 'base', 'high'] as CaseId[]) {
    const ps = portfolioSteady(four, reportSettings(caseId));
    add({ id: `t74-rev-${caseId}`, group: '§7.4 Four-card case, steady month', label: `Revenue, ${caseId}`, report: t74[caseId].rev, dashboard: ps.revenueUsd * fx, tolerance: 0.2 });
    add({ id: `t74-margin-${caseId}`, group: '§7.4 Four-card case, steady month', label: `Margin over bars and rent, ${caseId}`, report: t74[caseId].margin, dashboard: ps.marginUsd * fx, tolerance: 0.2 });
  }
  return rows;
}

export function checkStatus(r: CheckRow): { delta: number; ok: boolean } {
  const delta = r.unit === 'pct' ? r.dashboard - r.report : r.report === 0 ? 0 : (r.dashboard - r.report) / Math.abs(r.report);
  const tol = r.tolerance;
  return { delta, ok: Math.abs(delta) <= tol };
}
