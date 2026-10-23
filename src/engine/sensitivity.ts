import { portfolioSteady, portfolioTimeline } from './portfolio';
import { resolveCase } from './slot';
import type { Settings, Slot } from './types';

export type Metric = 'steady' | 'net12';

export interface Perturbation {
  id: string;
  label: string;
  lowLabel: string;
  highLabel: string;
  /** Metrics this input can move. */
  metrics: Metric[];
  low: (s: Settings) => Settings;
  high: (s: Settings) => Settings;
}

const wi = (s: Settings, patch: Partial<Settings['whatIf']>): Settings => ({ ...s, whatIf: { ...s.whatIf, ...patch } });

export const PERTURBATIONS: Perturbation[] = [
  {
    id: 'share',
    label: 'Share of traffic won',
    lowLabel: '−30%',
    highLabel: '+30%',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { demand: s.whatIf.demand * 0.7 }),
    high: (s) => wi(s, { demand: s.whatIf.demand * 1.3 }),
  },
  {
    id: 'price',
    label: 'Market token prices',
    lowLabel: '−20%',
    highLabel: '+20%',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { price: s.whatIf.price * 0.8 }),
    high: (s) => wi(s, { price: s.whatIf.price * 1.2 }),
  },
  {
    id: 'throughput',
    label: 'Throughput per card',
    lowLabel: '−20%',
    highLabel: '+20%',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { throughput: s.whatIf.throughput * 0.8 }),
    high: (s) => wi(s, { throughput: s.whatIf.throughput * 1.2 }),
  },
  {
    id: 'availability',
    label: 'Availability',
    lowLabel: '−5 pts',
    highLabel: '+3 pts',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { availabilityDelta: s.whatIf.availabilityDelta - 0.05 }),
    high: (s) => wi(s, { availabilityDelta: s.whatIf.availabilityDelta + 0.03 }),
  },
  {
    id: 'rent',
    label: 'Rent per GPU-hour',
    lowLabel: '+25%',
    highLabel: '−25%',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { rent: s.whatIf.rent * 1.25 }),
    high: (s) => wi(s, { rent: s.whatIf.rent * 0.75 }),
  },
  {
    id: 'bar',
    label: 'Lium bar',
    lowLabel: '+15%',
    highLabel: '−15%',
    metrics: ['steady', 'net12'],
    low: (s) => wi(s, { bar: s.whatIf.bar * 1.15 }),
    high: (s) => wi(s, { bar: s.whatIf.bar * 0.85 }),
  },
  {
    id: 'latency',
    label: 'Latency penalty from India',
    lowLabel: '+10 pts',
    highLabel: '−5 pts',
    metrics: ['steady', 'net12'],
    low: (s) => ({ ...s, override: { ...s.override, latencyLoss: Math.min(0.9, resolveCase(s).latencyLoss + 0.1) } }),
    high: (s) => ({ ...s, override: { ...s.override, latencyLoss: Math.max(0, resolveCase(s).latencyLoss - 0.05) } }),
  },
  {
    id: 'listing',
    label: 'Month the endpoint is listed',
    lowLabel: '+2 months',
    highLabel: '−2 months',
    metrics: ['net12'],
    low: (s) => ({ ...s, override: { ...s.override, listingMonth: resolveCase(s).listingMonth + 2 } }),
    high: (s) => ({ ...s, override: { ...s.override, listingMonth: Math.max(1, resolveCase(s).listingMonth - 2) } }),
  },
  {
    id: 'trend',
    label: 'Monthly price trend',
    lowLabel: '−2 pts',
    highLabel: '+2 pts',
    metrics: ['net12'],
    low: (s) => ({ ...s, override: { ...s.override, priceTrend: resolveCase(s).priceTrend - 0.02 } }),
    high: (s) => ({ ...s, override: { ...s.override, priceTrend: resolveCase(s).priceTrend + 0.02 } }),
  },
];

export interface TornadoRow {
  id: string;
  label: string;
  lowLabel: string;
  highLabel: string;
  lowDeltaUsd: number;
  highDeltaUsd: number;
  span: number;
}

export function metricValue(slots: Slot[], settings: Settings, metric: Metric): number {
  return metric === 'steady' ? portfolioSteady(slots, settings).marginUsd : portfolioTimeline(slots, settings).netUsd;
}

export function tornado(slots: Slot[], settings: Settings, metric: Metric): { base: number; rows: TornadoRow[] } {
  const base = metricValue(slots, settings, metric);
  const rows = PERTURBATIONS.filter((p) => p.metrics.includes(metric)).map((p) => {
    const lowDeltaUsd = metricValue(slots, p.low(settings), metric) - base;
    const highDeltaUsd = metricValue(slots, p.high(settings), metric) - base;
    return { id: p.id, label: p.label, lowLabel: p.lowLabel, highLabel: p.highLabel, lowDeltaUsd, highDeltaUsd, span: Math.abs(highDeltaUsd - lowDeltaUsd) };
  });
  rows.sort((a, b) => b.span - a.span);
  return { base, rows };
}
