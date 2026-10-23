import { type V, rep, est, drv, REPORT_DATE } from './provenance';

/** OpenRouter market snapshot per model, from the report (22 Oct 2026). Phase 2 replaces it with live data. */
export interface MarketRow {
  modelId: string;
  paidTokensPerDayB: V;
  paidTokens14dMedianB?: V;
  endpoints: V;
  inMin: V;
  inMed: V;
  outMin: V;
  outMed: V;
  outMax: V;
  cacheHit: V;
  /** Revenue per fully busy hour on one PRO 6000 at median prices, USD (report §6.1). */
  busyHourMedianUsd: V | null;
  busyHourLowestUsd: V | null;
  /** Revenue of a typical new endpoint at median prices before any card limit, USD a month (report §6.3). Null where the card was the limit. */
  typicalMedianUsd: V | null;
  /** Request shape the report states explicitly (only two models). */
  statedShape?: { avgTokens: V; outShare: V };
}

const S53 = 'Report §5.3';
const S61 = 'Report §6.1';
const asOf = REPORT_DATE;

const row = (
  modelId: string,
  tokensB: number,
  n: number,
  inMin: number,
  inMed: number,
  outMin: number,
  outMed: number,
  outMax: number,
  cache: number,
  busyMed: number | null,
  busyLow: number | null,
  typical: number | null,
  extra: Partial<MarketRow> = {},
): MarketRow => ({
  modelId,
  paidTokensPerDayB: rep(tokensB, S53, 'Seven-day average to 21 Oct 2026', asOf),
  endpoints: rep(n, S53, undefined, asOf),
  inMin: rep(inMin, S53, '$ per million input tokens, cheapest endpoint', asOf),
  inMed: rep(inMed, S53, '$ per million input tokens, median across endpoints', asOf),
  outMin: rep(outMin, S53, '$ per million output tokens, cheapest endpoint', asOf),
  outMed: rep(outMed, S53, '$ per million output tokens, median', asOf),
  outMax: rep(outMax, S53, '$ per million output tokens, dearest endpoint', asOf),
  cacheHit: rep(cache, S53, 'Share of input tokens served from cache', asOf),
  busyHourMedianUsd: busyMed === null ? null : rep(busyMed, S61, 'Per busy hour at median prices, one PRO 6000', asOf),
  busyHourLowestUsd: busyLow === null ? null : rep(busyLow, S61, 'Per busy hour at the lowest listed price', asOf),
  typicalMedianUsd: typical === null ? null : rep(typical, 'Report §6.3', 'Typical-entrant month at median prices: 0.3/N of paid traffic, no latency penalty, no card limit', asOf),
  ...extra,
});

export const MARKET: MarketRow[] = [
  row('qwen38_27b', 61.9, 20, 0.035, 0.2, 1.25, 2.3, 3.2, 0.72, 13.65, 4.98, 5097, {
    statedShape: {
      avgTokens: rep(15300, S53, 'A Qwen3.8-27B request averages 15.3k tokens', asOf),
      outShare: rep(0.041, S53, '4.1% of tokens are output', asOf),
    },
  }),
  row('qwen36_35b', 110, 10, 0.05, 0.1, 0.7, 1.0, 1.8, 0.63, 9.83, 5.37, 6740, {
    paidTokens14dMedianB: rep(31.6, S53, 'Spiky: one app, Sekai, drives much of the seven-day volume', asOf),
  }),
  row('gemma4_26b', 52.8, 13, 0.042, 0.1, 0.22, 0.33, 0.6, 0.24, 7.24, 3.41, 3551),
  row('gemma4_31b', 80, 12, 0.1, 0.15, 0.32, 0.4, 1.15, 0.27, 3.75, 2.62, null),
  row('gptoss_120b', 73.5, 22, 0.03, 0.1, 0.15, 0.5, 0.95, 0.41, 5.41, 1.62, 3602, {
    statedShape: {
      avgTokens: rep(4300, S53, 'A gpt-oss-120b request averages 4.3k tokens', asOf),
      outShare: rep(0.118, S53, '11.8% of tokens are output', asOf),
    },
  }),
  row('gptoss_20b', 28.1, 10, 0.018, 0.03, 0.09, 0.15, 0.3, 0.18, 3.82, 2.29, 1225),
  row('llama33_70b', 5.2, 10, 0.1, 0.45, 0.32, 0.72, 2.25, 0.28, 6.03, 1.51, 1777),
  row('mistral_nemo', 38.3, 5, 0.017, 0.029, 0.02, 0.03, 0.15, 0.02, 2.18, 1.26, null),
  {
    ...row('flashnext', 93, 1, 0.15, 0.15, 0.47, 0.47, 0.47, 0.88, null, null, null),
    busyHourMedianUsd: est(5.48, 'Report §5.3', 'Derived from "₹3.8L ($4k)" card-saturated month ÷ 730 h; the report does not model this model otherwise', asOf),
  },
  row('nemotron35', 19.8, 7, 0.1, 0.15, 0.4, 0.5, 1.25, 0.75, null, null, null),
  row('qwen35_122b', 1.5, 4, 0.26, 0.28, 2.08, 2.24, 3.2, 0.0, null, null, null),
];

export const MARKET_BY_MODEL: Record<string, MarketRow> = Object.fromEntries(MARKET.map((m) => [m.modelId, m]));

/** Cache-read price as a fraction of the input price. Calibrated so the report's Qwen3.8-27B figures reproduce. */
export const CACHE_PRICE_FRACTION = drv(
  0.25,
  'Report §5.3 and §6.1',
  'A 25% cache-read price reproduces the report\'s Qwen3.8-27B typical-entrant gross within 1%. InferenceX falls back to 10% when a catalog omits it. Live OpenRouter cache prices replace this in phase 2.',
);

export interface RentRef {
  gpuId: string;
  modelId: string;
  rentUsdHr: number;
  busyHourMedianUsd: number;
  busyHourLowestUsd: number;
  breakEvenMedian: number;
}

/** Report §7.1: what a rented card earns per busy hour. Used to scale throughput across cards. */
export const RENT_REFS: RentRef[] = [
  { gpuId: 'rtx5090', modelId: 'qwen38_27b', rentUsdHr: 0.72, busyHourMedianUsd: 16.3, busyHourLowestUsd: 5.94, breakEvenMedian: 0.04 },
  { gpuId: 'pro6000', modelId: 'qwen38_27b', rentUsdHr: 1.64, busyHourMedianUsd: 13.65, busyHourLowestUsd: 4.98, breakEvenMedian: 0.12 },
  { gpuId: 'b200', modelId: 'qwen38_27b', rentUsdHr: 7.68, busyHourMedianUsd: 21.71, busyHourLowestUsd: 7.92, breakEvenMedian: 0.35 },
  { gpuId: 'pro6000', modelId: 'qwen36_35b', rentUsdHr: 1.64, busyHourMedianUsd: 9.83, busyHourLowestUsd: 5.37, breakEvenMedian: 0.17 },
  { gpuId: 'pro6000', modelId: 'gemma4_26b', rentUsdHr: 1.64, busyHourMedianUsd: 7.24, busyHourLowestUsd: 3.41, breakEvenMedian: 0.23 },
  { gpuId: 'pro6000', modelId: 'gptoss_120b', rentUsdHr: 1.64, busyHourMedianUsd: 5.41, busyHourLowestUsd: 1.62, breakEvenMedian: 0.3 },
  { gpuId: 'h100', modelId: 'gptoss_120b', rentUsdHr: 3.75, busyHourMedianUsd: 5.48, busyHourLowestUsd: 1.64, breakEvenMedian: 0.68 },
  { gpuId: 'h200', modelId: 'gptoss_120b', rentUsdHr: 5.4, busyHourMedianUsd: 8.03, busyHourLowestUsd: 2.41, breakEvenMedian: 0.67 },
  { gpuId: 'b200', modelId: 'gptoss_120b', rentUsdHr: 7.68, busyHourMedianUsd: 30.34, busyHourLowestUsd: 9.1, breakEvenMedian: 0.25 },
  { gpuId: 'pro6000', modelId: 'llama33_70b', rentUsdHr: 1.64, busyHourMedianUsd: 6.03, busyHourLowestUsd: 1.51, breakEvenMedian: 0.27 },
  { gpuId: 'h100', modelId: 'llama33_70b', rentUsdHr: 3.75, busyHourMedianUsd: 7.73, busyHourLowestUsd: 1.94, breakEvenMedian: 0.49 },
  { gpuId: 'h200', modelId: 'llama33_70b', rentUsdHr: 5.4, busyHourMedianUsd: 9.05, busyHourLowestUsd: 2.27, breakEvenMedian: 0.6 },
  { gpuId: 'b200', modelId: 'llama33_70b', rentUsdHr: 7.68, busyHourMedianUsd: 24.21, busyHourLowestUsd: 6.07, breakEvenMedian: 0.32 },
];
