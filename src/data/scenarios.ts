import { type V, rep, est, drv, ph } from './provenance';

export type CaseId = 'low' | 'base' | 'high';
export const CASES: CaseId[] = ['low', 'base', 'high'];
export const CASE_LABEL: Record<CaseId, string> = { low: 'Low', base: 'Base', high: 'High' };

export interface Scenario {
  id: CaseId;
  /** Multiple of an equal 1/N share of a model's paid traffic that a typical new endpoint gets. */
  shareFactor: V;
  /** Share lost to the extra latency of serving from India. */
  latencyLoss: V;
  /** Monthly price trend, compounding from month 1 (launch premium fading as hosts arrive). */
  priceTrend: V;
  /** First calendar month the endpoint is listed. */
  listingMonth: V;
  queueMonths: V;
  reviewMonths: V;
}

export const SCENARIOS: Record<CaseId, Scenario> = {
  low: {
    id: 'low',
    shareFactor: drv(0.15, 'Report §6.6 vs §6.3', 'Back-solved: low-case Qwen3.8-27B steady month ₹2.00 lakh ÷ ₹4.83 lakh at 0.3×, after the 17% latency loss'),
    latencyLoss: rep(0.17, 'Report §6.2', 'Low case'),
    priceTrend: rep(-0.05, 'Report §6.5', 'A month, from month 1'),
    listingMonth: rep(7, 'Report §6.4', 'Queues of six months or more are reported'),
    queueMonths: rep(4, 'Report §6.4'),
    reviewMonths: rep(2, 'Report §6.4'),
  },
  base: {
    id: 'base',
    shareFactor: rep(0.3, 'Report §6.2', 'Median endpoint share × N is 0.29-0.36'),
    latencyLoss: rep(0.11, 'Report §6.2', 'Base case'),
    priceTrend: rep(-0.02, 'Report §6.5'),
    listingMonth: rep(4, 'Report §6.4'),
    queueMonths: rep(2, 'Report §6.4'),
    reviewMonths: rep(1, 'Report §6.4'),
  },
  high: {
    id: 'high',
    shareFactor: rep(1, 'Report §7.2', 'The high case gets a full equal 1/N share, which overflows the PRO 6000 by about ₹6 lakh a month'),
    latencyLoss: rep(0.04, 'Report §6.2', 'High case'),
    priceTrend: rep(0.02, 'Report §6.5'),
    listingMonth: rep(2, 'Report §6.4'),
    queueMonths: rep(0, 'Report §6.4'),
    reviewMonths: rep(1, 'Report §6.4'),
  },
};

export const ELASTICITY_PRICE = rep(-0.63, 'Report §6.2', 'A 10% price cut buys about 7% more traffic');
export const ELASTICITY_LATENCY = rep(-0.46, 'Report §6.2');
export const ZDR_SHARE_BONUS = rep(0.13, 'Report §9.7', 'ZDR endpoints got about 13% more share in the routing data');
export const RAMP = [
  rep(0.3, 'Report §6.5', 'Share in the first listed month, as a fraction of steady'),
  rep(0.65, 'Report §6.5', 'Second listed month'),
];
export const FLOOR_MONTHS = rep(2, 'Report §6.5', 'The first two listed months are priced at the floor');
export const MONTHS = rep(12, 'Report §6.5', 'Twelve-month horizon used throughout the report');
export const LISTING_ODDS = { low: rep(0.05, 'Report §3.1'), default: rep(0.1, 'Report §6.4'), high: rep(0.15, 'Report §3.1') };
export const SECOND_MODEL_DELAY = rep(2, 'Report §6.5', 'A listed provider adds a model about two months after the first');

/** OpenRouter routing priority by measured uptime. Share multipliers are assumptions. */
export const UPTIME_TIERS = [
  { min: 0.95, label: 'Normal priority', mult: rep(1, 'Report §3.1', '95% or better gets normal priority') },
  { min: 0.8, label: 'Lower priority', mult: ph(0.5, 'No source yet', 'Report: 80-94% gets lower priority; the share penalty is a guess', undefined, '× the share') },
  { min: 0, label: 'Fallback traffic only', mult: ph(0.1, 'No source yet', 'Report: under 80% gets fallback traffic only; the share is a guess', undefined, '× the share') },
];

export interface AvailabilityPreset {
  id: string;
  label: string;
  availability: V;
  help: string;
}

export const AVAILABILITY_PRESETS: AvailabilityPreset[] = [
  {
    id: 'ups-gateway',
    label: 'UPS + off-site gateway + hourly card',
    availability: ph(0.99, 'No source yet', 'The report gives no uptime figure for the full protection stack', undefined, 'of hours up'),
    help: 'Report §9.4 layers 1-3: blips ride through, traffic is redirected during planned cuts.',
  },
  {
    id: 'ups-only',
    label: 'UPS only',
    availability: ph(0.97, 'No source yet', undefined, undefined, 'of hours up'),
    help: 'Clean shutdowns, but planned 5-9 hour cuts still take the endpoint down.',
  },
  {
    id: 'no-ups',
    label: 'No UPS (about 30 h offline a month)',
    availability: drv(0.959, 'Report §9.4', '30 hours offline in a 730-hour month'),
    help: 'The report\'s base-case outage figure.',
  },
  {
    id: 'dc',
    label: 'Data-centre host (rented card)',
    availability: ph(0.995, 'No source yet', 'Rented cards sit in data centres with redundant power; the report gives no figure', undefined, 'of hours up'),
    help: 'Default for rented cards.',
  },
  {
    id: 'bad',
    label: 'Bad month (66% uptime)',
    availability: est(0.66, 'Report §8.3', 'Sogni\'s observed uptime; the report says this is about what long cuts would do without a UPS'),
    help: 'A stress case: below OpenRouter\'s 80% line, traffic falls to fallback only.',
  },
];

export interface WorkloadShape {
  avgTokens: V;
  outShare: V;
  /** null means use the model's own cache-hit rate. */
  cacheHit: V | null;
}

export interface WorkloadPreset {
  id: string;
  label: string;
  help: string;
  /** null means use each model's calibrated shape from the report. */
  shape: WorkloadShape | null;
  /** Suggested tok/s/user floor when a custom floor is used. */
  floorTps: V;
  /** Batch workloads are sold by hours and priced at a discount instead of routed. */
  batch?: boolean;
}

export const WORKLOADS: WorkloadPreset[] = [
  {
    id: 'observed',
    label: 'As observed on OpenRouter',
    help: 'Each model\'s own request shape, calibrated to the report\'s busy-hour revenue.',
    shape: null,
    floorTps: ph(40, 'No source yet', 'Placeholder floor until OpenRouter speed data is pulled in phase 2', undefined, 'tok/s per user'),
  },
  {
    id: 'agentic',
    label: 'Agentic coding',
    help: 'Long multi-turn prompts, heavy prefix reuse, short outputs. Shape of the report\'s Qwen3.8-27B traffic.',
    shape: {
      avgTokens: rep(15300, 'Report §5.3', 'Qwen3.8-27B request average'),
      outShare: rep(0.041, 'Report §5.3'),
      cacheHit: null,
    },
    floorTps: ph(40, 'No source yet', undefined, undefined, 'tok/s per user'),
  },
  {
    id: 'chat',
    label: 'Chat',
    help: 'Short prompts, longer answers, some prefix reuse.',
    shape: { avgTokens: ph(3000, 'No source yet', undefined, undefined, 'tokens per request'), outShare: ph(0.15, 'No source yet', undefined, undefined, 'of tokens are output'), cacheHit: ph(0.3, 'No source yet', undefined, undefined, 'of the prompt read from cache') },
    floorTps: ph(40, 'No source yet', undefined, undefined, 'tok/s per user'),
  },
  {
    id: 'roleplay',
    label: 'Roleplay',
    help: 'Long-lived conversations with medium context and good prefix reuse.',
    shape: { avgTokens: ph(8000, 'No source yet', undefined, undefined, 'tokens per request'), outShare: ph(0.06, 'No source yet', undefined, undefined, 'of tokens are output'), cacheHit: ph(0.6, 'No source yet', undefined, undefined, 'of the prompt read from cache') },
    floorTps: ph(30, 'No source yet', undefined, undefined, 'tok/s per user'),
  },
  {
    id: 'batch',
    label: 'Batch / offline',
    help: 'No interactivity target, so the card runs flat out. Sold by hours at a discount, not routed.',
    shape: { avgTokens: ph(4000, 'No source yet', undefined, undefined, 'tokens per request'), outShare: ph(0.2, 'No source yet', undefined, undefined, 'of tokens are output'), cacheHit: ph(0, 'No source yet', undefined, undefined, 'of the prompt read from cache') },
    floorTps: ph(8, 'No source yet', undefined, undefined, 'tok/s per user'),
    batch: true,
  },
];

export const WORKLOAD_BY_ID: Record<string, WorkloadPreset> = Object.fromEntries(WORKLOADS.map((w) => [w.id, w]));

export const BATCH_DISCOUNT = ph(0.5, 'No source yet', 'Provider batch APIs commonly price at about half of realtime; assumed', undefined, '× the realtime price');

export interface QosPreset {
  id: 'report' | 'floor' | 'router';
  label: string;
  help: string;
}

export const QOS_PRESETS: QosPreset[] = [
  { id: 'report', label: 'Report convention', help: 'Each model runs at the operating point the report\'s busy-hour revenue is based on (15-25% of vendor peak), 16 users a card, held down to what the KV cache can carry. On the report-points basis the cache is not checked, as in the report.' },
  { id: 'router', label: 'OpenRouter-competitive', help: "Serve as many concurrent requests as keep every user at the median typical stream speed of the model's listed OpenRouter endpoints (live)." },
  { id: 'floor', label: 'Custom tok/s/user floor', help: 'Serve as many concurrent requests as keep every user above a speed you set.' },
];

export interface ThroughputPreset {
  id: 'report' | 'adjusted' | 'roofline';
  label: string;
  help: string;
}

export const THROUGHPUT_PRESETS: ThroughputPreset[] = [
  { id: 'report', label: "Report's points as printed", help: "The report's published throughput points as printed (a power law through them), moved between cards and weight formats by the roofline but not between request shapes. Reproduces the report." },
  { id: 'adjusted', label: 'Anchored to measurements', help: "The fitted roofline, scaled to a measurement wherever there is one: InferenceX's runs for the card and model if it measured them (curve and all), else the report's one measured point for that card. Everything else is the roofline alone; the report's own estimates are not used. Longer contexts, more concurrency and prompt-heavy requests move it as the physics says." },
  { id: 'roofline', label: 'Roofline only', help: "Ignore every anchor and compute speed from memory bandwidth, tensor throughput, link speed and the request shape, with the efficiencies fitted to InferenceX's measured runs." },
];

/** Tax and route costs. */
export const TAX = {
  fiat: rep(0.2517, 'Report §2.4', 'About 25.17% on profit after expenses'),
  crypto: rep(0.312, 'Report §2.4', '30% plus 4% cess of gross receipts, no deduction'),
};

export type CostLevel = 'low' | 'base' | 'high';

export const ROUTE_COSTS = {
  monitoringInr: { low: rep(2000, 'Report §2.2'), base: rep(3500, 'Report §2.2', 'Midpoint of ₹2-5k a month'), high: rep(5000, 'Report §2.2') },
  upsInr: { low: rep(40_000, 'Report §2.5'), base: rep(55_000, 'Report §6.4'), high: rep(70_000, 'Report §6.4') },
  failoverInr: { low: rep(5000, 'Report §2.2'), base: rep(7500, 'Report §2.2', 'Midpoint of the agreed ₹5-10k a month cap'), high: rep(10_000, 'Report §2.2') },
  upsAmortMonths: rep(12, 'Report §2.2', 'Written off over the 12-month horizon'),
};

export interface LiumBar {
  boxId: string;
  gpuId: string;
  netInr: V;
  afterTaxInr: V;
  /** Extra electricity of serving versus idling on Lium, INR a month. */
  extraPowerInr: V;
  grossUsd: V;
}

export const LIUM: Record<string, LiumBar> = {
  pro6000: {
    boxId: 'box-pro6000',
    gpuId: 'pro6000',
    netInr: rep(80_000, 'Report §2.1', 'After Lium fees and power; 30 days to 22 Oct 2026'),
    afterTaxInr: rep(53_300, 'Report §2.1', 'Under the agreed crypto rule'),
    extraPowerInr: drv(5648, 'Report §2.2', 'Back-solved from the base fiat tie of ₹92,460'),
    grossUsd: rep(943, 'Report §2.1'),
  },
  ada6000: {
    boxId: 'box-ada6000',
    gpuId: 'ada6000',
    netInr: rep(52_000, 'Report §2.1'),
    afterTaxInr: rep(34_800, 'Report §2.1'),
    extraPowerInr: drv(1138, 'Report §2.2', 'Back-solved from the base fiat tie of ₹63,226'),
    grossUsd: rep(597, 'Report §2.1'),
  },
};

export const PROGRAM = {
  buildInr: rep(10_000, 'Report §6.4', 'A few days of rented-card time to build and load-test'),
  queueInrPerMonth: rep(1000, 'Report §6.4', 'Stored image and weights'),
  stakeCapInr: rep(200_000, 'Report §9.2', 'Net spend cap of ₹2 lakh'),
  toolsInrPerMonth: rep(3500, 'Report §6.4', 'Monitoring and failover tools while a rented card runs'),
  reviewGpuId: 'rtx5090',
};

export const BUY_HORIZON_MONTHS = ph(24, 'No source yet', 'Planning horizon for buy-versus-rent payback; the report only fixes 12 months and a three-month rule', undefined, 'months');
export const RENT_RULES = {
  exitMultiple: rep(1.5, 'Report §9.2', 'Hand a rented card back when monthly revenue falls below 1.5× its rent'),
  buyAfterMonths: rep(3, 'Report §7.5', 'Buy only after a rented card stayed above break-even for three months'),
};
