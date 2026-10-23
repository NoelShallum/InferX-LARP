import type { Quant } from '../data/gpus';
import type { CaseId, CostLevel } from '../data/scenarios';

export type Currency = 'INR' | 'USD';
export type Route = 'fiat' | 'crypto';
export type Ownership = 'owned' | 'rent' | 'buy';

export type PricingMode = 'median' | 'floor' | 'multiplier' | 'batch' | 'custom';

export interface Pricing {
  mode: PricingMode;
  /** Multiple of the median blended price, used by 'multiplier'. */
  mult: number;
  /** Batch discount off the live price, 0.5 means half price. */
  batchDiscount: number;
  customIn: number;
  customOut: number;
}

export type DemandMode = 'share' | 'utilisation' | 'overflow';

export interface Demand {
  mode: DemandMode;
  /** Share of time the card is sold, for 'utilisation'. */
  utilisation: number;
  /** Extra multiplier on the scenario's share factor. */
  shareScale: number;
  /** Slot whose unserved traffic this slot picks up, for 'overflow'. */
  of: string;
}

export interface Slot {
  id: string;
  label: string;
  ownership: Ownership;
  /** Key into LIUM for owned boxes. */
  boxKey?: string;
  gpuId: string;
  count: number;
  /** 'lium' keeps an owned card on Lium (the status quo). */
  modelId: string;
  quant: Quant | 'auto';
  mtp: boolean;
  workloadId: string;
  pricing: Pricing;
  demand: Demand;
  /** Availability preset id, or 'global' to use the global setting. */
  availabilityId: string;
  /** Months after the primary listing that this slot's model is accepted. */
  listingOffset: number;
  /** Months after listing that a rented card is paid for. */
  rentStartOffset: number;
  /** Purchase price override in INR for 'buy' slots. */
  buyPriceInr?: number;
  rentUsdHrOverride?: number;
}

export interface WhatIf {
  /** Multiplies measured or estimated throughput. */
  throughput: number;
  /** Multiplies the share of paid traffic won. */
  demand: number;
  /** Multiplies every token price. */
  price: number;
  /** Multiplies rent. */
  rent: number;
  /** Multiplies the Lium bar. */
  bar: number;
  /** Added to availability, in fraction points. */
  availabilityDelta: number;
}

export const NEUTRAL_WHATIF: WhatIf = { throughput: 1, demand: 1, price: 1, rent: 1, bar: 1, availabilityDelta: 0 };

export interface Settings {
  caseId: CaseId;
  currency: Currency;
  fx: number;
  afterTax: boolean;
  route: Route;
  costLevel: CostLevel;
  qos: { mode: 'report' | 'floor'; floorTps: number };
  proEdition: 'workstation' | 'maxq';
  summerDerate: boolean;
  zdr: boolean;
  /** Use the 14-day median volume instead of the 7-day average where the report gives one. */
  medianVolume: boolean;
  /** Price the first two listed months at the floor when the slot prices at the median. */
  launchFloor: boolean;
  /** Apply the report's revert and hand-back rules in the timeline. */
  killRules: boolean;
  odds: number;
  availabilityId: string;
  availabilityCustom: number | null;
  /** Overrides on the scenario, null keeps the scenario value. */
  override: { shareFactor: number | null; latencyLoss: number | null; priceTrend: number | null; listingMonth: number | null };
  whatIf: WhatIf;
}

export interface ResolvedCase {
  shareFactor: number;
  latencyLoss: number;
  priceTrend: number;
  listingMonth: number;
  queueMonths: number;
  reviewMonths: number;
}
