import { FX_INR_PER_USD } from '../data/gpus';
import { NEUTRAL_WHATIF, type Demand, type Pricing, type Settings, type Slot } from './types';

export const DEFAULT_PRICING: Pricing = { mode: 'median', mult: 1, batchDiscount: 0.5, customIn: 0.1, customOut: 1 };
export const DEFAULT_DEMAND: Demand = { mode: 'share', utilisation: 0.5, shareScale: 1, of: '' };

export const DEFAULT_SETTINGS: Settings = {
  caseId: 'base',
  currency: 'INR',
  fx: FX_INR_PER_USD.v,
  afterTax: false,
  route: 'fiat',
  costLevel: 'base',
  qos: { mode: 'report', floorTps: 40 },
  marketSource: 'live',
  rentSource: 'live',
  rentView: 'r90',
  rentStat: 'cheapest',
  proEdition: 'workstation',
  summerDerate: false,
  zdr: false,
  medianVolume: false,
  launchFloor: true,
  killRules: true,
  odds: 0.1,
  availabilityId: 'ups-gateway',
  availabilityCustom: null,
  override: { shareFactor: null, latencyLoss: null, priceTrend: null, listingMonth: null },
  whatIf: { ...NEUTRAL_WHATIF },
};

let counter = 0;
export function newSlotId(): string {
  counter += 1;
  return `slot-${Date.now().toString(36)}-${counter}`;
}

export function makeSlot(p: Partial<Slot> & Pick<Slot, 'gpuId' | 'modelId' | 'ownership'>): Slot {
  return {
    id: p.id ?? newSlotId(),
    label: p.label ?? '',
    count: 1,
    quant: 'auto',
    mtp: false,
    workloadId: 'observed',
    pricing: { ...DEFAULT_PRICING },
    demand: { ...DEFAULT_DEMAND },
    availabilityId: p.ownership === 'rent' ? 'dc' : 'global',
    listingOffset: 0,
    rentStartOffset: 0,
    ...p,
  };
}

export interface Preset {
  id: string;
  label: string;
  help: string;
  slots: () => Slot[];
}

export const PRESETS: Preset[] = [
  {
    id: 'status-quo',
    label: 'Status quo: both boxes on Lium',
    help: 'Nothing moves. Every owned card earns its Lium bar, so the margin is zero by definition.',
    slots: () => [
      makeSlot({ id: 'box-pro6000', label: 'Owned PRO 6000', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'lium' }),
      makeSlot({ id: 'box-ada6000', label: 'Owned 6000 Ada', ownership: 'owned', boxKey: 'ada6000', gpuId: 'ada6000', modelId: 'lium' }),
    ],
  },
  {
    id: 'staged',
    label: 'Report plan: PRO 6000 on Qwen3.8-27B, Ada on Gemma 4',
    help: 'The staged plan from the report: the PRO 6000 serves Qwen3.8-27B, the Ada serves Gemma 4 26B-A4B once that model is accepted two months later.',
    slots: () => [
      makeSlot({ id: 'box-pro6000', label: 'Owned PRO 6000 · Qwen3.8-27B', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' }),
      makeSlot({ id: 'box-ada6000', label: 'Owned 6000 Ada · Gemma 4 26B-A4B', ownership: 'owned', boxKey: 'ada6000', gpuId: 'ada6000', modelId: 'gemma4_26b', listingOffset: 2 }),
    ],
  },
  {
    id: 'four-card',
    label: 'Report four-card case: add two rented cards',
    help: 'The staged plan plus a rented PRO 6000 on Qwen3.6-35B-A3B and a rented RTX 5090 that picks up Qwen3.8-27B overflow (report §7.4).',
    slots: () => [
      makeSlot({ id: 'box-pro6000', label: 'Owned PRO 6000 · Qwen3.8-27B', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'qwen38_27b' }),
      makeSlot({ id: 'box-ada6000', label: 'Owned 6000 Ada · Gemma 4 26B-A4B', ownership: 'owned', boxKey: 'ada6000', gpuId: 'ada6000', modelId: 'gemma4_26b', listingOffset: 2 }),
      makeSlot({ id: 'rent-pro6000', label: 'Rented PRO 6000 · Qwen3.6-35B-A3B', ownership: 'rent', gpuId: 'pro6000', modelId: 'qwen36_35b', listingOffset: 2, quant: 'fp8' }),
      makeSlot({
        id: 'rent-5090',
        label: 'Rented RTX 5090 · Qwen3.8-27B overflow',
        ownership: 'rent',
        gpuId: 'rtx5090',
        modelId: 'qwen38_27b',
        demand: { ...DEFAULT_DEMAND, mode: 'overflow', of: 'box-pro6000' },
      }),
    ],
  },
  {
    id: 'rent-trial',
    label: 'Rented trial: one RTX 5090 on Qwen3.8-27B',
    help: 'The report\'s first step: build and benchmark on a rented RTX 5090 before any owned box moves.',
    slots: () => [
      makeSlot({ id: 'box-pro6000', label: 'Owned PRO 6000', ownership: 'owned', boxKey: 'pro6000', gpuId: 'pro6000', modelId: 'lium' }),
      makeSlot({ id: 'box-ada6000', label: 'Owned 6000 Ada', ownership: 'owned', boxKey: 'ada6000', gpuId: 'ada6000', modelId: 'lium' }),
      makeSlot({ id: 'rent-5090', label: 'Rented RTX 5090 · Qwen3.8-27B', ownership: 'rent', gpuId: 'rtx5090', modelId: 'qwen38_27b' }),
    ],
  },
];

export const PRESET_BY_ID: Record<string, Preset> = Object.fromEntries(PRESETS.map((p) => [p.id, p]));
