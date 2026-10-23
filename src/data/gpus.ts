import { type V, type Prov, rep, est, srcd, ph, drv } from './provenance';

export type Quant = 'fp16' | 'fp8' | 'nvfp4' | 'mxfp4' | 'awq4';

export const QUANT_LABEL: Record<Quant, string> = {
  fp16: 'FP16',
  fp8: 'FP8',
  nvfp4: 'NVFP4',
  mxfp4: 'MXFP4',
  awq4: 'AWQ INT4',
};

export interface Gpu {
  id: string;
  name: string;
  short: string;
  arch: string;
  role: 'serve' | 'control-plane';
  vramGB: V;
  bwGBs: V;
  tdpW: V;
  /** Weight formats the card runs natively. */
  quants: Quant[];
  quantsProv: Prov;
  link: 'nvlink' | 'pcie';
  /** Market on-demand rent, USD per GPU-hour. */
  rentUsdHr: V | null;
  /** Indian cloud rent, INR per GPU-hour, for reference. */
  rentInrHr: V | null;
  /** New purchase price in India, INR. */
  buyInr: V | null;
  /** Cards you own today. */
  owned: number;
  /** InferenceX-app HW_REGISTRY cost assumptions, USD per GPU-hour, for reference. */
  ixCost: { owned: V; rent: V } | null;
  note?: string;
}

const NVIDIA_PRO = 'https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/';
const IX_SPECS = 'InferenceX-app packages/app/src/lib/catalog/gpu-specs.ts';
const IX_HW = 'InferenceX-app packages/constants/src/gpu-keys.ts (HW_REGISTRY)';

export const GPUS: Gpu[] = [
  {
    id: 'pro6000',
    name: 'RTX PRO 6000 Blackwell (Workstation, 600 W)',
    short: 'PRO 6000',
    arch: 'Blackwell',
    role: 'serve',
    vramGB: srcd(96, NVIDIA_PRO, '96 GB GDDR7 with ECC on all three editions'),
    bwGBs: rep(1792, 'Report §5.2', 'Quoted as 1,792 GB/s'),
    tdpW: srcd(600, NVIDIA_PRO, 'Workstation Edition 600 W; Max-Q 300 W; Server 400-600 W'),
    quants: ['nvfp4', 'mxfp4', 'fp8', 'awq4', 'fp16'],
    quantsProv: rep(0, 'Report §5.1', 'Report serves NVFP4 and FP8 on this card').p,
    link: 'pcie',
    rentUsdHr: rep(1.64, 'Report §7.1', 'Median on-demand rent across the marketplaces the report checked'),
    rentInrHr: rep(182, 'Report §7.1', 'E2E Networks list price, INR per hour'),
    buyInr: rep(1_500_000, 'Report §7.5', 'About ₹15 lakh ($15,800) new'),
    owned: 1,
    ixCost: null,
    note: 'Owned. Edition (Workstation vs Max-Q) is unconfirmed: run nvidia-smi -q -d POWER (report §5.2).',
  },
  {
    id: 'ada6000',
    name: 'RTX 6000 Ada Generation',
    short: '6000 Ada',
    arch: 'Ada Lovelace',
    role: 'serve',
    vramGB: rep(48, 'Report §5.1'),
    bwGBs: srcd(960, 'https://lenovopress.lenovo.com/lp1940.pdf', 'Lenovo product guide and Macnica table agree; report §5.2 also quotes 960 GB/s'),
    tdpW: srcd(300, 'https://lenovopress.lenovo.com/lp1940.pdf', 'Several sources list 300 W board power'),
    quants: ['fp8', 'awq4', 'nvfp4', 'mxfp4', 'fp16'],
    quantsProv: est(0, 'Report §5.2', 'Report: Ada has no FP4 compute. FP8 and AWQ are used on it, and NVFP4 weights fit (§5.1), so NVFP4 and MXFP4 are assumed to run weight-only through software kernels.').p,
    link: 'pcie',
    rentUsdHr: est(0.65, 'Report §4.6', 'Midpoint of the report\'s $0.5-0.8/h for an L40S or 6000 Ada'),
    rentInrHr: null,
    buyInr: ph(650_000, 'No source yet', 'Placeholder, enter a real quote'),
    owned: 1,
    ixCost: null,
    note: 'Owned.',
  },
  {
    id: 'rtx3050',
    name: 'RTX 3050 (control plane)',
    short: '3050',
    arch: 'Ampere',
    role: 'control-plane',
    vramGB: ph(8, 'Report §8.6', 'Report says Whisper-turbo and embedding models fit in 6-8 GB; exact card size not given'),
    bwGBs: ph(224, 'No source yet', 'Not used until a non-text probe is modelled'),
    tdpW: ph(130, 'No source yet'),
    quants: ['fp16', 'awq4'],
    quantsProv: ph(0, 'No source yet').p,
    link: 'pcie',
    rentUsdHr: null,
    rentInrHr: null,
    buyInr: null,
    owned: 1,
    ixCost: null,
    note: 'Owned. Stays the control plane; can host Whisper-turbo and embedding probes (report §8.6). Not an LLM serving card.',
  },
  {
    id: 'rtx5090',
    name: 'GeForce RTX 5090',
    short: 'RTX 5090',
    arch: 'Blackwell',
    role: 'serve',
    vramGB: srcd(32, 'https://www.techspot.com/specs/gpu/303357-nvidia-geforce-rtx-5090.html', '32 GB GDDR7, 512-bit'),
    bwGBs: srcd(1792, 'https://www.techradar.com/computing/gpu/nvidia-rtx-5090-vs-rtx-4090', 'About 1,792 GB/s'),
    tdpW: srcd(575, 'https://www.techspot.com/specs/gpu/303357-nvidia-geforce-rtx-5090.html'),
    quants: ['nvfp4', 'mxfp4', 'fp8', 'awq4', 'fp16'],
    quantsProv: est(0, 'Report §7.1', 'Report assumes NVFP4 weights on the 5090').p,
    link: 'pcie',
    rentUsdHr: rep(0.72, 'Report §7.1', 'Median on-demand rent'),
    rentInrHr: null,
    buyInr: ph(350_000, 'No source yet', 'Placeholder, enter a real quote'),
    owned: 0,
    ixCost: null,
  },
  {
    id: 'rtx4090',
    name: 'GeForce RTX 4090',
    short: 'RTX 4090',
    arch: 'Ada Lovelace',
    role: 'serve',
    vramGB: ph(24, 'No source yet', '24 GB as commonly listed; not checked this session'),
    bwGBs: srcd(1008, 'https://www.techradar.com/computing/gpu/nvidia-rtx-5090-vs-rtx-4090', 'A comparison table lists about 1,010 GB/s'),
    tdpW: srcd(450, 'https://www.techradar.com/computing/gpu/nvidia-rtx-5090-vs-rtx-4090', 'TGP 450 W'),
    quants: ['fp8', 'awq4', 'mxfp4', 'fp16'],
    quantsProv: ph(0, 'No source yet', 'Ada tensor cores support FP8; assumed').p,
    link: 'pcie',
    rentUsdHr: ph(0.35, 'No source yet', 'Placeholder; Vast.ai live prices replace it in phase 2'),
    rentInrHr: null,
    buyInr: null,
    owned: 0,
    ixCost: null,
    note: 'In the report only as a Sogni earnings row (§8.3).',
  },
  {
    id: 'l40s',
    name: 'NVIDIA L40S',
    short: 'L40S',
    arch: 'Ada Lovelace',
    role: 'serve',
    vramGB: ph(48, 'No source yet', '48 GB as commonly listed; not checked this session'),
    bwGBs: ph(864, 'No source yet', 'Not checked this session'),
    tdpW: ph(350, 'No source yet', 'Not checked this session'),
    quants: ['fp8', 'awq4', 'mxfp4', 'fp16'],
    quantsProv: ph(0, 'No source yet').p,
    link: 'pcie',
    rentUsdHr: est(0.65, 'Report §4.6', 'Midpoint of $0.5-0.8/h for an L40S or 6000 Ada'),
    rentInrHr: null,
    buyInr: null,
    owned: 0,
    ixCost: null,
  },
  {
    id: 'h100',
    name: 'NVIDIA H100 SXM',
    short: 'H100',
    arch: 'Hopper',
    role: 'serve',
    vramGB: srcd(80, IX_SPECS, '80 GB HBM3'),
    bwGBs: srcd(3350, IX_SPECS, '3.35 TB/s'),
    tdpW: srcd(700, IX_HW),
    quants: ['fp8', 'mxfp4', 'awq4', 'fp16'],
    quantsProv: srcd(0, IX_SPECS, 'Hopper has FP8 but no FP4 tensor support').p,
    link: 'nvlink',
    rentUsdHr: rep(3.75, 'Report §7.1', 'Median on-demand rent'),
    rentInrHr: rep(355, 'Report §7.1', 'E2E ₹350-400 and Yotta ₹356 per hour'),
    buyInr: null,
    owned: 0,
    ixCost: { owned: srcd(1.17, IX_HW, 'InferenceX TCO assumption, owned $/GPU-hr'), rent: srcd(2.0, IX_HW, 'InferenceX rental assumption, $/GPU-hr') },
  },
  {
    id: 'h200',
    name: 'NVIDIA H200 SXM',
    short: 'H200',
    arch: 'Hopper',
    role: 'serve',
    vramGB: srcd(141, IX_SPECS, '141 GB HBM3e'),
    bwGBs: srcd(4800, IX_SPECS, '4.8 TB/s'),
    tdpW: srcd(700, IX_HW),
    quants: ['fp8', 'mxfp4', 'awq4', 'fp16'],
    quantsProv: srcd(0, IX_SPECS, 'Hopper has FP8 but no FP4 tensor support').p,
    link: 'nvlink',
    rentUsdHr: rep(5.4, 'Report §7.1', 'Median on-demand rent'),
    rentInrHr: null,
    buyInr: null,
    owned: 0,
    ixCost: { owned: srcd(1.22, IX_HW), rent: srcd(2.9, IX_HW) },
  },
  {
    id: 'b200',
    name: 'NVIDIA B200 HGX',
    short: 'B200',
    arch: 'Blackwell',
    role: 'serve',
    vramGB: srcd(180, IX_SPECS, '180 GB HBM3e'),
    bwGBs: srcd(8000, IX_SPECS, '8 TB/s'),
    tdpW: srcd(1000, IX_HW),
    quants: ['nvfp4', 'mxfp4', 'fp8', 'awq4', 'fp16'],
    quantsProv: srcd(0, IX_SPECS, 'Blackwell has FP4 and FP8').p,
    link: 'nvlink',
    rentUsdHr: rep(7.68, 'Report §7.1', 'Median on-demand rent'),
    rentInrHr: null,
    buyInr: null,
    owned: 0,
    ixCost: { owned: srcd(1.73, IX_HW), rent: srcd(3.7, IX_HW) },
  },
  {
    id: 'v100',
    name: 'Tesla V100 32 GB (PCIe)',
    short: 'V100 32GB',
    arch: 'Volta',
    role: 'serve',
    vramGB: srcd(32, 'https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf', '32 GB HBM2 variant'),
    bwGBs: srcd(900, 'https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf', '900 GB/s on the PCIe card (V100S is 1,134 GB/s)'),
    tdpW: srcd(250, 'https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf', 'PCIe 250 W; SXM2 is 300 W'),
    quants: ['fp16', 'awq4'],
    quantsProv: srcd(0, 'https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf', 'Volta has no FP8, FP4 or BF16 tensor support, so FP16 or INT4 only').p,
    link: 'pcie',
    rentUsdHr: ph(0.2, 'No source yet', 'Placeholder; Vast.ai live prices replace it in phase 2'),
    rentInrHr: null,
    buyInr: null,
    owned: 0,
    ixCost: null,
    note: 'Volta has no FP8 KV cache and older attention kernels; throughput here is a rough scaling.',
  },
];

export const GPU_BY_ID: Record<string, Gpu> = Object.fromEntries(GPUS.map((g) => [g.id, g]));

/** Cost of editions and heat, applied to the owned PRO 6000 only. */
export const PRO6000_EDITION = {
  workstation: { label: 'Workstation 600 W', decode: 1, prov: rep(1, 'Report §2.5').p },
  maxq: {
    label: 'Max-Q 300 W',
    decode: 0.95,
    prov: rep(0.95, 'Report §5.2', 'Max-Q loses about 5% on decode and 28-48% on prefill').p,
  },
} as const;

export const SUMMER_DERATE = est(0.875, 'Report §2.5', 'Power-limiting to 400-450 W: throughput cut 10-15% from April to June (midpoint)');

export const OWNED_BOXES = [
  { id: 'box-pro6000', gpuId: 'pro6000', label: 'Owned PRO 6000 box' },
  { id: 'box-ada6000', gpuId: 'ada6000', label: 'Owned 6000 Ada box' },
] as const;

export const FX_INR_PER_USD = drv(94.73, 'Report §2.2', '₹92,460 = $976 in the report; live rate arrives in phase 2');

export const HOURS_PER_MONTH = rep(730, 'Report §6.1', '₹1,293 per busy hour × 730 h = ₹9.44 lakh, the report\'s saturated month');
