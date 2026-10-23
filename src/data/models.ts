import { type V, type Prov, rep, meas, est, srcd, drv, ph } from './provenance';
import type { Quant } from './gpus';
import { ARCH, activeFromConfigV, kvPerTokV, totalFromConfigV, type ArchInfo } from './arch';

export interface TputPoint {
  /** Concurrent requests. */
  c: number;
  /** Aggregate output tokens per second at that concurrency. */
  tps: V;
}

export interface MtpSpec {
  /** Acceptance length used (tokens per verify step) and the draft length it applies to. */
  al: V;
  k: number;
  /** Measured gain at a given concurrency, used to calibrate the fade-out. */
  measured?: { c: number; tps: V };
  note: string;
}

export interface Model {
  id: string;
  name: string;
  orSlug?: string;
  arch: 'dense' | 'moe';
  paramsB: V;
  activeB: V;
  /** Weight size in GB at quants the report lists. Other quants are derived from parameter count. */
  weights: Partial<Record<Quant, V>>;
  defaultQuant: Quant;
  /** KV bytes per context token at FP8 KV cache (full-attention layers; sliding-window and recurrent layers are in `net`). */
  kvBytesPerTok: V;
  /** What the Hugging Face config says about memory and compute; null if the config was not read. */
  net: ArchInfo | null;
  /** Measured concurrent requests on a PRO 6000 at 8K and 32K context (report §5.1). */
  conc8k: V | null;
  conc32k: V | null;
  pro6000: TputPoint[];
  ada: TputPoint[];
  /** Index into pro6000 of the point the report's busy-hour revenue corresponds to. */
  refIdx: number;
  /** Weight format the PRO 6000 and Ada throughput points were measured or estimated at. */
  pointsQuant: { pro6000: Quant; ada: Quant };
  /** Cards the report says the model runs on although the plain VRAM arithmetic says otherwise. */
  forceFit?: { gpuIds: string[]; why: string };
  mtp: MtpSpec | null;
  /** Whether the report models revenue for this model. */
  economics: boolean;
  /** Not in the report: priced and sized from live OpenRouter data alone, with only the checkpoints listed here. */
  liveOnly?: boolean;
  note?: string;
}

const T52 = 'Report §5.2';
const T51 = 'Report §5.1';

const kvGeneric = (bytes: number, why: string): V => ph(bytes, 'No source yet', why, undefined, 'bytes per token');

const HF = (repo: string) => `https://huggingface.co/${repo}`;
const SUM = 'Safetensors files of the checkpoint, summed from the Hugging Face file tree on 23 Oct 2026';
const NO_POINTS: Pick<Model, 'conc8k' | 'conc32k' | 'pro6000' | 'ada' | 'refIdx'> = { conc8k: null, conc32k: null, pro6000: [], ada: [], refIdx: 0 };

/**
 * Models the report leaves out because they need several data-centre GPUs, but which carry most of OpenRouter's traffic.
 * Their market comes from live OpenRouter data; their speed from InferenceX where it measured them and the fitted
 * roofline otherwise. Parameter counts are the model cards'; KV cache comes from `net` (see `withArch`).
 */
const LIVE_ONLY: Omit<Model, 'net'>[] = [
  {
    id: 'dsv41flash',
    name: 'DeepSeek V4.1 Flash',
    orSlug: 'deepseek/deepseek-v4.1-flash',
    arch: 'moe',
    paramsB: srcd(552, HF('deepseek-ai/DeepSeek-V4.1-Flash'), 'Model card: 552B backbone parameters, plus a 196B Engram lookup table'),
    activeB: srcd(16, HF('deepseek-ai/DeepSeek-V4.1-Flash'), 'Model card: 16B activated per token in decode, 8B in prefill'),
    weights: {
      nvfp4: srcd(429.0, HF('nvidia/DeepSeek-V4.1-Flash-NVFP4'), SUM),
      fp8: srcd(510.3, HF('deepseek-ai/DeepSeek-V4.1-Flash'), `${SUM}. The released checkpoint keeps attention and Engram in FP8 and experts in FP4; treated here as FP8.`),
    },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: ph(890, 'No source yet', 'Replaced by the model card figure', undefined, 'bytes per token'),
    ...NO_POINTS,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: true,
    liveOnly: true,
    note: 'The biggest market on OpenRouter. The Engram table (196 GB) stays on the GPUs here; hosting it in CPU memory would let it fit fewer cards but is not modelled.',
  },
  {
    id: 'glm53',
    name: 'GLM-5.3',
    orSlug: 'z-ai/glm-5.3',
    arch: 'moe',
    paramsB: ph(744, 'No source yet', 'Replaced by the figure derived from the config', undefined, 'billion parameters'),
    activeB: ph(40, 'No source yet', 'Replaced by the figure derived from the config', undefined, 'billion active parameters'),
    weights: {
      nvfp4: srcd(464.1, HF('nvidia/GLM-5.3-NVFP4'), SUM),
      fp8: srcd(755.6, HF('zai-org/GLM-5.3'), `${SUM}. The released checkpoint is FP8.`),
    },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: ph(55000, 'No source yet', 'Replaced by the figure from the config', undefined, 'bytes per token'),
    ...NO_POINTS,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: true,
    liveOnly: true,
  },
  {
    id: 'glm53flash',
    name: 'GLM-5.3 Flash',
    orSlug: 'z-ai/glm-5.3-flash',
    arch: 'moe',
    paramsB: srcd(320, HF('zai-org/GLM-5.3-Flash'), 'Model card: 320B total parameters'),
    activeB: srcd(18, HF('zai-org/GLM-5.3-Flash'), 'Model card: 18B active parameters'),
    weights: {
      nvfp4: srcd(204.4, HF('nvidia/GLM-5.3-Flash-NVFP4'), SUM),
      fp8: srcd(328.3, HF('zai-org/GLM-5.3-Flash'), `${SUM}. The released checkpoint is FP8.`),
    },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: ph(7000, 'No source yet', 'Replaced by the figure from the config', undefined, 'bytes per token'),
    ...NO_POINTS,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: true,
    liveOnly: true,
  },
  {
    id: 'minimaxm3',
    name: 'MiniMax-M3',
    orSlug: 'minimax/minimax-m3',
    arch: 'moe',
    paramsB: srcd(428, HF('MiniMaxAI/MiniMax-M3'), 'Model card: about 428B parameters'),
    activeB: srcd(23, HF('MiniMaxAI/MiniMax-M3'), 'Model card: about 23B activated parameters'),
    weights: {
      nvfp4: srcd(250.1, HF('nvidia/MiniMax-M3-NVFP4'), SUM),
      fp8: srcd(443.7, HF('MiniMaxAI/MiniMax-M3-MXFP8'), SUM),
    },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: ph(90000, 'No source yet', 'Replaced by the figure from the config', undefined, 'bytes per token'),
    ...NO_POINTS,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: true,
    liveOnly: true,
  },
  {
    id: 'mimo26flash',
    name: 'MiMo-V2.6 Flash',
    orSlug: 'xiaomi/mimo-v2.6-flash',
    arch: 'moe',
    paramsB: srcd(309, HF('XiaomiMiMo/MiMo-V2.6-Flash-RL'), 'Model card: 309B total parameters'),
    activeB: srcd(15, HF('XiaomiMiMo/MiMo-V2.6-Flash-RL'), 'Model card: 15B activated parameters'),
    weights: {
      mxfp4: srcd(172.9, HF('XiaomiMiMo/MiMo-V2.6-Flash-RL'), `${SUM}. The released checkpoint keeps attention in FP8 and experts in MXFP4.`),
    },
    defaultQuant: 'mxfp4',
    kvBytesPerTok: ph(11500, 'No source yet', 'Replaced by the figure from the config', undefined, 'bytes per token'),
    ...NO_POINTS,
    pointsQuant: { pro6000: 'mxfp4', ada: 'mxfp4' },
    mtp: null,
    economics: true,
    liveOnly: true,
    note: 'Only the released MXFP4 checkpoint is sized; the OpenRouter listing is the RL variant.',
  },
];

const RAW_MODELS: Omit<Model, 'net'>[] = [
  {
    id: 'qwen38_27b',
    name: 'Qwen3.8-27B',
    orSlug: 'qwen/qwen3.8-27b',
    arch: 'dense',
    paramsB: rep(27, T51, 'From the model name'),
    activeB: rep(27, T51, 'Dense'),
    weights: {
      nvfp4: meas(21.9, T51, 'Hugging Face file sizes summed on 22 Oct 2026'),
      fp8: meas(30.9, T51),
      awq4: meas(21, T51, 'AWQ build used on the Ada'),
    },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: drv(30_300, 'Report §5.1', '1.7M-token KV pool on a PRO 6000 at FP8 weights: (0.9×96 − 30.9 − 4 GB overhead) / 1.7M tokens'),
    conc8k: meas(32.5, T51, 'Measured 29-36 concurrent at 8K, first token under 10 s (midpoint)'),
    conc32k: meas(3, T51),
    pro6000: [
      { c: 16, tps: meas(862, T52, 'HARD: Friendli, NVSX') },
      { c: 64, tps: est(2250, T52, 'EST: 2,000-2,500 at 64 (midpoint)') },
    ],
    ada: [
      { c: 16, tps: est(300, T52, 'EST for the 6000 Ada') },
      { c: 64, tps: est(900, T52, 'EST: 800-1,000 (midpoint)') },
    ],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'awq4' },
    mtp: {
      al: ph(2.32, 'InferenceX qwen3.8next_mtp.yaml', 'Proxy: acceptance length of the sibling Qwen3.8-Flash-Next at 3 draft tokens (thinking on). No curve exists for Qwen3.8-27B.', undefined, 'tokens per step'),
      k: 3,
      measured: { c: 32, tps: meas(1752, T52, 'Measured with MTP speculative decoding at concurrency 32') },
      note: 'Fade-out calibrated so the gain at 32 concurrent matches the measured 1,752 tok/s.',
    },
    economics: true,
    note: 'Default is NVFP4 to match the report\'s measured 862 tok/s. The report recommends FP8 until NVFP4 passes the §5.5 quality checks; FP8 costs about 29% of decode speed here.',
  },
  {
    id: 'qwen36_35b',
    name: 'Qwen3.6-35B-A3B',
    arch: 'moe',
    paramsB: rep(35, T51, 'From the model name'),
    activeB: rep(3, T51, 'From the model name'),
    weights: { nvfp4: meas(23.4, T51), fp8: meas(37.5, T51) },
    defaultQuant: 'fp8',
    kvBytesPerTok: kvGeneric(30_300, 'Assumed equal to Qwen3.8-27B (same hybrid-attention family); verify from the Hugging Face config in phase 3'),
    conc8k: meas(125, T51, '125+ concurrent at 8K'),
    conc32k: meas(41, T51),
    pro6000: [
      { c: 16, tps: est(1000, T52, 'EST: 900-1,100 (midpoint)') },
      { c: 100, tps: est(1800, T52, 'Medium: 1,700-1,900 at 75-125 users with 8K prompts (midpoint)') },
    ],
    ada: [
      { c: 16, tps: est(700, T52, 'EST') },
      { c: 64, tps: est(1500, T52, 'EST') },
    ],
    refIdx: 0,
    pointsQuant: { pro6000: 'fp8', ada: 'nvfp4' },
    mtp: {
      al: ph(2.32, 'InferenceX qwen3.8next_mtp.yaml', 'Proxy: Qwen3.8-Flash-Next acceptance length. Whether this checkpoint ships an MTP head is not checked.', undefined, 'tokens per step'),
      k: 3,
      note: 'No measured MTP point. Fade-out uses a placeholder saturation concurrency.',
    },
    economics: true,
  },
  {
    id: 'gemma4_26b',
    name: 'Gemma 4 26B-A4B',
    orSlug: 'google/gemma-4-26b-a4b-it',
    arch: 'moe',
    paramsB: rep(26, T51, 'From the model name'),
    activeB: rep(4, T51, 'From the model name'),
    weights: { fp8: meas(26, T51, 'About 26 GB at FP8') },
    defaultQuant: 'fp8',
    kvBytesPerTok: kvGeneric(60_000, 'Generic guess for a sliding-window plus global-attention model; verify from the Hugging Face config in phase 3'),
    conc8k: meas(125, T51, '125+ concurrent at 8K'),
    conc32k: meas(13, T51),
    pro6000: [
      { c: 16, tps: est(1000, T52, 'EST') },
      { c: 64, tps: est(2125, T52, 'Medium: 2,000-2,250 (midpoint)') },
    ],
    ada: [
      { c: 16, tps: est(600, T52, 'EST') },
      { c: 64, tps: est(1300, T52, 'EST') },
    ],
    refIdx: 0,
    pointsQuant: { pro6000: 'fp8', ada: 'fp8' },
    mtp: null,
    economics: true,
  },
  {
    id: 'gemma4_31b',
    name: 'Gemma 4 31B',
    arch: 'dense',
    paramsB: rep(31, T51, 'From the model name'),
    activeB: rep(31, T51, 'Dense'),
    weights: { nvfp4: meas(32.6, T51), awq4: meas(20.5, T51) },
    defaultQuant: 'awq4',
    kvBytesPerTok: kvGeneric(100_000, 'Generic guess for a 31B dense GQA model; verify from the Hugging Face config in phase 3'),
    conc8k: meas(34, T51),
    conc32k: meas(2, T51),
    pro6000: [
      { c: 16, tps: est(300, T52, 'EST') },
      { c: 64, tps: est(850, T52, 'Medium to low confidence') },
    ],
    ada: [
      { c: 16, tps: est(200, T52, 'EST') },
      { c: 64, tps: est(500, T52, 'EST') },
    ],
    refIdx: 0,
    pointsQuant: { pro6000: 'awq4', ada: 'awq4' },
    mtp: null,
    economics: true,
  },
  {
    id: 'gptoss_120b',
    name: 'gpt-oss-120b',
    arch: 'moe',
    paramsB: rep(120, T51, 'From the model name'),
    activeB: ph(5.1, 'No source yet', 'Active parameters as remembered; not in the report', undefined, 'billion active parameters'),
    weights: { mxfp4: meas(66, T51, 'About 66 GB at MXFP4') },
    defaultQuant: 'mxfp4',
    kvBytesPerTok: kvGeneric(18_400, 'From the gpt-oss architecture as remembered (18 full-attention layers, 8 KV heads, head dim 64, FP8); verify in phase 3'),
    conc8k: meas(59, T51),
    conc32k: meas(5, T51),
    pro6000: [
      { c: 16, tps: est(900, T52, 'EST: 800-1,000 (midpoint)') },
      { c: 50, tps: est(1525, T52, 'Medium: 1,525 at 50. A tuned stack reaches about 4,600 [ANEC], not used.') },
    ],
    ada: [],
    refIdx: 1,
    pointsQuant: { pro6000: 'mxfp4', ada: 'mxfp4' },
    mtp: null,
    economics: true,
    note: 'Does not fit the 48 GB Ada. The report\'s busy-hour revenue matches the 50-concurrent point, not 16.',
  },
  {
    id: 'gptoss_20b',
    name: 'gpt-oss-20b',
    arch: 'moe',
    paramsB: rep(20, T51, 'From the model name'),
    activeB: ph(3.6, 'No source yet', 'Active parameters as remembered; not in the report', undefined, 'billion active parameters'),
    weights: { mxfp4: meas(13, T51, 'About 13 GB at MXFP4') },
    defaultQuant: 'mxfp4',
    kvBytesPerTok: kvGeneric(12_300, 'From the gpt-oss architecture as remembered (12 full-attention layers, 8 KV heads, head dim 64, FP8); verify in phase 3'),
    conc8k: est(100, T51, '100+ [EST]'),
    conc32k: null,
    pro6000: [
      { c: 16, tps: est(2500, T52, 'EST') },
      { c: 50, tps: est(3753, T52, 'Medium: 3,753 at 50') },
    ],
    ada: [
      { c: 16, tps: est(1500, T52, 'EST') },
      { c: 64, tps: est(3000, T52, 'EST') },
    ],
    refIdx: 1,
    pointsQuant: { pro6000: 'mxfp4', ada: 'mxfp4' },
    mtp: null,
    economics: true,
    note: 'The report\'s busy-hour revenue matches the 50-concurrent point, as for gpt-oss-120b.',
  },
  {
    id: 'llama33_70b',
    name: 'Llama 3.3 70B',
    arch: 'dense',
    paramsB: rep(70, T51, 'From the model name'),
    activeB: rep(70, T51, 'Dense'),
    weights: { nvfp4: meas(42.7, T51) },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(163_840, 'From the Llama 3 architecture as remembered (80 layers, 8 KV heads, head dim 128, FP8). Consistent with the report: about 15-30 users at 8K.'),
    conc8k: est(22, T51, 'Report: about 15-30 users at 8K (midpoint)'),
    conc32k: null,
    pro6000: [
      { c: 16, tps: est(700, T52, 'EST: 600-800 (midpoint)') },
      { c: 64, tps: est(1378, T52, 'Medium: 1,031 (AWQ) to 1,724 (FP4, max load); midpoint, concurrency assumed') },
    ],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: true,
    note: 'Does not fit the 48 GB Ada (1.5 GB of KV left).',
  },
  {
    id: 'mistral_nemo',
    name: 'Mistral Nemo 12B',
    arch: 'dense',
    paramsB: rep(12, 'Report §5.3', 'From the model name'),
    activeB: rep(12, 'Report §5.3', 'Dense'),
    weights: { fp8: ph(12.2, 'No source yet', 'About one byte per parameter; not in the report', undefined, 'GB of weights') },
    defaultQuant: 'fp8',
    kvBytesPerTok: kvGeneric(81_920, 'From the Mistral Nemo architecture as remembered (40 layers, 8 KV heads, head dim 128, FP8); verify in phase 3'),
    conc8k: null,
    conc32k: null,
    pro6000: [
      { c: 16, tps: ph(2100, 'No source yet', 'The report has no throughput row for Nemo. Placeholder; revenue is anchored to the report\'s §6.1 busy-hour figure.', undefined, 'tok/s') },
      { c: 64, tps: ph(5200, 'No source yet', 'Placeholder', undefined, 'tok/s') },
    ],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'fp8', ada: 'fp8' },
    mtp: null,
    economics: true,
    note: 'Commodity trap: 1.6x the bar at the median price, below it at the floor (report §6.1).',
  },
  {
    id: 'flashnext',
    name: 'Qwen3.8-Flash-Next 125B-A6B',
    orSlug: 'qwen/qwen3.8-flash',
    arch: 'moe',
    paramsB: rep(125, T51, 'From the model name'),
    activeB: rep(6, T51, 'From the model name'),
    weights: { nvfp4: drv(86, 'Report §5.1', '186 GB on disk minus about 100 GB of n-gram table kept in host RAM') },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(20_000, 'Placeholder: hybrid attention, long context. Verify from the Hugging Face config in phase 3'),
    conc8k: null,
    conc32k: null,
    pro6000: [{ c: 6, tps: est(341, T52, 'ANEC: 6 clients at 100K context, with the n-gram table in host RAM') }],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    forceFit: { gpuIds: ['pro6000'], why: 'Report §5.1: runs on a 96 GB card only with about 100 GB of its n-gram table in host RAM' },
    mtp: null,
    economics: true,
    note: 'Needs about 100 GB of spare host RAM. Single provider on OpenRouter today (Alibaba), so a listing here fills a gap.',
  },
  {
    id: 'nemotron35',
    name: 'Nemotron 3.5 Super 120B-A12B',
    arch: 'moe',
    paramsB: rep(120, T51, 'From the model name'),
    activeB: rep(12, T51, 'From the model name'),
    weights: { nvfp4: est(77.5, T51, '75-80 GB (midpoint)') },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(8_000, 'Placeholder: hybrid Mamba MoE, very small KV per token'),
    conc8k: meas(70, T51, 'Measured on Nemotron 3 Super'),
    conc32k: meas(7, T51),
    pro6000: [
      { c: 16, tps: est(325, T52, 'EST: 300-350 (midpoint)') },
      { c: 64, tps: est(675, T52, 'EST: 650-700 (midpoint)') },
    ],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: false,
    note: 'The report does not model its revenue (slow on one card).',
  },
  {
    id: 'mistral_small4',
    name: 'Mistral Small 4 119B',
    arch: 'moe',
    paramsB: rep(119, T51, 'From the model name'),
    activeB: ph(6.5, 'No source yet', 'Placeholder', undefined, 'billion active parameters'),
    weights: { nvfp4: meas(70.8, T51) },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(60_000, 'Placeholder; report: about 12 GB of KV left on a PRO 6000'),
    conc8k: meas(19, T51),
    conc32k: meas(3, T51),
    pro6000: [],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: false,
    note: 'No throughput or price data in the report.',
  },
  {
    id: 'devstral2',
    name: 'Devstral 2 123B',
    arch: 'dense',
    paramsB: rep(123, T51, 'From the model name'),
    activeB: rep(123, T51, 'Dense'),
    weights: { nvfp4: meas(75, T51) },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(200_000, 'Placeholder; report: about 8 GB of KV left on a PRO 6000'),
    conc8k: est(8, T51, 'A few users'),
    conc32k: null,
    pro6000: [],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: false,
    note: 'About 20 tok/s per user on a PRO 6000 (report §5.4).',
  },
  {
    id: 'qwen35_122b',
    name: 'Qwen3.5-122B-A10B',
    arch: 'moe',
    paramsB: rep(122, T51, 'From the model name'),
    activeB: rep(10, T51, 'From the model name'),
    weights: { nvfp4: meas(83.5, T51) },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: kvGeneric(30_000, 'Placeholder; report: 0-3 GB of KV left on a PRO 6000'),
    conc8k: null,
    conc32k: null,
    pro6000: [],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: false,
    note: 'Barely fits a PRO 6000 and carries only 1.5B paid tokens a day.',
  },
  ...LIVE_ONLY,
];

/** Replace placeholder memory figures with what each model's Hugging Face config says. */
function withArch(m: Omit<Model, 'net'>): Model {
  const net = ARCH[m.id] ?? null;
  if (!net) return { ...m, net: null };
  const out: Model = { ...m, net, kvBytesPerTok: kvPerTokV(net) };
  if (m.paramsB.p.kind === 'placeholder') {
    const t = totalFromConfigV(net);
    if (t) out.paramsB = t;
  }
  if (m.activeB.p.kind === 'placeholder') {
    const a = activeFromConfigV(net);
    if (a) out.activeB = a;
  }
  if (m.mtp) {
    const head = net.mtpLayers > 0 ? `The checkpoint ships an MTP head (config: ${net.mtpLayers} layer${net.mtpLayers > 1 ? 's' : ''}).` : 'The config shows no MTP head, so this checkpoint may need a separate draft model.';
    out.mtp = { ...m.mtp, note: `${m.mtp.note} ${head}` };
  }
  return out;
}

export const MODELS: Model[] = RAW_MODELS.map(withArch);

/**
 * Models that are not in the catalog but have measured data that calibrates the roofline. Qwen3.5-397B-A17B is the
 * only model InferenceX measured on the RTX PRO 6000 (four cards, tensor-parallel over PCIe).
 */
export const CAL_MODELS: Model[] = [
  withArch({
    id: 'qwen35_397b',
    name: 'Qwen3.5-397B-A17B',
    arch: 'moe',
    paramsB: rep(397, 'InferenceX display name', 'From the model name'),
    activeB: ph(17, 'No source yet', 'Replaced by the figure derived from the config', undefined, 'billion active parameters'),
    weights: { nvfp4: srcd(251.2, 'https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4', 'Safetensors files of the NVFP4 checkpoint InferenceX serves, summed from the Hugging Face file tree on 23 Oct 2026') },
    defaultQuant: 'nvfp4',
    kvBytesPerTok: ph(15_000, 'No source yet', 'Replaced by the figure from the config', undefined, 'bytes per token'),
    conc8k: null,
    conc32k: null,
    pro6000: [],
    ada: [],
    refIdx: 0,
    pointsQuant: { pro6000: 'nvfp4', ada: 'nvfp4' },
    mtp: null,
    economics: false,
    note: 'Calibration only: too big for one card, measured by InferenceX on four RTX PRO 6000s.',
  }),
];
export const CAL_MODEL_BY_ID: Record<string, Model> = Object.fromEntries(CAL_MODELS.map((m) => [m.id, m]));

export const MODEL_BY_ID: Record<string, Model> = Object.fromEntries(MODELS.map((m) => [m.id, m]));

/** Bytes per parameter used to derive weight sizes the report does not list. */
export const BYTES_PER_PARAM: Record<Quant, V> = {
  fp16: drv(2, 'Arithmetic', '16 bits per weight'),
  fp8: drv(1, 'Arithmetic', '8 bits per weight'),
  nvfp4: drv(0.56, 'Arithmetic', '4 bits plus block scales, rounded up'),
  mxfp4: drv(0.55, 'Arithmetic', '4 bits plus block scales, rounded up'),
  awq4: drv(0.6, 'Arithmetic', '4 bits plus group scales and unquantised layers, rounded up'),
};

export interface ProvOnly {
  p: Prov;
}

/** InferenceX measured acceptance length per draft length (SPEED-Bench coding, thinking on). */
export const GOLDEN_AL_QWEN38_FLASH_NEXT: { k: number; al: V }[] = [
  [1, 1.65],
  [2, 2.06],
  [3, 2.32],
  [4, 2.47],
  [5, 2.59],
  [6, 2.66],
].map(([k, al]) => ({
  k: k as number,
  al: meas(al as number, 'InferenceX inferencex-e2e/infx/golden_al_distribution/qwen3.8next_mtp.yaml', 'Measured on B300 with vLLM native MTP, SPEED-Bench coding, thinking on'),
}));
