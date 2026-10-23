import bench from '../../data/inferencex/bench.json';
import type { Quant } from './gpus';
import { MODEL_BY_ID, CAL_MODEL_BY_ID } from './models';
import type { Prov } from './provenance';

/** InferenceX (SemiAnalysis) measured runs, compacted by `npm run data:inferencex`. */
export interface IxPoint {
  /** Concurrent requests across the whole node. */
  conc: number;
  /** Output tokens per second per GPU. */
  outPerGpu: number;
  /** Tokens per second one user sees (1 / median inter-token latency). */
  user: number | null;
  ttft50: number | null;
  ttft90: number | null;
}

export interface IxCurve {
  modelId: string;
  /** InferenceX display name. */
  modelName: string;
  gpuId: string;
  framework: string;
  precision: string;
  quant: Quant;
  gpus: number;
  tp: number;
  isl: number;
  osl: number;
  date: string;
  run: string | null;
  pts: IxPoint[];
}

export const IX_SNAPSHOT_AT: string = bench.generatedAt;
export const IX_RUN_URL = 'https://github.com/SemiAnalysisAI/InferenceX/actions/runs/';

const HW: Record<string, string> = { rtx6000pro: 'pro6000', h100: 'h100', h200: 'h200', b200: 'b200' };
/** Stacks a small provider can run. TensorRT-LLM and Dynamo runs are tuned beyond that. */
const OPEN_STACKS = ['vllm', 'sglang'];
const MIN_ANCHOR_POINTS = 3;

type RawPoint = [conc: number, out: number, user: number | null, ttft50: number | null, ttft90: number | null];

interface RawCurve {
  hardware: string;
  framework: string;
  precision: string;
  spec: string;
  type: string;
  tp: number;
  ep: number;
  dpAttn: boolean;
  gpus: number;
  isl?: number;
  osl?: number;
  date: string;
  run?: string;
  pts: RawPoint[];
}

/** Which of our weight formats a benchmark's precision label means for this model. */
function quantOf(modelId: string, precision: string): Quant | null {
  const m = MODEL_BY_ID[modelId] ?? CAL_MODEL_BY_ID[modelId];
  if (!m) return null;
  if (precision === 'fp8') return 'fp8';
  if (precision === 'bf16') return 'fp16';
  if (precision === 'fp4') return m.weights.mxfp4 ? 'mxfp4' : 'nvfp4';
  return null;
}

/** A measured point the dashboard ignores, with the reason. */
export interface IxDropped {
  modelName: string;
  gpuId: string;
  precision: string;
  gpus: number;
  isl: number;
  osl: number;
  conc: number;
  outPerGpu: number;
  reason: string;
  run: string | null;
}

export const IX_DROPPED: IxDropped[] = [];

/**
 * A benchmark run that crashed or stalled shows up as a point that breaks the physics: node throughput collapsing as
 * concurrency rises, or a user seeing a much slower stream at low load than at high load. Such points are dropped.
 */
const COLLAPSE = 0.5;
const USER_INVERSION = 1.5;

function sane(pts: RawPoint[], gpus: number): { keep: RawPoint[]; drop: { p: RawPoint; reason: string }[] } {
  const sorted = [...pts].sort((a, b) => a[0] - b[0]);
  const keep: RawPoint[] = [];
  const drop: { p: RawPoint; reason: string }[] = [];
  sorted.forEach((p, i) => {
    const node = p[1] * gpus;
    const lower = sorted.slice(0, i).reduce((m, q) => Math.max(m, q[1] * gpus), 0);
    if (lower > 0 && node < COLLAPSE * lower) {
      drop.push({ p, reason: `throughput fell to ${Math.round((100 * node) / lower)}% of the best at lower concurrency` });
      return;
    }
    const slower = p[2] != null && sorted.slice(i + 1).some((q) => q[2] != null && q[2] > USER_INVERSION * p[2]!);
    if (slower) {
      drop.push({ p, reason: 'a lone user saw a far slower stream than users at higher concurrency' });
      return;
    }
    keep.push(p);
  });
  return { keep, drop };
}

function build(): IxCurve[] {
  const out: IxCurve[] = [];
  const models = bench.models as unknown as Record<string, { id: string; curves: RawCurve[] }>;
  for (const [name, m] of Object.entries(models)) {
    for (const r of m.curves) {
      // Plain tensor-parallel runs of fixed-length requests; speculative decoding, expert parallelism and agentic traces are different workloads.
      if (r.type !== 'single_turn' || r.spec !== 'none' || r.dpAttn || r.ep > 1 || !r.isl || !r.osl || !HW[r.hardware] || !OPEN_STACKS.includes(r.framework)) continue;
      const quant = quantOf(m.id, r.precision);
      if (!quant) continue;
      const { keep, drop } = sane(r.pts, r.gpus);
      for (const d of drop) IX_DROPPED.push({ modelName: name, gpuId: HW[r.hardware]!, precision: r.precision, gpus: r.gpus, isl: r.isl, osl: r.osl, conc: d.p[0], outPerGpu: d.p[1], reason: d.reason, run: r.run ?? null });
      if (keep.length === 0) continue;
      out.push({
        modelId: m.id,
        modelName: name,
        gpuId: HW[r.hardware]!,
        framework: r.framework,
        precision: r.precision,
        quant,
        gpus: r.gpus,
        tp: r.tp,
        isl: r.isl,
        osl: r.osl,
        date: r.date,
        run: r.run ?? null,
        pts: keep.map(([conc, outPerGpu, user, ttft50, ttft90]) => ({ conc, outPerGpu, user, ttft50, ttft90 })),
      });
    }
  }
  return out;
}

export const IX_CURVES: IxCurve[] = build();

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)]! : 0;
};

/** Measured curves for exactly this model, card, weight format and GPU count. */
export function ixCurves(modelId: string, gpuId: string, quant: Quant, gpus: number): IxCurve[] {
  return IX_CURVES.filter((c) => c.modelId === modelId && c.gpuId === gpuId && c.quant === quant && c.gpus === gpus);
}

/**
 * The measured curve closest to the request's context length (8K/1K or 1K/1K), taking the stack that served most per GPU
 * where several were run. Null where this config was never measured.
 */
export function bestIxCurve(modelId: string, gpuId: string, quant: Quant, gpus: number, ctx: number): IxCurve | null {
  // A curve left with one or two points after the sanity filter cannot say how speed changes with load.
  const all = ixCurves(modelId, gpuId, quant, gpus).filter((c) => c.pts.length >= MIN_ANCHOR_POINTS);
  if (all.length === 0) return null;
  const dist = (c: IxCurve) => Math.abs(Math.log((c.isl + c.osl / 2) / Math.max(1, ctx)));
  const best = Math.min(...all.map(dist));
  const near = all.filter((c) => dist(c) < best + 1e-9);
  return near.sort((a, b) => median(b.pts.map((p) => p.outPerGpu)) - median(a.pts.map((p) => p.outPerGpu)))[0]!;
}

export function ixProv(c: IxCurve): Prov {
  return {
    kind: 'measured',
    src: c.run ? `${IX_RUN_URL}${c.run}` : 'InferenceX',
    note: `InferenceX, ${c.framework}, ${c.gpus} × ${c.gpuId}, ${c.precision.toUpperCase()}, ${c.isl / 1024}K in / ${c.osl / 1024}K out, ${c.date}`,
    asOf: c.date,
  };
}
