import { GPU_BY_ID } from '../data/gpus';
import { IX_CURVES, type IxCurve } from '../data/ix';
import { CAL_MODEL_BY_ID, MODEL_BY_ID } from '../data/models';
import { weightsGB } from './capacity';
import { ROOF_CLASSES, roofClassOf, servingPoint, type RoofClass, type RoofParams } from './roofline';

/** One InferenceX measurement set against the roofline: the rows the efficiencies were fitted to. */
export interface FitRow {
  curve: IxCurve;
  conc: number;
  outM: number;
  userM: number | null;
  ttftM: number | null;
}

/**
 * Runs at the weight formats the dashboard serves: 4-bit and FP8 for the catalog models, and 4-bit only for the 397B
 * model, whose 4-bit run is the one measured on the PRO 6000. Its 8-GPU FP8 and BF16 runs are far outside what a small
 * fleet rents and stay out of the fit.
 */
export function fitRows(): FitRow[] {
  return IX_CURVES.filter((c) => c.quant !== 'fp16' && (c.modelId !== 'qwen35_397b' || c.precision === 'fp4')).flatMap((curve) =>
    curve.pts.map((p) => ({ curve, conc: p.conc, outM: p.outPerGpu, userM: p.user, ttftM: p.ttft50 })),
  );
}

export interface FitPrediction {
  /** Output tokens per second per GPU. */
  out: number;
  /** Tokens per second one user sees, from the decode step alone. */
  user: number;
  ttft: number;
}

export function predictRow(r: FitRow, P: Record<RoofClass, RoofParams> = ROOF_CLASSES): FitPrediction | null {
  const c = r.curve;
  const model = MODEL_BY_ID[c.modelId] ?? CAL_MODEL_BY_ID[c.modelId];
  const gpu = GPU_BY_ID[c.gpuId];
  const w = model ? weightsGB(model, c.quant) : null;
  if (!model || !gpu || !w) return null;
  const shape = { avgTokens: c.isl + c.osl, outShare: c.osl / (c.isl + c.osl), cacheHit: 0 };
  const sp = servingPoint({ model, gpu, quant: c.quant, count: c.gpus, weightsGB: w.gb }, shape, r.conc, P[roofClassOf(gpu)]);
  return { out: sp.tps / c.gpus, user: 1 / sp.step.t, ttft: sp.ttftP50 };
}

/** Mean squared log error over output speed, speed per user and time to first token. */
export function fitLoss(rows: FitRow[], P: Record<RoofClass, RoofParams> = ROOF_CLASSES): number {
  let L = 0;
  let n = 0;
  for (const r of rows) {
    const p = predictRow(r, P);
    if (!p) continue;
    L += Math.log(p.out / r.outM) ** 2 + (r.userM ? 0.5 * Math.log(p.user / r.userM) ** 2 : 0) + (r.ttftM ? 0.5 * Math.log(p.ttft / r.ttftM) ** 2 : 0);
    n += 1;
  }
  return n ? L / n : 0;
}

export interface FitGroup {
  key: string;
  model: string;
  gpuId: string;
  precision: string;
  gpus: number;
  n: number;
  /** Median of predicted over measured. 1 is a perfect fit; below 1 the roofline is too slow. */
  out: number;
  user: number;
  ttft: number;
}

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)]! : NaN;
};

/** Residuals per model, card, weight format and GPU count. */
export function fitGroups(rows: FitRow[] = fitRows(), P: Record<RoofClass, RoofParams> = ROOF_CLASSES): FitGroup[] {
  const by = new Map<string, { r: FitRow; p: FitPrediction }[]>();
  for (const r of rows) {
    const p = predictRow(r, P);
    if (!p) continue;
    const k = `${r.curve.modelName}|${r.curve.gpuId}|${r.curve.precision}|${r.curve.gpus}`;
    if (!by.has(k)) by.set(k, []);
    by.get(k)!.push({ r, p });
  }
  return [...by.entries()]
    .map(([key, v]) => ({
      key,
      model: v[0]!.r.curve.modelName,
      gpuId: v[0]!.r.curve.gpuId,
      precision: v[0]!.r.curve.precision,
      gpus: v[0]!.r.curve.gpus,
      n: v.length,
      out: median(v.map((x) => x.p.out / x.r.outM)),
      user: median(v.filter((x) => x.r.userM).map((x) => x.p.user / x.r.userM!)),
      ttft: median(v.filter((x) => x.r.ttftM).map((x) => x.p.ttft / x.r.ttftM!)),
    }))
    .sort((a, b) => a.key.localeCompare(b.key));
}
