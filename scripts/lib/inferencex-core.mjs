/**
 * InferenceX measured benchmark rows (https://inferencex.semianalysis.com/api/v1), compacted to what the roofline
 * needs: per configuration and concurrency, output speed, speed per user and time to first token.
 */

/** InferenceX display model name → catalog model id. A null id keeps the model as a calibration-only entry. */
export const IX_MODELS = {
  'gpt-oss-120b': { id: 'gptoss_120b' },
  'Llama-3.3-70B-Instruct-FP8': { id: 'llama33_70b' },
  'Qwen3.8-Flash-Next': { id: 'flashnext' },
  'Qwen3.8-27B': { id: 'qwen38_27b' },
  'Qwen-3.5-397B-A17B': { id: 'qwen35_397b', calibrationOnly: true },
  'MiniMax-M3': { id: 'minimaxm3' },
  'DeepSeek-V4.1-Flash': { id: 'dsv41flash' },
};

/** InferenceX hardware key → catalog GPU id. Cards the catalog does not have are kept under their own key. */
export const IX_RUN_URL = 'https://github.com/SemiAnalysisAI/InferenceX/actions/runs/';

export const IX_HARDWARE = {
  rtx6000pro: 'pro6000',
  h100: 'h100',
  h200: 'h200',
  b200: 'b200',
};

const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null);
const r4 = (x) => (x === null ? null : Number(x.toPrecision(5)));

/** One compact row, or null for rows the roofline cannot use (multi-node, disaggregated, no throughput). */
export function compactRow(row) {
  if (row.is_multinode || row.disagg) return null;
  const m = row.metrics ?? {};
  const gpus = num(row.num_decode_gpu) ?? 1;
  const out = num(m.output_tput_per_gpu) ?? (num(m.output_tput_tps) !== null ? m.output_tput_tps / gpus : null);
  if (out === null || !num(row.conc)) return null;
  const itl = num(m.median_itl) ?? num(m.mean_itl);
  const user = num(m.median_intvty) ?? (itl ? 1 / itl : null);
  const runId = typeof row.run_url === 'string' ? (row.run_url.match(/runs\/(\d+)/) ?? [])[1] : null;
  const out_ = {
    hardware: row.hardware,
    framework: row.framework,
    precision: row.precision,
    spec: row.spec_method,
    type: row.benchmark_type,
    tp: row.decode_tp,
    ep: row.decode_ep,
    dpAttn: Boolean(row.decode_dp_attention),
    gpus,
    isl: num(row.isl),
    osl: num(row.osl),
    conc: row.conc,
    outPerGpu: r4(out),
    inPerGpu: r4(num(m.input_tput_per_gpu) ?? (num(m.input_tput_tps) !== null ? m.input_tput_tps / gpus : null)),
    user: r4(user),
    ttft50: r4(num(m.median_ttft)),
    ttft90: r4(num(m.p90_ttft)),
    kvPoolTokens: num(m.kv_cache_pool_tokens),
    date: row.date,
    run: runId ?? null,
  };
  // Null fields add bytes and nothing else.
  return Object.fromEntries(Object.entries(out_).filter(([, v]) => v !== null));
}

export function compactRows(rows) {
  return rows.map(compactRow).filter(Boolean);
}

/** Grouping key: one curve is one configuration swept over concurrency on one date. */
export function curveKey(r) {
  return [r.hardware, r.framework, r.precision, r.spec, r.type, r.tp, r.ep, r.dpAttn ? 'dp' : 'tp', r.gpus, r.isl, r.osl, r.date].join('|');
}

/**
 * Curves: the shared fields once, and each point as [concurrency, output per GPU, speed per user, median first token,
 * 90th-percentile first token]. A third the size of one object per point.
 */
export function toCurves(rows) {
  const by = new Map();
  for (const r of rows) {
    const k = curveKey(r);
    if (!by.has(k)) by.set(k, []);
    by.get(k).push(r);
  }
  return [...by.values()].map((v) => {
    const r0 = v[0];
    v.sort((a, b) => a.conc - b.conc);
    const c = {
      hardware: r0.hardware,
      framework: r0.framework,
      precision: r0.precision,
      spec: r0.spec,
      type: r0.type,
      tp: r0.tp,
      ep: r0.ep,
      dpAttn: r0.dpAttn,
      gpus: r0.gpus,
      isl: r0.isl,
      osl: r0.osl,
      date: r0.date,
      run: r0.run,
      pts: v.map((r) => [r.conc, r.outPerGpu, r.user ?? null, r.ttft50 ?? null, r.ttft90 ?? null]),
    };
    return Object.fromEntries(Object.entries(c).filter(([, x]) => x !== undefined && x !== null));
  });
}
