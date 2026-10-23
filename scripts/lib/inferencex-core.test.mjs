import { describe, expect, it } from 'vitest';
import { compactRow, compactRows, toCurves } from './inferencex-core.mjs';

const row = (over = {}, metrics = {}) => ({
  hardware: 'h200',
  framework: 'vllm',
  model: 'gptoss120b',
  precision: 'fp4',
  spec_method: 'none',
  disagg: false,
  is_multinode: false,
  decode_tp: 8,
  decode_ep: 1,
  decode_dp_attention: false,
  num_decode_gpu: 8,
  benchmark_type: 'single_turn',
  isl: 8192,
  osl: 1024,
  conc: 16,
  date: '2026-05-30',
  run_url: 'https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123456/attempts/2',
  metrics: { output_tput_per_gpu: 327.4, input_tput_per_gpu: 2600, median_intvty: 174, median_ttft: 0.13, p90_ttft: 0.3, kv_cache_pool_tokens: 1e7, ...metrics },
  ...over,
});

describe('InferenceX rows', () => {
  it('keeps the figures the roofline needs and the run id, not the whole URL', () => {
    const c = compactRow(row());
    expect(c).toMatchObject({ hardware: 'h200', framework: 'vllm', gpus: 8, conc: 16, outPerGpu: 327.4, user: 174, ttft50: 0.13, run: '123456' });
  });
  it('drops multi-node and disaggregated runs, which are not one node', () => {
    expect(compactRow(row({ is_multinode: true }))).toBeNull();
    expect(compactRow(row({ disagg: true }))).toBeNull();
  });
  it('drops rows without a throughput', () => {
    expect(compactRow(row({}, { output_tput_per_gpu: null }))).toBeNull();
  });
  it('derives output per GPU and speed per user for agentic runs that report totals', () => {
    const c = compactRow(row({ benchmark_type: 'agentic_traces', isl: null, osl: null, num_decode_gpu: 4 }, { output_tput_per_gpu: undefined, output_tput_tps: 400, median_intvty: undefined, median_itl: 0.01 }));
    expect(c.outPerGpu).toBe(100);
    expect(c.user).toBe(100);
  });
  it('omits null fields', () => {
    expect('kvPoolTokens' in compactRow(row({}, { kv_cache_pool_tokens: null }))).toBe(false);
  });
});

describe('InferenceX curves', () => {
  const rows = compactRows([row({ conc: 32 }), row({ conc: 4 }), row({ conc: 16 }), row({ conc: 16, decode_tp: 4, num_decode_gpu: 4 })]);
  const curves = toCurves(rows);
  it('groups points by configuration and orders them by concurrency', () => {
    expect(curves).toHaveLength(2);
    const tp8 = curves.find((c) => c.tp === 8);
    expect(tp8.pts.map((p) => p[0])).toEqual([4, 16, 32]);
  });
  it('writes each point as concurrency, output per GPU, speed per user and first-token times', () => {
    const tp4 = curves.find((c) => c.tp === 4);
    expect(tp4.pts[0]).toEqual([16, 327.4, 174, 0.13, 0.3]);
  });
});
