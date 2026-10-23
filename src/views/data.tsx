import { useMemo, useState } from 'preact/hooks';
import { FX_INR_PER_USD, GPUS, HOURS_PER_MONTH, SUMMER_DERATE } from '../data/gpus';
import { CACHE_PRICE_FRACTION, MARKET } from '../data/market';
import { BYTES_PER_PARAM, GOLDEN_AL_QWEN38_FLASH_NEXT, MODELS } from '../data/models';
import { liveRegister, OPENROUTER, OR_DAY, VAST, VAST_DAY } from '../data/live';
import { KIND_LABEL, KIND_ORDER, collectProvenance, countByKind, type ProvKind } from '../data/provenance';
import * as SC from '../data/scenarios';
import { GPU_BY_ID, QUANT_LABEL } from '../data/gpus';
import { bandwidthCeilingTps, decodeTps, ROOF_PROV } from '../engine/roofline';
import { weightsGB, GPU_MEM_UTIL, OVERHEAD_GB_PER_GPU, MIN_KV_TOKENS, ROOFLINE_MAX_CONC, REPORT_TEST_CTX, DRAFT_COST, MTP_DEFAULT_CSAT, VOLTA_KERNEL_PENALTY, MAX_EXTRAPOLATION } from '../engine/capacity';
import { checkStatus, runCrossChecks, type CheckRow } from '../engine/crosscheck';
import { fmtInr, fmtPct } from '../engine/units';
import { Field, SelectInput } from '../ui/controls';
import { ProvBadge } from '../ui/prov';

const ALL = {
  gpus: GPUS,
  models: MODELS,
  market: MARKET,
  scenarios: SC.SCENARIOS,
  lium: SC.LIUM,
  program: SC.PROGRAM,
  routeCosts: SC.ROUTE_COSTS,
  workloads: SC.WORKLOADS,
  availability: SC.AVAILABILITY_PRESETS,
  uptimeTiers: SC.UPTIME_TIERS,
  live: liveRegister(),
  constants: {
    fx: FX_INR_PER_USD,
    hoursPerMonth: HOURS_PER_MONTH,
    cachePriceFraction: CACHE_PRICE_FRACTION,
    summerDerate: SUMMER_DERATE,
    priceElasticity: SC.ELASTICITY_PRICE,
    latencyElasticity: SC.ELASTICITY_LATENCY,
    zdrBonus: SC.ZDR_SHARE_BONUS,
    ramp: SC.RAMP,
    floorMonths: SC.FLOOR_MONTHS,
    batchDiscount: SC.BATCH_DISCOUNT,
    gpuMemUtil: GPU_MEM_UTIL,
    overheadGbPerGpu: OVERHEAD_GB_PER_GPU,
    minKvTokens: MIN_KV_TOKENS,
    rooflineMaxConc: ROOFLINE_MAX_CONC,
    reportTestCtx: REPORT_TEST_CTX,
    roofline: ROOF_PROV,
    draftCost: DRAFT_COST,
    mtpDefaultCsat: MTP_DEFAULT_CSAT,
    voltaPenalty: VOLTA_KERNEL_PENALTY,
    maxExtrapolation: MAX_EXTRAPOLATION,
    bytesPerParam: BYTES_PER_PARAM,
    goldenAl: GOLDEN_AL_QWEN38_FLASH_NEXT,
    buyHorizon: SC.BUY_HORIZON_MONTHS,
    rentRules: SC.RENT_RULES,
    listingOdds: SC.LISTING_ODDS,
    secondModelDelay: SC.SECOND_MODEL_DELAY,
    tax: SC.TAX,
  },
};

const fmtVal = (v: unknown): string => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)))) : String(v));

function fmtCheck(r: CheckRow, x: number): string {
  if (r.unit === 'pct') return fmtPct(x, 1);
  return fmtInr(x, true);
}

const LIMITS = [
  { t: 'Prefill efficiency is a placeholder.', d: 'Prefill speed is compute-bound and its share of peak tensor throughput is not fitted to any measurement; the report\'s measured concurrency at 8K with first token under 10 s lands within a third to one-and-a-third times the roofline. Time to first token treats arrivals as a steady stream (an M/D/1 queue); bursts of agent requests arriving together would wait longer.' },
  { t: 'Multi-GPU scaling is a roofline model with placeholder links.', d: 'Nodes of 2, 4 and 8 GPUs split the weights and pay an all-reduce per layer; the link latency and bandwidth are placeholders. InferenceX has measured 8-GPU curves for H100, H200 and B200 on frontier models; they are not wired in yet.' },
  { t: 'Most of the report\'s throughput points are its own estimates.', d: 'Only one is a measurement (Qwen3.8-27B at 16 concurrent, and with MTP at 32); gpt-oss at 50 is stated at Medium confidence and the rest are flagged estimates. The roofline sits within 25% of those three and disagrees with several estimates; Llama 3.3 70B at 16 concurrent is above what the memory bus can deliver. The Throughput view marks these.' },
  { t: 'Cards other than the PRO 6000 and Ada carry the PRO 6000\'s kernel efficiency.', d: 'The report\'s per-busy-hour revenue gives ratios for the 5090, H100, H200 and B200 on a few models; everywhere else the roofline ratio is used, assuming the memory efficiency fitted on the PRO 6000 carries over. No measurement checks that.' },
  { t: 'Expert routing is assumed uniform.', d: 'A mixture-of-experts step reads the experts its batch touches; with uniform routing that is the most it can read. Real routers are skewed, so decode on these models may be faster than modelled.' },
  { t: 'KV figures assume an FP8 cache and ignore prefix sharing.', d: 'Each user is charged a full context. Sliding-window layers are capped at their window and recurrent state is counted in fp32, as the configs ask.' },
  { t: 'The V100 runs FP16 or INT4 only.', d: 'It has no FP8 or FP4, and older attention kernels. Its throughput carries a flat kernel penalty on top of the roofline.' },
  { t: 'The adjusted basis assumes the report\'s points describe observed traffic.', d: 'The report does not say what request shape its throughput points were measured at. The adjusted basis takes each model\'s observed traffic as that shape, pins the roofline to the report\'s number at its reference concurrency, and follows the roofline from there; at that point on observed traffic it reproduces the report exactly.' },
  { t: 'Request shape and cache price are calibrated, not observed.', d: 'In the report-snapshot mode they are back-solved from the report\'s busy-hour and typical-entrant figures. They reproduce the report, and match its stated 4.1% and 11.8% output shares for the two models where it states them.' },
  { t: 'Live data is a snapshot.', d: 'Prices and traffic are as of the last data run (shown on the Prices view). Refresh with npm run data; trends need at least two snapshots on different days.' },
  { t: 'Vast.ai coverage is thin for some cards.', d: 'The 6000 Ada has no offer at reliability 0.90 or above, and several multi-GPU sizes have none. Where a filter leaves no offer the dashboard falls back to the report or catalog price and says so on the card.' },
  { t: 'Tax after the first step is approximate for rented cards on the crypto route.', d: 'Gains over a Lium bar are exact on either route. Rented-card margin on the crypto route is taxed on gross receipts.' },
];


interface RoofRow {
  model: string;
  gpu: string;
  quant: string;
  c: number;
  report: number;
  kind: string;
  roof: number;
  ceiling: number;
}

/** The roofline against every throughput point the report publishes, at the report's 8K test context. */
function roofRows(): RoofRow[] {
  const out: RoofRow[] = [];
  for (const m of MODELS) {
    for (const gid of ['pro6000', 'ada6000'] as const) {
      const pts = gid === 'pro6000' ? m.pro6000 : m.ada;
      const q = m.pointsQuant[gid === 'pro6000' ? 'pro6000' : 'ada'];
      const w = weightsGB(m, q);
      if (!w) continue;
      for (const p of pts) {
        const i = { model: m, gpu: GPU_BY_ID[gid]!, quant: q, count: 1, weightsGB: w.gb, ctx: REPORT_TEST_CTX.v };
        out.push({ model: m.name, gpu: GPU_BY_ID[gid]!.short, quant: QUANT_LABEL[q], c: p.c, report: p.tps.v, kind: p.tps.p.kind, roof: decodeTps(i, p.c), ceiling: bandwidthCeilingTps(i, p.c) });
      }
    }
  }
  return out;
}

export function DataView() {
  const rows = useMemo(() => collectProvenance(ALL), []);
  const counts = useMemo(() => countByKind(rows), [rows]);
  const checks = useMemo(() => runCrossChecks(), []);
  const roof = useMemo(() => roofRows(), []);
  const above = roof.filter((r) => r.report > r.ceiling);
  const [kind, setKind] = useState<ProvKind | 'all'>('placeholder');
  const [q, setQ] = useState('');
  const [limit, setLimit] = useState(120);
  const [onlyMiss, setOnlyMiss] = useState(false);

  const filtered = rows.filter((r) => (kind === 'all' || r.prov.kind === kind) && (q === '' || `${r.path} ${r.prov.src} ${r.prov.note ?? ''}`.toLowerCase().includes(q.toLowerCase())));
  const ok = checks.filter((c) => checkStatus(c).ok).length;
  const worst = checks.reduce((a, c) => (c.unit === 'pct' ? a : Math.max(a, Math.abs(checkStatus(c).delta))), 0);
  const groups = Array.from(new Set(checks.map((c) => c.group)));

  return (
    <div class="stack">
      <section class="card" aria-label="Provenance summary">
        <h2>Where every number comes from</h2>
        <div class="sub">{rows.length} tagged values across the GPU, model, market and scenario data. Anything computed on top of them inherits the weakest tag, shown on each card.</div>
        <div class="tiles">
          {KIND_ORDER.filter((k) => counts[k] > 0).map((k) => (
            <button key={k} type="button" class="tile" style={{ textAlign: 'left', cursor: 'pointer' }} onClick={() => { setKind(k); setLimit(120); }} aria-pressed={kind === k}>
              <div class="t-label">
                <ProvBadge p={{ kind: k, src: KIND_LABEL[k] }} />
              </div>
              <div class="t-value">{counts[k]}</div>
              <div class="t-sub">{fmtPct(counts[k] / rows.length)} of values</div>
            </button>
          ))}
        </div>
      </section>

      <section class="card" aria-label="Live data">
        <h2>Live data</h2>
        <div class="sub">Two scripts collect it and save dated snapshots under data/live, so the dashboard works offline and history builds up.</div>
        <div class="kv" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
          <div>
            <span>OpenRouter snapshot</span>
            <b>
              {OR_DAY} · {Object.keys(OPENROUTER.models).length} models
            </b>
          </div>
          <div>
            <span>Vast.ai snapshot</span>
            <b>
              {VAST_DAY} · {Object.keys(VAST.gpus).length} GPUs · {Object.values(VAST.gpus).reduce((a, g) => a + (g.nodes['1']?.views.any.count ?? 0), 0)} single-card offers
            </b>
          </div>
        </div>
        <p class="small mt">
          Refresh with <code>npm run data</code> (both) or <code>npm run data:openrouter</code> and <code>npm run data:vast</code>. The Vast.ai script needs a Chromium; set <code>CHROME_PATH</code> if Playwright has none, or <code>HEADED=1</code> to watch it. Commit the changed files in <code>data/live</code>.
        </p>
      </section>

      <section class="card" aria-label="Value register">
        <div class="spread">
          <h2>Register of values</h2>
          <div class="row">
            <Field label="Show">
              <SelectInput<ProvKind | 'all'> label="Provenance kind" value={kind} options={[{ id: 'all', label: 'All kinds' }, ...KIND_ORDER.filter((k) => counts[k] > 0).map((k) => ({ id: k as ProvKind | 'all', label: `${KIND_LABEL[k]} (${counts[k]})` }))]} onChange={(v) => { setKind(v); setLimit(120); }} />
            </Field>
            <Field label="Search">
              <input class="input" aria-label="Search the register" value={q} onInput={(e) => { setQ(e.currentTarget.value); setLimit(120); }} placeholder="path, source or note" />
            </Field>
          </div>
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>Value</th>
                <th class="num">Number</th>
                <th>Kind</th>
                <th>Source</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, limit).map((r, i) => (
                <tr key={`${r.path}-${i}`}>
                  <td style={{ fontFamily: 'var(--mono)', fontSize: '12px' }}>{r.path}</td>
                  <td class="num">{fmtVal(r.value)}</td>
                  <td>
                    <ProvBadge p={r.prov} />
                  </td>
                  <td>{r.prov.src}</td>
                  <td class="small">{r.prov.note ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div class="spread mt">
          <span class="small">
            Showing {Math.min(limit, filtered.length)} of {filtered.length}
          </span>
          {filtered.length > limit ? (
            <button type="button" class="btn" onClick={() => setLimit(limit + 200)}>
              Show more
            </button>
          ) : null}
        </div>
      </section>

      <section class="card" aria-label="Cross-check against the report">
        <div class="spread">
          <h2>
            Cross-check against the report: {ok} of {checks.length} within tolerance
          </h2>
          <label class="check">
            <input type="checkbox" checked={onlyMiss} onChange={(e) => setOnlyMiss(e.currentTarget.checked)} />
            <span>Only show those that differ by more than 1%</span>
          </label>
        </div>
        <div class="sub">
          The dashboard's engine run under the report's own conditions (no outage haircut, no what-ifs) next to the report's printed figures. Largest rupee deviation is {fmtPct(worst, 1)}. These checks run in the test suite, so a change to the engine that drifts from the report fails the build.
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>Figure</th>
                <th class="num">Report</th>
                <th class="num">Dashboard</th>
                <th class="num">Difference</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const items = checks.filter((c) => c.group === g).filter((c) => !onlyMiss || Math.abs(checkStatus(c).delta) > 0.01);
                if (items.length === 0) return null;
                return (
                  <>
                    <tr key={`g-${g}`}>
                      <td colSpan={5} style={{ fontWeight: 600, background: 'var(--surface-2)' }}>
                        {g}
                      </td>
                    </tr>
                    {items.map((c) => {
                      const st = checkStatus(c);
                      return (
                        <tr key={c.id}>
                          <td>{c.label}</td>
                          <td class="num">{fmtCheck(c, c.report)}</td>
                          <td class="num">{fmtCheck(c, c.dashboard)}</td>
                          <td class={`num${st.ok ? '' : ' miss'}`}>{c.unit === 'pct' ? `${(st.delta * 100).toFixed(1)} pts` : `${st.delta >= 0 ? '+' : ''}${(st.delta * 100).toFixed(1)}%`}</td>
                          <td class="small">{c.note ?? ''}</td>
                        </tr>
                      );
                    })}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section class="card" aria-label="Roofline against the report's throughput points">
        <h2>Roofline against the report's throughput points</h2>
        <div class="sub">
          {roof.length} points on the PRO 6000 and the Ada, at {REPORT_TEST_CTX.v} tokens of context. The roofline's memory efficiency and step overhead were fitted to three of them: Qwen3.8-27B at 16 concurrent (measured), and gpt-oss-120b and gpt-oss-20b at 50 (stated at Medium confidence). The rest are the report's own estimates. {above.length} of them {above.length === 1 ? 'is' : 'are'} above the bus ceiling, the speed the memory bus could deliver at full bandwidth with nothing to read but the weights.
        </div>
        <details class="more">
          <summary>Show all {roof.length} points</summary>
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>GPU</th>
                  <th>Weights</th>
                  <th class="num">Concurrent</th>
                  <th class="num">Report</th>
                  <th>Report source</th>
                  <th class="num">Roofline</th>
                  <th class="num">Roofline ÷ report</th>
                  <th class="num">Bus ceiling</th>
                </tr>
              </thead>
              <tbody>
                {roof.map((r, i) => (
                  <tr key={i}>
                    <td>{r.model}</td>
                    <td>{r.gpu}</td>
                    <td>{r.quant}</td>
                    <td class="num">{r.c}</td>
                    <td class="num">{Math.round(r.report)}</td>
                    <td>
                      <ProvBadge p={{ kind: r.kind as ProvKind, src: r.kind === 'measured' ? 'Report §5.2' : 'Report §5.2 estimate' }} compact />
                    </td>
                    <td class="num">{Math.round(r.roof)}</td>
                    <td class="num">{(r.roof / r.report).toFixed(2)}</td>
                    <td class={r.report > r.ceiling ? 'num miss' : 'num'}>{Math.round(r.ceiling)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      <section class="card" aria-label="Known limitations">
        <h2>What this version does not do yet</h2>
        <ul class="plain">
          {LIMITS.map((l) => (
            <li key={l.t}>
              <b>{l.t}</b> {l.d}
            </li>
          ))}
        </ul>
        <h3 class="mt">Build order</h3>
        <ol class="plain">
          <li>
            <b>Phase 1:</b> the interface, the portfolio planner and the economics engine on the report's static numbers.
          </li>
          <li>
            <b>Phase 2:</b> live Vast.ai (browser scrape, then a script) and OpenRouter prices with dated history.
          </li>
          <li>
            <b>Phase 3 (this version):</b> model architecture from Hugging Face configs, a roofline for decode, prefill and time to first token fitted to the report's measured points, three throughput bases, a first-token limit in the quality bar.
          </li>
          <li>
            <b>Next:</b> InferenceX measured curves for multi-GPU nodes, a measurement run on your own cards to fit prefill efficiency, AMD and A100.
          </li>
        </ol>
      </section>
    </div>
  );
}
