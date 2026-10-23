import { useMemo, useState } from 'preact/hooks';
import { FX_INR_PER_USD, GPUS, HOURS_PER_MONTH, SUMMER_DERATE } from '../data/gpus';
import { CACHE_PRICE_FRACTION, MARKET } from '../data/market';
import { BYTES_PER_PARAM, GOLDEN_AL_QWEN38_FLASH_NEXT, MODELS } from '../data/models';
import { liveRegister, OPENROUTER, OR_DAY, VAST, VAST_DAY } from '../data/live';
import { KIND_LABEL, KIND_ORDER, collectProvenance, countByKind, type ProvKind } from '../data/provenance';
import * as SC from '../data/scenarios';
import { GPU_BY_ID, QUANT_LABEL } from '../data/gpus';
import { bandwidthCeilingTps, decodeTps, ROOF_PROV } from '../engine/roofline';
import { IX_CURVES, IX_SNAPSHOT_AT } from '../data/ix';
import { ARCH_SNAPSHOT_AT } from '../data/arch';
import { fitGroups, fitLoss, fitRows } from '../engine/ix-fit';
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
  { t: 'InferenceX covers a few cards and models.', d: 'Measured curves exist for gpt-oss-120b and Llama 3.3 70B on H100, H200 and B200 at 1 to 8 GPUs, and for Qwen3.5-397B on B200 and four RTX PRO 6000s. Nothing is measured for the 6000 Ada, 5090, 4090, L40S or V100: they inherit the fitted constants of the nearest family (Hopper for the older cards, workstation Blackwell for the 5090).' },
  { t: 'InferenceX runs are tuned stacks.', d: 'SemiAnalysis sweeps recipes (vLLM or SGLang here; TensorRT-LLM and Dynamo runs are left out). A small provider running defaults may get less than the measured curve.' },
  { t: 'The RTX PRO 6000 has four measured points.', d: 'All from Qwen3.5-397B on four cards over PCIe. They fix the PCIe all-reduce latency and bandwidth and the mixture-of-experts prefill efficiency; dense prefill on the card is still a placeholder, and the card\'s memory efficiency is carried over from B200.' },
  { t: 'Time to first token is an approximation.', d: 'It treats arrivals as a steady stream (an M/D/1 queue) in front of prefill. The fit to InferenceX medians is within about a third on dense models and looser on very large mixture-of-experts models; bursts of agent requests arriving together would wait longer.' },
  { t: 'The report\'s throughput points are estimates except one.', d: 'Only Qwen3.8-27B at 16 concurrent is a measurement. gpt-oss at 50 is stated at Medium confidence and the rest are flagged estimates. The adjusted basis ignores the estimates and uses the fitted roofline; it uses the measured point as an anchor for that card. Several estimates sit above what the memory bus could deliver (Llama 3.3 70B at 16 concurrent). The Throughput view marks them.' },
  { t: 'Expert routing is assumed uniform.', d: 'A mixture-of-experts step reads the experts its batch touches; with uniform routing that is the most it can read. Real routers are skewed, so decode on these models may be faster than modelled. The fitted memory efficiency absorbs part of this.' },
  { t: 'KV figures assume an FP8 cache and ignore prefix sharing.', d: 'Each user is charged a full context. Sliding-window layers are capped at their window and recurrent state is counted in fp32, as the configs ask.' },
  { t: 'The V100 runs FP16 or INT4 only.', d: 'It has no FP8 or FP4, and older attention kernels. Its throughput carries a flat kernel penalty on top of the roofline.' },
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
  const fit = useMemo(() => {
    const rows = fitRows();
    return { n: rows.length, loss: fitLoss(rows), groups: fitGroups(rows) };
  }, []);
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
        <h2>Collected data</h2>
        <div class="sub">Four scripts collect it and save snapshots under data/, so the dashboard works offline. Prices keep dated history.</div>
        <div class="kv" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
          <div>
            <span>OpenRouter snapshot</span>
            <b>
              {OR_DAY} · {Object.keys(OPENROUTER.models).length} models
            </b>
          </div>
          <div>
            <span>InferenceX snapshot</span>
            <b>
              {IX_SNAPSHOT_AT.slice(0, 10)} · {IX_CURVES.length} measured curves
            </b>
          </div>
          <div>
            <span>Hugging Face configs</span>
            <b>
              {ARCH_SNAPSHOT_AT.slice(0, 10)} · {MODELS.length} models
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
          Refresh with <code>npm run data</code> (Vast.ai, OpenRouter and InferenceX) or one at a time with <code>npm run data:vast</code>, <code>data:openrouter</code>, <code>data:inferencex</code> and <code>data:hf</code>. The Vast.ai script needs a Chromium; set <code>CHROME_PATH</code> if Playwright has none, or <code>HEADED=1</code> to watch it. Commit the changed files in <code>data/live</code>.
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

      <section class="card" aria-label="Roofline against InferenceX">
        <h2>Roofline against InferenceX's measured runs</h2>
        <div class="sub">
          The roofline's efficiencies were fitted to these {fit.n} measured points (vLLM and SGLang runs, {IX_CURVES.length} curves, read {IX_SNAPSHOT_AT.slice(0, 10)}). Each row is the median of predicted ÷ measured over the concurrencies in one curve: 1.00 is a perfect fit, below 1 the roofline is too slow. Mean squared log error over output speed, speed per user and time to first token: {fit.loss.toFixed(2)}.
        </div>
        <details class="more">
          <summary>Show all {fit.groups.length} groups</summary>
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Card</th>
                  <th>Weights</th>
                  <th class="num">GPUs</th>
                  <th class="num">Points</th>
                  <th class="num">Output speed</th>
                  <th class="num">Speed per user</th>
                  <th class="num">First token</th>
                </tr>
              </thead>
              <tbody>
                {fit.groups.map((g) => (
                  <tr key={g.key}>
                    <td>{g.model}</td>
                    <td>{GPU_BY_ID[g.gpuId]?.short ?? g.gpuId}</td>
                    <td>{g.precision.toUpperCase()}</td>
                    <td class="num">{g.gpus}</td>
                    <td class="num">{g.n}</td>
                    <td class="num">{g.out.toFixed(2)}</td>
                    <td class="num">{Number.isFinite(g.user) ? g.user.toFixed(2) : '—'}</td>
                    <td class="num">{Number.isFinite(g.ttft) ? g.ttft.toFixed(2) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      <section class="card" aria-label="Roofline against the report's throughput points">
        <h2>Roofline against the report's throughput points</h2>
        <div class="sub">
          {roof.length} points on the PRO 6000 and the Ada, at {REPORT_TEST_CTX.v} tokens of context. The roofline was fitted to InferenceX, not to these, so this is an independent check of the report: only Qwen3.8-27B at 16 concurrent is a measurement, gpt-oss at 50 is stated at Medium confidence and the rest are the report's own estimates. {above.length} of them {above.length === 1 ? 'is' : 'are'} above the bus ceiling, the speed the memory bus could deliver at full bandwidth with nothing to read but the weights.
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
            <b>Phase 3:</b> model architecture from Hugging Face configs, a roofline for decode, prefill and time to first token, three throughput bases, a first-token limit in the quality bar.
          </li>
          <li>
            <b>Phase 4 (this version):</b> InferenceX measured runs fit the roofline per card family and anchor it where a card and model were measured; multi-GPU nodes use the fitted all-reduce model.
          </li>
          <li>
            <b>Next:</b> a measurement run on your own cards to fit dense prefill on the PRO 6000 and Ada, then AMD and A100.
          </li>
        </ol>
      </section>
    </div>
  );
}
