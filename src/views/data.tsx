import { useMemo, useState } from 'preact/hooks';
import { FX_INR_PER_USD, GPUS, HOURS_PER_MONTH, SUMMER_DERATE } from '../data/gpus';
import { CACHE_PRICE_FRACTION, MARKET } from '../data/market';
import { BYTES_PER_PARAM, GOLDEN_AL_QWEN38_FLASH_NEXT, MODELS } from '../data/models';
import { liveRegister, OPENROUTER, OR_DAY, VAST, VAST_DAY } from '../data/live';
import { KIND_LABEL, KIND_ORDER, collectProvenance, countByKind, type ProvKind } from '../data/provenance';
import * as SC from '../data/scenarios';
import { GPU_MEM_UTIL, OVERHEAD_GB_PER_GPU, MIN_KV_TOKENS, MULTI_GPU_EFF, DRAFT_COST, MTP_DEFAULT_CSAT, VOLTA_KERNEL_PENALTY, MAX_EXTRAPOLATION } from '../engine/capacity';
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
    multiGpuEff: MULTI_GPU_EFF,
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
  { t: 'Prefill is not modelled.', d: 'The output rate is anchored to the report. The prefill speed a card needs at full load is shown as information on each card, but it does not cap revenue or time to first token yet. The roofline engine (phase 3) adds both.' },
  { t: 'Multi-GPU scaling is a placeholder.', d: 'Nodes of 2, 4 and 8 GPUs scale throughput by a flat efficiency per doubling. InferenceX has measured 8-GPU curves for H100, H200 and B200 on frontier models; wiring them in is phase 3.' },
  { t: 'Throughput for most cards is scaled, not measured.', d: 'Only the PRO 6000 and the Ada have report throughput points. The report\'s per-busy-hour revenue gives ratios for the 5090, H100, H200 and B200 on a few models. Everything else uses memory-bandwidth scaling, flagged as a placeholder.' },
  { t: 'KV bytes per token are placeholders for most models.', d: 'Only Qwen3.8-27B has a figure derived from the report. The rest come from architecture as remembered or generic guesses until the Hugging Face configs are pulled in.' },
  { t: 'The V100 runs FP16 or INT4 only.', d: 'It has no FP8 or FP4, and older attention kernels. Its throughput is a rough scaling with a flat kernel penalty.' },
  { t: 'Request shape and cache price are calibrated, not observed.', d: 'They are back-solved from the report\'s busy-hour and typical-entrant figures. They reproduce the report, and match its stated 4.1% and 11.8% output shares for the two models where it states them.' },
  { t: 'Live data is a snapshot.', d: 'Prices and traffic are as of the last data run (shown on the Prices view). Refresh with npm run data; trends need at least two snapshots on different days.' },
  { t: 'Vast.ai coverage is thin for some cards.', d: 'The 6000 Ada has no offer at reliability 0.90 or above, and several multi-GPU sizes have none. Where a filter leaves no offer the dashboard falls back to the report or catalog price and says so on the card.' },
  { t: 'Live request shape and report-calibrated capacity can disagree.', d: 'Capacity is anchored to the report; the live request shape changes what each output token earns but not how fast the card runs. Prefill, which ties the two together, arrives with the roofline engine.' },
  { t: 'Tax after the first step is approximate for rented cards on the crypto route.', d: 'Gains over a Lium bar are exact on either route. Rented-card margin on the crypto route is taxed on gross receipts.' },
];

export function DataView() {
  const rows = useMemo(() => collectProvenance(ALL), []);
  const counts = useMemo(() => countByKind(rows), [rows]);
  const checks = useMemo(() => runCrossChecks(), []);
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
            <b>Phase 2 (this version):</b> live Vast.ai (browser scrape, then a script) and OpenRouter prices with dated history.
          </li>
          <li>
            <b>Phase 3:</b> roofline throughput, prefill and time to first token, a PRO 6000 × Qwen3.8-27B slice end to end, then every model and card.
          </li>
        </ol>
      </section>
    </div>
  );
}
