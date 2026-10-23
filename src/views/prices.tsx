import { useState } from 'preact/hooks';
import { GPU_BY_ID, GPUS } from '../data/gpus';
import {
  OPENROUTER,
  OPENROUTER_HISTORY,
  OR_DAY,
  RENT_VIEW_LABEL,
  VAST,
  VAST_DAY,
  VAST_HISTORY,
  liveRent,
  orTrend,
  vastTrend,
  type OrField,
  type TrendPoint,
} from '../data/live';
import { MARKET } from '../data/market';
import { MODEL_BY_ID } from '../data/models';
import { LIUM, ROUTE_COSTS } from '../data/scenarios';
import { tieInr } from '../engine/bars';
import { DEFAULT_PRICING } from '../engine/defaults';
import { calibratedShape, priceRatio } from '../engine/pricing';
import { marketFor } from '../engine/sources';
import { fmtInr, fmtNum, fmtPct, fmtTokens } from '../engine/units';
import { settings } from '../state';
import { ChartCard, LineChart } from '../ui/charts';
import { Field, SelectInput } from '../ui/controls';
import { SERVE_GPUS } from '../ui/options';
import { ProvBadge, ProvSummary, Val } from '../ui/prov';

function Two({ live, report, fmt }: { live: number | null | undefined; report: number | null | undefined; fmt: (n: number) => string }) {
  return (
    <>
      <div>{live != null ? fmt(live) : '—'}</div>
      {report != null ? <div class="small">report {fmt(report)}</div> : null}
    </>
  );
}

const dayNumber = (d: string) => Math.round(Date.parse(`${d}T00:00:00Z`) / 86_400_000);
const dayLabel = (n: number) => new Date(n * 86_400_000).toISOString().slice(5, 10);

function Trend({ points, label, fmt, snapshots }: { points: TrendPoint[]; label: string; fmt: (v: number) => string; snapshots: number }) {
  if (points.length < 2) {
    return (
      <p class="small">
        {snapshots} snapshot{snapshots === 1 ? '' : 's'} saved so far{points.length === 1 ? ` (${points[0]!.date})` : ''}. Run <code>npm run data</code> on another day and commit the result; this chart draws itself from the second snapshot.
      </p>
    );
  }
  const pts = points.map((p) => ({ x: dayNumber(p.date), y: p.value }));
  return (
    <LineChart
      ariaLabel={`${label} over time`}
      series={[{ id: 't', label, color: 'var(--s1)', pts, width: 3, markers: true }]}
      xFmt={dayLabel}
      yFmt={fmt}
      xTicks={pts.map((p) => p.x)}
      tipTitle={dayLabel}
      height={240}
    />
  );
}

export function PricesView() {
  const s = settings.value;
  const liveSettings = { ...s, marketSource: 'live' as const };
  const [epModel, setEpModel] = useState('qwen38_27b');
  const [trendGpu, setTrendGpu] = useState('pro6000');
  const [trendModel, setTrendModel] = useState('qwen38_27b');
  const [trendField, setTrendField] = useState<OrField>('outMedian');

  const modelIds = Array.from(new Set([...MARKET.map((m) => m.modelId), ...Object.keys(OPENROUTER.models)])).filter((id) => MODEL_BY_ID[id]);
  const orModelsLive = Object.keys(OPENROUTER.models).length;
  const vastCount = Object.keys(VAST.gpus).length;
  const ep = OPENROUTER.models[epModel];

  const fieldFmt: Record<OrField, (v: number) => string> = {
    outMedian: (v) => `$${v.toFixed(2)}`,
    inMedian: (v) => `$${v.toFixed(3)}`,
    outMin: (v) => `$${v.toFixed(2)}`,
    tokensPerDay: (v) => `${v.toFixed(0)}B`,
    endpoints: (v) => `${v.toFixed(0)}`,
    p50Tps: (v) => `${v.toFixed(0)} tok/s`,
  };
  const fieldLabel: Record<OrField, string> = {
    outMedian: 'Median output price per million tokens',
    inMedian: 'Median input price per million tokens',
    outMin: 'Cheapest output price per million tokens',
    tokensPerDay: 'Paid tokens a day, billions',
    endpoints: 'Endpoints listed',
    p50Tps: 'Median typical stream speed, tokens per second',
  };

  return (
    <div class="stack">
      <div class="notice">
        <b>Live data.</b> OpenRouter as of {OR_DAY} ({orModelsLive} models) and Vast.ai as of {VAST_DAY} ({vastCount} GPUs). OpenRouter comes from its public JSON; Vast.ai is read through a real browser session because its search refuses plain requests. Refresh both with <code>npm run data</code>, then commit the new files in <code>data/live</code>. The numbers the report printed on 22 Oct 2026 stay beside the live ones, and you can switch every view back to them under Assumptions.
      </div>

      <section class="card" aria-label="OpenRouter prices">
        <h2>OpenRouter: what each model sells for</h2>
        <div class="sub">
          Dollars per million tokens, live over the last full days, with the report's snapshot under each figure. Traffic is paid (standard) traffic. Stream speed is the median of each endpoint's typical (p50) speed. The floor ratio is the cheapest endpoint's blended price over the median's at the model's request shape.
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>Model</th>
                <th class="num">Endpoints</th>
                <th class="num">Paid tokens a day</th>
                <th class="num">Input min / median</th>
                <th class="num">Output min / median / max</th>
                <th class="num">Cache hit</th>
                <th class="num">Stream tok/s</th>
                <th class="num">Output share</th>
                <th class="num">Floor ÷ median</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {modelIds.map((id) => {
                const model = MODEL_BY_ID[id]!;
                const rep = MARKET.find((m) => m.modelId === id);
                const lm = marketFor(id, liveSettings);
                const live = lm?.live;
                const o = OPENROUTER.models[id];
                const row = lm?.row ?? rep;
                if (!row) return null;
                const shape = calibratedShape(model, rep ?? row);
                const floor = priceRatio(row, { ...DEFAULT_PRICING, mode: 'floor' }, live?.shape?.outShare ?? shape.outShare, live?.shape?.cacheHit ?? shape.cacheHit, live?.cacheFraction ?? shape.cacheFraction);
                return (
                  <tr key={id}>
                    <td>{model.name}</td>
                    <td class="num">
                      <Two live={live ? row.endpoints.v : null} report={rep?.endpoints.v} fmt={(n) => String(n)} />
                    </td>
                    <td class="num">
                      <Two live={live ? row.paidTokensPerDayB.v * 1e9 : null} report={rep ? rep.paidTokensPerDayB.v * 1e9 : null} fmt={fmtTokens} />
                    </td>
                    <td class="num">
                      <div>{live ? `${row.inMin.v} / ${row.inMed.v}` : '—'}</div>
                      {rep ? <div class="small">report {rep.inMin.v} / {rep.inMed.v}</div> : null}
                    </td>
                    <td class="num">
                      <div>{live ? `${row.outMin.v} / ${row.outMed.v} / ${row.outMax.v}` : '—'}</div>
                      {rep ? <div class="small">report {rep.outMin.v} / {rep.outMed.v} / {rep.outMax.v}</div> : null}
                    </td>
                    <td class="num">
                      <Two live={live ? row.cacheHit.v : null} report={rep?.cacheHit.v} fmt={(n) => fmtPct(n)} />
                    </td>
                    <td class="num">{o && o.summary.p50TpsMedian !== null ? `${fmtNum(o.summary.p50TpsMedian, 0)} (${o.summary.speedEndpoints})` : '—'}</td>
                    <td class="num">{live?.shape ? fmtPct(live.shape.outShare, 1) : '—'}</td>
                    <td class="num">{floor.toFixed(2)}</td>
                    <td>
                      <ProvSummary provs={[row.endpoints.p, row.outMed.p, row.paidTokensPerDayB.p]} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p class="small mt">The number in brackets under stream speed is how many endpoints reported one. Batch variants exist for {Object.values(OPENROUTER.models).filter((m) => m.batch).length} of these models: {Object.entries(OPENROUTER.models).filter(([, m]) => m.batch).map(([id, m]) => `${MODEL_BY_ID[id]?.name ?? id} at ${Math.round(m.batch!.outRatio * 100)}% of the live output price`).join(', ') || 'none'}.</p>
      </section>

      <section class="card" aria-label="Who sells a model">
        <div class="spread">
          <div>
            <h2>Who sells it</h2>
            <div class="sub">Every endpoint listed for one model, cheapest output price first.</div>
          </div>
          <Field label="Model">
            <SelectInput label="Model" value={epModel} options={modelIds.map((id) => ({ id, label: MODEL_BY_ID[id]!.name }))} onChange={setEpModel} />
          </Field>
        </div>
        {ep ? (
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Weights</th>
                  <th class="num">Input</th>
                  <th class="num">Output</th>
                  <th class="num">Cache read</th>
                  <th class="num">Uptime, 1 day</th>
                  <th class="num">Stream tok/s</th>
                  <th class="num">Latency ms</th>
                  <th>Zero retention</th>
                </tr>
              </thead>
              <tbody>
                {ep.endpoints.map((e, i) => (
                  <tr key={i} style={e.status !== 0 ? { color: 'var(--muted)' } : undefined}>
                    <td>
                      {e.provider}
                      {e.status !== 0 ? <span class="small"> (degraded)</span> : null}
                    </td>
                    <td>{e.quantization}</td>
                    <td class="num">{e.inPerM ?? '—'}</td>
                    <td class="num">{e.outPerM ?? '—'}</td>
                    <td class="num">{e.cachePerM ?? '—'}</td>
                    <td class="num">{e.uptime1d !== null ? `${e.uptime1d}%` : '—'}</td>
                    <td class="num">{e.p50Tps ?? '—'}</td>
                    <td class="num">{e.p50LatencyMs !== null ? fmtNum(e.p50LatencyMs, 0) : '—'}</td>
                    <td>{e.zdr === null ? '—' : e.zdr ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p class="small">No live data for this model.</p>
        )}
      </section>

      <section class="card" aria-label="Vast.ai rent">
        <h2>Vast.ai: what a GPU costs to rent</h2>
        <div class="sub">
          On-demand, per GPU-hour, counting offers that pass the filter chosen under Assumptions: <b>{RENT_VIEW_LABEL[s.rentView]}</b>. Larger nodes show the cheapest per-GPU price at that size. A dash means no offer passes. The report's median is shown for comparison.
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th class="num">Report / catalog</th>
                <th class="num">Cheapest, 1 card</th>
                <th class="num">Median, 1 card</th>
                <th class="num">Offers</th>
                <th>Cheapest host</th>
                <th class="num">2 cards</th>
                <th class="num">4 cards</th>
                <th class="num">8 cards</th>
                <th class="num">Interruptible</th>
              </tr>
            </thead>
            <tbody>
              {SERVE_GPUS.map((g) => {
                const cheap = (n: number) => liveRent(g.id, n, s.rentView, 'cheapest');
                const one = cheap(1);
                const med = liveRent(g.id, 1, s.rentView, 'median');
                const v = VAST.gpus[g.id]?.nodes['1']?.views[s.rentView];
                const top = v?.cheapest[0];
                const bid = VAST.gpus[g.id]?.nodes['1']?.interruptible;
                const node = (n: number) => {
                  const l = cheap(n);
                  return l && !l.borrowedFromSingle ? `$${l.perGpu.toFixed(2)}` : '—';
                };
                return (
                  <tr key={g.id}>
                    <td>{g.short}</td>
                    <td class="num">{g.rentUsdHr ? <Val v={g.rentUsdHr} fmt={(x) => `$${x.toFixed(2)}`} /> : '—'}</td>
                    <td class="num">{one && !one.borrowedFromSingle ? `$${one.perGpu.toFixed(2)}` : '—'}</td>
                    <td class="num">{med && !med.borrowedFromSingle ? `$${med.perGpu.toFixed(2)}` : '—'}</td>
                    <td class="num">{v ? v.count : 0}</td>
                    <td class="small">{top ? `${top.geo ?? 'unknown'}, reliability ${top.reliability}${top.datacenter ? ', data-centre' : ''}` : '—'}</td>
                    <td class="num">{node(2)}</td>
                    <td class="num">{node(4)}</td>
                    <td class="num">{node(8)}</td>
                    <td class="num">{bid && bid.minPerGpu !== null ? `$${bid.minPerGpu.toFixed(2)}` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p class="small mt">
          Interruptible is the lowest minimum bid among verified hosts. A host can preempt it, which an endpoint scored on uptime cannot afford, so it is shown for reference only. Offers needed at least 100 GB of disk. Vast.ai names used per card are kept in the snapshot. <ProvBadge p={{ kind: 'live', src: 'Vast.ai (browser scrape)', asOf: VAST_DAY }} compact />
        </p>
      </section>

      <div class="grid cols-2">
        <ChartCard
          title="Rent over time"
          subtitle={`Per GPU-hour, one card, ${RENT_VIEW_LABEL[s.rentView].toLowerCase()}, ${s.rentStat}.`}
          actions={
            <Field label="GPU">
              <SelectInput label="GPU for the rent trend" value={trendGpu} options={SERVE_GPUS.map((g) => ({ id: g.id, label: g.short }))} onChange={setTrendGpu} />
            </Field>
          }
          chart={<Trend points={vastTrend(VAST_HISTORY, trendGpu, 1, s.rentView, s.rentStat)} label={`${GPU_BY_ID[trendGpu]?.short ?? trendGpu} rent`} fmt={(v) => `$${v.toFixed(2)}`} snapshots={VAST_HISTORY.length} />}
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Date</th>
                  <th class="num">Per GPU-hour</th>
                </tr>
              </thead>
              <tbody>
                {vastTrend(VAST_HISTORY, trendGpu, 1, s.rentView, s.rentStat).map((p) => (
                  <tr key={p.date}>
                    <td>{p.date}</td>
                    <td class="num">${p.value.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
        <ChartCard
          title="Market over time"
          subtitle={fieldLabel[trendField]}
          actions={
            <div class="row">
              <Field label="Model">
                <SelectInput label="Model for the market trend" value={trendModel} options={modelIds.filter((id) => OPENROUTER.models[id]).map((id) => ({ id, label: MODEL_BY_ID[id]!.name }))} onChange={setTrendModel} />
              </Field>
              <Field label="Measure">
                <SelectInput<OrField> label="Measure" value={trendField} options={(Object.keys(fieldLabel) as OrField[]).map((k) => ({ id: k, label: fieldLabel[k] }))} onChange={setTrendField} />
              </Field>
            </div>
          }
          chart={<Trend points={orTrend(OPENROUTER_HISTORY, trendModel, trendField)} label={fieldLabel[trendField]} fmt={fieldFmt[trendField]} snapshots={OPENROUTER_HISTORY.length} />}
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Date</th>
                  <th class="num">{fieldLabel[trendField]}</th>
                </tr>
              </thead>
              <tbody>
                {orTrend(OPENROUTER_HISTORY, trendModel, trendField).map((p) => (
                  <tr key={p.date}>
                    <td>{p.date}</td>
                    <td class="num">{fieldFmt[trendField](p.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
      </div>

      <section class="card" aria-label="GPU specifications and purchase prices">
        <h2>GPUs: specifications and purchase price</h2>
        <div class="sub">InferenceX figures are its own TCO assumptions for data-centre cards, shown for reference. Rent above is live; the report's rent is kept in the first column of the Vast.ai table.</div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th class="num">Memory</th>
                <th class="num">Bandwidth</th>
                <th class="num">Power</th>
                <th class="num">Indian cloud ₹/h</th>
                <th class="num">InferenceX own / rent $/h</th>
                <th class="num">New, INR</th>
                <th>Owned</th>
              </tr>
            </thead>
            <tbody>
              {GPUS.map((g) => (
                <tr key={g.id}>
                  <td>{g.name}</td>
                  <td class="num">
                    <Val v={g.vramGB} fmt={(x) => `${x} GB`} />
                  </td>
                  <td class="num">
                    <Val v={g.bwGBs} fmt={(x) => `${fmtNum(x)} GB/s`} />
                  </td>
                  <td class="num">
                    <Val v={g.tdpW} fmt={(x) => `${x} W`} />
                  </td>
                  <td class="num">{g.rentInrHr ? <Val v={g.rentInrHr} fmt={(x) => `₹${x}`} /> : '—'}</td>
                  <td class="num">
                    {g.ixCost ? (
                      <>
                        ${g.ixCost.owned.v.toFixed(2)} / ${g.ixCost.rent.v.toFixed(2)} <ProvBadge p={g.ixCost.owned.p} compact />
                      </>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td class="num">{g.buyInr ? <Val v={g.buyInr} fmt={(x) => fmtInr(x)} /> : '—'}</td>
                  <td>{g.owned > 0 ? `${g.owned} owned` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section class="card" aria-label="Lium bar">
        <h2>The Lium bar: what each owned box earns today</h2>
        <div class="sub">An inference route has to beat this after its own costs and tax. Net is after Lium fees and power, for the 30 days to 22 Oct 2026.</div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>Box</th>
                <th class="num">Lium net a month</th>
                <th class="num">After crypto tax</th>
                <th class="num">Gross, USD</th>
                <th class="num">Fiat tie, low / base / high</th>
                <th class="num">Crypto tie, base</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(LIUM).map(([key, l]) => (
                <tr key={key}>
                  <td>{key === 'pro6000' ? 'RTX PRO 6000' : 'RTX 6000 Ada'}</td>
                  <td class="num">
                    <Val v={l.netInr} fmt={(x) => fmtInr(x, false)} />
                  </td>
                  <td class="num">
                    <Val v={l.afterTaxInr} fmt={(x) => fmtInr(x, false)} />
                  </td>
                  <td class="num">
                    <Val v={l.grossUsd} fmt={(x) => `$${x}`} />
                  </td>
                  <td class="num">
                    {(['low', 'base', 'high'] as const).map((lv) => fmtInr(tieInr(key, 'fiat', lv).totalInr, false)).join(' / ')} <ProvBadge p={{ kind: 'derived', src: 'Report §2.2', note: 'Computed from the inputs above and the route costs below' }} compact />
                  </td>
                  <td class="num">{fmtInr(tieInr(key, 'crypto', s.costLevel).totalInr, false)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p class="small mt">
          Route costs a month at low / base / high: monitoring {ROUTE_COSTS.monitoringInr.low.v} / {ROUTE_COSTS.monitoringInr.base.v} / {ROUTE_COSTS.monitoringInr.high.v} rupees, UPS {fmtInr(ROUTE_COSTS.upsInr.low.v)} / {fmtInr(ROUTE_COSTS.upsInr.base.v)} / {fmtInr(ROUTE_COSTS.upsInr.high.v)} written off over {ROUTE_COSTS.upsAmortMonths.v} months, failover cap {fmtInr(ROUTE_COSTS.failoverInr.low.v)} / {fmtInr(ROUTE_COSTS.failoverInr.base.v)} / {fmtInr(ROUTE_COSTS.failoverInr.high.v)}. <ProvBadge p={ROUTE_COSTS.monitoringInr.base.p} compact />
        </p>
      </section>
    </div>
  );
}

