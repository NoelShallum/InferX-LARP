import { GPUS } from '../data/gpus';
import { MARKET } from '../data/market';
import { MODEL_BY_ID } from '../data/models';
import { LIUM, ROUTE_COSTS } from '../data/scenarios';
import { tieInr } from '../engine/bars';
import { DEFAULT_PRICING } from '../engine/defaults';
import { calibratedShape, priceRatio } from '../engine/pricing';
import { fmtInr, fmtNum, fmtPct, fmtTokens } from '../engine/units';
import { settings } from '../state';
import { Val, ProvBadge, ProvSummary } from '../ui/prov';

export function PricesView() {
  const s = settings.value;
  return (
    <div class="stack">
      <div class="notice">
        <b>Today these are the report's snapshot of 22 Oct 2026.</b> Live prices arrive in phase 2: a headless-browser scrape of Vast.ai (done once live, then scripted) and an OpenRouter script, both saved as dated snapshots so trends show here. Until then every price carries the Report badge.
      </div>

      <section class="card" aria-label="OpenRouter prices">
        <h2>OpenRouter: what each model sells for</h2>
        <div class="sub">Dollars per million tokens. Paid tokens are the seven-day average to 21 Oct 2026. The floor ratio is the cheapest endpoint's blended price over the median's at the model's request shape.</div>
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
                <th class="num">Floor ÷ median</th>
                <th class="num">Per busy hour, median / lowest</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {MARKET.map((m) => {
                const model = MODEL_BY_ID[m.modelId]!;
                const shape = calibratedShape(model, m);
                const floor = priceRatio(m, { ...DEFAULT_PRICING, mode: 'floor' }, shape.outShare, shape.cacheHit, shape.cacheFraction);
                return (
                  <tr key={m.modelId}>
                    <td>{model.name}</td>
                    <td class="num">{m.endpoints.v}</td>
                    <td class="num">
                      {fmtTokens(m.paidTokensPerDayB.v * 1e9)}
                      {m.paidTokens14dMedianB ? <span class="small"> (14-day median {m.paidTokens14dMedianB.v}B)</span> : null}
                    </td>
                    <td class="num">
                      {m.inMin.v} / {m.inMed.v}
                    </td>
                    <td class="num">
                      {m.outMin.v} / {m.outMed.v} / {m.outMax.v}
                    </td>
                    <td class="num">{fmtPct(m.cacheHit.v)}</td>
                    <td class="num">{m.busyHourMedianUsd ? floor.toFixed(2) : '—'}</td>
                    <td class="num">{m.busyHourMedianUsd ? `$${m.busyHourMedianUsd.v.toFixed(2)} / ${m.busyHourLowestUsd ? '$' + m.busyHourLowestUsd.v.toFixed(2) : '—'}` : 'not modelled'}</td>
                    <td>
                      <ProvSummary provs={[m.paidTokensPerDayB.p, m.inMed.p, m.outMed.p, m.cacheHit.p, ...(m.busyHourMedianUsd ? [m.busyHourMedianUsd.p] : [])]} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section class="card" aria-label="GPU rent and purchase prices">
        <h2>GPUs: specifications, rent and purchase price</h2>
        <div class="sub">Rent is the median on-demand price across the marketplaces the report checked. InferenceX figures are its own TCO assumptions for data-centre cards, shown for reference.</div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th class="num">Memory</th>
                <th class="num">Bandwidth</th>
                <th class="num">Power</th>
                <th class="num">Rent $/h</th>
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
                  <td class="num">{g.rentUsdHr ? <Val v={g.rentUsdHr} fmt={(x) => `$${x.toFixed(2)}`} /> : '—'}</td>
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
