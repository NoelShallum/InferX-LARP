import { useState } from 'preact/hooks';
import { NEUTRAL_WHATIF } from '../engine/types';
import { tornado, type Metric } from '../engine/sensitivity';
import { fmtPct, fmtX } from '../engine/units';
import { settings, slots, steady, updateSettings } from '../state';
import { ChartCard, Tornado } from '../ui/charts';
import { Field, Segmented, Slider } from '../ui/controls';
import { disp, fmtD, M, money, taxF } from '../ui/money';
import { modelName } from '../ui/options';
import { ProvBadge } from '../ui/prov';

export function SensitivityView() {
  const s = settings.value;
  const [metric, setMetric] = useState<Metric>('net12');
  const f = taxF();
  const t = tornado(slots.value, s, metric);
  const st = steady.value;
  const wi = s.whatIf;
  const setWi = (patch: Partial<typeof wi>) => updateSettings({ whatIf: { ...wi, ...patch } });
  const active = Object.entries(wi).some(([k, v]) => v !== (NEUTRAL_WHATIF as unknown as Record<string, number>)[k]);
  const items = t.rows.map((r) => ({ id: r.id, label: r.label, lowLabel: r.lowLabel, highLabel: r.highLabel, low: disp(r.lowDeltaUsd * f), high: disp(r.highDeltaUsd * f) }));
  const serving = st.results.filter((r) => r.kind === 'serve' && r.ok);

  return (
    <div class="stack">
      <div class="notice">
        Each bar moves one input at a time from the current plan and shows what happens to the result. Blue means the change helps, red means it hurts. Bars are sorted by how much that input matters.
      </div>
      <div class="filters" role="region" aria-label="Sensitivity options">
        <Field label="Result measured">
          <Segmented<Metric> label="Result measured" value={metric} options={[{ id: 'net12', label: '12-month net' }, { id: 'steady', label: 'Steady month' }]} onChange={setMetric} />
        </Field>
        <div class="field">
          <span class="label">Base</span>
          <b>
            <M usd={t.base * f} signed /> {metric === 'net12' ? 'over 12 months' : 'a month'}
          </b>
        </div>
      </div>
      <div class="grid cols-2">
        <ChartCard
          title="What moves the result"
          subtitle={`Change in ${metric === 'net12' ? '12-month net' : 'steady-month gain'} for each swing, ${s.afterTax ? 'after tax' : 'pre-tax'}.`}
          chart={
            <>
              <Tornado items={items} ariaLabel="Tornado chart of how each input changes the result" fmt={(v) => fmtD(v)} />
              <p class="small">
                Availability −5 points drops a 99% endpoint below OpenRouter's 95% line, where routing priority falls. The report gives no figure for that penalty, so the dashboard halves share <ProvBadge p={{ kind: 'placeholder', src: 'No source yet', note: 'Share multiplier below 95% uptime is a guess', value: '0.5', unit: '× share' }} compact />.
              </p>
            </>
          }
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Input</th>
                  <th>Bad swing</th>
                  <th class="num">Change</th>
                  <th>Good swing</th>
                  <th class="num">Change</th>
                </tr>
              </thead>
              <tbody>
                {t.rows.map((r) => (
                  <tr key={r.id}>
                    <td>{r.label}</td>
                    <td>{r.lowLabel}</td>
                    <td class="num">{money(r.lowDeltaUsd * f)}</td>
                    <td>{r.highLabel}</td>
                    <td class="num">{money(r.highDeltaUsd * f)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
        <section class="card" aria-label="What-if sliders">
          <div class="spread">
            <h2>What-if</h2>
            <button type="button" class="btn" disabled={!active} onClick={() => updateSettings({ whatIf: { ...NEUTRAL_WHATIF } })}>
              Reset what-if
            </button>
          </div>
          <div class="sub">These multipliers apply to every view while they are moved, so the whole dashboard answers "what if throughput were 20% lower".</div>
          <div class="stack">
            <Field label="Throughput per card">
              <Slider label="Throughput multiplier" min={0.5} max={1.5} step={0.05} value={wi.throughput} format={fmtX} onChange={(v) => setWi({ throughput: v })} />
            </Field>
            <Field label="Share of traffic won">
              <Slider label="Share multiplier" min={0.25} max={2} step={0.05} value={wi.demand} format={fmtX} onChange={(v) => setWi({ demand: v })} />
            </Field>
            <Field label="Market token prices (everyone's)">
              <Slider label="Market price multiplier" min={0.5} max={1.5} step={0.05} value={wi.price} format={fmtX} onChange={(v) => setWi({ price: v })} />
            </Field>
            <Field label="Rent per GPU-hour">
              <Slider label="Rent multiplier" min={0.5} max={1.5} step={0.05} value={wi.rent} format={fmtX} onChange={(v) => setWi({ rent: v })} />
            </Field>
            <Field label="Lium bar">
              <Slider label="Lium bar multiplier" min={0.7} max={1.3} step={0.05} value={wi.bar} format={fmtX} onChange={(v) => setWi({ bar: v })} />
            </Field>
            <Field label="Availability change">
              <Slider label="Availability change in points" min={-0.3} max={0.05} step={0.01} value={wi.availabilityDelta} format={(v) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(0)} pts`} onChange={(v) => setWi({ availabilityDelta: v })} />
            </Field>
          </div>
        </section>
      </div>
      <section class="card" aria-label="Break-even by card">
        <h2>Break-even by card</h2>
        <div class="sub">How full a card must be, and what price it needs, to cover its Lium bar or its rent. The marker is break-even; the bar is where the plan puts the card.</div>
        {serving.length === 0 ? (
          <p class="small">No card is serving tokens in this plan.</p>
        ) : (
          <div class="grid cols-2">
            {serving.map((r) => (
              <div key={r.slot.id} class="stack">
                <div class="spread">
                  <b>{r.slot.label || r.gpu.short}</b>
                  <span class="small">
                    {r.gpu.short} · {modelName(r.slot.modelId)}
                  </span>
                </div>
                <div class="meter" role="img" aria-label={`Card used ${fmtPct(r.utilisation)}; break-even ${r.breakEvenUtil === null ? 'not reachable' : fmtPct(r.breakEvenUtil)}`}>
                  <i style={{ width: `${Math.min(100, r.utilisation * 100)}%`, background: r.breakEvenUtil !== null && r.utilisation >= r.breakEvenUtil ? 'var(--accent)' : 'var(--s2)' }} />
                  {r.breakEvenUtil !== null ? <b style={{ left: `${Math.min(100, r.breakEvenUtil * 100)}%` }} /> : null}
                </div>
                <div class="small">
                  Card used {fmtPct(r.utilisation)} · break-even {r.breakEvenUtil === null ? 'not reachable even when full' : fmtPct(r.breakEvenUtil)} · price needed {r.breakEvenPriceScale === null ? 'not reachable' : `${fmtX(r.breakEvenPriceScale)} the current price`}
                </div>
              </div>
            ))}
          </div>
        )}
        <p class="small mt">
          Break-even price accounts for the report's price elasticity (−0.63) <ProvBadge p={{ kind: 'report', src: 'Report §6.2' }} compact />: a lower price wins some traffic back, a higher one loses it, and above 1.2× the median the report has no data.
        </p>
      </section>
    </div>
  );
}
