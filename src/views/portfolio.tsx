import { useState } from 'preact/hooks';
import { GPU_BY_ID, QUANT_LABEL } from '../data/gpus';
import { MODEL_BY_ID } from '../data/models';
import { provValue, type Prov } from '../data/provenance';
import { CASES, CASE_LABEL, PROGRAM, type CaseId } from '../data/scenarios';
import { makeSlot, PRESETS } from '../engine/defaults';
import { programCosts } from '../engine/portfolio';
import type { SlotResult } from '../engine/slot';
import type { Demand, Pricing, Slot } from '../engine/types';
import { fmtNum, fmtPct, fmtTokens, fmtTps, fmtX } from '../engine/units';
import { addSlot, evRows, loadPreset, presetId, removeSlot, setSlot, settings, slots, steady, timeline, timelinesByCase } from '../state';
import { ChartCard, HBars, LineChart, type BarItem } from '../ui/charts';
import { Field, NumberInput, SelectInput, Slider, Toggle } from '../ui/controls';
import { disp, fmtD, M, money, taxF } from '../ui/money';
import { AVAIL_OPTIONS, COUNT_OPTIONS, GPU_OPTIONS_ALL, GPU_OPTIONS_RENT, MODEL_OPTIONS, MODEL_OPTIONS_OWNED, QUANT_OPTIONS, WORKLOAD_OPTIONS, gpuShort, modelName } from '../ui/options';
import { ProvBadge, ProvSummary } from '../ui/prov';

const CASE_COLOR: Record<CaseId, string> = { low: 'var(--s2)', base: 'var(--s1)', high: 'var(--s3)' };

function Tile({ label, value, sub, prov }: { label: string; value: preact.ComponentChildren; sub?: preact.ComponentChildren; prov?: preact.ComponentChildren }) {
  return (
    <div class="tile">
      <div class="t-label">
        {label} {prov}
      </div>
      <div class="t-value">{value}</div>
      {sub ? <div class="t-sub">{sub}</div> : null}
    </div>
  );
}

function PricingEditor({ slot }: { slot: Slot }) {
  const p = slot.pricing;
  const set = (patch: Partial<Pricing>) => setSlot(slot.id, { pricing: { ...p, ...patch } });
  return (
    <>
      <Field label="Price strategy">
        <SelectInput
          label="Price strategy"
          value={p.mode}
          options={[
            { id: 'median', label: 'Median (floor in launch months)' },
            { id: 'floor', label: 'Floor: cheapest endpoint' },
            { id: 'multiplier', label: 'Multiple of median' },
            { id: 'batch', label: 'Batch discount' },
            { id: 'custom', label: 'My own $/M tokens' },
          ]}
          onChange={(v) => set({ mode: v })}
        />
      </Field>
      {p.mode === 'multiplier' ? (
        <Field label="Multiple of median">
          <NumberInput label="Multiple of median price" value={p.mult} min={0.1} max={3} step={0.05} suffix="×" onChange={(v) => set({ mult: v })} />
        </Field>
      ) : null}
      {p.mode === 'batch' ? (
        <Field label="Discount off live price">
          <NumberInput label="Batch discount" value={Math.round(p.batchDiscount * 100)} min={0} max={95} step={5} suffix="%" onChange={(v) => set({ batchDiscount: v / 100 })} />
        </Field>
      ) : null}
      {p.mode === 'custom' ? (
        <Field label="My price, $ per M tokens in / out">
          <span class="row">
            <NumberInput label="Input price per million tokens" value={p.customIn} min={0} step={0.01} width={70} onChange={(v) => set({ customIn: v })} />
            <NumberInput label="Output price per million tokens" value={p.customOut} min={0} step={0.05} width={70} onChange={(v) => set({ customOut: v })} />
          </span>
        </Field>
      ) : null}
    </>
  );
}

function DemandEditor({ slot }: { slot: Slot }) {
  const d = slot.demand;
  const set = (patch: Partial<Demand>) => setSlot(slot.id, { demand: { ...d, ...patch } });
  const others = slots.value.filter((s) => s.id !== slot.id && s.demand.mode !== 'overflow' && s.modelId !== 'lium');
  return (
    <>
      <Field label="Demand">
        <SelectInput
          label="Demand model"
          value={d.mode}
          options={[
            { id: 'share', label: 'Marketplace share model' },
            { id: 'utilisation', label: 'Hours sold (utilisation)' },
            { id: 'overflow', label: 'Overflow from another card', disabled: others.length === 0 },
          ]}
          onChange={(v) => set({ mode: v, of: v === 'overflow' ? (others[0]?.id ?? '') : d.of })}
        />
      </Field>
      {d.mode === 'share' ? (
        <Field label="Share multiplier">
          <NumberInput label="Share multiplier" value={d.shareScale} min={0} max={5} step={0.1} suffix="×" onChange={(v) => set({ shareScale: v })} />
        </Field>
      ) : null}
      {d.mode === 'utilisation' ? (
        <Field label="Card sold">
          <Slider label="Share of capacity sold" value={d.utilisation} min={0} max={1} step={0.01} format={(v) => fmtPct(v)} onChange={(v) => set({ utilisation: v })} />
        </Field>
      ) : null}
      {d.mode === 'overflow' ? (
        <Field label="Picks up traffic from">
          <SelectInput label="Overflow source" value={d.of} options={others.map((o) => ({ id: o.id, label: o.label || o.id }))} onChange={(v) => set({ of: v })} />
        </Field>
      ) : null}
    </>
  );
}

function SlotEditor({ slot }: { slot: Slot }) {
  const model = MODEL_BY_ID[slot.modelId];
  const owned = slot.ownership === 'owned';
  const gpu = GPU_BY_ID[slot.gpuId]!;
  return (
    <details class="more" open={!slot.label}>
      <summary>Edit card</summary>
      <div class="slot-grid">
        <Field label="Name">
          <input class="input" aria-label="Card name" value={slot.label} onInput={(e) => setSlot(slot.id, { label: e.currentTarget.value })} />
        </Field>
        {!owned ? (
          <>
            <Field label="GPU">
              <SelectInput label="GPU" value={slot.gpuId} options={slot.ownership === 'rent' ? GPU_OPTIONS_RENT : GPU_OPTIONS_ALL} onChange={(v) => setSlot(slot.id, { gpuId: v })} />
            </Field>
            <Field label="GPUs in the node">
              <SelectInput label="GPU count" value={String(slot.count)} options={COUNT_OPTIONS} onChange={(v) => setSlot(slot.id, { count: Number(v) })} />
            </Field>
          </>
        ) : null}
        <Field label="Model">
          <SelectInput label="Model" value={slot.modelId} options={owned ? MODEL_OPTIONS_OWNED : MODEL_OPTIONS} onChange={(v) => setSlot(slot.id, { modelId: v })} />
        </Field>
        {slot.modelId !== 'lium' ? (
          <>
            <Field label="Weights">
              <SelectInput label="Weight format" value={slot.quant} options={QUANT_OPTIONS} onChange={(v) => setSlot(slot.id, { quant: v })} />
            </Field>
            <Field label="Workload">
              <SelectInput label="Workload preset" value={slot.workloadId} options={WORKLOAD_OPTIONS} onChange={(v) => setSlot(slot.id, { workloadId: v })} />
            </Field>
            <PricingEditor slot={slot} />
            <DemandEditor slot={slot} />
            <Field label="Availability">
              <SelectInput label="Availability" value={slot.availabilityId} options={[{ id: 'global', label: 'Use the global setting' }, ...AVAIL_OPTIONS]} onChange={(v) => setSlot(slot.id, { availabilityId: v })} />
            </Field>
            <Field label="Accepted months after the first listing">
              <NumberInput label="Listing offset in months" value={slot.listingOffset} min={0} max={11} step={1} onChange={(v) => setSlot(slot.id, { listingOffset: Math.max(0, Math.round(v)) })} />
            </Field>
            {slot.ownership === 'rent' ? (
              <Field label={<>Rent $ per GPU-hour {gpu.rentUsdHr ? <ProvBadge p={gpu.rentUsdHr.p} compact /> : null}</>}>
                <NumberInput label="Rent per GPU-hour in dollars" value={slot.rentUsdHrOverride ?? gpu.rentUsdHr?.v ?? 0} min={0} step={0.01} width={80} onChange={(v) => setSlot(slot.id, { rentUsdHrOverride: v })} />
              </Field>
            ) : null}
            {slot.ownership === 'buy' ? (
              <Field label={<>Purchase price per card, INR {gpu.buyInr ? <ProvBadge p={gpu.buyInr.p} compact /> : null}</>}>
                <NumberInput label="Purchase price per card in rupees" value={slot.buyPriceInr ?? gpu.buyInr?.v ?? 0} min={0} step={10000} width={110} onChange={(v) => setSlot(slot.id, { buyPriceInr: v })} />
              </Field>
            ) : null}
            <div class="field">
              <span class="label">Speculative decoding</span>
              <Toggle checked={slot.mtp} onChange={(v) => setSlot(slot.id, { mtp: v })} title={model?.mtp?.note}>
                {model?.mtp ? 'MTP on' : 'No MTP evidence for this model'}
              </Toggle>
            </div>
          </>
        ) : null}
      </div>
      {!owned ? (
        <div class="row mt">
          <button type="button" class="btn danger" onClick={() => removeSlot(slot.id)}>
            Remove card
          </button>
        </div>
      ) : null}
    </details>
  );
}

function SlotCard({ r }: { r: SlotResult }) {
  const s = settings.value;
  const slot = r.slot;
  const gpu = r.gpu;
  const cap = r.capacity;
  const owned = slot.ownership === 'owned';
  const marginUsd = s.afterTax ? r.afterTaxMarginUsd : r.marginUsd;
  const comp = owned ? r.barUsd : r.costUsd;
  const tag = slot.ownership === 'owned' ? 'Owned' : slot.ownership === 'rent' ? 'Rented' : 'To buy';
  const lium = r.kind === 'lium';
  const marginLabel = lium ? 'earns its Lium bar' : owned ? 'gain over the Lium bar' : slot.ownership === 'rent' ? 'margin over rent' : 'margin over ownership cost';
  const model = MODEL_BY_ID[slot.modelId];
  return (
    <article class={`slot${lium ? ' off' : ''}`} aria-label={slot.label || 'Card'}>
      <div class="slot-head">
        <div class="slot-title">
          <span class="row" style={{ gap: '6px' }}>
            <span class={`chip${owned ? ' owned' : ''}`}>{tag}</span>
            <span>{slot.label || `${gpu.short} · ${modelName(slot.modelId)}`}</span>
          </span>
          <small>
            {gpu.short}
            {slot.count > 1 ? ` ×${slot.count}` : ''} · {modelName(slot.modelId)}
            {cap?.ok ? ` · ${QUANT_LABEL[cap.quant]}` : ''}
            {slot.mtp && cap?.mtp.on ? ' · MTP' : ''}
          </small>
        </div>
        <div class="slot-margin">
          {lium ? (
            <>
              <b>
                <M usd={r.barUsd} />
              </b>
              <small>a month on Lium</small>
            </>
          ) : r.ok ? (
            <>
              <b class={marginUsd >= 0 ? 'pos' : 'neg'}>
                <M usd={marginUsd} signed />
              </b>
              <small>{marginLabel}</small>
            </>
          ) : (
            <span class="chip bad">Cannot serve</span>
          )}
        </div>
      </div>
      {!r.ok ? <div class="warnbox">{r.reason}</div> : null}
      {r.ok && !lium && r.hasPlaceholder ? (
        <div class="row" style={{ marginTop: '6px' }}>
          <span class="chip warn">⚠ Some inputs here are placeholders</span>
          <span class="small">{placeholdersIn(r.evidence.map((e) => ({ label: e.label, prov: e.prov })), cap?.multipliers ?? [])}</span>
        </div>
      ) : null}
      {r.ok && !lium && cap ? (
        <div class="kv">
          <div>
            <span>Revenue a month</span>
            <b>
              <M usd={r.revenueUsd} />
            </b>
          </div>
          <div>
            <span>{owned ? 'Lium bar' : 'Cost'}</span>
            <b>
              <M usd={comp} />
            </b>
          </div>
          <div>
            <span>Card used</span>
            <b>{fmtPct(r.utilisation)}</b>
          </div>
          <div>
            <span>Output speed</span>
            <b>{fmtTps(cap.tpsNode)} tok/s</b>
          </div>
          <div>
            <span>Users at once</span>
            <b>
              {fmtNum(cap.cNode, 0)} · {fmtTps(cap.userTps)} tok/s each
            </b>
          </div>
          {cap.serving ? (
            <div>
              <span>Prefill takes · first token p90</span>
              <b>
                {fmtPct(cap.serving.share)} of GPU time · {cap.serving.ttftP90 >= 10 ? cap.serving.ttftP90.toFixed(0) : cap.serving.ttftP90.toFixed(1)} s
              </b>
            </div>
          ) : null}
          <div>
            <span>If fully busy</span>
            <b>
              <M usd={r.capacityRevenueUsd * r.availability} />
            </b>
          </div>
          <div>
            <span>Traffic share won</span>
            <b>{slot.demand.mode === 'share' ? fmtPct(r.shareOfTraffic, 2) : '—'}</b>
          </div>
          <div>
            <span>Blended price</span>
            <b>
              ${r.blendedUsdPerM.toFixed(3)}/M · {fmtX(r.priceRatio)} median
            </b>
          </div>
          <div>
            <span>Break-even card use</span>
            <b>{r.breakEvenUtil === null ? 'not reachable' : fmtPct(r.breakEvenUtil)}</b>
          </div>
          <div>
            <span>Break-even price</span>
            <b>
              {r.breakEvenPriceScale === null ? 'not reachable' : `${fmtX(r.breakEvenPriceScale)} current`}
              {r.breakEvenPriceScale !== null && r.breakEvenPriceScale * r.priceRatio < r.floorRatio ? (
                <span class="small" title="The cheapest listed endpoint is the lowest price the report has data for">
                  {' '}
                  (below the cheapest endpoint)
                </span>
              ) : null}
            </b>
          </div>
          <div>
            <span>Availability</span>
            <b>
              {fmtPct(r.availability, 1)} · {r.uptimeTier}
            </b>
          </div>
          {slot.ownership === 'buy' ? (
            <div>
              <span>Payback by serving</span>
              <b>{r.paybackServingMonths === null ? 'never' : `${r.paybackServingMonths.toFixed(1)} months`}</b>
            </div>
          ) : null}
          {slot.ownership === 'buy' && r.paybackLiumMonths !== null ? (
            <div>
              <span>Payback on Lium alone</span>
              <b>{r.paybackLiumMonths.toFixed(1)} months</b>
            </div>
          ) : null}
        </div>
      ) : null}
      {r.ok && !lium && cap ? (
        <>
          {cap.kvLimited ? (
            <div class="warnbox">
              {cap.cPerGpu > cap.kvConcPerGpu * 1.0001
                ? `The KV pool holds about ${fmtNum(cap.kvConcPerGpu, 0)} users per GPU at ${fmtTokens(r.shape?.avgTokens ?? 0)}-token context, fewer than the ${fmtNum(cap.cPerGpu, 0)} this operating point assumes.`
                : `The KV pool is the limit here: it holds about ${fmtNum(cap.kvConcPerGpu, 0)} users per GPU at ${fmtTokens(r.shape?.avgTokens ?? 0)}-token context, so the operating point stops there although the quality bar would allow more.`}
            </div>
          ) : null}
          {cap.extrapolated ? <div class="warnbox">This operating point is beyond the highest measured concurrency; throughput is extrapolated.</div> : null}
          {cap.belowFloor ? <div class="warnbox">{cap.ttftLimited ? 'Even one user waits longer for a first token than the limit allows on this card.' : 'Even one user falls below the tok/s/user floor on this card.'}</div> : null}
          {cap.ttftLimited && !cap.belowFloor ? <div class="warnbox">Concurrency is held down by the time-to-first-token limit, not the speed floor: prefill queues requests before they start.</div> : null}
          {cap.basisNote ? <div class="warnbox">{cap.basisNote}</div> : null}
          {r.aboveElasticityData ? <div class="warnbox">Priced above 1.2× the median, where the report has no data; share is a guess.</div> : null}
          <details class="evidence">
            <summary>
              Evidence and assumptions <ProvSummary provs={[...r.evidence.map((e) => e.prov), ...cap.multipliers.map((m) => m.prov)]} items={[...cap.multipliers.map((m) => ({ label: m.name, prov: m.prov })), ...r.evidence.map((e) => ({ label: e.label, prov: e.prov }))]} compact />
            </summary>
            <ul>
              {cap.multipliers.map((m, i) => (
                <li key={`m${i}`}>
                  <span>
                    {m.name} ×{m.value.toFixed(2)}
                  </span>
                  <ProvBadge p={m.prov} />
                </li>
              ))}
              {r.evidence.map((e, i) => (
                <li key={i}>
                  <span>{e.label}</span>
                  <ProvBadge p={e.prov} />
                </li>
              ))}
              <li>
                <span>
                  Uncached prefill at full load: {fmtNum(r.prefillTpsAtCapacity, 0)} tok/s
                  {cap.serving ? ` against ${fmtNum(cap.serving.prefill.tokPerSec, 0)} tok/s the card can prefill` : ''}
                </span>
                <ProvBadge p={cap.serving ? { kind: 'placeholder', src: 'Roofline', note: 'Prefill speed is compute-bound; its efficiency is not fitted to any measurement', value: fmtNum(cap.serving.prefill.tokPerSec, 0), unit: 'tok/s prefill' } : { kind: 'report', src: 'Not modelled on the report\'s own numbers', note: 'Switch the throughput basis to see prefill limits' }} />
              </li>
            </ul>
          </details>
        </>
      ) : null}
      <SlotEditor slot={slot} />
      {model && !model.economics ? <div class="warnbox">{model.note}</div> : null}
    </article>
  );
}

function AddCard() {
  const [gpu, setGpu] = useState('rtx5090');
  const rentable = GPU_OPTIONS_RENT.some((g) => g.id === gpu);
  return (
    <div class="row">
      <span class="small">Add a card:</span>
      <SelectInput label="GPU to add" value={gpu} options={GPU_OPTIONS_ALL} onChange={setGpu} />
      <button
        type="button"
        class="btn"
        disabled={!rentable}
        title={rentable ? 'Rent this GPU by the hour' : 'No market rent is known for this GPU'}
        onClick={() => addSlot(makeSlot({ ownership: 'rent', gpuId: gpu, modelId: 'qwen38_27b', label: `Rented ${gpuShort(gpu)}` }))}
      >
        Rent it
      </button>
      <button type="button" class="btn" onClick={() => addSlot(makeSlot({ ownership: 'buy', gpuId: gpu, modelId: 'qwen38_27b', label: `New ${gpuShort(gpu)}` }))}>
        Buy it
      </button>
    </div>
  );
}

type Pc = ReturnType<typeof programCosts>['costs'];

/** The placeholders behind a card, each with the value it carries, for the line under its header. */
function placeholdersIn(items: { label: string; prov: Prov }[], multipliers: { name: string; prov: Prov }[]): string {
  const all = [...multipliers.map((m) => ({ label: m.name, prov: m.prov })), ...items].filter((i) => i.prov.kind === 'placeholder');
  const text = [...new Set(all.map((i) => (provValue(i.prov) ? `${i.label} ${provValue(i.prov)}` : i.label)))];
  if (text.length === 0) return 'See the evidence list below.';
  return `In use: ${text.slice(0, 4).join('; ')}${text.length > 4 ? `; and ${text.length - 4} more` : ''}.`;
}

export function PortfolioView() {
  const [compare, setCompare] = useState(false);
  const s = settings.value;
  const st = steady.value;
  const tl = timeline.value;
  const ev = evRows.value;
  const tlc = timelinesByCase.value;
  const f = taxF();
  const evCase = ev.find((e) => e.caseId === s.caseId)!;
  const gain = s.afterTax ? st.afterTaxMarginUsd : st.marginUsd;
  const allProvs = st.results.flatMap((r) => [...r.evidence.map((e) => e.prov), ...(r.capacity?.multipliers.map((m) => m.prov) ?? [])]);
  const allItems = st.results.flatMap((r) => [...r.evidence.map((e) => ({ label: e.label, prov: e.prov })), ...(r.capacity?.multipliers.map((m) => ({ label: m.name, prov: m.prov })) ?? [])]);
  const capUsd = PROGRAM.stakeCapInr.v / s.fx;
  const showStake = tl.dependsOnListing && (tl.peakStakeUsd > 0 || tl.program.rejectedUsd > 0);
  const lium = st.results.filter((r) => r.kind === 'lium').length;

  const barItems: BarItem[] = st.results
    .filter((r) => r.kind === 'serve' && r.ok)
    .map((r) => {
      const comp = r.slot.ownership === 'owned' ? r.barUsd : r.costUsd;
      const m = s.afterTax ? r.afterTaxMarginUsd : r.marginUsd;
      return {
        id: r.slot.id,
        label: `${r.slot.ownership === 'owned' ? 'Owned' : r.slot.ownership === 'rent' ? 'Rented' : 'New'} ${r.gpu.short}${r.slot.count > 1 ? ` ×${r.slot.count}` : ''}`,
        sub: modelName(r.slot.modelId),
        value: disp(r.revenueUsd),
        compare: disp(comp),
        valueLabel: `${m >= 0 ? '+' : ''}${money(m)}`,
        tip: [
          { label: 'Revenue', value: money(r.revenueUsd) },
          { label: r.slot.ownership === 'owned' ? 'Lium bar' : 'Cost', value: money(comp) },
          { label: 'Margin', value: money(r.marginUsd) },
          { label: 'Card used', value: fmtPct(r.utilisation) },
        ],
      };
    });

  const months = tl.months.map((m) => m.m);
  const series = CASES.filter((c) => compare || c === s.caseId).map((c) => ({
    id: c,
    label: CASE_LABEL[c],
    color: CASE_COLOR[c],
    dim: compare && c !== s.caseId,
    width: c === s.caseId ? 3 : 2,
    pts: tlc[c].months.map((m) => ({ x: m.m, y: disp(m.cumGainUsd * f) })),
  }));
  const pcs = CASES.map((c) => ({ c, p: programCosts(slots.value, s, c).costs }));
  const stageRows: [string, (p: Pc) => number][] = [
    ['Build and load-test', (p) => p.buildUsd],
    ['Queue', (p) => p.queueUsd],
    ['Review card and tools', (p) => p.reviewUsd],
    ['Bridge month', (p) => p.bridgeUsd],
  ];

  return (
    <div class="stack">
      <div class="row spread">
        <div class="row">
          <Field label="Plan">
            <SelectInput label="Plan preset" value={presetId.value} options={PRESETS.map((p) => ({ id: p.id, label: p.label }))} onChange={loadPreset} />
          </Field>
        </div>
        <AddCard />
      </div>
      <div class="small">{PRESETS.find((p) => p.id === presetId.value)?.help}</div>

      <div class="top-grid">
        <div class="hero" aria-label="Headline result">
          <div class="h-label">
            Gain per steady month, {CASE_LABEL[s.caseId].toLowerCase()} case {s.afterTax ? '(after tax)' : '(pre-tax)'} <ProvSummary provs={allProvs} items={allItems} />
          </div>
          <div class={`h-value ${gain >= 0 ? 'pos' : 'neg'}`}>
            <M usd={gain} signed />
          </div>
          <div class="h-sub">
            {lium === st.results.length ? 'Both boxes stay on Lium, so the gain is zero by definition.' : 'Revenue minus the Lium income given up and any rent, once every card is listed and at full share.'}
          </div>
        </div>
        <div class="tiles">
          <Tile label="Steady revenue" value={<M usd={st.revenueUsd} />} sub={`${st.servingCount} card${st.servingCount === 1 ? '' : 's'} serving tokens`} />
          <Tile label="Lium bars + rent" value={<M usd={st.barUsd + st.rentAndCostUsd} />} sub={`bars ${money(st.barUsd)} · rent ${money(st.rentAndCostUsd)}`} />
          <Tile label="Cards busy" value={fmtPct(st.utilisation)} sub="share of capacity sold" />
          <Tile label="12-month net" value={<M usd={tl.netUsd * f} signed />} sub={tl.dependsOnListing ? `listed in month ${tl.primaryListing}` : 'nothing to list'} />
          <Tile
            label={`Expected value at ${fmtPct(s.odds)} odds`}
            value={<M usd={evCase.evUsd * f} signed />}
            sub={evCase.breakEvenOdds === null ? 'never pays in this case' : `pays above ${fmtPct(evCase.breakEvenOdds, 2)} odds`}
          />
          {showStake ? <Tile label="Peak stake used" value={<M usd={tl.peakStakeUsd} />} sub={`cap ${money(capUsd)}`} /> : null}
        </div>
      </div>

      <div class="slots">
        {st.results.map((r) => (
          <SlotCard key={r.slot.id} r={r} />
        ))}
        <div class="card" style={{ borderStyle: 'dashed' }}>
          <h3>RTX 3050 · control plane</h3>
          <p class="small">Owned, earns almost nothing on Lium and stays the control plane. It can host Whisper-turbo and embedding probes (report §8.6). Not modelled as an LLM card.</p>
        </div>
      </div>

      <div class="grid cols-2">
        <ChartCard
          title="Cumulative result over 12 months"
          subtitle={`Net of stage costs and rent, against staying on Lium. ${s.afterTax ? 'After tax.' : 'Pre-tax.'}`}
          actions={
            <Toggle checked={compare} onChange={setCompare} title="Low, base and high differ by an order of magnitude, so the chart zooms to the selected case unless you compare">
              Compare cases
            </Toggle>
          }
          legend={
            <div class="legend" aria-label="Scenario legend">
              {series.map((x) => (
                <span key={x.id} style={{ display: 'inline-flex', gap: '6px', alignItems: 'center' }}>
                  <i style={{ display: 'inline-block', width: '12px', height: '3px', borderRadius: '2px', background: x.color }} />
                  {x.label}
                </span>
              ))}
            </div>
          }
          chart={
            <LineChart
              ariaLabel="Cumulative net result by month for the selected scenario"
              series={series}
              xFmt={(x) => `M${x}`}
              yFmt={(y) => fmtD(y)}
              xTicks={months}
              xDomain={[1, 12]}
              zeroLine
              vrefs={[{ x: tl.primaryListing, label: 'listed' }]}
              tipTitle={(x) => `Month ${x}`}
              height={290}
              endLabels
            />
          }
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Month</th>
                  {CASES.map((c) => (
                    <th key={c} class="num">
                      {CASE_LABEL[c]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {months.map((m) => (
                  <tr key={m}>
                    <td>{m}</td>
                    {CASES.map((c) => (
                      <td key={c} class="num">
                        {money(tlc[c].months[m - 1]!.cumGainUsd * f)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
        <ChartCard
          title="Steady month by card"
          subtitle="Revenue against the Lium bar (owned) or rent (rented). The tick marks the bar."
          chart={barItems.length > 0 ? <HBars items={barItems} ariaLabel="Steady-month revenue by card with its bar or rent" valueName="Revenue" compareName="Lium bar or rent" fmt={(v) => fmtD(v)} /> : <p class="small">No card is serving tokens in this plan.</p>}
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Card</th>
                  <th class="num">Revenue</th>
                  <th class="num">Bar or rent</th>
                  <th class="num">Margin</th>
                  <th class="num">Card used</th>
                </tr>
              </thead>
              <tbody>
                {st.results
                  .filter((r) => r.kind === 'serve' && r.ok)
                  .map((r) => (
                    <tr key={r.slot.id}>
                      <td>{r.slot.label}</td>
                      <td class="num">{money(r.revenueUsd)}</td>
                      <td class="num">{money(r.slot.ownership === 'owned' ? r.barUsd : r.costUsd)}</td>
                      <td class="num">{money(s.afterTax ? r.afterTaxMarginUsd : r.marginUsd)}</td>
                      <td class="num">{fmtPct(r.utilisation)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          }
        />
      </div>

      {showStake || tl.dependsOnListing ? (
        <div class="grid cols-2">
          {showStake ? (
            <ChartCard
              title="Stake used against the cap"
              subtitle="Costs paid minus revenue received on rented cards and the listing programme. Owned boxes stay on Lium until they move, so they do not draw on the stake."
              chart={
                <LineChart
                  ariaLabel="Cumulative net spend against the stake cap, by month"
                  series={[{ id: 'stake', label: 'Stake used', color: 'var(--s1)', pts: tl.months.map((m) => ({ x: m.m, y: disp(m.stakeUsedUsd) })), width: 3 }]}
                  xFmt={(x) => `M${x}`}
                  yFmt={(y) => fmtD(y)}
                  xTicks={months}
                  xDomain={[1, 12]}
                  hrefs={[{ y: disp(capUsd), label: `cap ${money(capUsd)}` }]}
                  zeroLine
                  yDomain={[Math.min(0, ...tl.months.map((m) => disp(m.stakeUsedUsd))), Math.max(disp(capUsd) * 1.1, ...tl.months.map((m) => disp(m.stakeUsedUsd)))]}
                  tipTitle={(x) => `Month ${x}`}
                  height={250}
                />
              }
              table={
                <table class="t">
                  <thead>
                    <tr>
                      <th>Month</th>
                      <th class="num">Programme</th>
                      <th class="num">Rent</th>
                      <th class="num">Rented revenue less rent</th>
                      <th class="num">Stake used</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tl.months.map((m) => (
                      <tr key={m.m}>
                        <td>{m.m}</td>
                        <td class="num">{money(m.programUsd + m.toolsUsd)}</td>
                        <td class="num">{money(m.rentUsd)}</td>
                        <td class="num">{money(m.revenueUsd - m.rentUsd)}</td>
                        <td class="num">{money(m.stakeUsedUsd)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              }
            />
          ) : null}
          {tl.dependsOnListing ? (
            <section class="card" aria-label="Cost of trying">
              <h2>What it costs to try</h2>
              <div class="sub">
                Build, queue, a rented review card and one bridge month while the owned box finishes its Lium rental. Same stages as the report's §6.4 <ProvBadge p={PROGRAM.buildInr.p} compact />.
              </div>
              <div class="tablewrap">
                <table class="t">
                  <thead>
                    <tr>
                      <th>Stage</th>
                      {CASES.map((c) => (
                        <th key={c} class="num" style={c === s.caseId ? { color: 'var(--ink)' } : undefined}>
                          {CASE_LABEL[c]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {stageRows.map(([label, fn]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        {pcs.map(({ c, p }) => (
                          <td key={c} class="num">
                            {money(fn(p) * f)}
                          </td>
                        ))}
                      </tr>
                    ))}
                    <tr style={{ fontWeight: 600 }}>
                      <td>Total if listed</td>
                      {pcs.map(({ c, p }) => (
                        <td key={c} class="num">
                          {money((p.buildUsd + p.queueUsd + p.reviewUsd + p.bridgeUsd) * f)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td>If reviewed and turned down</td>
                      {pcs.map(({ c, p }) => (
                        <td key={c} class="num">
                          {money(p.rejectedUsd * f)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td>If never reviewed</td>
                      {pcs.map(({ c, p }) => (
                        <td key={c} class="num">
                          {money(p.neverUsd * f)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </div>
      ) : null}

      <section class="card" aria-label="Expected value of applying">
        <h2>Expected value of applying to OpenRouter</h2>
        <div class="sub">Same rule as the report: as many applications are reviewed and turned down as are listed, the rest are still waiting after 12 months. Set the odds under Assumptions.</div>
        {tl.dependsOnListing ? (
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Case</th>
                  <th class="num">Listed in</th>
                  <th class="num">If listed, 12 months</th>
                  <th class="num">Turned down</th>
                  <th class="num">Never reviewed</th>
                  {evCase.evAtOdds.map((o) => (
                    <th key={o.odds} class="num">
                      EV at {fmtPct(o.odds)}
                    </th>
                  ))}
                  <th class="num">Break-even odds</th>
                </tr>
              </thead>
              <tbody>
                {ev.map((e) => (
                  <tr key={e.caseId} style={e.caseId === s.caseId ? { fontWeight: 600 } : undefined}>
                    <td>{CASE_LABEL[e.caseId]}</td>
                    <td class="num">month {e.firstListed}</td>
                    <td class="num">{money(e.listedNetUsd * f)}</td>
                    <td class="num">{money(e.rejectedUsd * f)}</td>
                    <td class="num">{money(e.neverUsd * f)}</td>
                    {e.evAtOdds.map((o) => (
                      <td key={o.odds} class="num">
                        {money(o.usd * f)}
                      </td>
                    ))}
                    <td class="num">{e.breakEvenOdds === null ? 'none' : fmtPct(e.breakEvenOdds, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p class="small">Nothing in this plan needs an OpenRouter listing.</p>
        )}
      </section>
    </div>
  );
}
