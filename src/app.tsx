import { FX_INR_PER_USD, PRO6000_EDITION, SUMMER_DERATE } from './data/gpus';
import { AVAILABILITY_PRESETS, CASES, CASE_LABEL, LISTING_ODDS, QOS_PRESETS, ROUTE_COSTS, SCENARIOS, ZDR_SHARE_BONUS, type CaseId, type CostLevel } from './data/scenarios';
import { NEUTRAL_WHATIF } from './engine/types';
import { resolveCase } from './engine/slot';
import { fmtPct } from './engine/units';
import { goTo, resetAll, settings, theme, updateSettings, view, VIEWS } from './state';
import { Field, NumberInput, SelectInput, Segmented, Toggle } from './ui/controls';
import { KindLegend, Popover, ProvBadge } from './ui/prov';
import { CapacityView } from './views/capacity';
import { DataView } from './views/data';
import { PortfolioView } from './views/portfolio';
import { PricesView } from './views/prices';
import { SensitivityView } from './views/sensitivity';
import { ThroughputView } from './views/throughput';
import { ViabilityView } from './views/viability';

function OverrideNumber({
  label,
  effective,
  scenarioValue,
  provSrc,
  onChange,
  scale = 1,
  suffix,
  step,
  min,
  max,
}: {
  label: string;
  effective: number;
  scenarioValue: { v: number; p: Parameters<typeof ProvBadge>[0]['p'] };
  provSrc?: boolean;
  onChange: (v: number | null) => void;
  scale?: number;
  suffix?: string;
  step?: number;
  min?: number;
  max?: number;
}) {
  const changed = Math.abs(effective - scenarioValue.v) > 1e-9;
  return (
    <Field
      label={
        <>
          {label} {provSrc === false ? null : <ProvBadge p={scenarioValue.p} compact />}
        </>
      }
    >
      <span class="row" style={{ gap: '6px' }}>
        <NumberInput value={Number((effective * scale).toFixed(4))} label={label} step={step} min={min} max={max} suffix={suffix} onChange={(v) => onChange(v / scale)} />
        {changed ? (
          <button type="button" class="btn ghost" onClick={() => onChange(null)} title={`Back to the scenario value ${(scenarioValue.v * scale).toFixed(2)}`}>
            reset
          </button>
        ) : null}
      </span>
    </Field>
  );
}

function Assumptions() {
  const s = settings.value;
  const sc = SCENARIOS[s.caseId];
  const eff = resolveCase(s);
  const whatIfActive = Object.entries(s.whatIf).some(([k, v]) => v !== (NEUTRAL_WHATIF as unknown as Record<string, number>)[k]);
  return (
    <details class="more">
      <summary>
        Assumptions{whatIfActive ? ' · what-if active' : ''} · {CASE_LABEL[s.caseId]} case
      </summary>
      <div class="assump">
        <Field label={<>Scenario share of traffic <ProvBadge p={sc.shareFactor.p} compact /></>}>
          <span class="row" style={{ gap: '6px' }}>
            <NumberInput value={Number(eff.shareFactor.toFixed(3))} label="Share factor" step={0.05} min={0} max={3} suffix="× equal share" onChange={(v) => updateSettings({ override: { ...s.override, shareFactor: v } })} />
            {s.override.shareFactor !== null ? <button type="button" class="btn ghost" onClick={() => updateSettings({ override: { ...s.override, shareFactor: null } })}>reset</button> : null}
          </span>
        </Field>
        <OverrideNumber label="Latency penalty from India" effective={eff.latencyLoss} scenarioValue={sc.latencyLoss} scale={100} suffix="% share" step={1} min={0} max={80} onChange={(v) => updateSettings({ override: { ...s.override, latencyLoss: v } })} />
        <OverrideNumber label="Monthly price trend" effective={eff.priceTrend} scenarioValue={sc.priceTrend} scale={100} suffix="% a month" step={0.5} min={-20} max={20} onChange={(v) => updateSettings({ override: { ...s.override, priceTrend: v } })} />
        <OverrideNumber label="First listed month" effective={eff.listingMonth} scenarioValue={sc.listingMonth} suffix="month" step={1} min={1} max={12} onChange={(v) => updateSettings({ override: { ...s.override, listingMonth: v === null ? null : Math.round(v) } })} />
        <Field label={<>Owned PRO 6000 edition <ProvBadge p={PRO6000_EDITION[s.proEdition].prov} compact /></>}>
          <SelectInput label="PRO 6000 edition" value={s.proEdition} options={[{ id: 'workstation', label: 'Workstation 600 W' }, { id: 'maxq', label: 'Max-Q 300 W (decode ×0.95)' }]} onChange={(v) => updateSettings({ proEdition: v })} />
        </Field>
        <Field label={<>Bar cost level <ProvBadge p={ROUTE_COSTS.upsInr.base.p} compact /></>}>
          <Segmented label="Bar cost level" value={s.costLevel} options={(['low', 'base', 'high'] as CostLevel[]).map((c) => ({ id: c, label: CASE_LABEL[c] }))} onChange={(v) => updateSettings({ costLevel: v })} />
        </Field>
        <Field label={<>Listing odds for expected value <ProvBadge p={LISTING_ODDS.default.p} compact /></>}>
          <span class="row">
            <input type="range" aria-label="Listing odds" min={1} max={30} step={1} value={Math.round(s.odds * 100)} style={{ accentColor: 'var(--accent)' }} onInput={(e) => updateSettings({ odds: e.currentTarget.valueAsNumber / 100 })} />
            <b>{fmtPct(s.odds)}</b>
          </span>
        </Field>
        <Field label={<>FX, INR per USD <ProvBadge p={FX_INR_PER_USD.p} compact /></>}>
          <NumberInput value={s.fx} label="INR per USD" step={0.1} min={50} max={150} width={90} onChange={(v) => updateSettings({ fx: v })} />
        </Field>
        <div class="stack">
          <Toggle checked={s.summerDerate} onChange={(v) => updateSettings({ summerDerate: v })} title={SUMMER_DERATE.p.note}>
            Summer power cap on owned boxes (×{SUMMER_DERATE.v}) <ProvBadge p={SUMMER_DERATE.p} compact />
          </Toggle>
          <Toggle checked={s.zdr} onChange={(v) => updateSettings({ zdr: v })}>
            Zero-data-retention endpoint (+{fmtPct(ZDR_SHARE_BONUS.v)} share) <ProvBadge p={ZDR_SHARE_BONUS.p} compact />
          </Toggle>
          <Toggle checked={s.medianVolume} onChange={(v) => updateSettings({ medianVolume: v })} title="Qwen3.6-35B-A3B's seven-day average is about 3.5× its 14-day median">
            Use 14-day median traffic where the report gives one
          </Toggle>
          <Toggle checked={s.launchFloor} onChange={(v) => updateSettings({ launchFloor: v })}>
            Price the first two listed months at the floor
          </Toggle>
          <Toggle checked={s.killRules} onChange={(v) => updateSettings({ killRules: v })}>
            Apply the report's revert and hand-back rules
          </Toggle>
        </div>
      </div>
    </details>
  );
}

function Filters() {
  const s = settings.value;
  return (
    <div class="filters" role="region" aria-label="Filters that apply to every view">
      <Field label="Scenario">
        <Segmented label="Scenario" value={s.caseId} options={CASES.map((c) => ({ id: c as CaseId, label: CASE_LABEL[c] }))} onChange={(v) => updateSettings({ caseId: v })} />
      </Field>
      <Field label="Currency">
        <Segmented label="Currency" value={s.currency} options={[{ id: 'INR', label: '₹ INR' }, { id: 'USD', label: '$ USD' }]} onChange={(v) => updateSettings({ currency: v })} />
      </Field>
      <Field label="Tax">
        <Segmented label="Tax view" value={s.afterTax ? 'after' : 'pre'} options={[{ id: 'pre', label: 'Pre-tax' }, { id: 'after', label: 'After tax' }]} onChange={(v) => updateSettings({ afterTax: v === 'after' })} />
      </Field>
      <Field label="Payout route">
        <SelectInput label="Payout route" value={s.route} options={[{ id: 'fiat', label: 'Fiat (e.g. OpenRouter wire)' }, { id: 'crypto', label: 'Crypto (31.2% of gross)' }]} onChange={(v) => updateSettings({ route: v })} />
      </Field>
      <Field label="Quality bar">
        <span class="row" style={{ gap: '6px' }}>
          <SelectInput label="Quality bar" value={s.qos.mode} options={QOS_PRESETS.map((q) => ({ id: q.id, label: q.label }))} onChange={(v) => updateSettings({ qos: { ...s.qos, mode: v } })} />
          {s.qos.mode === 'floor' ? <NumberInput label="Minimum tokens per second per user" value={s.qos.floorTps} min={1} max={200} step={1} width={70} suffix="tok/s/user" onChange={(v) => updateSettings({ qos: { ...s.qos, floorTps: v } })} /> : null}
        </span>
      </Field>
      <Field label="Availability">
        <span class="row" style={{ gap: '6px' }}>
          <SelectInput label="Availability preset" value={s.availabilityCustom === null ? s.availabilityId : 'custom'} options={[...AVAILABILITY_PRESETS.filter((a) => a.id !== 'dc').map((a) => ({ id: a.id, label: `${a.label} · ${fmtPct(a.availability.v, 1)}` })), { id: 'custom', label: 'Custom' }]} onChange={(v) => (v === 'custom' ? updateSettings({ availabilityCustom: 0.97 }) : updateSettings({ availabilityId: v, availabilityCustom: null }))} />
          {s.availabilityCustom !== null ? <NumberInput label="Custom availability" value={Math.round(s.availabilityCustom * 1000) / 10} min={30} max={100} step={0.5} width={70} suffix="%" onChange={(v) => updateSettings({ availabilityCustom: Math.min(1, Math.max(0.3, v / 100)) })} /> : null}
        </span>
      </Field>
    </div>
  );
}

const VIEW_COMPONENT = {
  portfolio: PortfolioView,
  viability: ViabilityView,
  capacity: CapacityView,
  throughput: ThroughputView,
  sensitivity: SensitivityView,
  prices: PricesView,
  data: DataView,
} as const;

export function App() {
  const V = VIEW_COMPONENT[view.value];
  return (
    <div class="shell">
      <header class="topbar">
        <div class="brand">
          <b>Inferx</b>
          <span>Inference economics for a small GPU fleet</span>
        </div>
        <div class="row">
          <Segmented label="Theme" value={theme.value} options={[{ id: 'system', label: 'Auto' }, { id: 'light', label: 'Light' }, { id: 'dark', label: 'Dark' }]} onChange={(v) => (theme.value = v)} />
          <button type="button" class="btn" onClick={() => resetAll()} title="Restore the report's base assumptions and plan">
            Reset
          </button>
        </div>
      </header>
      <nav class="nav" aria-label="Views">
        {VIEWS.map((v) => (
          <a
            key={v.id}
            href={`#${v.id}`}
            aria-current={view.value === v.id ? 'page' : undefined}
            onClick={(e) => {
              e.preventDefault();
              goTo(v.id);
            }}
          >
            {v.label}
          </a>
        ))}
      </nav>
      <Filters />
      <Assumptions />
      <KindLegend />
      <main>
        <V />
      </main>
      <footer class="small" style={{ marginTop: '28px' }}>
        Numbers are built on the report "Inference as a Service: Platforms, Models and Margins" (22 Oct 2026) and the InferenceX repositories. Phase 1: static report data, live Vast.ai and OpenRouter prices and the roofline engine come next. Not financial or legal advice.
      </footer>
      <Popover />
    </div>
  );
}
