import { useMemo, useState } from 'preact/hooks';
import { GPUS, QUANT_LABEL } from '../data/gpus';
import { makeSlot } from '../engine/defaults';
import { evaluateSlot, type SlotResult } from '../engine/slot';
import { fmtPct, fmtTps, fmtX } from '../engine/units';
import { addSlot, goTo, setSlot, settings, slots } from '../state';
import { ChartCard, Heatmap, type HeatCell } from '../ui/charts';
import { Field, SelectInput, Segmented, Toggle } from '../ui/controls';
import { M, money } from '../ui/money';
import { COUNT_OPTIONS, ECON_MODELS, WORKLOAD_OPTIONS, gpuShort, modelName } from '../ui/options';
import { ProvBadge } from '../ui/prov';

type Metric = 'ratio' | 'margin' | 'breakeven' | 'used';

const SHORT: Record<string, string> = {
  qwen38_27b: 'Qwen3.8-27B',
  qwen36_35b: 'Qwen3.6-35B-A3B',
  gemma4_26b: 'Gemma 4 26B-A4B',
  gemma4_31b: 'Gemma 4 31B',
  gptoss_120b: 'gpt-oss-120b',
  gptoss_20b: 'gpt-oss-20b',
  llama33_70b: 'Llama 3.3 70B',
  mistral_nemo: 'Mistral Nemo 12B',
  flashnext: 'Qwen3.8-Flash-Next',
};

interface RowDef {
  id: string;
  label: string;
  gpuId: string;
  ownership: 'owned' | 'rent';
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
/** Streams slower than this feel slow to a person; an assumption, not a report figure. */
const SLOW_TPS = 25;

export function ViabilityView() {
  const s = settings.value;
  const [count, setCount] = useState('1');
  const [metric, setMetric] = useState<Metric>('ratio');
  const [workload, setWorkload] = useState('observed');
  const [mtp, setMtp] = useState(false);
  const [sel, setSel] = useState<string | null>(null);

  const rowDefs: RowDef[] = useMemo(() => {
    const out: RowDef[] = [];
    for (const g of GPUS.filter((x) => x.role === 'serve')) {
      if (g.owned > 0) out.push({ id: `own|${g.id}`, label: `${g.short} · owned`, gpuId: g.id, ownership: 'owned' });
    }
    for (const g of GPUS.filter((x) => x.role === 'serve' && x.rentUsdHr)) {
      out.push({ id: `rent|${g.id}`, label: `${g.short} · rent${Number(count) > 1 ? ` ×${count}` : ''}`, gpuId: g.id, ownership: 'rent' });
    }
    return out;
  }, [count]);

  const results = useMemo(() => {
    const m = new Map<string, SlotResult>();
    for (const r of rowDefs) {
      for (const mod of ECON_MODELS) {
        const slot = makeSlot({
          id: 'grid',
          ownership: r.ownership,
          boxKey: r.ownership === 'owned' ? r.gpuId : undefined,
          gpuId: r.gpuId,
          modelId: mod.id,
          count: r.ownership === 'rent' ? Number(count) : 1,
          workloadId: workload,
          mtp: mtp && !!mod.mtp,
          availabilityId: r.ownership === 'rent' ? 'dc' : 'global',
        });
        m.set(`${r.id}|${mod.id}`, evaluateSlot(slot, s));
      }
    }
    return m;
  }, [rowDefs, s, count, workload, mtp]);

  const marginOf = (r: SlotResult) => (s.afterTax ? r.afterTaxMarginUsd : r.marginUsd);
  const marginScale = useMemo(() => {
    const vals = [...results.values()].filter((r) => r.ok).map((r) => Math.abs(r.marginUsd));
    vals.sort((a, b) => a - b);
    return vals.length ? Math.max(1, vals[Math.floor(vals.length * 0.9)] ?? 1) : 1;
  }, [results]);

  const cells: HeatCell[] = [];
  for (const r of rowDefs) {
    for (const mod of ECON_MODELS) {
      const res = results.get(`${r.id}|${mod.id}`)!;
      const key = { row: r.id, col: mod.id };
      const title = `${r.label} · ${SHORT[mod.id]}`;
      if (!res.ok) {
        cells.push({ ...key, t: null, text: '—', tipTitle: title, tip: [{ label: 'Cannot serve', value: res.reason }] });
        continue;
      }
      const comp = r.ownership === 'owned' ? res.barUsd : res.costUsd;
      const ratio = comp > 0 ? res.revenueUsd / comp : Infinity;
      let t: number | null;
      let text: string;
      if (metric === 'ratio') {
        t = clamp(Math.log2(Math.max(ratio, 1 / 64)) / 3, -1, 1);
        text = fmtX(ratio);
      } else if (metric === 'margin') {
        t = clamp(res.marginUsd / marginScale, -1, 1);
        text = money(marginOf(res));
      } else if (metric === 'breakeven') {
        t = res.breakEvenUtil === null ? 1 : res.breakEvenUtil;
        text = res.breakEvenUtil === null ? '>100%' : fmtPct(res.breakEvenUtil);
      } else {
        t = res.utilisation;
        text = fmtPct(res.utilisation);
      }
      const user = res.capacity?.userTps ?? 0;
      const slow = user > 0 && user < SLOW_TPS;
      cells.push({
        ...key,
        t,
        text: slow ? `${text}*` : text,
        tipTitle: title,
        tip: [
          { label: 'Revenue a month', value: money(res.revenueUsd) },
          { label: r.ownership === 'owned' ? 'Lium bar' : 'Rent', value: money(comp) },
          { label: 'Margin', value: money(marginOf(res)) },
          { label: 'Card used', value: fmtPct(res.utilisation) },
          { label: 'Break-even card use', value: res.breakEvenUtil === null ? 'not reachable' : fmtPct(res.breakEvenUtil) },
          { label: 'Weights', value: res.capacity ? QUANT_LABEL[res.capacity.quant] : '' },
          { label: 'Output speed', value: `${fmtTps(res.capacity?.tpsNode ?? 0)} tok/s` },
          { label: 'Per user', value: `${fmtTps(user)} tok/s${slow ? ' (slow)' : ''}` },
          ...(res.hasPlaceholder ? [{ label: 'Inputs', value: 'includes placeholders' }] : []),
        ],
      });
    }
  }

  const selRes = sel ? results.get(sel) : null;
  const selRow = sel ? rowDefs.find((r) => sel.startsWith(`${r.id}|`)) : undefined;
  const selModelId = sel && selRow ? sel.slice(selRow.id.length + 1) : '';
  const metricLabel: Record<Metric, string> = {
    ratio: 'Revenue ÷ cost',
    margin: 'Margin a month',
    breakeven: 'Break-even card use',
    used: 'Card used',
  };

  const addFromSel = () => {
    if (!selRow || !selModelId) return;
    if (selRow.ownership === 'rent') {
      addSlot(makeSlot({ ownership: 'rent', gpuId: selRow.gpuId, modelId: selModelId, count: Number(count), workloadId: workload, mtp: mtp && !!ECON_MODELS.find((m) => m.id === selModelId)?.mtp, label: `Rented ${gpuShort(selRow.gpuId)} · ${modelName(selModelId)}` }));
    } else {
      const target = slots.value.find((x) => x.ownership === 'owned' && x.gpuId === selRow.gpuId);
      if (target) setSlot(target.id, { modelId: selModelId, label: `Owned ${gpuShort(selRow.gpuId)} · ${modelName(selModelId)}` });
    }
    goTo('portfolio');
  };

  const buyRes = useMemo(() => {
    if (!selRow || !selModelId) return null;
    const g = GPUS.find((x) => x.id === selRow.gpuId);
    if (!g || !g.buyInr) return null;
    const slot = makeSlot({ id: 'b', ownership: 'buy', gpuId: g.id, modelId: selModelId, count: 1, workloadId: workload, mtp: mtp && !!ECON_MODELS.find((m) => m.id === selModelId)?.mtp, availabilityId: 'global' });
    return { res: evaluateSlot(slot, s), gpu: g };
  }, [selRow, selModelId, s, workload, mtp]);

  const scaleNote =
    metric === 'ratio' ? 'Blue is above break-even, red is below; the colour saturates at 8× or 1/8×. * marks a stream slower than 25 tok/s per user at the operating point.' : metric === 'margin' ? `Blue earns more than it costs, red less; saturates at ${money(marginScale)}.` : metric === 'breakeven' ? 'Darker means a fuller card is needed to break even.' : 'Darker means a busier card.';

  return (
    <div class="stack">
      <div class="notice">
        Every cell uses the global scenario, quality bar, tax view and route, with the typical-entrant marketplace share. Owned rows are compared with the Lium bar; rented rows with market rent. Hatched cells cannot serve that model.
      </div>
      <div class="filters" role="region" aria-label="Grid options">
        <Field label="Colour shows">
          <Segmented<Metric> label="Metric" value={metric} options={[{ id: 'ratio', label: 'Revenue ÷ cost' }, { id: 'margin', label: 'Margin' }, { id: 'breakeven', label: 'Break-even use' }, { id: 'used', label: 'Card used' }]} onChange={setMetric} />
        </Field>
        <Field label="Workload">
          <SelectInput label="Workload" value={workload} options={WORKLOAD_OPTIONS} onChange={setWorkload} />
        </Field>
        <Field label="GPUs per rented node">
          <SelectInput label="GPUs per rented node" value={count} options={COUNT_OPTIONS} onChange={setCount} />
        </Field>
        <Toggle checked={mtp} onChange={setMtp}>
          MTP where available
        </Toggle>
      </div>
      <ChartCard
        title={`GPU × model: ${metricLabel[metric]}`}
        subtitle={scaleNote}
        chart={<Heatmap rows={rowDefs.map((r) => ({ id: r.id, label: r.label }))} cols={ECON_MODELS.map((m) => ({ id: m.id, label: SHORT[m.id] ?? m.name }))} cells={cells} mode={metric === 'ratio' || metric === 'margin' ? 'diverging' : 'sequential'} ariaLabel={`Grid of GPUs against models showing ${metricLabel[metric]}`} selected={sel} onSelect={(r, c) => setSel(`${r}|${c}`)} />}
        table={
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                {ECON_MODELS.map((m) => (
                  <th key={m.id} class="num">
                    {SHORT[m.id]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rowDefs.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  {ECON_MODELS.map((m) => (
                    <td key={m.id} class="num">
                      {cells.find((c) => c.row === r.id && c.col === m.id)?.text ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        }
      />
      {selRes && selRow ? (
        <section class="card" aria-label="Selected cell">
          <div class="spread">
            <h2>
              {selRow.label} on {modelName(selModelId)}
            </h2>
            <button type="button" class="btn primary" disabled={!selRes.ok} onClick={addFromSel}>
              {selRow.ownership === 'rent' ? 'Add as a rented card' : 'Put my owned box on this model'}
            </button>
          </div>
          {selRes.ok ? (
            <>
              <div class="kv">
                <div>
                  <span>Revenue a month</span>
                  <b>
                    <M usd={selRes.revenueUsd} />
                  </b>
                </div>
                <div>
                  <span>{selRow.ownership === 'owned' ? 'Lium bar' : 'Rent'}</span>
                  <b>
                    <M usd={selRow.ownership === 'owned' ? selRes.barUsd : selRes.costUsd} />
                  </b>
                </div>
                <div>
                  <span>Margin</span>
                  <b>
                    <M usd={marginOf(selRes)} signed />
                  </b>
                </div>
                <div>
                  <span>Card used</span>
                  <b>{fmtPct(selRes.utilisation)}</b>
                </div>
                <div>
                  <span>Break-even card use</span>
                  <b>{selRes.breakEvenUtil === null ? 'not reachable' : fmtPct(selRes.breakEvenUtil)}</b>
                </div>
                <div>
                  <span>Break-even price</span>
                  <b>{selRes.breakEvenPriceScale === null ? 'not reachable' : `${fmtX(selRes.breakEvenPriceScale)} the current price`}</b>
                </div>
                <div>
                  <span>Weights</span>
                  <b>{selRes.capacity ? QUANT_LABEL[selRes.capacity.quant] : ''}</b>
                </div>
                <div>
                  <span>Output speed</span>
                  <b>{fmtTps(selRes.capacity?.tpsNode ?? 0)} tok/s</b>
                </div>
              </div>
              {selRes.capacity?.multipliers.some((m) => m.prov.kind === 'placeholder') ? (
                <div class="warnbox">
                  This cell scales throughput from another card by memory bandwidth, which is a placeholder until the roofline engine arrives. <ProvBadge p={{ kind: 'placeholder', src: 'Bandwidth scaling' }} />
                </div>
              ) : null}
              {buyRes && selRow.ownership === 'rent' ? (
                <div class="mt">
                  <h3>Buy instead of rent</h3>
                  <p class="small">
                    A new {buyRes.gpu.short} costs {money(((buyRes.gpu.buyInr?.v ?? 0) / s.fx))} <ProvBadge p={buyRes.gpu.buyInr!.p} compact />. Serving this model it would earn {money(buyRes.res.revenueUsd)} a month and pay itself back in{' '}
                    {buyRes.res.paybackServingMonths === null ? 'never' : `${buyRes.res.paybackServingMonths.toFixed(1)} months`}
                    {buyRes.res.paybackLiumMonths ? `, against ${buyRes.res.paybackLiumMonths.toFixed(1)} months on Lium alone` : ''}. The report's rule: buy only after a rented card has stayed above break-even for three months.
                  </p>
                </div>
              ) : null}
            </>
          ) : (
            <div class="warnbox">{selRes.reason}</div>
          )}
        </section>
      ) : (
        <p class="small">Select a cell for its numbers and to add it to the portfolio.</p>
      )}
    </div>
  );
}
