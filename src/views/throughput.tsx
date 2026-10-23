import { useState } from 'preact/hooks';
import { GPU_BY_ID, QUANT_LABEL } from '../data/gpus';
import { GOLDEN_AL_QWEN38_FLASH_NEXT, MODEL_BY_ID } from '../data/models';
import { THROUGHPUT_PRESETS, WORKLOAD_BY_ID } from '../data/scenarios';
import { DRAFT_COST, REPORT_TEST_CTX, capacityFor, chooseQuant, curveFor, mtpInfo, weightsGB } from '../engine/capacity';
import { makeSlot } from '../engine/defaults';
import { bandwidthCeilingTps, decodeTps, prefillRate, requestParts } from '../engine/roofline';
import { slotShape } from '../engine/slot';
import { floorFor } from '../engine/sources';
import { fmtNum, fmtPct, fmtTps } from '../engine/units';
import { settings } from '../state';
import { ChartCard, LineChart, type ExtraPoint, type LineSeries } from '../ui/charts';
import { Field, SelectInput, Toggle } from '../ui/controls';
import { ECON_MODELS, SERVE_GPUS, WORKLOAD_OPTIONS } from '../ui/options';
import { ProvBadge, Val } from '../ui/prov';

const GPU_COLOR: Record<string, string> = {
  pro6000: 'var(--s1)',
  ada6000: 'var(--s2)',
  rtx5090: 'var(--s3)',
  rtx4090: 'var(--s4)',
  l40s: 'var(--s5)',
  h100: 'var(--s6)',
  h200: 'var(--s7)',
  b200: 'var(--s8)',
  v100: 'var(--ink-2)',
};

const fmtSec = (x: number) => (!Number.isFinite(x) ? '—' : x >= 10 ? `${x.toFixed(0)} s` : `${x.toFixed(1)} s`);

export function ThroughputView() {
  const s = settings.value;
  const [modelId, setModelId] = useState('qwen38_27b');
  const [workloadId, setWorkloadId] = useState('observed');
  const [picked, setPicked] = useState<string[]>(['pro6000', 'ada6000', 'rtx5090', 'h100']);
  const [mtp, setMtp] = useState(false);
  const model = MODEL_BY_ID[modelId]!;
  const pro = GPU_BY_ID['pro6000']!;
  const wl = WORKLOAD_BY_ID[workloadId]!;
  const basisInfo = THROUGHPUT_PRESETS.find((t) => t.id === s.throughput)!;

  const floorNow = floorFor(modelId, s);
  const qosNow = s.qos.mode === 'report' ? { mode: 'report' as const, floorTps: s.qos.floorTps } : { mode: 'floor' as const, floorTps: wl.batch ? wl.floorTps.v : floorNow.floor };
  const series: LineSeries[] = [];
  const points: ExtraPoint[] = [];
  const unfit: string[] = [];
  const notes: string[] = [];
  const rows: { gpu: string; c: number; tps: number; user: number; kind: string; src: string }[] = [];
  const ops: { gpu: string; color: string; quant: string; c: number; tps: number; user: number; share: number | null; ttft: number | null; limit: string; bound: string }[] = [];
  const prefill: { gpu: string; color: string; rate: number; seconds: number; burst8k: number; reported: number | null }[] = [];
  let shapeNow = { avgTokens: 0, outShare: 0, cacheHit: 0 };

  for (const id of picked) {
    const gpu = GPU_BY_ID[id]!;
    const fit = chooseQuant(model, gpu, 1, 'auto');
    if (!fit.fits) {
      unfit.push(`${gpu.short}: ${fit.reason}`);
      continue;
    }
    const slot = makeSlot({ gpuId: id, modelId, ownership: 'rent', workloadId });
    const sh = slotShape(slot, s);
    const ob = slotShape({ ...slot, workloadId: 'observed' }, s);
    if (!sh || !ob) continue;
    shapeNow = { avgTokens: sh.avgTokens, outShare: sh.outShare, cacheHit: sh.cacheHit };
    const obsNow = { avgTokens: ob.avgTokens, outShare: ob.outShare, cacheHit: ob.cacheHit };
    const color = GPU_COLOR[id] ?? 'var(--ink-2)';
    const cap = capacityFor({
      model,
      gpu,
      proGpu: pro,
      count: 1,
      quant: 'auto',
      mtp,
      qos: { ...qosNow, ttftP90Sec: s.qos.ttftP90Sec },
      extraMult: s.whatIf.throughput,
      shape: shapeNow,
      baseShape: obsNow,
      basis: s.throughput,
      batch: wl.batch,
      sweepSteps: 36,
    });
    if (!cap.ok) {
      unfit.push(`${gpu.short}: ${cap.reason}`);
      continue;
    }
    if (cap.basisNote) notes.push(`${gpu.short}: ${cap.basisNote}`);
    series.push({ id, label: `${gpu.short} · ${QUANT_LABEL[fit.quant]}`, color, pts: cap.sweep.map((p) => ({ x: p.user, y: p.tps })), width: 2 });
    if (cap.curve) {
      const info = mtp ? mtpInfo(model, curveFor(model, pro, chooseQuant(model, pro, 1, model.pointsQuant.pro6000).quant, pro)) : null;
      cap.curve.pts.forEach((p, i) => {
        const kind = cap.curve!.baseProvs[i]?.kind ?? 'derived';
        const g = mtp && info ? 1 + (info.s1 - 1) * (Number.isFinite(info.cSat) ? Math.max(0, 1 - p.c / info.cSat) : 1) : 1;
        const tps = p.tps * g * s.whatIf.throughput;
        points.push({
          x: tps / p.c,
          y: tps,
          color,
          label: gpu.short,
          hollow: kind !== 'measured',
          tipTitle: `${gpu.short}: the report's point at ${p.c} concurrent`,
          tip: [
            { label: 'Output', value: `${fmtNum(tps)} tok/s` },
            { label: 'Per user', value: `${fmtTps(tps / p.c)} tok/s` },
            { label: 'Source', value: `${kind === 'measured' ? 'measured' : kind === 'report-est' ? 'report estimate' : kind}${cap.curve!.source === 'scaled' ? ', scaled from PRO 6000' : ''}` },
          ],
        });
        rows.push({ gpu: gpu.short, c: p.c, tps, user: tps / p.c, kind, src: cap.curve!.baseProvs[i]?.src ?? '' });
      });
    }
    if (mtp && id === 'pro6000' && model.mtp?.measured) {
      const m = model.mtp.measured;
      points.push({ x: m.tps.v / m.c, y: m.tps.v, color, label: 'MTP measured', tipTitle: 'Measured with MTP', tip: [{ label: 'Output', value: `${fmtNum(m.tps.v)} tok/s` }, { label: 'Concurrent', value: String(m.c) }, { label: 'Source', value: 'measured' }] });
    }
    points.push({
      x: cap.userTps,
      y: cap.tpsNode,
      color,
      label: gpu.short,
      ring: true,
      hollow: true,
      tipTitle: `${gpu.short} operating point`,
      tip: [
        { label: 'Output', value: `${fmtNum(cap.tpsNode)} tok/s` },
        { label: 'Concurrent', value: fmtNum(cap.cNode, 0) },
        { label: 'Per user', value: `${fmtTps(cap.userTps)} tok/s` },
        ...(cap.serving ? [{ label: 'Prefill share', value: fmtPct(cap.serving.share) }, { label: 'First token, p90', value: fmtSec(cap.serving.ttftP90) }] : []),
      ],
    });
    const limit = cap.belowFloor ? (cap.ttftLimited ? 'first-token time (cannot meet)' : 'speed floor (cannot meet)') : cap.kvLimited ? 'KV memory' : cap.ttftLimited ? 'first-token time' : s.qos.mode === 'report' ? 'report operating point' : cap.extrapolated ? 'end of curve' : 'speed floor';
    ops.push({ gpu: gpu.short, color, quant: QUANT_LABEL[fit.quant], c: cap.cNode, tps: cap.tpsNode, user: cap.userTps, share: cap.serving?.share ?? null, ttft: cap.serving?.ttftP90 ?? null, limit, bound: cap.bound ?? '—' });

    const w = weightsGB(model, fit.quant)!;
    const parts = requestParts(shapeNow);
    const pf = prefillRate({ model, gpu, quant: fit.quant, count: 1, weightsGB: w.gb, ctx: parts.ctx }, parts.input);
    const p8 = prefillRate({ model, gpu, quant: fit.quant, count: 1, weightsGB: w.gb, ctx: 8192 }, 8192);
    prefill.push({ gpu: gpu.short, color, rate: pf.tokPerSec, seconds: parts.fresh / pf.tokPerSec, burst8k: 10 / p8.seconds, reported: id === 'pro6000' ? (model.conc8k?.v ?? null) : null });
  }

  const info = model.mtp ? mtpInfo(model, curveFor(model, pro, chooseQuant(model, pro, 1, model.pointsQuant.pro6000).quant, pro)) : null;
  const xs = series.flatMap((x) => x.pts.map((p) => p.x));
  const xd: [number, number] = xs.length ? [Math.max(1, Math.min(...xs) * 0.85), Math.max(...xs) * 1.1] : [1, 100];

  // Roofline against the report's own points, on the cards the report measured, at the report's test context.
  const checks: { gpu: string; quant: string; c: number; report: number; kind: string; src: string; roof: number; ceiling: number }[] = [];
  for (const gid of ['pro6000', 'ada6000'] as const) {
    const gpu = GPU_BY_ID[gid]!;
    const pts = gid === 'pro6000' ? model.pro6000 : model.ada;
    const q = model.pointsQuant[gid === 'pro6000' ? 'pro6000' : 'ada'];
    const w = weightsGB(model, q);
    if (!w) continue;
    for (const p of pts) {
      const i = { model, gpu, quant: q, count: 1, weightsGB: w.gb, ctx: REPORT_TEST_CTX.v };
      checks.push({ gpu: gpu.short, quant: QUANT_LABEL[q], c: p.c, report: p.tps.v, kind: p.tps.p.kind, src: p.tps.p.src, roof: decodeTps(i, p.c), ceiling: bandwidthCeilingTps(i, p.c) });
    }
  }
  const shapeParts = requestParts(shapeNow);

  return (
    <div class="stack">
      <div class="notice">
        <b>{basisInfo.label}.</b> {basisInfo.help} Filled dots are the report's measurements, hollow dots its estimates, and the ring is the operating point under the current quality bar. Change the basis in the filter row above.
      </div>
      <div class="filters" role="region" aria-label="Throughput inputs">
        <Field label="Model">
          <SelectInput
            label="Model"
            value={modelId}
            options={ECON_MODELS.map((m) => ({ id: m.id, label: m.name }))}
            onChange={(v) => {
              setModelId(v);
              if (!MODEL_BY_ID[v]?.mtp) setMtp(false);
            }}
          />
        </Field>
        <Field label="Request">
          <SelectInput label="Workload preset" value={workloadId} options={WORKLOAD_OPTIONS} onChange={setWorkloadId} />
        </Field>
        <Toggle checked={mtp} onChange={setMtp} title={model.mtp?.note}>
          {model.mtp ? 'MTP on' : 'No MTP evidence for this model'}
        </Toggle>
        <div class="field">
          <span class="label">GPUs</span>
          <div class="row" role="group" aria-label="GPUs to plot">
            {SERVE_GPUS.map((g) => {
              const on = picked.includes(g.id);
              return (
                <button key={g.id} type="button" class="btn" aria-pressed={on} style={{ borderColor: on ? GPU_COLOR[g.id] : undefined, borderWidth: on ? '2px' : undefined }} onClick={() => setPicked(on ? picked.filter((x) => x !== g.id) : picked.length < 8 ? [...picked, g.id] : picked)}>
                  <span style={{ display: 'inline-block', width: '10px', height: '3px', background: GPU_COLOR[g.id], marginRight: '6px', verticalAlign: 'middle' }} />
                  {g.short}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      <p class="small">
        Request: {fmtNum(shapeNow.avgTokens)} tokens on average, {fmtPct(shapeNow.outShare, 1)} of them output, {fmtPct(shapeNow.cacheHit)} of the prompt read from the prefix cache.
      </p>
      <ChartCard
        title={`${model.name}: throughput against speed per user`}
        subtitle={s.qos.mode === 'report' ? "Quality bar: the report's convention (the operating point behind its busy-hour revenue)." : s.qos.mode === 'router' && floorNow.v ? `Quality bar: the median OpenRouter stream for this model, ${fmtTps(floorNow.floor)} tok/s per user.` : `Quality bar: at least ${fmtTps(floorNow.floor)} tok/s per user.`}
        legend={
          <div class="legend" aria-label="GPU legend">
            {series.map((x) => (
              <span key={x.id} style={{ display: 'inline-flex', gap: '6px', alignItems: 'center' }}>
                <i style={{ display: 'inline-block', width: '12px', height: '3px', borderRadius: '2px', background: x.color }} />
                {x.label}
              </span>
            ))}
          </div>
        }
        chart={
          series.length ? (
            <LineChart
              ariaLabel={`Aggregate output tokens per second against tokens per second per user for ${model.name}`}
              series={series}
              points={points}
              xScale="log"
              xDomain={xd}
              xFmt={(x) => fmtNum(x)}
              yFmt={(y) => fmtNum(y)}
              xLabel="Tokens per second per user (log scale)"
              yLabel="Output tokens per second"
              hover="nearest"
              height={340}
              vrefs={s.qos.mode === 'report' ? [] : [{ x: qosNow.floorTps, label: 'floor' }]}
              xTicks={[1, 2, 5, 10, 20, 50, 100, 200, 500].filter((t) => t >= xd[0] * 0.95 && t <= xd[1] * 1.05)}
            />
          ) : (
            <p class="small">No selected GPU can serve this model.</p>
          )
        }
        table={
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th class="num">Concurrent</th>
                <th class="num">Output tok/s</th>
                <th class="num">Per user tok/s</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.gpu}</td>
                  <td class="num">{r.c}</td>
                  <td class="num">{fmtNum(r.tps)}</td>
                  <td class="num">{fmtTps(r.user)}</td>
                  <td>
                    <ProvBadge p={{ kind: r.kind as 'measured', src: r.src }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      />
      {unfit.length ? <div class="warnbox">{unfit.join(' · ')}</div> : null}
      {notes.length ? <div class="warnbox">{notes.join(' · ')}</div> : null}

      <section class="card" aria-label="Operating points">
        <h2>Operating point on each card</h2>
        <div class="sub">What the current quality bar allows: how many requests run at once, what each user sees, how much GPU time prefill takes and how long the slowest tenth wait for a first token.</div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th>Weights</th>
                <th class="num">Concurrent</th>
                <th class="num">Output tok/s</th>
                <th class="num">Per user</th>
                <th class="num">Prefill share</th>
                <th class="num">First token p90</th>
                <th>Held back by</th>
                <th>Step is</th>
              </tr>
            </thead>
            <tbody>
              {ops.map((o) => (
                <tr key={o.gpu}>
                  <td>
                    <i style={{ display: 'inline-block', width: '10px', height: '3px', background: o.color, marginRight: '6px', verticalAlign: 'middle' }} />
                    {o.gpu}
                  </td>
                  <td>{o.quant}</td>
                  <td class="num">{fmtNum(o.c, 0)}</td>
                  <td class="num">{fmtNum(o.tps)}</td>
                  <td class="num">{fmtTps(o.user)}</td>
                  <td class="num">{o.share === null ? '—' : fmtPct(o.share)}</td>
                  <td class="num">{o.ttft === null ? '—' : fmtSec(o.ttft)}</td>
                  <td>{o.limit}</td>
                  <td>{o.bound === 'memory' ? 'memory-bound' : o.bound === 'compute' ? 'compute-bound' : o.bound === 'comm' ? 'link-bound' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {s.throughput === 'report' ? <p class="small mt">On the report's own numbers prefill and first-token time are not modelled; switch the basis to see them.</p> : null}
      </section>

      <section class="card" aria-label="Roofline against the report">
        <h2>Roofline against the report's points for {model.name}</h2>
        <div class="sub">
          The roofline reads the weights and cache once per step at {fmtNum(REPORT_TEST_CTX.v)} tokens of context <ProvBadge p={REPORT_TEST_CTX.p} compact />. Its memory efficiency was fitted to three of the report's points (Qwen3.8-27B at 16 concurrent, measured; gpt-oss-120b and gpt-oss-20b at 50); every other row is an independent check. The ceiling is the memory bus at full bandwidth with nothing but the weights to read.
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th>Weights</th>
                <th class="num">Concurrent</th>
                <th class="num">Report tok/s</th>
                <th>Report source</th>
                <th class="num">Roofline tok/s</th>
                <th class="num">Roofline ÷ report</th>
                <th class="num">Bus ceiling</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c, i) => (
                <tr key={i}>
                  <td>{c.gpu}</td>
                  <td>{c.quant}</td>
                  <td class="num">{c.c}</td>
                  <td class="num">{fmtNum(c.report)}</td>
                  <td>
                    <ProvBadge p={{ kind: c.kind as 'measured', src: c.src }} />
                  </td>
                  <td class="num">{fmtNum(c.roof)}</td>
                  <td class="num">{(c.roof / c.report).toFixed(2)}</td>
                  <td class="num">
                    {fmtNum(c.ceiling)}
                    {c.report > c.ceiling ? <b title="The report's figure is higher than the memory bus can deliver at full bandwidth with no cache to read"> ⚠ report is above it</b> : null}
                  </td>
                </tr>
              ))}
              {checks.length === 0 ? (
                <tr>
                  <td colSpan={8} class="small">
                    The report has no throughput points for this model.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section class="card" aria-label="Prefill">
        <h2>Prefill and time to first token</h2>
        <div class="sub">
          Prefill pushes the prompt through the tensor cores, so it is bound by compute rather than memory. This request brings {fmtNum(shapeParts.fresh)} fresh prompt tokens after the prefix cache. The last two columns line the roofline up with the report's measured concurrency at 8K context with first token under 10 s, a burst of requests arriving together.
        </div>
        <div class="tablewrap">
          <table class="t">
            <thead>
              <tr>
                <th>GPU</th>
                <th class="num">Prefill tok/s</th>
                <th class="num">Fresh prompt takes</th>
                <th class="num">Burst of 8K prompts in 10 s</th>
                <th class="num">Report measured</th>
              </tr>
            </thead>
            <tbody>
              {prefill.map((p) => (
                <tr key={p.gpu}>
                  <td>
                    <i style={{ display: 'inline-block', width: '10px', height: '3px', background: p.color, marginRight: '6px', verticalAlign: 'middle' }} />
                    {p.gpu}
                  </td>
                  <td class="num">{fmtNum(p.rate)}</td>
                  <td class="num">{fmtSec(p.seconds)}</td>
                  <td class="num">{fmtNum(p.burst8k, 0)}</td>
                  <td class="num">{p.reported === null ? '—' : fmtNum(p.reported, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p class="small mt">Prefill efficiency (share of peak tensor throughput) is a placeholder: no measurement fixes it. Across the models the report measured, the roofline's burst lands between a third and one-and-a-third times the report's concurrency.</p>
      </section>

      {model.mtp ? (
        <section class="card" aria-label="Speculative decoding">
          <h2>Speculative decoding (MTP) for {model.name}</h2>
          <div class="sub">
            Gain at one user is acceptance length ÷ (1 + draft steps × {DRAFT_COST.v} cost per draft step) <ProvBadge p={DRAFT_COST.p} compact />, fading to nothing as concurrency makes decode compute-bound.
          </div>
          <div class="kv" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
            <div>
              <span>Acceptance length used</span>
              <b>
                <Val v={model.mtp.al} fmt={(x) => `${x} at k=${model.mtp!.k}`} />
              </b>
            </div>
            <div>
              <span>Gain at one user</span>
              <b>{info ? `${info.s1.toFixed(2)}×` : '—'}</b>
            </div>
            <div>
              <span>Stops helping near</span>
              <b>{info && Number.isFinite(info.cSat) ? `${Math.round(info.cSat)} concurrent` : '—'}</b>
            </div>
            <div>
              <span>Calibration</span>
              <b>{info?.calibrated ? 'to the measured point' : 'placeholder'}</b>
            </div>
          </div>
          <p class="small mt">{model.mtp.note}</p>
          <h3 class="mt">InferenceX measured acceptance length, Qwen3.8-Flash-Next (SPEED-Bench coding, thinking on)</h3>
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Draft tokens</th>
                  {GOLDEN_AL_QWEN38_FLASH_NEXT.map((g) => (
                    <th key={g.k} class="num">
                      {g.k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Acceptance length</td>
                  {GOLDEN_AL_QWEN38_FLASH_NEXT.map((g) => (
                    <td key={g.k} class="num">
                      {g.al.v.toFixed(2)} <ProvBadge p={g.al.p} compact />
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
