import { useState } from 'preact/hooks';
import { GPU_BY_ID, QUANT_LABEL } from '../data/gpus';
import { GOLDEN_AL_QWEN38_FLASH_NEXT, MODEL_BY_ID } from '../data/models';
import { DRAFT_COST, MAX_EXTRAPOLATION, capacityFor, mtpGain, mtpInfo, tpsAt, curveFor, chooseQuant } from '../engine/capacity';
import { fmtNum, fmtTps } from '../engine/units';
import { settings } from '../state';
import { ChartCard, LineChart, type ExtraPoint, type LineSeries } from '../ui/charts';
import { Field, SelectInput, Toggle } from '../ui/controls';
import { ECON_MODELS, SERVE_GPUS } from '../ui/options';
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

export function ThroughputView() {
  const s = settings.value;
  const models = ECON_MODELS.filter((m) => m.pro6000.length > 0);
  const [modelId, setModelId] = useState('qwen38_27b');
  const [picked, setPicked] = useState<string[]>(['pro6000', 'ada6000', 'rtx5090', 'h100']);
  const [mtp, setMtp] = useState(false);
  const model = MODEL_BY_ID[modelId]!;
  const pro = GPU_BY_ID['pro6000']!;

  const series: LineSeries[] = [];
  const points: ExtraPoint[] = [];
  const unfit: string[] = [];
  const rows: { gpu: string; c: number; tps: number; user: number; kind: string; src: string }[] = [];
  for (const id of picked) {
    const gpu = GPU_BY_ID[id]!;
    const fit = chooseQuant(model, gpu, 1, 'auto');
    if (!fit.fits) {
      unfit.push(`${gpu.short}: ${fit.reason}`);
      continue;
    }
    const curve = curveFor(model, gpu, fit.quant, pro);
    if (!curve) continue;
    const proFit = chooseQuant(model, pro, 1, model.pointsQuant.pro6000);
    const info = mtp ? mtpInfo(model, curveFor(model, pro, proFit.quant, pro)) : null;
    const color = GPU_COLOR[id] ?? 'var(--ink-2)';
    const cLast = curve.pts[curve.pts.length - 1]!.c;
    const cMax = cLast * MAX_EXTRAPOLATION.v;
    const pts: { x: number; y: number }[] = [];
    const steps = 36;
    for (let i = 0; i <= steps; i++) {
      const c = Math.exp(Math.log(1) + (Math.log(cMax) - Math.log(1)) * (i / steps));
      const g = mtp ? mtpGain(info, c) : 1;
      const tps = tpsAt(curve, c) * g * s.whatIf.throughput;
      pts.push({ x: tps / c, y: tps });
    }
    series.push({ id, label: `${gpu.short} · ${QUANT_LABEL[fit.quant]}`, color, pts, width: 2 });
    curve.pts.forEach((p, i) => {
      const kind = curve.baseProvs[i]?.kind ?? 'derived';
      const g = mtp ? mtpGain(info, p.c) : 1;
      const tps = p.tps * g * s.whatIf.throughput;
      points.push({
        x: tps / p.c,
        y: tps,
        color,
        label: gpu.short,
        hollow: kind !== 'measured',
        tipTitle: `${gpu.short} at ${p.c} concurrent`,
        tip: [
          { label: 'Output', value: `${fmtNum(tps)} tok/s` },
          { label: 'Per user', value: `${fmtTps(tps / p.c)} tok/s` },
          { label: 'Source', value: `${kind === 'measured' ? 'measured' : kind === 'report-est' ? 'report estimate' : kind}${curve.source === 'scaled' ? ', scaled from PRO 6000' : ''}` },
        ],
      });
      rows.push({ gpu: gpu.short, c: p.c, tps, user: tps / p.c, kind, src: curve.baseProvs[i]?.src ?? '' });
    });
    if (mtp && id === 'pro6000' && model.mtp?.measured) {
      const m = model.mtp.measured;
      points.push({ x: m.tps.v / m.c, y: m.tps.v, color, label: 'MTP measured', tipTitle: 'Measured with MTP', tip: [{ label: 'Output', value: `${fmtNum(m.tps.v)} tok/s` }, { label: 'Concurrent', value: String(m.c) }, { label: 'Source', value: 'measured' }] });
    }
    const cap = capacityFor({ model, gpu, proGpu: pro, count: 1, quant: 'auto', mtp, qos: s.qos, extraMult: s.whatIf.throughput, avgContextTokens: 15300 });
    if (cap.ok) {
      points.push({ x: cap.userTps, y: cap.tpsNode, color, label: gpu.short, ring: true, hollow: true, tipTitle: `${gpu.short} operating point`, tip: [{ label: 'Output', value: `${fmtNum(cap.tpsNode)} tok/s` }, { label: 'Concurrent', value: fmtNum(cap.cNode, 0) }, { label: 'Per user', value: `${fmtTps(cap.userTps)} tok/s` }] });
    }
  }

  const info = model.mtp ? mtpInfo(model, curveFor(model, pro, chooseQuant(model, pro, 1, model.pointsQuant.pro6000).quant, pro)) : null;
  const xs = series.flatMap((x) => x.pts.map((p) => p.x));
  const xd: [number, number] = xs.length ? [Math.max(1, Math.min(...xs) * 0.85), Math.max(...xs) * 1.1] : [1, 100];

  return (
    <div class="stack">
      <div class="notice">
        Each curve is a power law fitted through the report's published points for that card. Filled dots are measurements, hollow dots are the report's own estimates, and the ring is the operating point under the current quality bar. Cards other than the PRO 6000 and Ada are scaled, which is a placeholder until phase 3.
      </div>
      <div class="filters" role="region" aria-label="Throughput inputs">
        <Field label="Model">
          <SelectInput label="Model" value={modelId} options={models.map((m) => ({ id: m.id, label: m.name }))} onChange={(v) => { setModelId(v); if (!MODEL_BY_ID[v]?.mtp) setMtp(false); }} />
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
      <ChartCard
        title={`${model.name}: throughput against speed per user`}
        subtitle={s.qos.mode === 'floor' ? `Quality bar: at least ${s.qos.floorTps} tok/s per user.` : "Quality bar: the report's convention (the operating point behind its busy-hour revenue)."}
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
              vrefs={s.qos.mode === 'floor' ? [{ x: s.qos.floorTps, label: 'floor' }] : []}
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
