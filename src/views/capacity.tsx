import { useState } from 'preact/hooks';
import { GPU_BY_ID, GPUS, QUANT_LABEL, type Quant } from '../data/gpus';
import { MODELS, MODEL_BY_ID } from '../data/models';
import { ARCH_SNAPSHOT_AT } from '../data/arch';
import { GPU_MEM_UTIL, OVERHEAD_GB_PER_GPU, chooseQuant, kvUsersAt, MIN_KV_TOKENS, seqKvBytes } from '../engine/capacity';
import { fmtNum, fmtTokens } from '../engine/units';
import { ChartCard, StackedBar } from '../ui/charts';
import { Field, SelectInput, Segmented } from '../ui/controls';
import { COUNT_OPTIONS, QUANT_OPTIONS, SERVE_GPUS } from '../ui/options';
import { ProvBadge, Val } from '../ui/prov';

const CTX = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
const ctxLabel = (c: number) => `${c / 1024}K`;

export function CapacityView() {
  const [gpuId, setGpuId] = useState('pro6000');
  const [count, setCount] = useState('1');
  const [modelId, setModelId] = useState('qwen38_27b');
  const [quant, setQuant] = useState<Quant | 'auto'>('auto');
  const [kv, setKv] = useState<'fp8' | 'bf16'>('fp8');
  const [ctx, setCtx] = useState(16384);

  const gpu = GPU_BY_ID[gpuId]!;
  const model = MODEL_BY_ID[modelId]!;
  const n = Number(count);
  const kvFactor = kv === 'bf16' ? 2 : 1;
  const fit = chooseQuant(model, gpu, n, quant);
  const kvBytes = model.kvBytesPerTok.v * kvFactor;
  const usersAtCtx = fit.fits || fit.kvPoolGB > 0 ? kvUsersAt(model, fit, ctx, kvFactor) : 0;
  const kvTokens = usersAtCtx * ctx;
  const net = model.net;
  const perUserFixed = net ? seqKvBytes(model, 0, kvFactor) + (net.window ? net.slidingTokBytes * net.window * kvFactor : 0) : 0;
  const totalVram = gpu.vramGB.v * n;
  const unused = totalVram * (1 - GPU_MEM_UTIL.v);

  const segments = [
    { id: 'w', label: 'Weights', value: fit.weightsGB, color: 'var(--s1)', valueLabel: `${fit.weightsGB.toFixed(1)} GB` },
    { id: 'o', label: 'Overhead', value: fit.overheadGB, color: 'var(--s2)', valueLabel: `${fit.overheadGB.toFixed(0)} GB` },
    { id: 'k', label: 'KV pool', value: fit.kvPoolGB, color: 'var(--s3)', valueLabel: `${fit.kvPoolGB.toFixed(1)} GB` },
    { id: 'u', label: 'Held back', value: unused, color: 'var(--axis)', valueLabel: `${unused.toFixed(0)} GB` },
  ].filter((s) => s.value > 0);

  const rowsForModel = SERVE_GPUS.flatMap((g) => [1, 2, 4, 8].map((c) => ({ g, c, f: chooseQuant(model, g, c, quant) }))).filter((r) => r.c === 1 || r.f.fits);
  const rowsForGpu = MODELS.map((m) => ({ m, f: chooseQuant(m, gpu, n, quant) }));

  return (
    <div class="stack">
      <div class="filters" role="region" aria-label="Capacity inputs">
        <Field label="GPU">
          <SelectInput label="GPU" value={gpuId} options={SERVE_GPUS.map((g) => ({ id: g.id, label: g.short }))} onChange={setGpuId} />
        </Field>
        <Field label="GPUs in the node">
          <SelectInput label="GPU count" value={count} options={COUNT_OPTIONS} onChange={setCount} />
        </Field>
        <Field label="Model">
          <SelectInput label="Model" value={modelId} options={MODELS.map((m) => ({ id: m.id, label: m.name }))} onChange={setModelId} />
        </Field>
        <Field label="Weights">
          <SelectInput label="Weight format" value={quant} options={QUANT_OPTIONS} onChange={setQuant} />
        </Field>
        <Field label="KV cache format">
          <Segmented label="KV format" value={kv} options={[{ id: 'fp8', label: 'FP8' }, { id: 'bf16', label: 'BF16 (2×)' }]} onChange={setKv} />
        </Field>
        <Field label="Context per user">
          <SelectInput label="Context per user" value={String(ctx)} options={CTX.map((c) => ({ id: String(c), label: ctxLabel(c) }))} onChange={(v) => setCtx(Number(v))} />
        </Field>
      </div>

      <div class="grid cols-2">
        <ChartCard
          title="Where the memory goes"
          subtitle={`${gpu.short}${n > 1 ? ` ×${n}` : ''}: ${totalVram.toFixed(0)} GB in total. The serving engine may use ${(GPU_MEM_UTIL.v * 100).toFixed(0)}% of it.`}
          chart={<StackedBar segments={segments} total={totalVram} totalLabel={`${totalVram.toFixed(0)} GB`} ariaLabel={`Memory split for ${model.name} on ${gpu.short}`} />}
          table={
            <table class="t">
              <thead>
                <tr>
                  <th>Part</th>
                  <th class="num">GB</th>
                </tr>
              </thead>
              <tbody>
                {segments.map((s) => (
                  <tr key={s.id}>
                    <td>{s.label}</td>
                    <td class="num">{s.value.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
        <section class="card" aria-label="What fits">
          <h2>{fit.fits ? `Fits as ${QUANT_LABEL[fit.quant]}` : 'Does not fit'}</h2>
          <div class="sub">{fit.reason}</div>
          <div class="kv" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
            <div>
              <span>Context tokens that fit</span>
              <b>{fmtTokens(kvTokens)}</b>
            </div>
            <div>
              <span>Users at {ctxLabel(ctx)} each</span>
              <b>{fmtNum(usersAtCtx, usersAtCtx < 10 ? 1 : 0)}</b>
            </div>
            <div>
              <span>KV bytes per token</span>
              <b>
                {(kvBytes / 1000).toFixed(1)} KB <ProvBadge p={model.kvBytesPerTok.p} compact />
              </b>
            </div>
            {net && perUserFixed > 0 ? (
              <div>
                <span>Fixed per user (window and state)</span>
                <b>{(perUserFixed / 1e6).toFixed(0)} MB</b>
              </div>
            ) : null}
            <div>
              <span>Weights</span>
              <b>
                {fit.weightsGB.toFixed(1)} GB <ProvBadge p={fit.weightsProv} compact />
              </b>
            </div>
            {model.conc8k ? (
              <div>
                <span>Measured at 8K on a PRO 6000</span>
                <b>
                  <Val v={model.conc8k} fmt={(x) => `${x} users`} />
                </b>
              </div>
            ) : null}
            {model.conc32k ? (
              <div>
                <span>Measured at 32K on a PRO 6000</span>
                <b>
                  <Val v={model.conc32k} fmt={(x) => `${x} users`} />
                </b>
              </div>
            ) : null}
          </div>
          <p class="small mt">
            Overhead is {OVERHEAD_GB_PER_GPU.v} GB per GPU and a model counts as fitting only if at least {fmtTokens(MIN_KV_TOKENS.v)} tokens of KV remain. Both are assumptions <ProvBadge p={OVERHEAD_GB_PER_GPU.p} compact />. Prefix sharing between users can stretch KV further; this counts every user's full context. The measured users above are limited by first-token latency, not memory.
          </p>
          {model.note ? <p class="small">{model.note}</p> : null}
        </section>
      </div>

      <div class="grid cols-2">
        <section class="card" aria-label={`${model.name} on every GPU`}>
          <h2>{model.name} on every GPU</h2>
          <div class="sub">Single cards, plus larger nodes only where they make the model fit.</div>
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>GPU</th>
                  <th>Weights</th>
                  <th class="num">KV pool</th>
                  <th class="num">Users at {ctxLabel(ctx)}</th>
                  <th>Fit</th>
                </tr>
              </thead>
              <tbody>
                {rowsForModel.map(({ g, c, f }) => (
                  <tr key={`${g.id}-${c}`}>
                    <td>
                      {g.short}
                      {c > 1 ? ` ×${c}` : ''}
                    </td>
                    <td>{QUANT_LABEL[f.quant]}</td>
                    <td class="num">{f.kvPoolGB.toFixed(1)} GB</td>
                    <td class="num">{fmtNum(kvUsersAt(model, f, ctx, kvFactor), 0)}</td>
                    <td class={f.fits ? '' : 'miss'}>{f.fits ? (f.forced ? 'Fits with host-RAM offload' : 'Fits') : f.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section class="card" aria-label={`Every model on ${gpu.short}`}>
          <h2>
            Every model on {gpu.short}
            {n > 1 ? ` ×${n}` : ''}
          </h2>
          <div class="sub">Weights are the report's file sizes where it lists them, derived from parameter count otherwise.</div>
          <div class="tablewrap">
            <table class="t">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Weights</th>
                  <th class="num">KV pool</th>
                  <th class="num">Users at {ctxLabel(ctx)}</th>
                  <th>Fit</th>
                </tr>
              </thead>
              <tbody>
                {rowsForGpu.map(({ m, f }) => (
                  <tr key={m.id}>
                    <td>{m.name}</td>
                    <td>{QUANT_LABEL[f.quant]}</td>
                    <td class="num">{f.kvPoolGB.toFixed(1)} GB</td>
                    <td class="num">{fmtNum(kvUsersAt(m, f, ctx, kvFactor), 0)}</td>
                    <td class={f.fits ? '' : 'miss'}>{f.fits ? (f.forced ? 'Fits with host-RAM offload' : 'Fits') : f.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
      {net ? (
        <section class="card" aria-label="Architecture">
          <h2>{model.name}: what the config says</h2>
          <div class="sub">
            Read from <a href={net.configUrl} target="_blank" rel="noreferrer">{net.repo}</a> on {ARCH_SNAPSHOT_AT.slice(0, 10)}. Only the layers that keep a full cache grow with context; sliding-window layers stop at their window and linear-attention or Mamba layers keep a fixed state.
          </div>
          <div class="kv" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
            <div>
              <span>Layers</span>
              <b>
                {net.layers} ({[net.layerMix.full ? `${net.layerMix.full} full attention` : '', net.layerMix.sliding ? `${net.layerMix.sliding} sliding ${net.window}` : '', net.layerMix.linear ? `${net.layerMix.linear} linear` : '', net.layerMix.mamba ? `${net.layerMix.mamba} Mamba` : '', net.layerMix.other ? `${net.layerMix.other} expert-only` : ''].filter(Boolean).join(', ')})
              </b>
            </div>
            <div>
              <span>Recurrent state per user</span>
              <b>{net.stateBytes > 0 ? `${(net.stateBytes / 1e6).toFixed(0)} MB` : 'none'}</b>
            </div>
            <div>
              <span>Experts</span>
              <b>{net.moe ? `${net.moe.topK} of ${net.moe.experts}` : 'dense'}</b>
            </div>
            <div>
              <span>Active parameters</span>
              <b>
                <Val v={model.activeB} fmt={(x) => `${x}B`} />
              </b>
            </div>
            <div>
              <span>Speculative-decoding head</span>
              <b>{net.mtpLayers > 0 ? `yes, ${net.mtpLayers} layer${net.mtpLayers > 1 ? 's' : ''}` : 'none in the config'}</b>
            </div>
            <div>
              <span>Maximum context</span>
              <b>{net.maxPositions ? fmtTokens(net.maxPositions) : '—'}</b>
            </div>
          </div>
          {net.note ? <p class="small">{net.note}</p> : null}
        </section>
      ) : null}
      <p class="small">
        Catalog: {GPUS.filter((g) => g.role === 'serve').length} serving GPUs, {MODELS.length} models. KV figures come from each model's Hugging Face config; the report's own users-at-8K and users-at-32K figures are limited by first-token time, not memory, so they are not comparable.
      </p>
    </div>
  );
}
