import type { ComponentChildren } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { extent, max, min } from 'd3-array';
import { scaleBand, scaleLinear, scaleLog } from 'd3-scale';
import { line as d3line, curveMonotoneX } from 'd3-shape';

export function useWidth(): [{ current: HTMLDivElement | null }, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = () => setW(Math.max(260, Math.floor(el.getBoundingClientRect().width)));
    apply();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export interface TipRow {
  color?: string;
  label: string;
  value: string;
}
export interface TipData {
  x: number;
  y: number;
  title?: string;
  rows: TipRow[];
}

export function Tip({ tip, width }: { tip: TipData | null; width: number }) {
  if (!tip) return null;
  const flip = tip.x > width * 0.6;
  return (
    <div class="tooltip" role="status" style={{ left: `${flip ? tip.x - 12 : tip.x + 12}px`, top: `${Math.max(0, tip.y - 8)}px`, transform: flip ? 'translateX(-100%)' : undefined }}>
      {tip.title ? <div class="tt-title">{tip.title}</div> : null}
      {tip.rows.map((r, i) => (
        <div class="tt-row" key={i}>
          {r.color ? <span class="tt-key" style={{ borderColor: r.color }} /> : null}
          <span>{r.label}</span>
          <b>{r.value}</b>
        </div>
      ))}
    </div>
  );
}

/** Card with a chart/table switch. Every chart has a table twin. */
export function ChartCard({
  title,
  subtitle,
  legend,
  chart,
  table,
  actions,
}: {
  title: string;
  subtitle?: ComponentChildren;
  legend?: ComponentChildren;
  chart: ComponentChildren;
  table: ComponentChildren;
  actions?: ComponentChildren;
}) {
  const [mode, setMode] = useState<'chart' | 'table'>('chart');
  return (
    <section class="card" aria-label={title}>
      <div class="chart-head">
        <div>
          <h2>{title}</h2>
          {subtitle ? <div class="sub">{subtitle}</div> : null}
        </div>
        <div class="row">
          {actions}
          <div class="viewtoggle" role="group" aria-label={`${title} view`}>
            <button type="button" aria-pressed={mode === 'chart'} onClick={() => setMode('chart')}>
              Chart
            </button>
            <button type="button" aria-pressed={mode === 'table'} onClick={() => setMode('table')}>
              Table
            </button>
          </div>
        </div>
      </div>
      {mode === 'chart' ? (
        <>
          {legend}
          {chart}
        </>
      ) : (
        <div class="tablewrap">{table}</div>
      )}
    </section>
  );
}

export interface LineSeries {
  id: string;
  label: string;
  color: string;
  pts: { x: number; y: number }[];
  width?: number;
  dim?: boolean;
  /** Draw only markers, no line. */
  markersOnly?: boolean;
  markers?: boolean;
}

export interface ExtraPoint {
  x: number;
  y: number;
  color: string;
  label: string;
  hollow?: boolean;
  ring?: boolean;
  tip: TipRow[];
  tipTitle?: string;
}

interface LineChartProps {
  series: LineSeries[];
  ariaLabel: string;
  xFmt: (x: number) => string;
  yFmt: (y: number) => string;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  xScale?: 'linear' | 'log';
  xDomain?: [number, number];
  yDomain?: [number, number];
  xTicks?: number[];
  zeroLine?: boolean;
  hrefs?: { y: number; label: string }[];
  vrefs?: { x: number; label: string }[];
  points?: ExtraPoint[];
  hover?: 'x' | 'nearest';
  endLabels?: boolean;
  tipTitle?: (x: number) => string;
  curve?: boolean;
}

export function LineChart(p: LineChartProps) {
  const [ref, w] = useWidth();
  const [tip, setTip] = useState<TipData | null>(null);
  const [cursor, setCursor] = useState<{ x: number } | null>(null);
  const h = p.height ?? 280;
  const m = { l: 62, r: p.endLabels ? 96 : 18, t: 12, b: p.xLabel ? 46 : 30 };
  const iw = Math.max(50, w - m.l - m.r);
  const ih = h - m.t - m.b;

  const all = p.series.flatMap((s) => s.pts);
  const extra = p.points ?? [];
  const xs = [...all.map((d) => d.x), ...extra.map((d) => d.x)];
  const ys = [...all.map((d) => d.y), ...extra.map((d) => d.y), ...(p.hrefs?.map((r) => r.y) ?? []), ...(p.zeroLine ? [0] : [])];
  const xd = p.xDomain ?? ((extent(xs) as [number, number]) ?? [0, 1]);
  const yd0 = p.yDomain ?? ((extent(ys) as [number, number]) ?? [0, 1]);
  const pad = (yd0[1] - yd0[0]) * 0.08 || 1;
  const yd: [number, number] = p.yDomain ?? [yd0[0] - (yd0[0] < 0 ? pad : 0), yd0[1] + pad];
  const x = (p.xScale === 'log' ? scaleLog().domain(xd).range([0, iw]) : scaleLinear().domain(xd).range([0, iw])) as ((v: number) => number) & { ticks: (n?: number) => number[] };
  const y = scaleLinear().domain(yd).nice().range([ih, 0]);
  const xt = p.xTicks ?? (x.ticks(Math.max(3, Math.floor(iw / 90))) as number[]);
  const yt = y.ticks(5);
  const gen = d3line<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y((d) => y(d.y));
  if (p.curve) gen.curve(curveMonotoneX);

  const xsAll = Array.from(new Set(all.map((d) => d.x))).sort((a, b) => a - b);

  const onMove = (e: PointerEvent) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = e.clientX - r.left - m.l;
    const py = e.clientY - r.top - m.t;
    if (p.hover === 'nearest') {
      let best: { d: number; item: ExtraPoint | { x: number; y: number; s: LineSeries } } | null = null;
      for (const s of p.series) for (const d of s.pts) {
        const dist = Math.hypot(x(d.x) - px, y(d.y) - py);
        if (!best || dist < best.d) best = { d: dist, item: { x: d.x, y: d.y, s } };
      }
      for (const d of extra) {
        const dist = Math.hypot(x(d.x) - px, y(d.y) - py);
        if (!best || dist < best.d) best = { d: dist, item: d };
      }
      if (best && best.d < 48) {
        const it = best.item;
        if ('s' in it) setTip({ x: x(it.x) + m.l, y: y(it.y) + m.t, title: `${it.s.label}`, rows: [{ label: p.xLabel ?? 'x', value: p.xFmt(it.x) }, { label: p.yLabel ?? 'y', value: p.yFmt(it.y) }] });
        else setTip({ x: x(it.x) + m.l, y: y(it.y) + m.t, title: it.tipTitle ?? it.label, rows: it.tip });
      } else setTip(null);
      return;
    }
    if (xsAll.length === 0) return;
    const xv = (x as unknown as { invert: (v: number) => number }).invert ? (x as unknown as { invert: (v: number) => number }).invert(px) : 0;
    let nearest = xsAll[0]!;
    for (const v of xsAll) if (Math.abs(v - xv) < Math.abs(nearest - xv)) nearest = v;
    setCursor({ x: nearest });
    const rows: TipRow[] = [];
    for (const s of p.series) {
      const d = s.pts.find((q) => q.x === nearest);
      if (d) rows.push({ color: s.color, label: s.label, value: p.yFmt(d.y) });
    }
    setTip({ x: x(nearest) + m.l, y: m.t + 20, title: p.tipTitle ? p.tipTitle(nearest) : p.xFmt(nearest), rows });
  };
  const leave = () => {
    setTip(null);
    setCursor(null);
  };

  return (
    <div class="chart" ref={ref} onPointerLeave={leave}>
      <svg width={w} height={h} role="img" aria-label={p.ariaLabel} onPointerMove={onMove as unknown as (e: Event) => void}>
        <g transform={`translate(${m.l},${m.t})`}>
          {yt.map((t) => (
            <g key={t}>
              <line class="gridline" x1={0} x2={iw} y1={y(t)} y2={y(t)} />
              <text x={-8} y={y(t)} text-anchor="end" dominant-baseline="middle">
                {p.yFmt(t)}
              </text>
            </g>
          ))}
          {p.zeroLine && yd[0] < 0 && yd[1] > 0 ? <line class="axisline" x1={0} x2={iw} y1={y(0)} y2={y(0)} /> : null}
          <line class="axisline" x1={0} x2={iw} y1={ih} y2={ih} />
          {xt.map((t) => (
            <text key={t} x={x(t)} y={ih + 16} text-anchor="middle">
              {p.xFmt(t)}
            </text>
          ))}
          {p.xLabel ? (
            <text x={iw / 2} y={ih + 38} text-anchor="middle">
              {p.xLabel}
            </text>
          ) : null}
          {p.hrefs?.map((r) => (
            <g key={r.label}>
              <line class="axisline" x1={0} x2={iw} y1={y(r.y)} y2={y(r.y)} />
              <text x={iw} y={y(r.y) - 4} text-anchor="end">
                {r.label}
              </text>
            </g>
          ))}
          {p.vrefs?.map((r) => (
            <g key={r.label}>
              <line class="axisline" x1={x(r.x)} x2={x(r.x)} y1={0} y2={ih} />
              <text x={x(r.x) + 4} y={10}>
                {r.label}
              </text>
            </g>
          ))}
          {cursor ? <line class="axisline" x1={x(cursor.x)} x2={x(cursor.x)} y1={0} y2={ih} /> : null}
          {p.series.map((s) => (
            <g key={s.id} opacity={s.dim ? 0.45 : 1}>
              {s.markersOnly ? null : <path d={gen(s.pts) ?? ''} fill="none" stroke={s.color} stroke-width={s.width ?? 2} stroke-linecap="round" stroke-linejoin="round" />}
              {(s.markers || s.markersOnly) && s.pts.map((d, i) => <circle key={i} cx={x(d.x)} cy={y(d.y)} r={4} fill={s.color} stroke="var(--surface)" stroke-width={2} />)}
              {p.endLabels && s.pts.length > 0 ? (
                <>
                  <circle cx={x(s.pts[s.pts.length - 1]!.x)} cy={y(s.pts[s.pts.length - 1]!.y)} r={4} fill={s.color} stroke="var(--surface)" stroke-width={2} />
                  <text x={x(s.pts[s.pts.length - 1]!.x) + 8} y={y(s.pts[s.pts.length - 1]!.y)} dominant-baseline="middle" class="strong">
                    {s.label}
                  </text>
                </>
              ) : null}
            </g>
          ))}
          {extra.map((d, i) => (
            <circle key={i} cx={x(d.x)} cy={y(d.y)} r={d.ring ? 7 : 5} fill={d.hollow ? 'var(--surface)' : d.color} stroke={d.color} stroke-width={d.hollow || d.ring ? 2 : 2} />
          ))}
          {p.hover === 'nearest' ? null : <rect class="hit" x={0} y={0} width={iw} height={ih} />}
        </g>
      </svg>
      <Tip tip={tip} width={w} />
    </div>
  );
}

export interface BarItem {
  id: string;
  label: string;
  sub?: string;
  value: number;
  compare?: number;
  valueLabel: string;
  compareLabel?: string;
  color?: string;
  tip?: TipRow[];
}

export function HBars({ items, ariaLabel, compareName, valueName, fmt }: { items: BarItem[]; ariaLabel: string; compareName: string; valueName: string; fmt: (v: number) => string }) {
  const [ref, w] = useWidth();
  const [tip, setTip] = useState<TipData | null>(null);
  const rowH = 44;
  const labelW = Math.min(230, Math.max(120, w * 0.34));
  const m = { l: labelW, r: 120, t: 6, b: 24 };
  const iw = Math.max(40, w - m.l - m.r);
  const h = m.t + m.b + rowH * items.length;
  const top = max(items.flatMap((i) => [i.value, i.compare ?? 0])) ?? 1;
  const x = scaleLinear().domain([0, top * 1.05 || 1]).range([0, iw]);
  const ticks = x.ticks(Math.max(2, Math.floor(iw / 110)));
  return (
    <div class="chart" ref={ref} onPointerLeave={() => setTip(null)}>
      <svg width={w} height={h} role="img" aria-label={ariaLabel}>
        <g transform={`translate(${m.l},${m.t})`}>
          {ticks.map((t) => (
            <g key={t}>
              <line class="gridline" x1={x(t)} x2={x(t)} y1={0} y2={rowH * items.length} />
              <text x={x(t)} y={rowH * items.length + 16} text-anchor="middle">
                {fmt(t)}
              </text>
            </g>
          ))}
          <line class="axisline" x1={0} x2={0} y1={0} y2={rowH * items.length} />
          {items.map((it, i) => {
            const cy = i * rowH + rowH / 2;
            const bw = Math.max(0, x(it.value));
            const r = Math.min(4, bw / 2);
            return (
              <g
                key={it.id}
                onPointerMove={(e: PointerEvent) => {
                  const rc = ref.current?.getBoundingClientRect();
                  if (rc) setTip({ x: e.clientX - rc.left, y: e.clientY - rc.top, title: it.label, rows: it.tip ?? [{ label: valueName, value: it.valueLabel }] });
                }}
              >
                <text x={-10} y={cy - (it.sub ? 6 : 0)} text-anchor="end" dominant-baseline="middle" class="strong">
                  {it.label}
                </text>
                {it.sub ? (
                  <text x={-10} y={cy + 9} text-anchor="end" dominant-baseline="middle">
                    {it.sub}
                  </text>
                ) : null}
                {bw > 0 ? <path d={`M0,${cy - 10} H${bw - r} a${r},${r} 0 0 1 ${r},${r} V${cy + 10 - r} a${r},${r} 0 0 1 ${-r},${r} H0 Z`} fill={it.color ?? 'var(--s1)'} /> : null}
                {it.compare !== undefined ? <line x1={x(it.compare)} x2={x(it.compare)} y1={cy - 16} y2={cy + 16} stroke="var(--ink)" stroke-width={2} /> : null}
                <text x={Math.max(bw, it.compare !== undefined ? x(it.compare) : 0) + 8} y={cy} dominant-baseline="middle" class="strong">
                  {it.valueLabel}
                </text>
                <rect class="hit" x={-m.l} y={i * rowH} width={w} height={rowH} />
              </g>
            );
          })}
        </g>
      </svg>
      <div class="small" style={{ marginTop: '2px' }}>
        <span style={{ display: 'inline-block', width: '10px', height: '10px', background: 'var(--s1)', borderRadius: '2px', marginRight: '6px' }} />
        {valueName}
        <span style={{ display: 'inline-block', width: '2px', height: '12px', background: 'var(--ink)', margin: '0 6px 0 16px', verticalAlign: 'middle' }} />
        {compareName}
      </div>
      <Tip tip={tip} width={w} />
    </div>
  );
}

export interface Segment {
  id: string;
  label: string;
  value: number;
  color: string;
  valueLabel: string;
}

export function StackedBar({ segments, ariaLabel, total, totalLabel }: { segments: Segment[]; ariaLabel: string; total: number; totalLabel: string }) {
  const [ref, w] = useWidth();
  const [tip, setTip] = useState<TipData | null>(null);
  const h = 54;
  const x = scaleLinear().domain([0, total]).range([0, w]);
  let acc = 0;
  return (
    <div class="chart" ref={ref} onPointerLeave={() => setTip(null)}>
      <div class="legend">
        {segments.map((s) => (
          <span key={s.id} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <i class="sw" style={{ background: s.color, display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px' }} />
            {s.label} <b style={{ color: 'var(--ink)' }}>{s.valueLabel}</b>
          </span>
        ))}
      </div>
      <svg width={w} height={h} role="img" aria-label={ariaLabel}>
        {segments.map((s) => {
          const x0 = x(acc);
          const wd = Math.max(0, x(s.value) - 2);
          acc += s.value;
          const fits = wd > s.valueLabel.length * 7 + 14;
          return (
            <g
              key={s.id}
              onPointerMove={(e: PointerEvent) => {
                const rc = ref.current?.getBoundingClientRect();
                if (rc) setTip({ x: e.clientX - rc.left, y: e.clientY - rc.top - 10, title: s.label, rows: [{ color: s.color, label: 'Size', value: s.valueLabel }] });
              }}
            >
              <rect x={x0} y={8} width={wd} height={30} rx={0} fill={s.color} />
              {fits ? (
                <text x={x0 + wd / 2} y={24} text-anchor="middle" dominant-baseline="middle" style={{ fill: '#fff', fontWeight: 600 }}>
                  {s.valueLabel}
                </text>
              ) : null}
              <rect class="hit" x={x0} y={0} width={Math.max(wd, 6)} height={h} />
            </g>
          );
        })}
        <text x={0} y={h - 4}>
          0
        </text>
        <text x={w} y={h - 4} text-anchor="end">
          {totalLabel}
        </text>
      </svg>
      <Tip tip={tip} width={w} />
    </div>
  );
}

export interface HeatCell {
  row: string;
  col: string;
  /** -1..1 for diverging, 0..1 for sequential; null for no data. */
  t: number | null;
  text: string;
  tip: TipRow[];
  tipTitle: string;
}

export function Heatmap({
  rows,
  cols,
  cells,
  mode,
  ariaLabel,
  selected,
  onSelect,
}: {
  rows: { id: string; label: string; group?: string }[];
  cols: { id: string; label: string }[];
  cells: HeatCell[];
  mode: 'diverging' | 'sequential';
  ariaLabel: string;
  selected?: string | null;
  onSelect?: (row: string, col: string) => void;
}) {
  const [ref, w] = useWidth();
  const [tip, setTip] = useState<TipData | null>(null);
  const labelW = 150;
  const headH = 96;
  const cw = Math.max(54, Math.min(104, (w - labelW - 8) / Math.max(1, cols.length)));
  const rh = 38;
  const h = headH + rows.length * rh + 6;
  const W = labelW + cw * cols.length + 8;
  const map = new Map(cells.map((c) => [`${c.row}|${c.col}`, c]));
  const fill = (t: number | null): string => {
    if (t === null) return 'url(#nodata)';
    if (mode === 'diverging') {
      const a = Math.min(1, Math.abs(t));
      return t >= 0 ? `color-mix(in oklab, var(--div-pos) ${(a * 100).toFixed(0)}%, var(--mid))` : `color-mix(in oklab, var(--div-neg) ${(a * 100).toFixed(0)}%, var(--mid))`;
    }
    return `color-mix(in oklab, var(--seq-3) ${(Math.min(1, Math.max(0, t)) * 100).toFixed(0)}%, var(--seq-0))`;
  };
  const textFill = (t: number | null): string => {
    if (t === null) return 'var(--muted)';
    const a = mode === 'diverging' ? Math.abs(t) : t;
    return a > 0.55 ? '#ffffff' : 'var(--ink)';
  };
  return (
    <div class="chart" ref={ref} onPointerLeave={() => setTip(null)} style={{ overflowX: 'auto' }}>
      <svg width={Math.max(w, W)} height={h} role="grid" aria-label={ariaLabel}>
        <defs>
          <pattern id="nodata" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="8" height="8" fill="var(--surface-2)" />
            <line x1="0" y1="0" x2="0" y2="8" stroke="var(--axis)" stroke-width="2" />
          </pattern>
        </defs>
        {cols.map((c, j) => (
          <text key={c.id} transform={`translate(${labelW + j * cw + cw / 2},${headH - 8}) rotate(-38)`} text-anchor="start" class="strong" style={{ fontSize: '11px' }}>
            {c.label}
          </text>
        ))}
        {rows.map((r, i) => (
          <g key={r.id}>
            <text x={labelW - 8} y={headH + i * rh + rh / 2} text-anchor="end" dominant-baseline="middle" class="strong">
              {r.label}
            </text>
            {cols.map((c, j) => {
              const cell = map.get(`${r.id}|${c.id}`);
              const t = cell ? cell.t : null;
              const key = `${r.id}|${c.id}`;
              const sel = selected === key;
              return (
                <g
                  key={c.id}
                  class="heat-cell"
                  tabIndex={0}
                  role="gridcell"
                  aria-label={cell ? `${cell.tipTitle}: ${cell.text}` : `${r.label}, ${c.label}: no data`}
                  onClick={() => onSelect?.(r.id, c.id)}
                  onKeyDown={(e: KeyboardEvent) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelect?.(r.id, c.id);
                    }
                  }}
                  onPointerMove={(e: PointerEvent) => {
                    const rc = ref.current?.getBoundingClientRect();
                    if (rc && cell) setTip({ x: e.clientX - rc.left + (ref.current?.scrollLeft ?? 0), y: e.clientY - rc.top, title: cell.tipTitle, rows: cell.tip });
                  }}
                  onFocus={() => {
                    if (cell) setTip({ x: labelW + j * cw + cw / 2, y: headH + i * rh, title: cell.tipTitle, rows: cell.tip });
                  }}
                  onBlur={() => setTip(null)}
                >
                  <rect class="cellbg" x={labelW + j * cw + 1} y={headH + i * rh + 1} width={cw - 2} height={rh - 2} rx={4} fill={fill(t)} stroke={sel ? 'var(--ink)' : 'none'} stroke-width={sel ? 2 : 0} />
                  <text x={labelW + j * cw + cw / 2} y={headH + i * rh + rh / 2} text-anchor="middle" dominant-baseline="middle" style={{ fill: textFill(t), fontWeight: 600 }}>
                    {cell ? cell.text : '—'}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <Tip tip={tip} width={Math.max(w, W)} />
    </div>
  );
}

export interface TornadoItem {
  id: string;
  label: string;
  lowLabel: string;
  highLabel: string;
  low: number;
  high: number;
}

export function Tornado({ items, ariaLabel, fmt }: { items: TornadoItem[]; ariaLabel: string; fmt: (v: number) => string }) {
  const [ref, w] = useWidth();
  const [tip, setTip] = useState<TipData | null>(null);
  const rowH = 40;
  const labelW = Math.min(210, Math.max(130, w * 0.3));
  const m = { l: labelW, r: 70, t: 6, b: 26 };
  const iw = Math.max(60, w - m.l - m.r);
  const h = m.t + m.b + rowH * items.length;
  const lo = min(items.flatMap((i) => [i.low, i.high, 0])) ?? 0;
  const hi = max(items.flatMap((i) => [i.low, i.high, 0])) ?? 1;
  const padSpan = (hi - lo) * 0.16;
  const x = scaleLinear().domain([lo - padSpan, hi + padSpan]).nice().range([0, iw]);
  const ticks = x.ticks(Math.max(2, Math.floor(iw / 110)));
  const col = (v: number) => (v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)');
  const bar = (v: number, cy: number) => {
    const x0 = x(0);
    const x1 = x(v);
    const left = Math.min(x0, x1);
    const wd = Math.abs(x1 - x0);
    const r = Math.min(4, wd / 2);
    const dirRight = x1 >= x0;
    const d = dirRight
      ? `M${left},${cy - 10} H${left + wd - r} a${r},${r} 0 0 1 ${r},${r} V${cy + 10 - r} a${r},${r} 0 0 1 ${-r},${r} H${left} Z`
      : `M${left + wd},${cy - 10} H${left + r} a${r},${r} 0 0 0 ${-r},${r} V${cy + 10 - r} a${r},${r} 0 0 0 ${r},${r} H${left + wd} Z`;
    return wd > 0.5 ? <path d={d} fill={col(v)} /> : null;
  };
  return (
    <div class="chart" ref={ref} onPointerLeave={() => setTip(null)}>
      <svg width={w} height={h} role="img" aria-label={ariaLabel}>
        <g transform={`translate(${m.l},${m.t})`}>
          {ticks.map((t) => (
            <g key={t}>
              <line class="gridline" x1={x(t)} x2={x(t)} y1={0} y2={rowH * items.length} />
              <text x={x(t)} y={rowH * items.length + 16} text-anchor="middle">
                {fmt(t)}
              </text>
            </g>
          ))}
          <line class="axisline" x1={x(0)} x2={x(0)} y1={0} y2={rowH * items.length} />
          {items.map((it, i) => {
            const cy = i * rowH + rowH / 2;
            return (
              <g
                key={it.id}
                onPointerMove={(e: PointerEvent) => {
                  const rc = ref.current?.getBoundingClientRect();
                  if (rc)
                    setTip({
                      x: e.clientX - rc.left,
                      y: e.clientY - rc.top,
                      title: it.label,
                      rows: [
                        { label: `If ${it.lowLabel}`, value: fmt(it.low) },
                        { label: `If ${it.highLabel}`, value: fmt(it.high) },
                      ],
                    });
                }}
              >
                <text x={-10} y={cy} text-anchor="end" dominant-baseline="middle" class="strong">
                  {it.label}
                </text>
                {bar(it.low, cy)}
                {bar(it.high, cy)}
                <text x={x(it.low) + (it.low < 0 ? -6 : 6)} y={cy} text-anchor={it.low < 0 ? 'end' : 'start'} dominant-baseline="middle">
                  {it.lowLabel}
                </text>
                <text x={x(it.high) + (it.high < 0 ? -6 : 6)} y={cy} text-anchor={it.high < 0 ? 'end' : 'start'} dominant-baseline="middle">
                  {it.highLabel}
                </text>
                <rect class="hit" x={-m.l} y={i * rowH} width={w} height={rowH} />
              </g>
            );
          })}
        </g>
      </svg>
      <Tip tip={tip} width={w} />
    </div>
  );
}

export { scaleBand };
