import { signal } from '@preact/signals';
import { KIND_GLYPH, KIND_HELP, KIND_LABEL, KIND_ORDER, type Prov, type ProvKind, type V } from '../data/provenance';

interface PopState {
  x: number;
  y: number;
  title: string;
  lines: string[];
}

const pop = signal<PopState | null>(null);

export function Popover() {
  const p = pop.value;
  if (!p) return null;
  const left = Math.max(8, Math.min(p.x, window.innerWidth - 336));
  const top = p.y + 18 > window.innerHeight - 140 ? Math.max(8, p.y - 110) : p.y + 18;
  return (
    <div class="popover" role="tooltip" style={{ left: `${left}px`, top: `${top}px` }}>
      <b>{p.title}</b>
      {p.lines.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
    </div>
  );
}

function show(el: Element, title: string, lines: string[]) {
  const r = el.getBoundingClientRect();
  pop.value = { x: r.left, y: r.top, title, lines };
}

const hide = () => {
  pop.value = null;
};

function describe(p: Prov): { title: string; lines: string[] } {
  const lines = [p.src];
  if (p.note) lines.push(p.note);
  if (p.asOf) lines.push(`As of ${p.asOf}`);
  lines.push(KIND_HELP[p.kind]);
  return { title: KIND_LABEL[p.kind], lines };
}

/** Marks where a number comes from. Placeholders are loud on purpose. */
export function ProvBadge({ p, compact = false }: { p: Prov; compact?: boolean }) {
  const d = describe(p);
  const label = `${KIND_LABEL[p.kind]}. ${p.src}${p.note ? `. ${p.note}` : ''}`;
  return (
    <span
      class={`prov${compact ? ' compact' : ''}`}
      data-kind={p.kind}
      tabIndex={0}
      role="note"
      aria-label={label}
      onMouseEnter={(e) => show(e.currentTarget, d.title, d.lines)}
      onFocus={(e) => show(e.currentTarget, d.title, d.lines)}
      onMouseLeave={hide}
      onBlur={hide}
    >
      <span aria-hidden="true">{KIND_GLYPH[p.kind]}</span>
      <span class="txt">{p.kind === 'report-est' ? 'Est.' : KIND_LABEL[p.kind]}</span>
    </span>
  );
}

const WEAKNESS: ProvKind[] = ['placeholder', 'report-est', 'derived', 'report', 'sourced', 'measured', 'live', 'user'];

export function weakest(provs: Prov[]): Prov | null {
  let best: Prov | null = null;
  let bestRank = Infinity;
  for (const p of provs) {
    const r = WEAKNESS.indexOf(p.kind);
    if (r < bestRank) {
      best = p;
      bestRank = r;
    }
  }
  return best;
}

/** One badge summarising a list of inputs by their weakest member. */
export function ProvSummary({ provs, compact = false }: { provs: Prov[]; compact?: boolean }) {
  const w = weakest(provs);
  if (!w) return null;
  const counts = Object.fromEntries(KIND_ORDER.map((k) => [k, 0])) as Record<ProvKind, number>;
  provs.forEach((p) => (counts[p.kind] += 1));
  const lines = KIND_ORDER.filter((k) => counts[k] > 0).map((k) => `${counts[k]} ${KIND_LABEL[k].toLowerCase()}`);
  const d = { title: `Weakest input: ${KIND_LABEL[w.kind].toLowerCase()}`, lines: [lines.join(', '), 'Open the evidence list for each input.'] };
  return (
    <span
      class={`prov${compact ? ' compact' : ''}`}
      data-kind={w.kind}
      tabIndex={0}
      role="note"
      aria-label={`${d.title}. ${d.lines.join('. ')}`}
      onMouseEnter={(e) => show(e.currentTarget, d.title, d.lines)}
      onFocus={(e) => show(e.currentTarget, d.title, d.lines)}
      onMouseLeave={hide}
      onBlur={hide}
    >
      <span aria-hidden="true">{KIND_GLYPH[w.kind]}</span>
      <span class="txt">{w.kind === 'placeholder' ? `${counts.placeholder} placeholder${counts.placeholder === 1 ? '' : 's'}` : w.kind === 'report-est' ? 'Est.' : KIND_LABEL[w.kind]}</span>
    </span>
  );
}

export function Val({ v, fmt, compact = true }: { v: V; fmt?: (x: number) => string; compact?: boolean }) {
  const n = typeof v.v === 'number' ? (fmt ? fmt(v.v) : String(v.v)) : String(v.v);
  return (
    <span>
      {n} <ProvBadge p={v.p} compact={compact} />
    </span>
  );
}

export function KindLegend() {
  return (
    <div class="legend-prov" aria-label="What the provenance badges mean">
      <span>Where numbers come from:</span>
      {KIND_ORDER.filter((k) => k !== 'live' && k !== 'user').map((k) => (
        <span key={k} title={KIND_HELP[k]}>
          <ProvBadge p={{ kind: k, src: KIND_HELP[k] }} />
        </span>
      ))}
    </div>
  );
}
