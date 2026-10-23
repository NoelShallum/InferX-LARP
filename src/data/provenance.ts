/**
 * Every number the dashboard uses is wrapped in V<T> = { v, p } so the UI can show where it came from.
 *
 *  measured     published benchmark measurement (report [HARD] figures, InferenceX runs)
 *  report       a figure stated in the report (snapshot data, agreed rules, quoted prices)
 *  report-est   a figure the report itself flags as an estimate or anecdote ([EST], [ANEC], "Medium")
 *  sourced      public datasheet / web page / repo checked while building this dashboard
 *  derived      computed from other values (the note says how)
 *  placeholder  assumption with no reference yet; to be replaced by live data or the roofline engine
 *  user         entered by the person using the dashboard
 *  live         fetched by a collection script (phase 2), with a timestamp
 */
export type ProvKind =
  | 'measured'
  | 'report'
  | 'report-est'
  | 'sourced'
  | 'derived'
  | 'placeholder'
  | 'user'
  | 'live';

export interface Prov {
  kind: ProvKind;
  src: string;
  note?: string;
  asOf?: string;
  /** What the dashboard is using right now, as text. Stamped on every placeholder so the badge can say it. */
  value?: string;
  /** What the value is measured in, where it is not obvious from the note. */
  unit?: string;
}

export interface V<T = number> {
  v: T;
  p: Prov;
}

export const REPORT_NAME = 'Report: Inference as a Service — Platforms, Models and Margins (22 Oct 2026)';
export const REPORT_DATE = '2026-10-22';

export const KIND_ORDER: ProvKind[] = [
  'placeholder',
  'report-est',
  'derived',
  'report',
  'measured',
  'sourced',
  'live',
  'user',
];

export const KIND_LABEL: Record<ProvKind, string> = {
  measured: 'Measured',
  report: 'Report',
  'report-est': 'Report estimate',
  sourced: 'Sourced',
  derived: 'Derived',
  placeholder: 'Placeholder',
  user: 'Your input',
  live: 'Live',
};

export const KIND_GLYPH: Record<ProvKind, string> = {
  measured: '●',
  report: '▣',
  'report-est': '◐',
  sourced: '◆',
  derived: 'ƒ',
  placeholder: '⚠',
  user: '✎',
  live: '↻',
};

export const KIND_HELP: Record<ProvKind, string> = {
  measured: 'A published benchmark measurement.',
  report: 'Stated in your report (snapshot data, agreed rule or quoted price).',
  'report-est': 'The report itself flags this as an estimate or anecdote.',
  sourced: 'Checked against a public datasheet, page or repository while building this.',
  derived: 'Computed from other values; the note shows how.',
  placeholder: 'An assumption with no reference yet. Treat as a guess until replaced.',
  user: 'Entered by you.',
  live: 'Fetched by a collection script, with a timestamp.',
};

const mk = <T>(kind: ProvKind) => (v: T, src: string, note?: string, asOf?: string): V<T> => ({
  v,
  p: { kind, src, ...(note ? { note } : {}), ...(asOf ? { asOf } : {}) },
});

export const meas = mk<number>('measured');
export const rep = mk<number>('report');
export const est = mk<number>('report-est');
export const srcd = mk<number>('sourced');
export const drv = mk<number>('derived');
const fmtPlaceholder = (v: number): string => (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4))));

/**
 * An assumption with no reference. It records the value it carries (and its unit) so a badge can state what the
 * dashboard is using, not only that it is a guess.
 */
export const ph = (v: number, src: string, note?: string, asOf?: string, unit?: string): V<number> => {
  const out = mk<number>('placeholder')(v, src, note, asOf);
  out.p.value = fmtPlaceholder(v);
  if (unit) out.p.unit = unit;
  return out;
};
export const live = mk<number>('live');

export const repS = mk<string>('report');
export const phS = (v: string, src: string, note?: string, asOf?: string, unit?: string): V<string> => {
  const out = mk<string>('placeholder')(v, src, note, asOf);
  out.p.value = v;
  if (unit) out.p.unit = unit;
  return out;
};

/** "0.6 × of roofline speed": the value in use with its unit, or null when the provenance does not state one. */
export function provValue(p: Prov): string | null {
  if (p.value === undefined) return null;
  return p.unit ? `${p.value} ${p.unit}` : p.value;
}
export const repB = mk<boolean>('report');

export function isV(x: unknown): x is V<unknown> {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Record<string, unknown>;
  if (!('v' in o) || !('p' in o)) return false;
  const p = o['p'] as Record<string, unknown> | null;
  return typeof p === 'object' && p !== null && typeof p['kind'] === 'string' && typeof p['src'] === 'string';
}

export interface ProvRow {
  path: string;
  value: unknown;
  prov: Prov;
}

/** Walk any nested data and list every V<T> leaf with its path. Used by the Data audit view. */
export function collectProvenance(root: unknown, base = ''): ProvRow[] {
  const out: ProvRow[] = [];
  const walk = (node: unknown, path: string) => {
    if (isV(node)) {
      out.push({ path, value: node.v, prov: node.p });
      return;
    }
    if (Array.isArray(node)) {
      node.forEach((n, i) => walk(n, `${path}[${i}]`));
      return;
    }
    if (typeof node === 'object' && node !== null) {
      for (const [k, v] of Object.entries(node)) walk(v, path ? `${path}.${k}` : k);
    }
  };
  walk(root, base);
  return out;
}

export function countByKind(rows: ProvRow[]): Record<ProvKind, number> {
  const c = Object.fromEntries(KIND_ORDER.map((k) => [k, 0])) as Record<ProvKind, number>;
  for (const r of rows) c[r.prov.kind] += 1;
  return c;
}
