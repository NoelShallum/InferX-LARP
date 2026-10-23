import { describe, expect, it } from 'vitest';
import { ROOF_CLASSES, type RoofClass, type RoofParams } from './roofline';
import { fitGroups, fitLoss, fitRows } from './ix-fit';

/**
 * Refit the roofline's efficiencies to InferenceX:  FIT=1 npx vitest run src/engine/roofline.fit.test.ts
 * It prints the constants to paste into ROOF_CLASSES. The ordinary run only checks the shipped constants still fit.
 */
type P = Record<RoofClass, RoofParams>;

function search(P: P, rows: ReturnType<typeof fitRows>) {
  const knobs: { name: string; get: () => number; set: (v: number) => void }[] = [];
  const cap = (v: number) => Math.min(1, v);
  for (const c of ['hopper', 'blackwell-dc'] as RoofClass[]) {
    knobs.push({ name: `${c}.etaMem`, get: () => P[c].etaMem, set: (v) => { P[c].etaMem = cap(v); if (c === 'blackwell-dc') P['blackwell-ws'].etaMem = P[c].etaMem; } });
    knobs.push({ name: `${c}.stepOverheadMs`, get: () => P[c].stepOverheadMs, set: (v) => { P[c].stepOverheadMs = v; if (c === 'blackwell-dc') P['blackwell-ws'].stepOverheadMs = v; } });
    knobs.push({ name: `${c}.etaPrefill`, get: () => P[c].etaPrefill, set: (v) => (P[c].etaPrefill = cap(v)) });
    knobs.push({ name: `${c}.etaPrefillMoe`, get: () => P[c].etaPrefillMoe, set: (v) => (P[c].etaPrefillMoe = cap(v)) });
  }
  knobs.push({ name: 'ws.etaPrefillMoe', get: () => P['blackwell-ws'].etaPrefillMoe, set: (v) => (P['blackwell-ws'].etaPrefillMoe = cap(v)) });
  const link = (key: 'allReduceUs' | 'linkGBs', k: 'nvlink' | 'pcie') => ({ name: `${key}.${k}`, get: () => P.hopper[key][k], set: (v: number) => { for (const c of Object.values(P)) c[key][k] = v; } });
  knobs.push(link('allReduceUs', 'nvlink'), link('allReduceUs', 'pcie'), link('linkGBs', 'nvlink'), link('linkGBs', 'pcie'));
  let best = fitLoss(rows, P);
  for (let iter = 0; iter < 40; iter++) {
    let improved = false;
    for (const k of knobs) {
      const v0 = k.get();
      let bv = v0;
      for (const f of [0.6, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.6]) {
        k.set(v0 * f);
        const l = fitLoss(rows, P);
        if (l < best - 1e-6) { best = l; bv = k.get(); improved = true; }
      }
      k.set(bv);
    }
    if (!improved) break;
  }
  return { best, knobs };
}

describe('roofline against InferenceX', () => {
  const rows = fitRows();
  it.runIf(process.env.FIT === '1')('refits the efficiencies', () => {
    const P: P = JSON.parse(JSON.stringify(ROOF_CLASSES));
    const { best, knobs } = search(P, rows);
    console.log(`rows ${rows.length}  start loss ${fitLoss(rows).toFixed(4)}  final loss ${best.toFixed(4)}`);
    console.log(knobs.map((k) => `${k.name} = ${k.get().toFixed(3)}`).join('\n'));
    console.log(JSON.stringify(P, null, 1));
  });
  it('has the measured rows the constants were fitted to', () => {
    expect(rows.length).toBeGreaterThan(350);
  });
  it('the shipped constants still fit them (mean squared log error under 0.33)', () => {
    expect(fitLoss(rows)).toBeLessThan(0.33);
  });
  it('no model, card and GPU count is off by more than 45% on median output speed', () => {
    for (const g of fitGroups(rows)) {
      expect(g.out, g.key).toBeGreaterThan(0.55);
      expect(g.out, g.key).toBeLessThan(1.45);
    }
  });
  it('tensor-parallel runs are not systematically over- or under-predicted as GPUs are added', () => {
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]!;
    const by = (n: number) => med(fitGroups(rows).filter((g) => g.gpus === n).map((g) => g.out));
    expect(by(8) / by(1)).toBeGreaterThan(0.75);
    expect(by(8) / by(1)).toBeLessThan(1.35);
  });
});
