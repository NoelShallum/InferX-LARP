import { describe, expect, it } from 'vitest';
import { collectProvenance, ph, phS, provValue } from './provenance';
import { ALL } from './register';

describe('placeholders state the value in use', () => {
  it('ph() stamps the value and unit it carries', () => {
    const v = ph(0.6000000001, 'No source yet', 'note', undefined, '× the roofline speed');
    expect(v.p.value).toBe('0.6');
    expect(provValue(v.p)).toBe('0.6 × the roofline speed');
    expect(provValue(phS('FP8', 'No source yet').p)).toBe('FP8');
  });
  it('a provenance without a stamped value says nothing rather than guessing', () => {
    expect(provValue({ kind: 'report', src: 'Report' })).toBeNull();
  });
  const placeholders = collectProvenance(ALL).filter((r) => r.prov.kind === 'placeholder');
  it('every placeholder in the register carries its value', () => {
    expect(placeholders.length).toBeGreaterThan(20);
    for (const r of placeholders) expect(provValue(r.prov), r.path).not.toBeNull();
  });
  it('and a unit, so "8" reads as 8 ₹ per kWh', () => {
    for (const r of placeholders) expect(r.prov.unit, `${r.path} = ${String(r.value)}`).toBeTruthy();
  });
  it('the stamped value is the value the register holds', () => {
    for (const r of placeholders) {
      if (typeof r.value === 'number' && r.prov.unit !== 'weight formats') expect(Number(r.prov.value), r.path).toBeCloseTo(r.value, 2);
    }
  });
});
