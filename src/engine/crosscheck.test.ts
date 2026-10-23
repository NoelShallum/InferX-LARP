import { describe, expect, it } from 'vitest';
import { checkStatus, runCrossChecks } from './crosscheck';

describe('dashboard engine against the report', () => {
  const rows = runCrossChecks();
  it('has the expected number of cross-checks', () => {
    expect(rows.length).toBeGreaterThan(60);
  });
  it('prints the comparison', () => {
    const lines = rows.map((r) => {
      const s = checkStatus(r);
      return `${s.ok ? 'ok  ' : 'MISS'} ${r.group} | ${r.label} | report ${r.report.toFixed(r.unit === 'pct' ? 3 : 0)} | dash ${r.dashboard.toFixed(r.unit === 'pct' ? 3 : 0)} | ${(s.delta * 100).toFixed(1)}%`;
    });
    console.log(lines.join('\n'));
  });
  it('every cross-check lands inside its stated tolerance', () => {
    const misses = rows.filter((r) => !checkStatus(r).ok).map((r) => `${r.id}: report ${r.report}, dashboard ${r.dashboard}`);
    expect(misses).toEqual([]);
  });
});
