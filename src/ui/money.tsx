import { settings } from '../state';
import { fmtInr, fmtUsd } from '../engine/units';
import { afterTaxFactor } from '../engine/portfolio';

/** Convert a USD amount into the display currency's unit. */
export const disp = (usd: number): number => (settings.value.currency === 'INR' ? usd * settings.value.fx : usd);

export const fmtD = (v: number, compact = true): string => (settings.value.currency === 'INR' ? fmtInr(v, compact) : fmtUsd(v, compact));

export const money = (usd: number, compact = true): string => fmtD(disp(usd), compact);

export const taxF = (): number => afterTaxFactor(settings.value);

export function M({ usd, signed = false, compact = true }: { usd: number; signed?: boolean; compact?: boolean }) {
  const s = money(usd, compact);
  const cls = signed ? (usd > 0.5 ? 'pos' : usd < -0.5 ? 'neg' : '') : '';
  return <span class={cls}>{signed && usd > 0.5 ? `+${s}` : s}</span>;
}
