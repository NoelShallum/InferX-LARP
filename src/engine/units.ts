import type { Currency } from './types';

const LAKH = 1e5;
const CRORE = 1e7;

export function fmtNum(n: number, digits = 0): string {
  if (!Number.isFinite(n)) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtInr(inr: number, compact = true): string {
  if (!Number.isFinite(inr)) return '—';
  const sign = inr < 0 ? '−' : '';
  const a = Math.abs(inr);
  if (!compact) return `${sign}₹${Math.round(a).toLocaleString('en-IN')}`;
  if (a >= CRORE) return `${sign}₹${(a / CRORE).toFixed(2)} cr`;
  if (a >= LAKH) return `${sign}₹${(a / LAKH).toFixed(a >= 10 * LAKH ? 1 : 2)} lakh`;
  if (a >= 1000) return `${sign}₹${(a / 1000).toFixed(a >= 10_000 ? 0 : 1)}k`;
  return `${sign}₹${Math.round(a)}`;
}

export function fmtUsd(usd: number, compact = true): string {
  if (!Number.isFinite(usd)) return '—';
  const sign = usd < 0 ? '−' : '';
  const a = Math.abs(usd);
  if (!compact) return `${sign}$${Math.round(a).toLocaleString('en-US')}`;
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 10_000) return `${sign}$${(a / 1000).toFixed(0)}k`;
  if (a >= 1000) return `${sign}$${(a / 1000).toFixed(1)}k`;
  if (a >= 100) return `${sign}$${Math.round(a)}`;
  return `${sign}$${a.toFixed(2)}`;
}

/** Format a USD amount in the chosen display currency. */
export function fmtMoney(usd: number, currency: Currency, fx: number, compact = true): string {
  return currency === 'INR' ? fmtInr(usd * fx, compact) : fmtUsd(usd, compact);
}

export function fmtPct(x: number, digits = 0): string {
  if (!Number.isFinite(x)) return '—';
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtX(x: number): string {
  if (!Number.isFinite(x)) return '—';
  return `${x.toFixed(x >= 10 ? 0 : 1)}×`;
}

export function fmtTps(x: number): string {
  if (!Number.isFinite(x)) return '—';
  return x >= 100 ? fmtNum(x) : x.toFixed(1);
}

export function fmtTokens(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return `${Math.round(n)}`;
}
