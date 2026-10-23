import { computed, effect, signal } from '@preact/signals';
import { DEFAULT_SETTINGS, PRESETS, PRESET_BY_ID } from './engine/defaults';
import { expectedValue, portfolioSteady, portfolioTimeline } from './engine/portfolio';
import { CASES } from './data/scenarios';
import type { Settings, Slot } from './engine/types';

const KEY = 'inferx.state.v1';

interface Persisted {
  settings: Settings;
  slots: Slot[];
  presetId: string;
  theme: 'system' | 'light' | 'dark';
}

function load(): Partial<Persisted> {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const p = JSON.parse(raw) as Partial<Persisted>;
    return p && typeof p === 'object' ? p : {};
  } catch {
    return {};
  }
}

const saved = load();
const DEFAULT_PRESET = 'staged';

export const settings = signal<Settings>({
  ...DEFAULT_SETTINGS,
  ...(saved.settings ?? {}),
  qos: { ...DEFAULT_SETTINGS.qos, ...(saved.settings?.qos ?? {}) },
  override: { ...DEFAULT_SETTINGS.override, ...(saved.settings?.override ?? {}) },
  whatIf: { ...DEFAULT_SETTINGS.whatIf, ...(saved.settings?.whatIf ?? {}) },
});
export const presetId = signal<string>(saved.presetId && PRESET_BY_ID[saved.presetId] ? saved.presetId : DEFAULT_PRESET);
export const slots = signal<Slot[]>(
  Array.isArray(saved.slots) && saved.slots.length > 0 ? saved.slots : PRESET_BY_ID[presetId.value]!.slots(),
);
export const theme = signal<'system' | 'light' | 'dark'>(saved.theme ?? 'system');

effect(() => {
  const payload: Persisted = { settings: settings.value, slots: slots.value, presetId: presetId.value, theme: theme.value };
  try {
    window.localStorage.setItem(KEY, JSON.stringify(payload));
  } catch {
    // Storage can be blocked or full; the dashboard works without it.
  }
});

effect(() => {
  const root = document.documentElement;
  if (theme.value === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', theme.value);
});

export const steady = computed(() => portfolioSteady(slots.value, settings.value));
export const timeline = computed(() => portfolioTimeline(slots.value, settings.value));
export const timelinesByCase = computed(() => Object.fromEntries(CASES.map((c) => [c, portfolioTimeline(slots.value, settings.value, c)])) as Record<(typeof CASES)[number], ReturnType<typeof portfolioTimeline>>);
export const evRows = computed(() => expectedValue(slots.value, settings.value, [0.05, 0.1, 0.15]));

export function updateSettings(patch: Partial<Settings>): void {
  settings.value = { ...settings.value, ...patch };
}

export function setSlot(id: string, patch: Partial<Slot>): void {
  slots.value = slots.value.map((s) => (s.id === id ? { ...s, ...patch } : s));
}

export function addSlot(slot: Slot): void {
  slots.value = [...slots.value, slot];
}

export function removeSlot(id: string): void {
  slots.value = slots.value.filter((s) => s.id !== id);
}

export function loadPreset(id: string): void {
  const p = PRESET_BY_ID[id];
  if (!p) return;
  presetId.value = id;
  slots.value = p.slots();
}

export function resetAll(): void {
  settings.value = { ...DEFAULT_SETTINGS, override: { ...DEFAULT_SETTINGS.override }, whatIf: { ...DEFAULT_SETTINGS.whatIf }, qos: { ...DEFAULT_SETTINGS.qos } };
  loadPreset(DEFAULT_PRESET);
}

export const VIEWS = [
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'viability', label: 'Viability grid' },
  { id: 'capacity', label: 'Capacity' },
  { id: 'throughput', label: 'Throughput' },
  { id: 'sensitivity', label: 'Sensitivity' },
  { id: 'prices', label: 'Prices' },
  { id: 'data', label: 'Data audit' },
] as const;

export type ViewId = (typeof VIEWS)[number]['id'];

const fromHash = (): ViewId => {
  const h = window.location.hash.replace(/^#\/?/, '');
  return (VIEWS.find((v) => v.id === h)?.id ?? 'portfolio') as ViewId;
};

export const view = signal<ViewId>(fromHash());
window.addEventListener('hashchange', () => {
  view.value = fromHash();
});

export { PRESETS };
