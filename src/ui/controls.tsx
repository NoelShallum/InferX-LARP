import type { ComponentChildren } from 'preact';

export function Field({ label, children, hint }: { label: ComponentChildren; children: ComponentChildren; hint?: string }) {
  return (
    <div class="field" title={hint}>
      <span class="label">{label}</span>
      {children}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { id: T; label: string; title?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div class="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={o.id === value} title={o.title} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SelectInput<T extends string>({
  value,
  options,
  onChange,
  label,
  id,
}: {
  value: T;
  options: { id: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  label: string;
  id?: string;
}) {
  return (
    <select class="input" id={id} aria-label={label} value={value} onChange={(e) => onChange(e.currentTarget.value as T)}>
      {options.map((o) => (
        <option key={o.id} value={o.id} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function NumberInput({
  value,
  onChange,
  label,
  min,
  max,
  step,
  width,
  suffix,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  width?: number;
  suffix?: string;
}) {
  return (
    <span class="row" style={{ gap: '4px' }}>
      <input
        class="input"
        type="number"
        aria-label={label}
        value={Number.isFinite(value) ? value : ''}
        min={min}
        max={max}
        step={step}
        style={width ? { width: `${width}px` } : undefined}
        onInput={(e) => {
          const n = e.currentTarget.valueAsNumber;
          if (Number.isFinite(n)) onChange(n);
        }}
      />
      {suffix ? <span class="small">{suffix}</span> : null}
    </span>
  );
}

export function Toggle({ checked, onChange, children, title }: { checked: boolean; onChange: (v: boolean) => void; children: ComponentChildren; title?: string }) {
  return (
    <label class="check" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.currentTarget.checked)} />
      <span>{children}</span>
    </label>
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  label,
  format,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  label: string;
  format: (v: number) => string;
}) {
  return (
    <div class="row" style={{ gap: '8px', flexWrap: 'nowrap' }}>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value} style={{ flex: 1, minWidth: '100px', accentColor: 'var(--accent)' }} onInput={(e) => onChange(e.currentTarget.valueAsNumber)} />
      <span style={{ minWidth: '56px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{format(value)}</span>
    </div>
  );
}
