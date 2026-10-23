import { GPUS, QUANT_LABEL, type Quant } from '../data/gpus';
import { MODELS } from '../data/models';
import { AVAILABILITY_PRESETS, WORKLOADS } from '../data/scenarios';

export const ECON_MODELS = MODELS.filter((m) => m.economics);

export const MODEL_OPTIONS = ECON_MODELS.map((m) => ({ id: m.id, label: m.name }));
export const MODEL_OPTIONS_OWNED = [{ id: 'lium', label: 'Stay on Lium (status quo)' }, ...MODEL_OPTIONS];
export const SERVE_GPUS = GPUS.filter((g) => g.role === 'serve');
export const RENTABLE_GPUS = SERVE_GPUS.filter((g) => g.rentUsdHr);
export const GPU_OPTIONS_RENT = RENTABLE_GPUS.map((g) => ({ id: g.id, label: g.short }));
export const GPU_OPTIONS_ALL = SERVE_GPUS.map((g) => ({ id: g.id, label: g.short }));
export const WORKLOAD_OPTIONS = WORKLOADS.map((w) => ({ id: w.id, label: w.label }));
export const QUANT_OPTIONS: { id: Quant | 'auto'; label: string }[] = [{ id: 'auto', label: 'Auto' }, ...(Object.keys(QUANT_LABEL) as Quant[]).map((q) => ({ id: q, label: QUANT_LABEL[q] }))];
export const AVAIL_OPTIONS = AVAILABILITY_PRESETS.map((a) => ({ id: a.id, label: a.label }));
export const COUNT_OPTIONS = ['1', '2', '4', '8'].map((n) => ({ id: n, label: `${n}×` }));

export const modelName = (id: string) => (id === 'lium' ? 'Stay on Lium' : (MODELS.find((m) => m.id === id)?.name ?? id));
export const gpuShort = (id: string) => GPUS.find((g) => g.id === id)?.short ?? id;
