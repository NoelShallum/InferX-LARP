import { round, summarise } from './stats.mjs';

/**
 * Catalog GPU id -> the canonical names Vast.ai uses (from its gpu_types endpoint).
 * minVramMb keeps the 16 GB Tesla V100 out.
 */
export const GPU_TARGETS = {
  pro6000: { names: ['RTX PRO 6000 WS', 'RTX PRO 6000 S', 'RTX PRO 6000'], note: 'Workstation, Server and unlabelled editions; Max-Q is excluded because it is a slower 300 W card' },
  ada6000: { names: ['RTX 6000Ada'] },
  rtx5090: { names: ['RTX 5090'], note: '5090 D and Ti excluded' },
  rtx4090: { names: ['RTX 4090'] },
  l40s: { names: ['L40S'] },
  h100: { names: ['H100 SXM'], note: 'SXM only; PCIe and NVL excluded' },
  h200: { names: ['H200'] },
  b200: { names: ['B200'] },
  v100: { names: ['Tesla V100', 'Tesla V100S'], minVramMb: 30000, note: '32 GB cards only' },
};

export const NODE_SIZES = [1, 2, 4, 8];

/** Reliability views the dashboard can switch between. The default is r90. */
export const VIEWS = {
  r90: { label: 'Verified, reliability at least 0.90', test: (o) => o.verified && o.reliability >= 0.9 },
  r98: { label: 'Verified, reliability at least 0.98', test: (o) => o.verified && o.reliability >= 0.98 },
  dc99: { label: 'Data-centre host, reliability at least 0.99', test: (o) => o.verified && o.datacenter && o.reliability >= 0.99 },
  any: { label: 'Any offer, no filter', test: () => true },
};

export function buildQuery({ names, numGpus, minVramMb, verified, type = 'on-demand', limit = 200, minDiskGb = 100 }) {
  const q = {
    rentable: { eq: true },
    rented: { eq: false },
    gpu_name: { in: names },
    num_gpus: { eq: numGpus },
    disk_space: { gte: minDiskGb },
    allocated_storage: 16,
    type,
    order: [['dph_total', 'asc']],
    limit,
    resource_type: 'gpu',
  };
  if (verified) q.verified = { eq: true };
  if (minVramMb) q.gpu_ram = { gte: minVramMb };
  return q;
}

/** Keep only what the dashboard needs from one Vast.ai offer. */
export function compactOffer(o) {
  const n = Math.max(1, o.num_gpus ?? 1);
  const dph = Number(o.dph_total);
  return {
    id: o.id,
    machineId: o.machine_id,
    gpuName: o.gpu_name,
    numGpus: n,
    dph: round(dph, 4),
    perGpu: round(dph / n, 4),
    reliability: round(o.reliability2 ?? o.reliability, 4),
    verified: o.verification === 'verified',
    datacenter: o.hosting_type === 1 || o.datacenter === 1 || o.datacenter === true,
    geo: o.geolocation ?? null,
    inetDown: o.inet_down === undefined ? null : Math.round(o.inet_down),
    inetUp: o.inet_up === undefined ? null : Math.round(o.inet_up),
    vramGb: o.gpu_ram === undefined ? null : round(o.gpu_ram / 1024, 1),
    cpuRamGb: o.cpu_ram === undefined ? null : round(o.cpu_ram / 1024, 0),
    diskGb: o.disk_space === undefined ? null : Math.round(o.disk_space),
    pcieBw: o.pcie_bw === undefined ? null : round(o.pcie_bw, 1),
    cudaMax: o.cuda_max_good ?? null,
  };
}

/** Per-view statistics over the compacted offers of one GPU at one node size. */
export function viewStats(offers) {
  const out = {};
  for (const [id, v] of Object.entries(VIEWS)) {
    const pass = offers.filter(v.test);
    const perGpu = pass.map((o) => o.perGpu);
    const s = summarise(perGpu);
    out[id] = {
      count: s.n,
      minPerGpu: round(s.min),
      p25PerGpu: round(s.p25),
      medianPerGpu: round(s.median),
      cheapest: [...pass].sort((a, b) => a.perGpu - b.perGpu).slice(0, 3),
    };
  }
  return out;
}

/** History copy: keep every statistic, and only the single cheapest offer per view. */
export function compactVastForHistory(payload) {
  const gpus = {};
  for (const [id, g] of Object.entries(payload.gpus)) {
    gpus[id] = { ...g, nodes: {} };
    for (const [n, entry] of Object.entries(g.nodes)) {
      const views = {};
      for (const [v, stats] of Object.entries(entry.views)) views[v] = { ...stats, cheapest: stats.cheapest.slice(0, 1) };
      gpus[id].nodes[n] = { ...entry, views };
    }
  }
  return { ...payload, gpus };
}

/** Interruptible (bid) offers: the minimum bid is what a host will accept. */
export function bidStats(rawOffers) {
  const bids = rawOffers.map((o) => (o.min_bid ?? o.dph_total) / Math.max(1, o.num_gpus ?? 1)).filter(Number.isFinite);
  const s = summarise(bids);
  return { count: s.n, minPerGpu: round(s.min), medianPerGpu: round(s.median) };
}
