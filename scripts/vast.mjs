/**
 * Vast.ai GPU rent snapshot.
 *
 * Vast.ai's search API refuses plain HTTP clients (403), so this opens its console in a real browser once,
 * lets the page establish its session, and then runs every search from inside that page.
 *
 *   npm run data:vast
 *
 * Environment:
 *   CHROME_PATH   path to a Chromium or Chrome executable (default: the one Playwright installed)
 *   HEADED=1      show the browser window (use this if the page asks you to prove you are human)
 *   GPUS=pro6000,rtx5090   limit to some catalog GPUs
 */
import { chromium } from 'playwright-core';
import { writeSnapshot, sleep } from './lib/snapshot.mjs';
import { GPU_TARGETS, NODE_SIZES, VIEWS, bidStats, buildQuery, compactOffer, compactVastForHistory, viewStats } from './lib/vast-core.mjs';

const CONSOLE = 'https://cloud.vast.ai/create/';
const wanted = process.env.GPUS ? process.env.GPUS.split(',') : Object.keys(GPU_TARGETS);

const browser = await chromium.launch({
  headless: process.env.HEADED !== '1',
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ['--no-sandbox'],
});
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 1000 },
  userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
});
const page = await ctx.newPage();

let sessionOk = false;
page.on('response', (r) => {
  if (r.url().includes('/api/v0/bundles/') && r.status() === 200) sessionOk = true;
});
await page.goto(CONSOLE, { waitUntil: 'domcontentloaded', timeout: 60000 });
for (let i = 0; i < 60 && !sessionOk; i++) await sleep(1000);
if (!sessionOk) {
  await browser.close();
  console.error('Vast.ai did not return offers to the page within 60 seconds. Run again with HEADED=1 and complete any check it shows.');
  process.exit(1);
}

async function search(q) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    const res = await page.evaluate(async (query) => {
      const r = await fetch('/api/v0/bundles/?q=' + encodeURIComponent(JSON.stringify(query)), { credentials: 'include' });
      return { status: r.status, text: await r.text() };
    }, q);
    if (res.status === 200) return JSON.parse(res.text).offers ?? [];
    if (res.status === 429 || res.status >= 500) await sleep(2000 * attempt);
    else throw new Error(`Vast.ai answered ${res.status} for ${JSON.stringify(q.gpu_name)} x${q.num_gpus}`);
  }
  throw new Error('Vast.ai kept refusing the search');
}

const gpus = {};
for (const id of wanted) {
  const t = GPU_TARGETS[id];
  if (!t) {
    console.warn(`Unknown GPU id ${id}, skipped`);
    continue;
  }
  gpus[id] = { vastNames: t.names, note: t.note ?? null, nodes: {} };
  for (const n of NODE_SIZES) {
    const verifiedRaw = await search(buildQuery({ names: t.names, numGpus: n, minVramMb: t.minVramMb, verified: true }));
    await sleep(350);
    const anyRaw = await search(buildQuery({ names: t.names, numGpus: n, minVramMb: t.minVramMb, verified: false, limit: 100 }));
    await sleep(350);
    const compact = [...verifiedRaw, ...anyRaw].map(compactOffer);
    const unique = [...new Map(compact.map((o) => [o.id, o])).values()];
    const entry = { views: viewStats(unique) };
    if (n === 1) {
      entry.interruptible = bidStats(await search(buildQuery({ names: t.names, numGpus: 1, minVramMb: t.minVramMb, verified: true, type: 'bid', limit: 100 })));
      await sleep(350);
    }
    gpus[id].nodes[n] = entry;
    const r90 = entry.views.r90;
    console.log(`${id.padEnd(8)} x${n}: ${String(r90.count).padStart(3)} offers at r90, cheapest ${r90.minPerGpu ?? '—'} per GPU-hour, median ${r90.medianPerGpu ?? '—'}`);
  }
}

await browser.close();
const now = new Date();
const payload = {
  source: 'vast.ai',
  generatedAt: now.toISOString(),
  query: { type: 'on-demand', verifiedForViews: true, minDiskGb: 100, allocatedStorageGb: 16, offersPerSearch: 200 },
  views: Object.fromEntries(Object.entries(VIEWS).map(([k, v]) => [k, v.label])),
  gpus,
};
console.log(writeSnapshot('vast', payload, now, compactVastForHistory));
