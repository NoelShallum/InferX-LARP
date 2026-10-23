import fs from 'node:fs';
import path from 'node:path';

export const LIVE_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../data/live');

/**
 * Write the full latest snapshot and one compact dated copy per UTC day (a second run on the same day replaces it).
 * History keeps the statistics trends need and drops the per-offer and per-endpoint lists, so the repository grows slowly.
 */
export function writeSnapshot(name, payload, now = new Date(), compact = (p) => p) {
  const day = now.toISOString().slice(0, 10);
  fs.mkdirSync(path.join(LIVE_DIR, 'history'), { recursive: true });
  fs.writeFileSync(path.join(LIVE_DIR, `${name}.latest.json`), JSON.stringify(payload, null, 1) + '\n');
  fs.writeFileSync(path.join(LIVE_DIR, 'history', `${name}-${day}.json`), JSON.stringify(compact(payload)) + '\n');
  return { latest: `data/live/${name}.latest.json`, dated: `data/live/history/${name}-${day}.json` };
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const REPO_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');

/** Write a JSON file under the repository, relative to its root. */
export function writeJson(rel, payload) {
  const file = path.join(REPO_DIR, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(payload, null, 1) + '\n');
  return rel;
}
