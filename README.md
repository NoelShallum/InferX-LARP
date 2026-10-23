# Inferx

Inference economics for a small GPU fleet: which GPUs, which models, rent or own, and what it earns against what the same card earns on Lium.

It is a static dashboard. All the maths runs in the browser, so every slider updates every view at once. Nothing is sent anywhere.

Built on three things:

- the report *Inference as a Service: Platforms, Models and Margins* (22 Oct 2026), which supplies the market data, the routing and price-elasticity findings, the Lium bars and the scenario cases;
- [SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX) and its [dashboard](https://github.com/SemiAnalysisAI/InferenceX-app), which supply GPU specifications, speculative-decoding acceptance lengths and the metric definitions (tokens per second per user, cost per million tokens, utilisation, availability);
- public datasheets for the cards InferenceX does not cover.

## What is in it

| View | Answers |
| --- | --- |
| **Portfolio** | The main screen. Your owned boxes plus rented and to-buy cards, each on a model. Steady-month gain over the Lium bar, a 12-month cash view for low, base and high cases, stake used against the ₹2 lakh cap, the cost of trying, and the expected value of applying to OpenRouter. |
| **Viability grid** | GPU against model: revenue ÷ cost, margin, break-even card use or card used. Owned rows compare with the Lium bar, rented rows with rent. Click a cell to add it to the portfolio. |
| **Capacity** | Memory split into weights, overhead and KV pool; how many users fit at a context length, counting sliding windows and recurrent state; every model on a GPU and every GPU for a model; what each model's Hugging Face config says. |
| **Throughput** | Output tokens per second against tokens per second per user, per GPU, for the request you pick, with MTP on or off. The operating point under the quality bar, how much GPU time prefill takes, time to first token, and the roofline set against every point the report publishes. |
| **Sensitivity** | A tornado of which inputs move the result, what-if sliders that apply to the whole dashboard, and break-even card use and price per card. |
| **Prices** | Live OpenRouter prices, traffic and stream speeds next to the report's snapshot, who sells each model, live Vast.ai rent by reliability tier, price trends once there are two snapshots, purchase prices and the Lium bar. |
| **Data audit** | A register of every tagged number (including the live ones) with where it came from, and a cross-check of the engine against 94 figures printed in the report. |

## Where numbers come from

Every number carries a badge, and anything computed on top of tagged inputs shows the weakest one.

| Badge | Meaning |
| --- | --- |
| **Measured** | A published benchmark measurement. |
| **Report** | Stated in the report (snapshot data, an agreed rule, a quoted price). |
| **Est.** | The report itself flags it as an estimate or anecdote. |
| **Derived** | Computed from other values; the note shows how. |
| **Sourced** | Checked against a public datasheet or repository. |
| **⚠ Placeholder** | An assumption with no reference yet. Treat as a guess. |
| **Live** | Fetched by a collection script (`npm run data`), with the snapshot date. |

Hugging Face config figures (KV bytes per token, recurrent state, expert counts, active parameters) are tagged **Sourced**, with the commit they were read at.

The **Data audit** view lists every placeholder so none hides.

## Live data

Two scripts collect prices. Both write a full `data/live/<source>.latest.json` that the dashboard imports at build time, and a compact dated copy in `data/live/history/` so trends show once there are two snapshots.

```bash
npm run data            # both
npm run data:vast       # Vast.ai GPU rent
npm run data:openrouter # OpenRouter prices, traffic, stream speed
npm run data:hf         # model architecture from Hugging Face configs (rarely needs re-running)
npm run data:inferencex # InferenceX measured throughput, speed per user and time to first token
```

**Vast.ai** refuses plain HTTP clients (403), so the script opens `cloud.vast.ai/create/` in a real Chromium once, lets the page set up its session, and then runs every search from inside that page. It needs a Chrome or Chromium on the machine; `playwright-core` does not download one.

| Variable | Effect |
| --- | --- |
| `CHROME_PATH` | Chrome or Chromium executable to use. |
| `HEADED=1` | Show the window, in case Vast asks you to prove you are human. |
| `GPUS=pro6000,rtx5090` | Only those catalog GPUs. |

For each GPU and node size it keeps the cheapest and the median on-demand offer under four reliability views: **r90** (verified, reliability ≥ 0.90, the default), **r98**, **dc99** (data-centre, ≥ 0.99) and **any**. Interruptible bids are stored separately and never used as rent. Region is saved on each offer, and the dashboard looks anywhere.

**OpenRouter** needs no browser. It reads the public API and the public stats endpoints: endpoint prices, cache-read prices, median stream speed per model (the basis of the OpenRouter-competitive quality bar), 7-day traffic and the request shape that follows from it. `MODELS=qwen38_27b,gptoss_120b` limits it.

In the dashboard, **Market source** and **Rent source** (Assumptions → Data sources) switch between live and the report's snapshot. A rented card with no live offer at the chosen tier falls back, in order, to the live per-GPU price of a single card, then to the report's price, and is flagged each time. With both switched to the report, the engine reproduces the report's figures.

**Hugging Face** needs no browser either. For each catalog model it reads `config.json` at a pinned commit and writes `data/hf/archs.json`: KV bytes per token (full-attention layers only), sliding-window size, recurrent state per sequence for linear-attention and Mamba layers, expert counts and active parameters. The cache is counted at one byte per element (an FP8 cache) and the recurrent state in fp32. Active parameters are checked against the published figures (gpt-oss-120b 5.2B against 5.1B, Qwen3.6-35B 3.0B against A3B, Nemotron 12.4B against A12B).

**InferenceX** is read from its public API (`/api/v1/benchmarks`). For gpt-oss-120b, Llama 3.3 70B, Qwen3.8-Flash-Next and Qwen3.5-397B it keeps vLLM and SGLang single-node runs on H100, H200, B200 and the RTX PRO 6000: concurrency, output speed per GPU, speed per user and time to first token, with the run id. The 397B model is not in the catalog; it is there because InferenceX measured it on four PRO 6000s over PCIe, the only measured data for that card.

Commit the new snapshot files to publish them: the Pages workflow builds from the repository.

## How the engine works

`src/engine` is plain TypeScript with no UI imports.

- `calibration.ts` — the report gives, per model, what a fully busy hour earns (§6.1) and what a typical new endpoint grosses (§6.3). With the §5.2 throughput point, those two figures pin the request's output share and the cache-read price. For Qwen3.8-27B and gpt-oss-120b the output share lands within half a point of the 4.1% and 11.8% the report states.
- `roofline.ts` — one decode step reads the weights the batch needs (only the experts it touches, for a mixture of experts) plus every sequence's cache and recurrent state, against the card's memory bandwidth; compute is checked against tensor throughput at the precision the weights run in; tensor-parallel nodes pay an all-reduce per layer. Prefill is compute-bound. A request's prefill takes a share of GPU time away from decode and queues in front of the first token. Its efficiencies (memory bandwidth, step overhead, prefill, all-reduce latency and bandwidth) are fitted per card family to 393 InferenceX measurement points, and the residuals are on the Data audit. Refit with `FIT=1 npx vitest run src/engine/roofline.fit.test.ts`.
- `capacity.ts` — fit (weights, overhead, KV pool), quantisation choice, three throughput bases, MTP gain from acceptance length, operating point under a speed floor and a first-token limit.
- `pricing.ts` — blended price per million tokens from request shape and cache price; price elasticity (−0.63) and its limits.
- `slot.ts` — one card on one model: capacity, demand (marketplace share model, hours sold, or overflow), revenue, cost or Lium bar, margin, tax, break-even.
- `bars.ts` — the Lium bar: what a fiat or crypto route must earn to tie with Lium after tax.
- `timeline.ts`, `portfolio.ts` — 12 months with listing delay, ramp, floor-price months, price trend and the report's revert and hand-back rules; programme costs; expected value of applying.
- `sensitivity.ts` — the tornado.
- `crosscheck.ts` — runs the engine under the report's own conditions next to the report's printed figures.

`npm test` runs the cross-checks, so a change that drifts from the report fails. With the report's inputs the engine lands within 1% on 75 of the 94 printed figures and within 5% on 92. The worst is 14%, on the report's own low-case gain. Figures the engine is calibrated to match are marked as such; the lowest-price figures are independent checks. The Data audit view lists every difference.

### Three throughput bases

| Basis | What it does |
| --- | --- |
| **Report's points as printed** | A power law through the report's published points. Reproduces the report. Does not know about request shape. |
| **Anchored to measurements** (default) | The fitted roofline, scaled to a measurement wherever there is one: InferenceX's runs for the card and model (the whole curve, at the concurrencies it measured), else the report's one measured point for that card. Everything else is the roofline alone; the report's own estimates are not used. |
| **Roofline only** | Every anchor ignored; speed from bandwidth, tensor throughput, link speed and the request shape. |

Most of the report's throughput points are its own estimates, and the fitted roofline disagrees with several. The Throughput view and the Data audit set them against the roofline and mark the ones that sit above what the memory bus could deliver (Llama 3.3 70B at 16 concurrent).

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # engine tests and the cross-check against the report
npm run build      # static site in dist/
npm run build:single   # one self-contained HTML file in dist-single/
npm run data       # refresh live prices (see Live data)
```

## Deploy

`.github/workflows/pages.yml` tests, builds and publishes to GitHub Pages on every push to `main`. In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions** once.

## Status

- **Phase 1:** the interface, the portfolio planner and the economics engine on the report's static numbers.
- **Phase 2:** live Vast.ai prices (a headless-browser scrape, then scripted) and an OpenRouter script, both saved as dated snapshots. The dashboard defaults to live data with the report as a switch, and has an OpenRouter-competitive quality bar from live stream speeds.
- **Phase 3:** model architecture from Hugging Face configs, a roofline for decode, prefill and time to first token, three throughput bases, a first-token limit in the quality bar, KV counting for sliding windows and recurrent state.
- **Phase 4 (this version):** InferenceX measured runs (single card and tensor-parallel) fit the roofline per card family and anchor it where a card and model were measured; multi-GPU nodes use the fitted all-reduce model.
- **Next:** a measurement run on your own cards to fit dense prefill and the Ada, AMD and A100.

What this version does not do yet is listed at the bottom of the Data audit view.

Not financial or legal advice.
