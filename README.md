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
| **Capacity** | Memory split into weights, overhead and KV pool; how many users fit at a context length; every model on a GPU and every GPU for a model. |
| **Throughput** | Output tokens per second against tokens per second per user, per GPU, with MTP on or off and the operating point under the current quality bar. |
| **Sensitivity** | A tornado of which inputs move the result, what-if sliders that apply to the whole dashboard, and break-even card use and price per card. |
| **Prices** | The report's OpenRouter snapshot, GPU rent and purchase prices, and the Lium bar. |
| **Data audit** | A register of every tagged number with where it came from, and a cross-check of the engine against 94 figures printed in the report. |

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
| **Live** | Fetched by a collection script (phase 2). |

The **Data audit** view lists every placeholder so none hides.

## How the engine works

`src/engine` is plain TypeScript with no UI imports.

- `calibration.ts` — the report gives, per model, what a fully busy hour earns (§6.1) and what a typical new endpoint grosses (§6.3). With the §5.2 throughput point, those two figures pin the request's output share and the cache-read price. For Qwen3.8-27B and gpt-oss-120b the output share lands within half a point of the 4.1% and 11.8% the report states.
- `capacity.ts` — fit (weights, overhead, KV pool), quantisation choice, throughput curve through the report's points, MTP gain from acceptance length, multi-GPU scaling.
- `pricing.ts` — blended price per million tokens from request shape and cache price; price elasticity (−0.63) and its limits.
- `slot.ts` — one card on one model: capacity, demand (marketplace share model, hours sold, or overflow), revenue, cost or Lium bar, margin, tax, break-even.
- `bars.ts` — the Lium bar: what a fiat or crypto route must earn to tie with Lium after tax.
- `timeline.ts`, `portfolio.ts` — 12 months with listing delay, ramp, floor-price months, price trend and the report's revert and hand-back rules; programme costs; expected value of applying.
- `sensitivity.ts` — the tornado.
- `crosscheck.ts` — runs the engine under the report's own conditions next to the report's printed figures.

`npm test` runs the cross-checks, so a change that drifts from the report fails. With the report's inputs the engine lands within 1% on 75 of the 94 printed figures and within 5% on 91. The worst is 14%, on the report's own low-case gain. Figures the engine is calibrated to match are marked as such; the lowest-price figures are independent checks. The Data audit view lists every difference.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # engine tests and the cross-check against the report
npm run build      # static site in dist/
npm run build:single   # one self-contained HTML file in dist-single/
```

## Deploy

`.github/workflows/pages.yml` tests, builds and publishes to GitHub Pages on every push to `main`. In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions** once.

## Status

- **Phase 1 (this version):** the interface, the portfolio planner and the economics engine on the report's static numbers.
- **Phase 2:** live Vast.ai prices (a headless-browser scrape, done once live, then scripted) and an OpenRouter script, both saved as dated snapshots so trends show.
- **Phase 3:** roofline throughput, prefill and time to first token, PRO 6000 × Qwen3.8-27B end to end, then every model and card. InferenceX measured curves for multi-GPU nodes.

What this version does not do yet is listed at the bottom of the Data audit view.

Not financial or legal advice.
