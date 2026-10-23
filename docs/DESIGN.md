# Design decisions

Agreed in the interview before building. Kept here so the reasons survive.

## What it is for

- **Main screen: the portfolio plan.** Owned PRO 6000, 6000 Ada and 3050, plus rented and to-buy cards, each assigned a model and workload. Not a per-GPU leaderboard.
- **Drill-downs:** viability grid (GPU × model), capacity planner (KV cache, concurrent users, context), throughput curves (with MTP), sensitivity and break-even. A live price board is always included.
- **Audience:** built in public. The repo and the dashboard are public, and the report's real numbers, including Lium earnings, are committed as they are.

## Throughput

- **Roofline model plus published points.** No benchmark harness for the owned boxes in v1.
- InferenceX has no measured data for one PRO 6000, Ada or 5090 on 27–35B models, so the report's published points are the anchors and the roofline fills the gaps.
- Default basis: the fitted roofline scaled to a measurement where one exists ("measured where it exists, scaling model elsewhere"): InferenceX's curve for the card and model, else the report's one measured point. The report's estimates are not used. The report's points as printed and the roofline alone are one switch away.
- The roofline's efficiencies are fitted per card family to 393 InferenceX points (H100, H200, B200, four PRO 6000s; gpt-oss-120b, Llama 3.3 70B, Qwen3.5-397B). Cards InferenceX has not measured inherit the nearest family; dense prefill on the PRO 6000 and Ada is a placeholder.
- Model architecture (KV per token, sliding windows, recurrent state, experts, active parameters) comes from Hugging Face configs, tagged Sourced with the commit. Where a config cannot settle a figure (DeepSeek V4.1 Flash's cache, a sparse indexer's key layout) the model card or a stated assumption is used and the value is tagged Derived with the assumption in its note.
- MiniMax-M3's InferenceX runs are held out of the fit (`HOLDOUT_MODELS` in `ix-fit.ts`) so the roofline's accuracy on an uncalibrated model is measured, not assumed; a regression test bounds it.
- A benchmark point that breaks the physics of a load test (throughput collapsing as concurrency rises, a lone user slower than a crowd) is dropped and listed, never silently.
- **Multi-GPU rentals are in scope now.** InferenceX measured curves are used where they exist; a tensor-parallel scaling model fills the rest.

## Demand and price

- **Marketplace share model with a utilisation override.** A typical new endpoint gets 0.3 of an equal 1/N share of a model's paid OpenRouter traffic. Price elasticity −0.63, latency elasticity −0.46.
- **Batch work** is sold either at a discount to the live price × hours sold, or at your own price per million tokens. Batch is not routed.
- **Rent price:** cheapest reliable on-demand Vast.ai offer by default (verified, reliability ≥ 0.90, any region; the region is saved); the median and stricter tiers are stored too.
- **Quality bar:** adjustable minimum tokens per second per user and p90 time to first token. The OpenRouter-competitive preset uses the median live stream speed per model (phase 2). The report's convention and a custom floor remain available.

## Workloads

Configurable presets: as observed on OpenRouter per model, agentic coding, chat, roleplay, batch. Each carries a request shape (average tokens, output share, cache hit) and a suggested speed floor.

## Speculative decoding

On/off toggle using acceptance lengths from InferenceX where they exist (Qwen3.8-Flash-Next) and a calibrated fade-out so the gain at 32 concurrent matches the report's measured 1,752 tok/s for Qwen3.8-27B. Models with no evidence have no MTP option.

## Time and risk

Steady month plus a 12-month timeline with low, base and high presets (listing month, ramp, floor-price months, monthly price trend). Listing odds and expected value are a toggle. The report's revert and hand-back rules can be switched on or off.

## Downtime

Scenario presets built from the report's outage evidence, plus a custom availability percentage. Availability feeds revenue and OpenRouter's 95% and 80% uptime tiers. A summer heat derate is a separate toggle.

## Tax and currency

INR and USD with a pre-tax and after-tax switch, using the report's agreed rule: crypto 31.2% of gross, fiat about 25.17% of profit. The rule is an editable input because the report flags that a CA may treat Lium differently.

## Provenance

Every value is tagged measured, report, report estimate, derived, sourced or placeholder. Placeholders are loud. The Data audit view lists all of them.

## Scope for now

- **Models:** the report's, from its §5 tables, plus the five busiest OpenRouter models it does not cover (DeepSeek V4.1 Flash, GLM-5.3, GLM-5.3 Flash, MiniMax-M3, MiMo-V2.6 Flash). Those have no report numbers: market from live OpenRouter data, speed from the roofline (MiniMax-M3 anchored to InferenceX), and rented nodes of 2, 4 and 8 cards because none fits one card.
- **GPUs:** the report's (PRO 6000, 6000 Ada, 3050, 5090, 4090, L40S, H100, H200, B200) plus the Tesla V100 32 GB. The 16 GB V100 is left out because it fits almost none of the models. AMD and A100 come later.

## Build order

1. Interface and planner on the report's static numbers, with the economics engine behind it. **Done.**
2. Vast.ai browser scrape (done once live, then scripted) and an OpenRouter script, with dated snapshots; replace the static prices. **Done.**
3. Roofline engine, starting with PRO 6000 × Qwen3.8-27B end to end, then every model and card. **Done** (Hugging Face configs, roofline, prefill and first-token time).
4. InferenceX measured curves (single card and tensor-parallel) fit the roofline and anchor it. **Done.**
5. Widen the models: the five highest-traffic OpenRouter models on nodes of 1, 2, 4 and 8 cards. **Done.**
6. Next: a measurement run on the owned cards, AMD and A100.

## Acceptance

"Report as inspiration only": no requirement to match the report. The engine nevertheless lands within 1% on 75 of the 94 printed figures it is checked against and within 5% on 91, and the Data audit view shows every difference.
