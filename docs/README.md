# docs/

The narrative layer of the thesis repo. Start with **[RESEARCH_LOG.md](RESEARCH_LOG.md)** — the six-phase chronology linking every question to its campaign, evidence, and conclusion.

| Directory | Contents |
|---|---|
| [`milestones/`](milestones/) | Date-stamped briefings and walkthroughs written *at the time*: the gradient-trap briefing (2026-03-16), the product-balanced walkthrough (2026-04-13), architecture choices (2026-04-11), the research update (2026-05-16), and the audit/results briefings (2026-05-21, 2026-05-23). Historical records — preserved as written. |
| [`reports/`](reports/) | Analysis reports: `gradient_diagnostics_analysis.md` (the forward/backward gain asymmetry theory + measurements), the diagnostic phase reports (`diagnostic_phase{1,2,3}_report.html`), the full audit report (`final_report.html`), and the sweep scorecard (`sweep_results_table.{md,pdf}`). |
| [`plans_handoffs/`](plans_handoffs/) | Working plans and status handoffs: the post-meeting follow-up plan (2026-05-23), the research status handoff (2026-07-04 — the most current statement of where things stand), and the simulation run log. |
| [`figures/`](figures/) | The **curated** figure set (~15 images) embedded by the campaign READMEs under `cluster/`. The full figure collection lives in `reports/figures/` (local-only). |
| `scratch/` | Session working notes — gitignored, local-only. Includes `readme_fact_sheets/`: the verified per-campaign fact sheets (with source-file provenance for every number) from which the campaign READMEs were written. |

## Reading order for a newcomer

1. Repo root `README.md` — what the thesis is, headline results.
2. `RESEARCH_LOG.md` — the story in six phases.
3. `cluster/README.md` → per-campaign READMEs — each experiment's question, findings, and how to reproduce it.
4. `reports/results/INDEX.md` — the evidence files behind every number.
5. `reports/gradient_diagnostics_analysis.md` + `thesis/main.pdf` — the mathematics.
