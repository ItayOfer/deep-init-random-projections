# FRONTIER — current state & active workstreams

> **The delegation hub.** This is the first document any new agent, terminal, or session reads after the onboarding chain (below). It answers: what is open, what is in flight, what is decided, and what can be picked up. **Update this file whenever a workstream starts, changes state, or closes** — it is the durable memory of the project's working edge.

_Last updated: 2026-07-17_

## Onboarding chain (for any new agent/terminal)

1. `README.md` — what the thesis is, headline results, repo map
2. `docs/RESEARCH_LOG.md` — the six-phase narrative and open problems
3. **this file** — what's happening now
4. Your task brief in `docs/plans_handoffs/briefs/` (if assigned one)
5. `CLAUDE.md` — conventions, cluster workflow, campaign checklist
6. Deep dives as needed: `cluster/<NN>_*/README.md` (per-campaign stories), `reports/results/INDEX.md` (evidence), `INITIALIZERS.md` (math reference)

## Current state (one paragraph)

The 12-architecture program is resolved and documented: He 10/12 (100L+BN open wall), V2 5/12 (depth ceiling ≈30L), rcfwd stable at all depths with fmnist/30L PASS @ ep74 but blocked at ≥50L by representation content, not gradient flow (probe-verified; see `cluster/09_rcfwd_rescale/README.md` incl. the three-requirements frame and the 2×2 table). The two live directions: a new experiment campaign (scope arriving from the advisor) and the dying-neurons proof (draft exists, details arriving).

## Active workstreams

| ID | Workstream | Status | Next action | Entry points |
|---|---|---|---|---|
| W1 | **Campaign 10 — rc frozen ends** (100L; train only last-3 vs only first-3 layers, head always frozen, raw vs corrected recipe) | **worked — awaiting oracle merge** | Two passes, 8/16 smoke jobs run and pulled (a mid-campaign design fix corrected `last3` from an asymmetric `{fc99,fc100,head}` to the symmetric `{fc98,fc99,fc100}`, head frozen in both conditions — `first3` was unaffected and never reran). **Raw** (`row_centered_he`): both `last3` cells **DEAD** (tail gradients underflow to exact float32 zero by ep7–17), both `first3` cells **STUCK** (fc1–fc3 gradients healthy, order 1–5, zero loss movement). Refutes the advisor's "both train fast" bet. **Corrected recipe** (`row_centered_forward_balanced`+`grad_rescale`, campaign 09's exact recipe — run to test whether the raw failure was a gradient-conditioning artifact, H1, or content death, H2): `last3` **starts learning** (11%→24%/22% train acc over 20 ep, no plateau — the raw death was a forward-scale artifact, H1-consistent for the tail); `first3` still fails, loss now *worsens* past ln(10) despite healthy gradients under both recipes (H2-consistent for the front — content/reachability bottleneck, not gradient conditioning). Net: the bottleneck is asymmetric by layer position. Audits not gated except `last3`/rcfwd, flagged as a real candidate. Branch `work/rc-frozen-ends` has everything (code, 8 JSONs, 3 figures, docs); oracle to spot-check the headline numbers, merge, and (only after that) fold into `docs/RESEARCH_LOG.md` | The brief (Outcome section filled); `cluster/10_rc_frozen_ends/README.md` §Findings + §H1 vs H2 (figures in `docs/figures/rcfrozen_*.png`); `reports/results/rcfrozen_*_smoke_*_100L[_rcfwd].json` |
| W2 | **Dying-neurons proof validation** — P[dead on dataset] → ½ with depth; advisor asked to fix + add bounds | **briefed — ready to spin up** | Worker executes [`briefs/2026-07-17_dying-neurons-proof-validation.md`](briefs/2026-07-17_dying-neurons-proof-validation.md) (top-tier model; output stays in `docs/scratch/proofs/`) | The brief; PDFs in `docs/scratch/proofs/`; `thesis/chapters/ch3_gradient_trap.tex` |
| W3 | Content-preserving init screen (α-family) | parked, ready | Run `scripts/content_profile_per_layer.py` over `partial_centered_he` α-grid; candidates must beat rcfwd's ℓ\*≈25 decay | `cluster/09_rcfwd_rescale/README.md` §content probes |
| W4 | cifar10/30L rcfwd audit extension | parked, cheap | Extend past 200 ep (was 0.92 and climbing) for the second PASS | `cluster/09_rcfwd_rescale/` |

**Known oracle notes on W2** (registered 2026-07-17, before the improvement round): the draft proves the boxed bound `P[dead] ∈ [½(1−(N choose 2)Δ_ℓ), ½]` via kernel-collapse → concentration → orthant symmetry. Two gaps to expect work on: (a) the expectation-vs-realization recursion across layers (Jensen gap in `E[χ(ρ)]` vs `χ(E[ρ])`); (b) no convergence *rate* — χ has quadratic tangency at ρ=1 so ε_ℓ ~ 1/ℓ², which would make the bound quantitative in (depth, N). Bonus experiment the repo can run cheaply: the proof's mechanism predicts row-centered nets (no ρ→1 collapse) have far fewer dataset-dead neurons than He at depth — measurable with existing machinery.

## Open problems (stable; details in `docs/RESEARCH_LOG.md`)

1. V2 depth ceiling (float32 overflow of the per-layer variance schedule).
2. The 100L+BN joint wall (every initializer × optimizer recipe fails).
3. Content-preserving initialization (beat He's per-layer content-decay profile; screen with the probe scripts before training).

## Known evidence gaps (from `reports/results/INDEX.md`)

`diagnostic_phase4.json` never retained · `v2_nobn_sgd_audit_*` never run · `he_sgd_lowlr` round-1 JSONs not retained · `geometry_product_balanced_*` outputs never pulled · `eta_star_recommended.json` predates the safety filter in `eta_sweep_pick.py`.

## The agentic cycle (how every new task runs)

1. **Define** — the oracle session + user write a brief from `briefs/TEMPLATE.md` → `docs/plans_handoffs/briefs/<yyyy-mm-dd>_<slug>.md`; add a row to this file.
2. **Spin up** — open a new terminal/agent in the repo root; its first instruction: *"Read `docs/plans_handoffs/briefs/<file>.md` and follow its onboarding chain."* Nothing else is needed — the repo is self-sufficient. **Model selection:** judgment work (architecture, research direction, proof math, verification, surprising-result triage) runs on the top-tier model; brief-execution work (campaign scaffolding, `.sub` edits, sync/submit/pull, plotting per spec) runs on the mid-tier model — the brief carries the intelligence. Verification always returns to the top-tier model before results enter the record. Sessions are cheap and disposable: prefer a fresh session per workstream (it re-becomes the oracle from this file + memory) over one long context.
3. **Isolate** — purely *additive* work (a new campaign dir, a new note, new results) goes on `main`; anything touching shared files (`src/`, existing docs, notebooks) goes on a branch `work/<slug>` that the oracle merges.
4. **Work** — under the standing rules in `CLAUDE.md`: every number traces to a file in `reports/results/`; seeds fixed; no gradient clipping in V2/rcfwd-family experiments; public-repo naming.
5. **Report back** — the worker fills the brief's *Outcome* section, updates its campaign/task README and the row in this file, commits (and pushes if on `main`).
6. **Close the loop** — the oracle verifies (spot-checks claims against the JSONs — the adversarial-verification pattern), merges branches, updates `docs/RESEARCH_LOG.md` and its memory, and marks the row done.

**Oracle continuity:** the coordinating session maintains this file + its private memory. If that session ends, a new one resumes losslessly from: memory dir → this file → RESEARCH_LOG. No knowledge lives only in a conversation.
