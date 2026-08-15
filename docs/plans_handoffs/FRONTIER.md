# FRONTIER — current state & active workstreams

> **The delegation hub.** This is the first document any new agent, terminal, or session reads after the onboarding chain (below). It answers: what is open, what is in flight, what is decided, and what can be picked up. **Update this file whenever a workstream starts, changes state, or closes** — it is the durable memory of the project's working edge.

_Last updated: 2026-08-15_

## Onboarding chain (for any new agent/terminal)

1. `README.md` — what the thesis is, headline results, repo map
2. `docs/RESEARCH_LOG.md` — the six-phase narrative and open problems
3. **this file** — what's happening now
4. Your task brief in `docs/plans_handoffs/briefs/` (if assigned one)
5. `CLAUDE.md` — conventions, cluster workflow, campaign checklist
6. Deep dives as needed: `cluster/<NN>_*/README.md` (per-campaign stories), `reports/results/INDEX.md` (evidence), `INITIALIZERS.md` (math reference)

## Current state (one paragraph)

The 12-architecture program is resolved and documented: He 10/12 (100L+BN open wall), V2 5/12 (depth ceiling ≈30L), rcfwd stable at all depths with fmnist/30L PASS @ ep74 but blocked at ≥50L by representation content, not gradient flow (probe-verified; see `cluster/09_rcfwd_rescale/README.md` incl. the three-requirements frame and the 2×2 table). **Campaign 10 is merged** (`da925b8`): neither a 3-layer window at the tail nor at the front trains a 100L row-centered net under the raw recipe, and under the rcfwd recipe the tail starts learning while the front gets worse — the bottleneck is asymmetric by layer position. An advisor meeting on 2026-08-15 opened three follow-ups, now briefed as W5 (2-layer windows + the missing corners of the recipe 2×2) and W6 (campaign 11: post-ReLU DC removal). The proof workstream W2 is awaiting a final oracle spot-check.

## Active workstreams

| ID | Workstream | Status | Next action | Entry points |
|---|---|---|---|---|
| W1 | **Campaign 10 — rc frozen ends** (100L; 3-layer window at the tail vs at the front, head frozen in both, raw vs rcfwd recipe) | **closed — merged into `main` 2026-08-15 (`da925b8`)** | Continued by W5. Headline: under `row_centered_he` both `last3` cells are **DEAD** (tail gradients underflow to exact float32 zero by ep7–17) and both `first3` cells **STUCK** (fc1–fc3 gradients healthy at 1–5, zero loss movement) — refuting the advisor's "both train fast" bet. Under campaign 09's rcfwd recipe `last3` **starts learning** (11%→24% fmnist, 11%→22% cifar10 over 20 ep, no plateau) while `first3` still fails with loss *climbing* past ln(10) at chance accuracy. Campaign-10 subs realigned to `--exclude=dgx04` on merge | `cluster/10_rc_frozen_ends/README.md` §Findings + §H1 vs H2; `reports/results/rcfrozen_*_smoke_*_100L[_rcfwd].json`; figures `docs/figures/rcfrozen_*.png` |
| W2 | **Dying-neurons proof validation** — P[dead on dataset] → ½ with depth; advisor asked to fix + add bounds | **worker done 2026-07-17 — awaiting oracle spot-check** | Oracle verifies suspects (a)&(b) against `docs/scratch/proofs/validation_report.md`, then hand to user/advisor; next brief = apply fixes to manuscript | Report + `chi_rate_check.py` in `docs/scratch/proofs/`; brief Outcome |
| W5 | **Campaign 10 continuation — 2-layer windows + recipe ablation** (advisor follow-ups 1 & 2) | **briefed — ready to spin up** | Worker executes [`briefs/2026-08-15_campaign10-2layer-windows-and-recipe-ablation.md`](briefs/2026-08-15_campaign10-2layer-windows-and-recipe-ablation.md) on `main` (additive). Two jobs: (a) redo the frozen windows at **2 layers** (`last2={fc99,fc100}`, `first2={fc1,fc2}`, head frozen in both) and train the working cell to **99% train accuracy** rather than a 20-ep smoke; (b) run the two **never-run corners** of the (init × grad_rescale) 2×2 — `raw+rescale` and `fwdbal` — so campaign 10's "forward-scale artifact" conclusion becomes a measurement. Init-time numbers already in hand (see the oracle note below); `raw+rescale` needs an LR ladder because its gradients are uniformly ≈9e-9 | The brief; `reports/results/recipe_decomposition_funnel.json`; `scripts/recipe_decomposition_funnel.py` |
| W6 | **Campaign 11 — post-ReLU DC removal** (`a = relu(Wx) − c·rms(a)`; advisor follow-up 3) | **worker done 2026-08-15 (init-time) — awaiting oracle verify + merge of `work/relu-shift`; cluster jobs NOT submitted** | Oracle: (1) verify + merge `work/relu-shift`; (2) run the 18 smoke subs (worker had no cluster access — exact command sequence in the brief's Outcome); (3) rule on the **per-sample-RMS** recommendation below. `src/` gate is PASS (bit-exact no-op vs `main`, 202 param tensors + a full 100L epoch, `relu_shift_noop_verification.json`) | [`cluster/11_relu_shift/README.md`](../../cluster/11_relu_shift/README.md); the brief's Outcome; `reports/results/relu_shift_*.json` (11 files); `scripts/relu_shift_*.py` (5) |
| W3 | Content-preserving init screen (α-family) | parked, ready | Run `scripts/content_profile_per_layer.py` over `partial_centered_he` α-grid; candidates must beat rcfwd's ℓ\*≈25 decay | `cluster/09_rcfwd_rescale/README.md` §content probes |
| W4 | cifar10/30L rcfwd audit extension | parked, cheap | Extend past 200 ep (was 0.92 and climbing) for the second PASS | `cluster/09_rcfwd_rescale/` |

**Oracle notes on W5** (registered 2026-08-15, measured before briefing — `reports/results/recipe_decomposition_funnel.json`, 100L width 500 seed 42): the rcfwd recipe bundles two independent interventions — **(A)** the init change `row_centered_he → row_centered_forward_balanced` (fixes the forward) and **(B)** `_GradRescale` (fixes the backward). Campaigns 09/10 only ever ran the `(raw, none)` and `(fwdbal, rescale)` corners, so "last3's recovery was a forward-scale artifact" is currently an inference. At init: `raw` act-RMS @L100 = 5.01e-9 with gradient max/min = 1.66e8; `raw+rescale` has **bit-identical activations** (`_GradRescale` is identity forward) but the **flattest** gradient profile of all four corners, 1.13×, pinned at ≈9e-9 ≈ r^100; `fwdbal` alone hits 4.61e8 at layer 1 and should abort; `rcfwd` is 1.35× at O(1). Consequence: any explanation of `last3` that runs through activations must be an (A) story, since (B) provably cannot touch the forward pass. Also confirmed: the two inits differ by ≈1.21^ℓ in scale with mean pairwise cosine ≈0.32 at *every* depth for both — same statistical geometry, different scale.

**Oracle notes on W6** (registered 2026-08-15 before the work; kept for the record): He kills **40.0%** of its neurons by layer 60 (dataset-dead, N=512), an independent empirical confirmation of W2's →½ prediction. DC removal eliminates dead neurons (0–1.4% for every `c ≤ 0.75`). The effect is **strongly non-monotone in `c`** and the theoretically exact `c = 1/√π` is *not* the best point. Two leads were flagged: what governs the non-monotonicity, and whether the implied gain 0.9083 at `c=1/√π` is really `√r = 0.9086`.

**W6 worker findings** (2026-08-15, `cluster/11_relu_shift/README.md`; both leads resolved):

- **The `√r` lead is refuted.** The per-layer forward gain has a closed form, `G(c) = √(1 − 2c/√π + c²)`, and at `c = 1/√π` it is `√(1 − 1/π) = r = 0.82565` — **not** `√r`. The measured 0.9083 is a geometric *mean* of a gain that drifts with depth (0.8381 @L10 → 0.9404 @L100) and merely crosses `√r` near L=60. Verified positively by `c = 0.75`, whose implied gain is depth-independent and equals `G(0.75) = 0.8463` to 4 dp at L=10/20/30/60/100.
- **Thesis-level consequence.** Exact DC removal costs *exactly* `r` per layer — the same constant row-centering pays — and `G(c) < 1` for every `c ∈ (0, 2/√π)`. So **requirements (i) and (ii) are provably incompatible under DC removal**: the only shift with unit forward gain is no shift. Belongs in the manuscript alongside the gain-coupling lock.
- **The non-monotonicity is a batch-statistic artifact, not a property of DC removal.** `rms` is a batch-global scalar, so the subtraction is *absolute*, which amplifies per-sample norm spread with depth; each sample then sits at an effective `c/t` on the U-shaped curve. Diagnostic `norm_heterogeneity_kappa` collapses to 0.32–0.68 exactly where the theory fails and stays ≥0.998 where it holds. **Control:** with a per-sample RMS the closed form is exact across the whole grid, monotonicity is restored, and `c = 1/√π` reaches **mean pairwise cosine 0.0037 at 100 layers** — the lowest in the project.
- **Open oracle decision:** whether to adopt the **per-sample RMS** form. One line of code; makes the family behave as designed; removes a BatchNorm-like batch-dependence with no running-stats fix. Deliberately *not* added to the grid (outside the brief's settled scope). It does **not** rescue requirement (ii) — `G(1/√π) = r` regardless of how the RMS is computed.
- **Fork recommendation: detached** (`relu_shift_detach=True`, now the config default) — it is the exact dual of row-centering and makes the backward bit-identical to plain He. Caveat carried honestly: the 2-epoch CPU pre-triage marginally favours the differentiable arm (0.826 vs 0.799 fmnist), so `*_diff` control jobs are in the grid.
- **Where the shift genuinely wins:** at 30L, `c = 0.1–0.25` beats **both** baselines on distance-correlation-to-input (0.691 vs He's 0.388 on cifar10) with **0.000** dead units vs He's 0.302–0.342. Unresolved gap: the brief's own norm-artifact caveat is not yet tested, so that is a measurement, not an interpretation. At 100L nothing survives — every arm at or near chance content.

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
