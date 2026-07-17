# Brief — dying-neurons-proof-validation (2026-07-17)

**Onboarding chain (read in order before starting):** `README.md` → `docs/RESEARCH_LOG.md` → `docs/plans_handoffs/FRONTIER.md` → this brief → `CLAUDE.md` (see "Working on proofs / the manuscript").

## Goal

Validate, line by line and down to notation level, the proof that in a deep He-initialized ReLU network the probability a neuron is **dead on the entire dataset** converges to 1/2 with depth — and produce a precise list of what must be fixed and which bounds must be added (the advisor's explicit request).

## Context

- **Source documents (local snapshots, gitignored):** `docs/scratch/proofs/dying_neurons_depth_proportion_draft.pdf` (the streamlined 5-page version, boxed result `P[dead] ∈ [½(1−(N choose 2)Δ_ℓ), ½]`) and `docs/scratch/proofs/dying_neurons_depth_DRAFT-2.pdf` (fuller draft, possibly more material). The Overleaf project is the source of truth; older iterations exist in `~/Downloads/` if version archaeology is needed.
- **Proof chain to validate:** (§2) kernel map χ(ρ)=K(α) iterates pairwise correlations to 1 — Lemma 1 (monotone convergence); (§2.2) Theorem 1: E[ρₙ] → µ_C/√(µ_A µ_B) in the wide limit (SLLN + CMT + DCT); (§3) moment identification giving E[ρ^(ℓ)] = χ(ρ^(ℓ−1)); (§3.2) concentration: sub-exponential summands → Bernstein-type `P(|ρ−E ρ| ≥ δ) ≤ C₁e^{−C₂nδ²}` + union bound over pairs; (§4) orthant probability (Lemma 2: P[same sign] = 1 − arccos(ρ)/π) + init symmetry → the boxed ½-interval.
- **Oracle-registered suspects** (verify or refute; do not assume): (a) the **recursion across layers mixes expectation and realization** — E[ρ^(ℓ)] = χ(ρ^(ℓ−1)) treats ρ^(ℓ−1) as deterministic; a correct treatment needs the conditional structure (given h^(ℓ−1), neurons are i.i.d.) plus control of the Jensen gap E[χ(ρ)] vs χ(E[ρ]) and error propagation through L layers; (b) **no convergence rate**: χ has quadratic tangency at ρ=1 (χ′(1)=1), so ε_ℓ = Θ(1/ℓ²)-type behavior must be derived to make Δ_ℓ = arccos(1−ε_ℓ)/π quantitative — this is the "add bounds" ask; (c) order of limits (width n → ∞ vs depth ℓ → ∞) is never pinned down — the concentration is per-layer at fixed n while Lemma 1 is depth-asymptotic; (d) Theorem 1's i.i.d. premise for the summands must be stated conditionally on the previous layer; (e) zero-bias assumption and the dataset-level dead definition must be used consistently (dead ⇔ z<0 for all N samples).
- **In-repo results to stay consistent with (cite, don't re-derive):** `thesis/chapters/ch3_gradient_trap.tex` — half-Gaussian moments lemma (`Pr(z>0)=1/2`, `E[a²]=σ²/2`) and the measured ~50%±0.4–1.7% per-input survival; `docs/reports/gradient_diagnostics_analysis.md` §4; the arc-cosine kernel in `src/rp_study/analysis/kernel.py` and INITIALIZERS.md. Note the distinction the manuscript already draws: per-input inactivity (=1/2 trivially) vs dataset-level permanent death (this proof) — the writeup must not conflate them.

## Deliverables

1. `docs/scratch/proofs/validation_report.md` — verdict per step (VALID / GAP / WRONG), each with the exact location (page/section/equation), the precise mathematical objection, and the minimal fix. Notation audit included: every symbol defined before use, consistent with `thesis/main.tex` macros (`\E`, `\Var`, `\relu`) and the manuscript's conventions (weight layout, widths d_ℓ, dataset size N).
2. A concrete "bounds to add" plan answering the advisor: the quantitative rate ε_ℓ (from χ's expansion at ρ=1: `1−χ(ρ) ≈ (1−ρ) − c(1−ρ)^{3/2}`-type analysis — derive the correct exponent, don't guess), the resulting explicit Δ_ℓ(N, ℓ, n), and where each bound slots into the existing structure.
3. (If time permits) a sanity numeric: iterate χ from ρ₀=0.5 and confirm the fitted decay of 1−ρ_ℓ matches the derived rate.

## Constraints

- Branch: none needed — deliverable 1–2 live in gitignored `docs/scratch/proofs/` (drafts stay private until ready; per CLAUDE.md). No manuscript edits in this task; that's the *next* brief after the user reviews the validation.
- Read-only with respect to the repo (except the report file). Mathematical claims must be argued, not asserted; where the draft is salvageable with a weaker statement, propose the weakest sufficient fix.
- Model: top-tier (judgment work).

## Definition of done

Validation report exists with a verdict for every numbered step of both PDFs' arguments, a notation-audit table, and the ranked fix list; the oracle spot-checks at least the two registered suspects (a) and (b) against the report before it goes to the user/advisor.

## Outcome  *(filled by the worker at the end)*

—
