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

**Done 2026-07-17.** Deliverables in `docs/scratch/proofs/` (gitignored): `validation_report.md` (full report)
and `chi_rate_check.py` (deliverable-3 numeric, re-runnable).

**Verdict:** the boxed result `P[dead] ∈ [½(1−(N choose 2)Δ_ℓ), ½] → ½` is **correct**, subject to a
limit-order restatement and one honest caveat. All arithmetic (moment identities §3, arc-cosine kernel,
Sheppard orthant formula, the ½·P(A) collapse) is correct; the gaps are structural.

**Both registered suspects confirmed:**
- **(a) expectation-vs-realization — GAP, central.** `E[ρ^(ℓ)] = χ(ρ^(ℓ−1))` puts a number on the left and a
  random variable on the right; Theorem 1 only proves the *conditional* statement `E[ρ^(ℓ)|h^(ℓ−1)] →
  χ(ρ^(ℓ−1))`. The unconditional recursion is `E[ρ^(ℓ)]=E[χ(ρ^(ℓ−1))] ≠ χ(E[ρ^(ℓ−1)])` (Jensen gap). Fix:
  infinite-width-per-layer induction (standard mean-field / Poole–Daniely–Schoenholz) makes ρ^(ℓ) deterministic
  per layer, killing the Jensen gap and fixing the order of limits (n→∞ then ℓ→∞). Written up in report §A.
- **(b) no rate — supplied.** χ′(1)=1 ⇒ algebraic convergence. Derived (not guessed): `1−χ(ρ) = (1−ρ) −
  (2√2/3π)(1−ρ)^{3/2} + O((1−ρ)²)` ⇒ `ε_ℓ = 1−ρ_ℓ ≈ (9π²/2)/ℓ² ≈ 44.41/ℓ²` and `Δ_ℓ ≈ 3/ℓ`, giving the
  explicit `P[dead] ∈ [½(1−3N(N−1)/(2ℓ)), ½]`. Numerically confirmed for four ρ₀ (constant is
  init-independent). Report §B.

**Bonus finds worth the advisor's attention:**
- The "ASK Ido" step (G5) **is valid but for the wrong reason**: `P[dead]=½P(A)` holds *exactly* because
  `A₋⊆A` and `w↦−w` symmetry, not because "P(Aᶜ)→0". Replace the justification.
- **Practical caveat:** the lower bound is non-vacuous only for depth `ℓ ≳ (3/2)N(N−1) = Θ(N²)` — vacuous for
  real datasets; the Θ(N²) is a loose union bound. Frame as an asymptotic-in-depth, fixed-N existence result.
- **Consistency (§E):** the ≈50%-flat ReLU-survival evidence in ch3 is *blind* to dataset-level death (both
  collapsed and non-collapsed regimes give ≈50% average firing), so it neither supports nor refutes the proof.
  The right empirical probe is the per-neuron *across-dataset* dead fraction — proposed as a cheap bonus
  experiment (§F): predicts He's dataset-dead fraction rises with depth while row-centered stays low.

Also flagged: broken `[?]` citation (§3.2), unused "ratio of expectations" clause with a UI gap (delete),
and two notation overloads (`σ` = ReLU vs std-dev; `X` = vector vs coordinate) to fix before manuscript entry.
Ranked fix list in report §G. No manuscript edits made (per constraint) — that is the next brief after review.
