# Research Log

Chronological narrative of the project: what each phase asked, what was run, what was found, and where the evidence lives. Dates are 2026. Companion indices: `reports/results/INDEX.md` (evidence), `cluster/README.md` (campaigns), `notebooks/README.md` (analyses).

---

## Phase 1 — Foundations: geometry, kernels, and the row-centering idea (Jan – Mar)

**Question.** How do deep ReLU networks under random initialization transform data geometry, and can row-centered weights prevent the arc-cosine-kernel collapse?

- Multi-layer random-projection + ReLU geometry experiments (`notebooks/01–04`, originally `notebooks/archive/Random_Projections.ipynb`).
- Formalized the three failure modes: geometric collapse via the arc-cosine kernel `K(α)`, gradient vanishing/explosion, dead neurons.
- Proved row-centered weights are blind to the DC component (`Σ_j W_ij = 0`) and derived the universal ReLU centering ratio `Var(a)/E[a²] = (π−1)/π ≈ 0.682`, giving the structural forward gain `g_fwd = √((π−1)/π) ≈ 0.826` per layer — **independent of weight variance**.
- Formal writeup: `thesis/main.tex` (Mar).

**Key artifacts:** `thesis/main.pdf`, `CONTEXT.md`, `notebooks/01–04`.

## Phase 2 — The gradient trap: forward/backward gain asymmetry (Mar 16 – Apr 11)

**Question.** Why do row-centered networks lose gradients even when geometry is preserved — and can any variance choice fix it?

- Discovered the invariant: with row centering, `g_fwd/g_bwd ≈ 0.826` is **locked regardless of variance** — no scalar variance achieves `g_fwd = g_bwd = 1` simultaneously. The geometry–gradient conflict is structural, not tunable.
- Registered 19 initializer variants (partial centering α, layer-balanced η schedules, kernel-preserving optimization) in the registry; all documented in `INITIALIZERS.md`.
- Geometry claim revised later (Phase 5): PCA "spread" was misleading; k-NN accuracy shows row-centered variants collapse class structure at depth just like He, via the opposite mechanism (**spread ≠ structure**).

**Key artifacts:** `docs/reports/gradient_diagnostics_analysis.md`, `notebooks/05–07`, `cluster/01_geometry/`.

## Phase 3 — Designing V1 and V2 (Apr 12 – May 4)

**Question.** If both gains can't be 1, can their *product* be — and can variance be redistributed across depth?

- **V1** `row_centered_product_balanced`: base variance `v* = 2√(π/(π−1))/d ≈ 2.422/d` sets `g_fwd · g_bwd = 1` exactly — the unique fixed point between the vanishing and exploding variants.
- **V2** `row_centered_layer_balanced_product_base`: per-layer schedule `s_l = s* · r^{η(l−(L+1)/2)}` (η = 0.5 default) redistributes variance across depth for more uniform per-layer gradients (ratio ~1900× → ~36× at L=50).
- In parallel, the He supervised baseline was built: 12-architecture grid sweep with per-architecture HP tuning → **5/12 PASS** (`cluster/02_he_tuning/`).

**Key artifacts:** `docs/milestones/2026-04-13_product_balanced_walkthrough.md`, `thesis/product_balanced_report/`, `docs/reports/sweep_results_table.md`, `notebooks/10–11`.

## Phase 4 — He diagnostics, final audit, and recovery (May 14 – 23)

**Question.** How far can carefully-tuned He go on the 12-architecture benchmark, and what exactly blocks depth 100?

- **Diagnostic phases 1–5** (May 14–16, `cluster/03_he_diagnostics/`): hypothesis-driven short runs selecting per-architecture recipes → `docs/reports/diagnostic_phase{1,2,3}_report.html`.
- **Final audit** (May 16–18, `cluster/04_he_final_audit/`): 200-epoch unified audit → **8/12 PASS** (`final_audit_merged.json`).
- **Recovery** (May 22–23, `cluster/05_sgd_recovery/`): the 100L/NoBN failures were an **Adam pathology** (second-moment underflow on tiny float32 gradients) — plain SGD rescued both → **He 10/12 PASS**. Both 100L/BN cases resisted every recipe (recovery3: Adam + plateau + warmup peaked 21–47% then drifted) → the **100L+BN open wall**.

**Key artifacts:** `docs/reports/final_report.html`, `notebooks/13_final_results.ipynb`, `cluster/04_he_final_audit/README.md`, `cluster/05_sgd_recovery/README.md`.

## Phase 5 — V2 on trial: the depth ceiling and double preconditioning (May 22 – 24)

**Question.** Does V2's per-layer scheduling extend trainable depth beyond He?

- V2 audit rounds 1–4 (`cluster/06_v2_row_centered/`, no gradient clipping — enforced by assertion): **5/12 PASS** — all four 30L architectures + fmnist/50L/BN.
- **Depth ceiling:** layer-1 weight std scales as `r^{−η(L−1)/2}`; at L=100 that is ~126× He, overflowing float32 by layer ~11. No η rescues it.
- **Double preconditioning:** V2's per-layer scaling composes with SGD's uniform step but fights Adam's adaptive scaling — switching fmnist/50L/BN from Adam to plain SGD took it from a stuck 33% to **99.95%**, and the per-layer gradient ratio from 5×10⁴ to 1×10².
- Geometry revision landed here too (k-NN, `notebooks/09_depth_geometry_comparison.ipynb`): He+BN and all row-centered variants sit at chance by 20 layers; He+NoBN keeps 0.639.

**Key artifacts:** `docs/reports/final_report.html` (V2 sections), `cluster/06_v2_row_centered/README.md`.

## Phase 6 — Follow-ups and the untested rcfwd idea (May 25)

**Question.** Can anything lift the V2 depth ceiling or crack 100L+BN — and can gradient flow be fixed *outside* the weights?

- **η\* sweep** (`scripts/eta_sweep_research.py` → `cluster/07_v2_eta_nobn/`): per-architecture gradient-ratio-minimizing η does not lift the ceiling; L=100 NoBN still overflows/diverges.
- **He low-LR probe** (`cluster/08_he_lowlr_probe/`): 100L/BN at LR ≤ 1e-6 survives numerically but stays frozen at chance — confirming the wall is not merely a stability issue.
- **rcfwd grad-rescale** (`cluster/09_rcfwd_rescale/`): new idea — initialize with `row_centered_forward_balanced` (forward gain exactly 1) and cancel the backward gain `1/r ≈ 1.21` with a custom autograd op (`_GradRescale`: identity forward, multiply gradient by `r` per layer). A closed-form, non-adaptive per-layer LR. At initialization it flattens the error-signal blowup from ~1e8× to ~1.2× and the gradient ratio from 5×10⁷ to ~6× (He-like). Implementation: `grad_rescale` config field + hook in `src/rp_study/models/classifiers.py`; figures in `reports/figures/rcfwd_rescale/`.

**Resolved (Jul 6).** Smokes: all six train stably — no NaN even at 100L NoBN (where V2 died at epoch 1), gradient ratios 2–18×. Audits (200 ep): **fmnist/30L PASSES at ep74 — matching tuned He (ep80) with an untuned fixed-LR recipe**; cifar10/30L climbs to 0.92 (out of epochs); ≥50L slow-to-frozen. The LR ladder killed the step-size explanation: learning speed is LR-insensitive across the stable range (0.01–0.1) with a NaN wall above. **Conclusion: gradient conditioning and representation content are separate bottlenecks.** rcfwd solves the first completely (stable, well-conditioned, 200-epoch-verified at every depth); the second — class structure destroyed by depth, as measured in Phase 1's geometry — is what blocks L ≥ 50, and no backward-pass fix can restore information the forward pass never delivered.

---

## Open problems

1. **V2 depth ceiling (L ≥ 50):** compounding per-layer variance scaling overflows float32; needs a reformulation (e.g., scale-free parameterization) rather than tuning.
2. **100L + BN joint wall:** fails for He and V2, under Adam and SGD, with and without warmup/plateau — the strongest structural finding; candidate explanations involve BN's interaction with near-zero gradients at extreme depth.
3. **Content-preserving initialization:** rcfwd closed the gradient-flow problem and thereby isolated the real frontier — an initialization whose *forward pass keeps delivering class structure* at depth. The per-layer probes (Jul 6) made the target quantitative: rcfwd's linear-probe profile matches He through ~5 layers, collapses to chance by ℓ≈25 (SNR ~ 0.826^ℓ), and trainability tracks the *noise-tail length* L−ℓ\* (tail ~10 → PASS, ~30 → crawl, ~80 → frozen). He's slowly-decaying profile is the benchmark to beat. Candidates (partial-α centering, structure-preserving hybrids) are screenable with `scripts/content_profile_per_layer.py` in minutes, before any training.
