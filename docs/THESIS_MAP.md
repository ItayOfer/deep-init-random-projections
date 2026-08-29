# Thesis map — where we are, and results → chapters

*2026-08-15, restructured 2026-08-29. Read this to re-enter the thesis: Part I is the story and the state, Part II is the writing plan. Read it instead of FRONTIER when the question is "where am I / what do I write next" rather than "what do I run next". Card references `[card §N, eq (N)]` point into `thesis/reference_card.pdf`.*

---

## Part I — Where we are

### The arc

**Act 1 — the problem (campaign 01 · chapters 1, 7).** A deep ReLU network at initialization destroys the geometry of its input: the arc-cosine kernel map `χ(ρ)` [card §3, eq (4)] pushes every pairwise correlation toward `ρ = 1` — measured mean pairwise cosine **0.993 at L=60** under He. The thesis question: can an initializer motivated by random projections prevent this collapse *and* train better? **Row centering** (each output neuron's weights sum to zero, so the layer annihilates the constant/DC component of its input) does prevent it: the iteration lands on a non-degenerate fixed point `ρ* ≈ 0.32` (**the 71° result**), stable from L=20 to L=60 [card §3]. The candidate had to satisfy three requirements: **(i)** avoid geometric collapse, **(ii)** keep gradients stable, **(iii)** preserve class content.

**Act 2 — the baseline is stronger than it looks (campaigns 02–05).** Tuning He across 12 architectures went 5/12 → 8/12 → **10/12 PASS** (recovery recipes; both 100L+BN cells remain open). Pass criterion for these counts: `eval_train_accuracy ≥ 0.995` AND `eval_train_loss ≤ 0.10`. The first warning sign: He trains almost everywhere *despite* fully collapsed geometry — geometry might not be what binds.

**Act 3 — the candidates, and the lock (campaigns 06–09).** **V2** (`row_centered_layer_balanced_product_base`, η=0.5): **5/12**, hard depth ceiling at 30L; no η rescues 100L (campaign 07). Meanwhile the reason row-centered nets are hard to train deep got a closed form: the **gain-coupling lock** — for a row-centered layer on post-ReLU input, `g_fwd/g_bwd = r = √((π−1)/π) ≈ 0.8256`, independent of weight variance [card §4, eq (7)], a direct consequence of the half-Gaussian moments and the centering ratio `(π−1)/π` [card §2, eq (1)–(3)]. You can slide along the family (backward-balanced, forward-balanced) but not escape the ratio. **rcfwd** (campaign 09) does the best possible thing within the lock: forward-balanced init + `_GradRescale` (×r in backward only) → flat gains both directions, numerically stable at every depth — and its best result ever against He is reaching the *same* bar **six epochs sooner** (fmnist/30L, ep74 vs tuned He's ep80). At ≥50L it is blocked by representation *content*, not gradient flow.

**Act 4 — the mechanism experiments (campaigns 10–12, 2026-08-15).** Three advisor asks, answered in `docs/reports/2026-08-15_campaign10_followup_synthesis.md`:
- **Frozen windows (10):** training only the first 2–3 of 100 frozen layers is *causally disconnected* — a 9-order-of-magnitude LR sweep leaves the loss at `ln 10` to within `2.08e-7` (the `r^100 ≈ 5e-9` reach law, [card §6]). Training only the last 2–3 reaches **0.9498 train / 0.1132 test** at 400 epochs: **capacity ≠ content** — the frozen random stack stays injective (memorizable) while class structure is destroyed.
- **Post-ReLU shift (11):** subtracting `c·rms(a)` after the ReLU is the *activation-space dual* of row centering — `W(a − c𝟙) = Wa − c(W𝟙)`, so on a row-centered weight the shift is exactly a no-op. Its forward gain `G(c)` is minimized at `c = 1/√π`, where `G = r` **exactly** [card §5, eq (8)]. Empirically: the apparent +6.6 pp CIFAR-10 win at 30L was a final-epoch artifact (+0.2 pp at best-epoch); at 100L the comparison was never made — 8 of 10 arms, He included, aborted on loss explosion under the minimal recipe.
- **Frozen readout (12):** ranking initializers by how much class content a trained 2-layer readout recovers — **He first at 30L and at 100L**; at 100L every arm is at chance. Content dies between depth 30 and 100 *for everyone*.
- **Dying neurons (proof track):** `P[dead] = ½·P[all N signs agree]` exactly; collapse rate `ε_ℓ ≈ 9π²/2ℓ²`, `Δ_ℓ ≈ 3/ℓ` [card §3 eq (5)–(6), §7 eq (9)]. Measured dead fractions 0.342/0.400/0.476 at L=30/60/100; the Slepian form of the bound gives 0.3881 vs measured 0.4000 at L=60 — tight to 3%.

### The ledger — proven · measured · retracted · open

**Proven** (derived + numerically verified; card § in brackets, thesis location per card §10 provenance):

| # | result | card | in the thesis? |
|---|---|---|---|
| P1 | Half-Gaussian moments, centering ratio `(π−1)/π` | §2 | ch3 (lemma + prop) |
| P2 | Arc-cosine collapse, algebraic rate `ε_ℓ ≈ 9π²/2ℓ²` | §3 | ch1 thm; rate in proofs draft only |
| P3 | Gain-coupling lock `g_fwd/g_bwd = r` | §4 | ch3 `prop:fb_ratio` |
| P4 | `G(c)`, minimum `G(1/√π) = r`, duality (shift ≡ row centering) | §5 | **not yet in any chapter** |
| P5 | `P[dead] = ½P(A)` exact; `→ ½` as `ρ→1` | §7 | draft `.tex` in `docs/scratch/proofs/` |
| — | **Slepian bound — the one item that is a sketch, not a proof** | §7 | the single research task left |

**Measured** (the load-bearing numbers; all re-derived by the 2026-08-15 adversarial audit — cite the synthesis, don't re-verify):

- He **10/12** · V2 **5/12** (ceiling 30L) · rcfwd **1 PASS** (same bar, 6 epochs sooner) · shift family **0** — synthesis §1, §4.
- Capacity ≠ content: **0.9498 train / 0.1132 test** (`rcfrozen_last3_audit` fmnist, 400 ep) — synthesis §4.1.
- Front window causally disconnected: loss = `ln 10` ± `2.08e-7` across 9 orders of LR — synthesis §4.2.
- Frozen readout: He first at both depths; **every** arm at chance at 100L — synthesis §5.2.
- Dead fractions 0.342/0.400/0.476 (L=30/60/100); Slepian 0.3881 vs 0.4000 — synthesis §3.3.
- The shift kills dying neurons (0–1.4% dead vs He's 34–48%) — and it doesn't help training — synthesis §4.3.

**Retracted** (all four on 2026-08-15; all four from reading train without test, or final epochs without the curve — synthesis §6):

1. "Shift beats He by +6.6 pp at 30L" → final-epoch artifact; +0.2 pp at best-epoch.
2. "The probes understate badly — a readout gets 0.83 where they said chance" → 0.83 was *train*; test is 0.1132.
3. "Training 3 layers beats training all 100 by ~4.8×" → true on train, reversed on test.
4. "He wins 100L end-to-end by 7–26 pp" → compared a completed 20-epoch He against arms that died at epochs 2–8; **100L end-to-end is simply unmeasured**.

**Open leads** (real, but they only matter on branch B of the fork below — synthesis §9): the 18 written-but-ungated 200-epoch 30L audits · the 100L arms under a recipe that doesn't explode (campaign 05's) · differentiable-vs-detached shift · the norm control · the per-sample-RMS variant · re-basing the α-screen on content metrics.

### Decoder — terms you will re-encounter

| term | meaning |
|---|---|
| `He` | baseline, `Var(W)·d = 2`, uncentered — collapses geometry, trains anyway |
| row centering | subtract each row's mean → `W𝟙 = 0`, layer is blind to the input's DC [card §1] |
| `V2` | `row_centered_layer_balanced_product_base`: row-centered + per-layer variance schedule, η=0.5 |
| `fwdbal` | `row_centered_forward_balanced`: rows rescaled so `g_fwd = 1` (pushes the cost to backward) |
| `rcfwd` | fwdbal init **+** `grad_rescale=r` (identity forward, ×r backward) — cancels the lock's gains |
| post-ReLU shift | `a ← relu(Wx) − c·rms(a)` on He weights — activation-space DC removal, dual to row centering |
| frozen window | 100L net frozen except 2–3 layers at one end (campaign 10) |
| frozen readout | 100L net frozen entirely; train a fresh 2-layer readout; measures *content* (campaign 12) |
| capacity ≠ content | a frozen random map stays injective (memorizable to 0.95 train) while class structure dies (0.11 test) |
| `r` | `√((π−1)/π) ≈ 0.8256` — the constant everything pays [card §9] |
| `G(c)` | forward gain of the shift; `min_c G = G(1/√π) = r` [card §5] |
| pass criterion | campaigns 01–10: `acc ≥ 0.995` AND `loss ≤ 0.10` (all headline X/12 counts). Campaign 10-onward (advisor, 2026-08-15): `acc ≥ 0.99`, loss dropped. **Never relabel the old counts.** |

### The reframe

> The thesis is **not** a search for an initializer that beats He — twelve campaigns say that search failed. It is a **no-go result with a constant**: every fix for ReLU's geometric collapse works by removing the DC component. Removing it costs exactly `r = √((π−1)/π) ≈ 0.826` of forward gain per layer — identically in weight space and activation space, because they are the same operation. And the fix, even when the cost is fully managed, buys nothing — because geometry was never the binding constraint.

Everything already run is evidence for this. **No further simulation is needed to write it.**

#### Why this reframe (the argument, so it survives any session)

Reached on 2026-08-15, not assumed. The case:

1. **Twelve campaigns, no win.** He 10/12; V2 5/12 with a 30L ceiling; rcfwd one PASS; the shift family none. The best result ever produced against He is *reaching the same bar six epochs sooner* — never a higher number.
2. **The one apparent win did not survive scrutiny.** Campaign 11's `c=0.10` looked like +6.6 pp on CIFAR-10 test; that was the final epoch of a curve oscillating 0.38–0.47, with He landing on its worst value. Mean-of-last-5: +2.1 pp; best-epoch: +0.2 pp; three of four arms go negative.
3. **All the candidates are one intervention, and its cost is provable.** Every candidate removes the ReLU DC. Row centering does it in weight space, the shift in activation space, and `W(a − c𝟙) = Wa − c(W𝟙)` makes them the same operation [card §5]. The operation costs exactly `r` of forward gain per layer. Not twelve separate disappointments — one constant, paid twelve times.
4. **The fix works as designed — and buys nothing.** DC removal genuinely delivers what it promises: the 71° fixed point instead of collapse, dead-neuron fractions of 0–1.4% against He's 34–48%. But content still dies at depth for *every* arm — including rcfwd, which cancels the lock's gain imbalance entirely and still fails on content at ≥50L, and including He itself, whose 100L readout is at chance. **Geometry is not the binding constraint; class content is, and no initializer choice measured here moves it.** So the constant is not the sole cause of failure — it is the *price* of a fix for the wrong constraint.
5. **The cost is transferable but not removable.** Rescaling weights by `1/G(c)` moves it from the forward pass to the backward with `g_fwd/g_bwd` invariant. That invariance is the signature of a lock, not an impossibility [card §5].

**What would overturn it.** A candidate that beats He on *held-out* accuracy, at matched epochs, under a robust estimator (best-epoch or mean-of-last-5), at a depth where collapse actually bites. Nothing measured so far comes close.

**Status: provisional.** The advisor has not agreed to this framing. Part II assumes he does.

### The fork — the advisor's decision

The synthesis's open leads and this map's "stop simulating" are not a contradiction; they are the two branches of one decision, which is the advisor's to make:

- **Branch A — accept the no-go framing.** The experimental programme is closed as *evidence for the no-go*, and what remains is ≈2.5 weeks of writing (Part II). The open leads are cited as future work.
- **Branch B — keep chasing the win.** Then the honest ordered next steps are: (1) the 18 written 200-epoch 30L audits (subs exist, one `sbatch` away), (2) the 100L arms under campaign 05's stable recipe, (3) differentiable-vs-detached. Each is cluster-days, not weeks — but each also only matters if a win is still the goal.

The case to bring to that meeting: five proven results, twelve campaigns of consistent evidence, a measured dying-neuron fraction matching the bound to 3% — and the choice above, stated as a choice.

---

## Part II — The plan (assumes branch A)

### The map

| ch | title | already written | to add | source (already exists) |
|---|---|---|---|---|
| 1 | Framework | arc-cosine theorem, angle contraction | — | solid, leave it |
| 2 | Initializers | catalog, 5 families | one entry: the post-ReLU shift family | `INITIALIZERS.md`, `cluster/11/README.md` |
| **3** | **Gradient trap** | half-Gaussian moments, centering ratio, DC blindness, **forward-backward gain ratio** | **`G(c)` theorem · duality theorem · lock generalized to DC removal** | `cluster/11/README.md` §"The closed form"; synthesis §3.1 |
| **4** | **Fixing the trap** | α-sweep, η-sweep only (277 lines — the weakest chapter) | **campaigns 09–12 as the evidence the fix buys nothing** | campaign READMEs 09/10/11/12; synthesis §4 |
| 5 | Geometry revisited | k-NN overturns PCA | **capacity ≠ content** (train 0.95 / test 0.11) — the same "the metric was wrong" lesson, one level deeper | synthesis §5.1, §5.3 |
| 6 | Kernel | biased ReLU kernel `K_β(α)` | probably fold into ch4; decide, don't expand | — |
| 7 | Angle map | two fixed points, the 71° result | — | solid, leave it |
| **NEW** | **Dying neurons** | *nothing in the thesis* | the whole chapter: `P[dead] → ½`, rate `ε_ℓ ≈ 9π²/2ℓ²`, `Δ_ℓ ≈ 3/ℓ`, Slepian bound, 47.6% measurement | `docs/scratch/proofs/dying_neurons_clean_proof.tex` + `oracle_spotcheck_addendum.md` |
| 8 | Conclusions | chronology of attempts | rewrite around the no-go statement | this file |

### Write order

1. **Ch3 additions — the spine.** Three theorems, all derived and numerically verified. Transcription into LaTeX, not research. Reuse the existing `\E \Var \Cov` macros. **~1 week.**
2. **New dying-neurons chapter.** The `.tex` already exists in `docs/scratch/proofs/`; it needs the Slepian replacement folded in and the empirical figure added. **~3 days.**
3. **Ch4 expansion.** Mostly transcription from four campaign READMEs. This is where campaigns 09–12 stop being a chronology and become an argument. **~4 days.**
4. **Ch5 insert + ch8 rewrite.** **~2 days.**

≈ 2.5 weeks of writing. (This estimates the writing work only — it is not a claim about total remaining runway, which is a separate question.)

### The one piece of actual research left

The **Slepian step** in the dying-neurons chapter is currently a sketch, not a proof: state the inequality, verify its hypotheses (equal variances after normalizing, one-sided event), and either make the `C√(ε log N)` constant explicit or keep the exact 1-D integral. Everything else on this map is writing.

### What not to do

- **Do not run more simulations without the fork being decided.** 189 result JSONs is already more than the document can absorb. The open leads matter only on branch B.
- **Do not open campaign 13.** `FRONTIER.md` is a machine for starting experiments; it is the wrong tool while writing.
- **Do not re-verify the numbers.** An adversarial audit re-derived 186 claims on 2026-08-15; the corrected figures are in `docs/reports/2026-08-15_campaign10_followup_synthesis.md`. Cite that, don't redo it.

### Before you start

Tell the advisor you are reframing from "find a better initializer" to "characterize why the natural fix cannot work, with the constant" — and put the fork to him explicitly. He should agree or redirect **before** two weeks of writing, not after.
