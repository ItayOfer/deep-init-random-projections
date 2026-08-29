# Thesis map — results → chapters

*2026-08-15. The guideline for finishing. Read this instead of FRONTIER when the question is "what do I write next" rather than "what do I run next".*

## The reframe

The thesis is **not** a search for an initializer that beats He — twelve campaigns say that search failed. It is a **no-go result with a constant**:

> Every fix for ReLU's geometric collapse works by removing the DC component. Removing it costs exactly `r = √((π−1)/π) ≈ 0.826` of forward gain per layer — identically in weight space and activation space, because they are the same operation. That cost is never repaid.

Everything already run is evidence for this. **No further simulation is needed to write it.**

## The map

| ch | title | already written | to add | source (already exists) |
|---|---|---|---|---|
| 1 | Framework | arc-cosine theorem, angle contraction | — | solid, leave it |
| 2 | Initializers | catalog, 5 families | one entry: the post-ReLU shift family | `INITIALIZERS.md`, `cluster/11/README.md` |
| **3** | **Gradient trap** | half-Gaussian moments, centering ratio, DC blindness, **forward-backward gain ratio** | **`G(c)` theorem · duality theorem · lock generalized to DC removal** | `cluster/11/README.md` §"The closed form"; synthesis §3.1 |
| **4** | **Fixing the trap** | α-sweep, η-sweep only (277 lines — the weakest chapter) | **campaigns 09–12 as the evidence the cost is never repaid** | campaign READMEs 09/10/11/12; synthesis §4 |
| 5 | Geometry revisited | k-NN overturns PCA | **capacity ≠ content** (train 0.95 / test 0.11) — the same "the metric was wrong" lesson, one level deeper | synthesis §5.1, §5.3 |
| 6 | Kernel | biased ReLU kernel `K_β(α)` | probably fold into ch4; decide, don't expand | — |
| 7 | Angle map | two fixed points, the 71° result | — | solid, leave it |
| **NEW** | **Dying neurons** | *nothing in the thesis* | the whole chapter: `P[dead] → ½`, rate `ε_ℓ ≈ 9π²/2ℓ²`, `Δ_ℓ ≈ 3/ℓ`, Slepian bound, 47.6% measurement | `docs/scratch/proofs/dying_neurons_clean_proof.tex` + `oracle_spotcheck_addendum.md` |
| 8 | Conclusions | chronology of attempts | rewrite around the no-go statement | this file |

## Write order

1. **Ch3 additions — the spine.** Three theorems, all derived and numerically verified. Transcription into LaTeX, not research. Reuse the existing `\E \Var \Cov` macros. **~1 week.**
2. **New dying-neurons chapter.** The `.tex` already exists in `docs/scratch/proofs/`; it needs the Slepian replacement folded in and the empirical figure added. **~3 days.**
3. **Ch4 expansion.** Mostly transcription from four campaign READMEs. This is where campaigns 09–12 stop being a chronology and become an argument. **~4 days.**
4. **Ch5 insert + ch8 rewrite.** **~2 days.**

≈ 2.5 weeks of writing. That is the whole remaining thesis.

## The one piece of actual research left

The **Slepian step** in the dying-neurons chapter is currently a sketch, not a proof: state the inequality, verify its hypotheses (equal variances after normalizing, one-sided event), and either make the `C√(ε log N)` constant explicit or keep the exact 1-D integral. Everything else on this map is writing.

## What not to do

- **Do not run more simulations.** 189 result JSONs is already more than the document can absorb. The open items from 2026-08-15 (the 100L recipe, the tuned-He baseline) matter only if you are still chasing a win over He — and under this reframe you are not.
- **Do not open campaign 13.** `FRONTIER.md` is a machine for starting experiments; it is the wrong tool for the next month.
- **Do not re-verify the numbers.** An adversarial audit re-derived 186 claims on 2026-08-15; the corrected figures are in `docs/reports/2026-08-15_campaign10_followup_synthesis.md`. Cite that, don't redo it.

## Before you start

Tell the advisor you are reframing from "find a better initializer" to "characterize why the natural fix cannot work, with the constant". He should agree or redirect **before** two weeks of writing, not after. The case: five theorems, twelve campaigns of consistent evidence, and a measured dying-neuron fraction matching the bound to 3%.
