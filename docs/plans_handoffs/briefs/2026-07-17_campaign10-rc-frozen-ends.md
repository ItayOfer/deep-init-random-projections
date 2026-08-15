# Brief — campaign10-rc-frozen-ends (2026-07-17)

**Onboarding chain (read in order before starting):** `README.md` → `docs/RESEARCH_LOG.md` → `docs/plans_handoffs/FRONTIER.md` → this brief → `CLAUDE.md` (esp. "Starting a new campaign" checklist).

## Goal

At 100 layers with the **plainest row-centered initialization** (`row_centered_he`), train only the **last 3 layers** (everything else frozen) and, separately, only the **first 3 layers** — to see where trainable signal can enter a deep row-centered network. Advisor's bet: training is fast in both cases.

## Context

- Follows campaign [09](../../../cluster/09_rcfwd_rescale/README.md): rcfwd showed gradient conditioning is solvable but representation content dies by layer ~25 in the row-centered family (probe chain in that README). This campaign surgically tests the two entry points for signal.
- Init: `row_centered_he` — He then subtract row means, **no** variance re-adjustment (registry name exactly; see INITIALIZERS.md). Per the gain-coupling lock (g_fwd/g_bwd = r): g_fwd ≈ 0.826, **g_bwd ≈ 1.0** — forward signal shrinks ~10⁻⁸ over 97 layers while gradients traverse at roughly unit scale. Both facts matter below.

## Experiment definition (confirmed with user 2026-07-17)

100L FC, width 500, **NoBN**, plain SGD lr 1e-2 (mom 0, wd 0, fixed LR), bs 256, seed 42, no clipping (assert), normalize inputs. Datasets: {fashion_mnist, cifar10}. Two conditions:

- **`last3`** — trainable: hidden layers 99, 100 **and the output head** (the head is the last layer; this matches the flow …98→99→100 all marked "train"). All other parameters frozen.
- **`first3`** — trainable: hidden layers 1, 2, 3. Everything else **including the output head** frozen at init (per the …→0→0 flow). *Flag: this head-frozen asymmetry is the literal reading of the spec — confirm with the advisor at the first opportunity; a `first3+head` variant is a natural addition if he intended the head trainable.*

Matrix: 2 conditions × 2 datasets × {smoke 20 ep, audit 200 ep} = 8 jobs. Smoke gates audit. `diagnostics_every=1` in smoke with per-layer gradient logging.

## Implementation notes (get these right)

1. **Freezing = `requires_grad=False` on the frozen Linear layers' parameters** — mathematically identical to "gradient deterministically 0" and cheaper. **Critical:** freezing weights must NOT stop backprop *through* those layers — gradients still flow to earlier trainable layers via the input-grad path; `requires_grad=False` on parameters preserves this. Do NOT use `torch.no_grad()`/detach on activations.
2. This needs a small shared-code change: add `ClassifierConfig.trainable_layers: Optional[list] = None` (None = all; else a spec like `["fc1","fc2","fc3"]` / `["fc99","fc100","head"]`), applied in `DeepFCClassifier` after init. **Touches `src/` → branch `work/rc-frozen-ends`** per the cycle rule. Keep it additive and default-None so nothing existing changes behavior.
3. Verify freezing in the smoke logs: frozen layers' `grad_norm_per_layer` entries should read as absent/zero for *parameter* grads; also assert at startup that the count of trainable tensors matches the condition (print it in the banner).
4. Runner: copy the campaign-09 skeleton per the CLAUDE.md checklist. Campaign dir `cluster/10_rc_frozen_ends/`; labels `rcfrozen_{first3,last3}_{smoke,audit}_{fmnist,cifar10}_100L` (= job name = JSON stem in flat `reports/results/`).
5. Numerical watch-items to log/report: activation RMS at the head (expect ~10⁻⁸ scale under this init — if the head's input is that small, head-weight gradients scale with it; report it, don't silently "fix" it), and per-layer grad norms of the trainable sets.

## Pre-registered predictions (so the outcome is informative either way)

- *Advisor's bet:* both conditions train fast.
- *Oracle's predictions from campaign-09 data:* **last3** — head sits on representations that are (a) content-dead (probe: chance by ℓ≈25) and (b) vanishingly small in scale (0.826^97) → stuck at chance. **first3** — gradients arrive at ~unit scale (g_bwd≈1) and the first layers sit on full-content inputs, but any learned change must survive 97 frozen scrambling layers forward → slow or stuck. Divergence between these predictions and the outcomes is the finding.

## Deliverables

1. Branch `work/rc-frozen-ends` with: the `trainable_layers` config + classifier support (additive), `cluster/10_rc_frozen_ends/` (runner + 8 subs), campaign `README.md` (standard shape; predictions section included).
2. Smoke results pulled to `reports/results/rcfrozen_*_smoke_*.json`; triage table in the Outcome section; audits only for conditions the smoke justifies (or on explicit user call).
3. Rows updated: `cluster/README.md`, `reports/results/INDEX.md` (when results land), FRONTIER W1.

## Constraints

Branch `work/rc-frozen-ends` (shared-file change); oracle merges. All campaign-09-era standing rules (seed 42, no clipping, flat results, label=jobname=stem). CPU 2-epoch local smoke of the freeze mechanics **before** any cluster sync. Model: mid-tier is fine for execution; triage judgment returns to the oracle.

## Definition of done

Local freeze-mechanics test passes (trainable-tensor count asserted per condition); 4 smoke JSONs pulled + committed (via merge); campaign README written with numbers traced to JSONs; Outcome filled; FRONTIER updated; oracle verification of the headline numbers done before RESEARCH_LOG is touched.

## Outcome  *(filled by the worker at the end)*

**Done:** branch `work/rc-frozen-ends` created. `ClassifierConfig.trainable_layers: Optional[List[str]] = None` added (additive, default None — no behavior change; `src/rp_study/config.py`). `DeepFCClassifier._freeze_except` + `trainable_tensor_count()` added and wired through `build_classifier` for `architecture="fc"` (`src/rp_study/models/classifiers.py`). `cluster/10_rc_frozen_ends/run_rc_frozen_ends.py` (copied campaign-09 skeleton, banner asserts trainable-tensor count matches the condition) + 8 `.sub` files + campaign `README.md` (predictions section included). Rows added to `cluster/README.md` and FRONTIER W1.

**Freeze-mechanics verification (local, CPU, before any cluster sync — required by the brief):** ran depth-100, width-500, `row_centered_he` builds for both conditions. Trainable-tensor count = 6 in both cases (3 layers × {weight,bias}), matching the assert now baked into the runner's `main()`. Post-backward `grad_norm_per_layer` is nonzero exactly at the named fc-layers and zero elsewhere in both conditions — confirms `requires_grad=False` suppresses only the frozen layers' own parameter gradients while backprop still traverses them (no `torch.no_grad()`/detach used). 2-epoch CPU runs complete (`status=completed`, no divergence) on both fashion_mnist and cifar10 for both conditions. Early numerical signal matching the brief's prediction: `last3`'s trainable gradients (fc99, fc100) sit at ~1e-4–1e-8 scale in the dry run, consistent with the predicted `0.826^97` forward-signal collapse reaching the head.

**Cluster run (user ran sync/submit/pull from their own terminal — this sandbox has no cluster SSH credentials).** All 4 smoke jobs (`rcfrozen_{first3,last3}_smoke_{fmnist,cifar10}_100L`) completed cleanly on the cluster: `status=completed`, `stop_reason=max_epochs`, no abort, banner assert `trainable tensors=6 (expect 6)` passed for every job. JSONs + `.out` logs pulled to `reports/results/` and `logs/slurm/10_rc_frozen_ends/`.

**Headline result — both conditions dead/stuck, by two different mechanisms, refuting the advisor's bet:**

| Cell | train acc ep1→ep20 | loss (all 20 ep) | trainable-layer grads | verdict |
|---|---|---|---|---|
| `last3`/fmnist | 0.1000 → 0.1000 (bit-exact) | 2.302585 = ln(10), bit-exact | fc99/fc100: 1e-10–1e-6 → exact float32 **zero** from ep3 | **DEAD** |
| `last3`/cifar10 | 0.1000 → 0.1000 (bit-exact) | 2.302585, bit-exact | fc99/fc100: 1e-10–1e-7 → exact float32 **zero** from ep8 | **DEAD** |
| `first3`/fmnist | 0.1019 → 0.1031 (noise) | 2.302585, bit-exact | fc1–fc3: 1.2–2.2, stable, never vanishing | **STUCK** |
| `first3`/cifar10 | 0.0983 → 0.1029 (noise) | 2.302585, bit-exact | fc1–fc3: 1.5–4.9, stable, never vanishing | **STUCK** |

`last3` confirms the oracle's prediction exactly (and sharper: the gradient doesn't just stay tiny, it underflows to exact `0.0` within a handful of epochs, permanently freezing training). `first3` refutes the "slow" half of the oracle's prediction: despite ~4,700 healthy, non-vanishing SGD steps (20 ep × ~235 batches, fmnist), the loss does not move even in the 6th decimal — not slow learning, zero learning, fully absorbed by the 97 frozen downstream layers. **Neither entry point trains.** Full mechanistic writeup in `cluster/10_rc_frozen_ends/README.md` §Findings.

**Triage call (raw recipe): audits not gated.** All four smoke cells show zero epochs of measurable progress (unlike campaign-09's promoted cells, which had partial-but-decreasing loss curves to extrapolate from) — `last3`'s gradient is identically zero so 200 more epochs is mathematically inert, and `first3` has no signal suggesting behavior would differ at 200 vs 20 epochs. The 4 raw-recipe audit `.sub` files remain available on explicit advisor call.

**Follow-up added to this same campaign (not a separate one), before oracle handoff, at the user's request:** a live disagreement existed about *why* campaign 09 needed `grad_rescale` at all — the advisor's worry (H1) was that the rescale might be masking/redirecting a real, trainable gradient rather than just conditioning its scale; the standing campaign-09 finding (H2) is that row-centered content death, independent of gradient conditioning, is the true bottleneck. Reran the identical `first3`/`last3` frozen design under campaign 09's exact corrected recipe (`row_centered_forward_balanced` init + `grad_rescale=r≈0.826`) to isolate the rescale as the only changed variable. Extended `run_rc_frozen_ends.py` with a recipe axis (`raw` vs `rcfwd`, selected by an `_rcfwd` label suffix, backward-compatible with the original 4 labels); local CPU 2-epoch mechanics check done before cluster sync; 4 new smoke jobs run and pulled.

**Result: asymmetric, not a clean win for either hypothesis.** `last3`/rcfwd **starts learning** — train accuracy climbs monotonically from ~11% to ~21% (fmnist) / ~19% (cifar10) over 20 epochs, gradients healthy throughout (no underflow, unlike the raw recipe). This shows the raw recipe's `last3` death was a **forward-scale artifact** (`row_centered_he`'s `0.826^L` decay reaching the head), not evidence of zero exploitable content at layer 99 — correcting the scale unlocks real, if slow, learning (H1-consistent for the tail, though the mechanism is forward-scale, not "the rescale was hiding a gradient" per se). `first3`/rcfwd **still fails** — gradients are healthy under both recipes (grad_rescale was never the limiting factor), but instead of the raw pass's flat inertness, the loss now climbs steadily past `ln(10)` (to 6.8–8.4) while accuracy stays pinned at chance: a real, well-scaled gradient is doing *something* to the network, just nothing useful, consistent with the 97 frozen downstream layers absorbing/scrambling any local change (H2-consistent for the front). Net: the bottleneck is asymmetric by layer position — a scale problem at the tail (fixable), a content/reachability problem at the front (not fixed by gradient conditioning). Full writeup in `cluster/10_rc_frozen_ends/README.md` §H1 vs H2, with 3 supporting figures (`docs/figures/rcfrozen_mechanisms.png`, `rcfrozen_rcfwd_mechanisms.png`, `rcfrozen_recipe_comparison.png`).

**Done:** `cluster/10_rc_frozen_ends/README.md` Findings + §H1 vs H2 + Evidence sections filled with numbers traced to all 8 JSONs, 3 figures generated (`scripts/rc_frozen_ends_plots.py`, `scripts/rc_frozen_ends_rcfwd_plots.py`) and curated into `docs/figures/`; `cluster/README.md` row 10 + campaign-details entry updated; `reports/results/INDEX.md` row updated; FRONTIER W1 updated. Branch `work/rc-frozen-ends` (pushed to origin) ready for the oracle to spot-check and merge.

**Not done / left open (as of the above):** `first3+head` variant (flagged above) not run. `last3`/rcfwd audit (200 ep) not run — flagged in the README as a real candidate given its still-improving trajectory, but promoting it is the oracle's/advisor's call, not made unilaterally here. RESEARCH_LOG untouched pending oracle spot-check per the definition of done.

**Design correction (2026-07-21), before oracle handoff:** the user caught that `last3` as defined above (`fc99, fc100, head` trainable) was asymmetric with `first3` (`fc1, fc2, fc3` trainable, head frozen) — the head was only ever trainable at one end. The intended design is symmetric: the head is always the fixed readout, never part of either trainable window. Corrected `TRAINABLE_LAYERS["last3"]` to `{fc98, fc99, fc100}` (head frozen in both conditions) in `run_rc_frozen_ends.py`. `first3` was unaffected (its head was already frozen) and was not rerun. Reran all 4 `last3` smoke jobs (both recipes × both datasets) under the corrected definition; local CPU 2-epoch mechanics check done first.

**Corrected results — the qualitative picture is unchanged, numbers updated:**

| Cell | train acc ep1→ep20 | loss (all 20 ep) | trainable-layer grads | verdict |
|---|---|---|---|---|
| `last3`/fmnist (raw) | 0.1000 → 0.1000 (bit-exact) | 2.302585, bit-exact | fc98/fc99/fc100: 1e-10–1e-5 → exact float32 **zero** from ep7 | **DEAD** |
| `last3`/cifar10 (raw) | 0.1000 → 0.1000 (bit-exact) | 2.302585, bit-exact | fc98/fc99/fc100: 1e-10–1e-5 → exact float32 **zero** from ep17 | **DEAD** |
| `last3`/fmnist (rcfwd) | 0.1087 → **0.2366** (steady climb) | 2.597 → **2.117** (steady fall) | fc98/fc99/fc100: 2.2–2.3 → 1.10–1.17, healthy | **LEARNING** |
| `last3`/cifar10 (rcfwd) | 0.1073 → **0.2200** (steady climb) | 2.603 → **2.144** (steady fall) | fc98/fc99/fc100: 1.75–1.83 → 1.17–1.24, healthy | **LEARNING** |

`first3` numbers (both recipes) are unchanged from above. All conclusions in §H1 vs H2 hold under the corrected definition — `last3`/rcfwd is if anything slightly stronger now (24%/22% vs. the previous asymmetric-design's 21%/19% at ep20), since `fc98` sits one layer earlier (marginally less forward-decayed) than the previous window. Regenerated all 3 figures (`scripts/rc_frozen_ends_plots.py`, `scripts/rc_frozen_ends_rcfwd_plots.py`) and rewrote `cluster/10_rc_frozen_ends/README.md`, `cluster/README.md`, `reports/results/INDEX.md`, and FRONTIER W1 with the corrected numbers and layer names throughout. The original `first3+head` flag is now moot (resolved by this correction — head is frozen in both conditions, symmetrically).

**Not done / left open (final):** a `head`-trainable-in-both variant (distinct from the resolved `first3+head` flag) was never run — low priority, likely to track existing results closely per the reasoning in the README's Evidence & gaps. `last3`/rcfwd audit (200 ep) still not run, still flagged as a real candidate, still the oracle's/advisor's call. RESEARCH_LOG untouched pending oracle spot-check per the definition of done.
