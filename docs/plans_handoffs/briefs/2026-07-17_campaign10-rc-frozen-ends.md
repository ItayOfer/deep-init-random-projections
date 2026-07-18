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

**Triage call: audits not gated.** All four smoke cells show zero epochs of measurable progress (unlike campaign-09's promoted cells, which had partial-but-decreasing loss curves to extrapolate from) — `last3`'s gradient is identically zero so 200 more epochs is mathematically inert, and `first3` has no signal suggesting behavior would differ at 200 vs 20 epochs. The 4 audit `.sub` files remain available on explicit advisor call.

**Done:** `cluster/10_rc_frozen_ends/README.md` Findings + Evidence sections filled with numbers traced to the JSONs; `cluster/README.md` row 10 + campaign-details entry updated; `reports/results/INDEX.md` row added; FRONTIER W1 to be updated next. Branch `work/rc-frozen-ends` ready for the oracle to spot-check and merge.

**Not done / left open:** `first3+head` variant (flagged above) not run — low priority given `last3`'s result already shows the head's gradient is scale-dead regardless of which upstream layers are trainable. RESEARCH_LOG untouched pending oracle spot-check per the definition of done.
