# Brief — campaign 10 continuation: 2-layer windows + recipe ablation (2026-08-15)

**Onboarding chain (read in order before starting):** `README.md` → `docs/RESEARCH_LOG.md` → `docs/plans_handoffs/FRONTIER.md` → this brief → `CLAUDE.md` (conventions). Task-specific deep dives: `cluster/10_rc_frozen_ends/README.md` (the whole campaign, incl. §H1 vs H2), `cluster/09_rcfwd_rescale/README.md` (the rcfwd recipe and the three-requirements frame).

## Goal

Answer two advisor follow-ups to campaign 10 in the same campaign directory: (1) redo the frozen-window experiments with **2-layer** windows instead of 3, and train the one working cell to a real accuracy target instead of a 20-epoch smoke; (2) run the **missing two corners** of the (initialization × grad_rescale) 2×2 so campaign 10's "it was a forward-scale artifact" conclusion becomes a measurement rather than an inference.

## Context

**Why now.** Advisor meeting after campaign 10 (2026-08-15). Two requests:

- *Window size.* Use 2 hidden layers per window, excluding the output head: `last2 = {fc99, fc100}` and `first2 = {fc1, fc2}`. Same symmetric design campaign 10 already corrected to — the head is frozen in **both** conditions and is never part of a trainable window.
- *Train to a target.* Once a cell shows a "successful formula", stop smoke-testing it and actually train it to **≈99% train accuracy**, tracking `eval_train_accuracy` and **not** `eval_train_loss`. Campaign 10 justifies this directly: its `first3`/rcfwd cell sat at chance accuracy while loss *climbed* to 6.8–8.4, so the two metrics decouple in exactly this regime.
- *Backward-only correction.* The advisor asked to "do the correction for the backward only, see what happens to the gradients in the forward, and check the activations — they will probably be small."

**What that third request means (settled with the user 2026-08-15, do not re-litigate).** The rcfwd recipe bundles **two independent interventions** and campaigns 09/10 only ever ran two of the four corners:

| corner | init | `grad_rescale` | ran in |
|---|---|---|---|
| `raw` | `row_centered_he` | `None` | c09, c10 "raw" |
| **`raw+rescale`** | `row_centered_he` | `r≈0.826` | **never** |
| **`fwdbal`** | `row_centered_forward_balanced` | `None` | **never** |
| `rcfwd` | `row_centered_forward_balanced` | `r≈0.826` | c09, c10 "rcfwd" |

"Correction for the backward only" = **`raw+rescale`**: leave the initialization alone (so the forward keeps its natural `0.826^ℓ` decay) and correct only the backward pass. `_GradRescale` is identity in the forward pass, so its activations are bit-identical to `raw` — the advisor's "the activations will probably be small" is true *by construction*, and the point of the corner is that it isolates (B) from (A).

**Prior art — already measured, do not re-derive.** `scripts/recipe_decomposition_funnel.py` → `reports/results/recipe_decomposition_funnel.json` measures all four corners at initialization (100L, width 500, batch 128, seed 42):

| corner | act RMS @L100 | ‖∂L/∂W‖ @L1 | @L100 | max/min |
|---|---|---|---|---|
| `raw` | 5.01e-9 | 1.91 | 1.15e-8 | 1.66e8 |
| `raw+rescale` | 5.01e-9 | 9.11e-9 | 9.48e-9 | **1.13** |
| `fwdbal` | 1.26 | 4.61e8 | 3.43 | 1.35e8 |
| `rcfwd` | 1.26 | 2.20 | 2.83 | 1.35 |

Read that table before designing anything. Three consequences that shape the run plan:

1. `raw+rescale` has the **flattest** gradient profile of all four corners (1.13× across 100 layers, better than rcfwd's 1.35×) but pins every layer at ≈9e-9 ≈ `r^100`. At the campaign's standing `lr=1e-2` it will not move. **It therefore needs an LR ladder** — the analytically-matched LR is ≈ `1e-2 / r^100 ≈ 2e6`; ladder around that, do not assume it.
2. `fwdbal` (no rescale) hits 4.6e8 at layer 1 and will almost certainly trip the runner's abort-on-explosion. Run it anyway — a clean documented abort *is* the result, and it is the corner that shows why the rescale exists. Give it one cheap job per dataset, not the full grid.
3. Because `_GradRescale` cannot touch the forward pass, the activation-RMS column has only two distinct values in the whole 2×2. Any story that explains `last3` via activations must therefore be an (A) story, not a (B) story.

**One more established fact worth not re-discovering:** `row_centered_forward_balanced` is `row_centered_he` with the rows rescaled to a larger target std — the per-layer activation-RMS ratio between them tracks `1.21^ℓ` to within 7% at ℓ=100, and mean pairwise cosine is ≈0.32 at *every* depth for both. The two recipes differ in scale, not in the statistical geometry of the representation.

## Deliverables

1. **Runner changes** in `cluster/10_rc_frozen_ends/run_rc_frozen_ends.py` (additive — must not change the meaning of any existing label):
   - `TRAINABLE_LAYERS` gains `"first2": ["fc1","fc2"]` and `"last2": ["fc99","fc100"]`. Keep `first3`/`last3` so the existing 8 JSONs stay reproducible.
   - `RECIPES` gains `"rawrescale": {"init_strategy": "row_centered_he", "grad_rescale": R}` and `"fwdbal": {"init_strategy": "row_centered_forward_balanced", "grad_rescale": None}`, with `RECIPE_SUFFIX` entries `"_rawrescale"` and `"_fwdbal"`. **Careful:** `_parse_label` currently detects the recipe with `label.endswith("_rcfwd")` — rewrite it to check the suffixes longest-first, and add a unit-style assertion that every label in `EXPERIMENT_LABELS` round-trips through `_parse_label` back to its own components. A silent mis-parse here would run the wrong recipe under the right filename, which is the single worst failure mode available in this campaign.
   - A `--target-train-accuracy` flag wired to `TrainingConfig.target_train_accuracy` (default `None`, so existing behavior is untouched).
2. **Smoke grid (20 ep, `diagnostics_every=1`, `log_grad_per_layer=True`)** — the 2-layer windows across the recipes that can plausibly move:
   - `{first2, last2} × {fmnist, cifar10} × {raw, rcfwd}` = 8 jobs at `lr=1e-2`.
   - `{first2, last2} × {fmnist, cifar10} × {rawrescale}` = 4 jobs, **each repeated over an LR ladder** (see Constraints).
   - `{last2} × {fmnist, cifar10} × {fwdbal}` = 2 jobs, expected to abort; documented as such.
3. **Target-accuracy runs** for every cell whose smoke shows monotone progress. Campaign 10 already identifies the front-runner: `last3`/rcfwd (11%→24% fmnist, 11%→22% cifar10 over 20 ep, no plateau). Run `last2` and `last3` under `rcfwd` with `--target-train-accuracy 0.99` and a large epoch budget (start 400; the existing audit subs are 200 ep / 6 h wall-time, so raise `--time` accordingly). **These are the runs that answer the advisor's actual question** — do not let the smoke grid crowd them out.
4. **`cluster/10_rc_frozen_ends/README.md` extended** with a new section covering: the 2-layer results next to the 3-layer ones, the 2×2 ablation table (init-time numbers from the JSON above + trained outcomes), and an explicit verdict on whether `last3`'s rcfwd recovery survives isolation of the two interventions.
5. **Record updates**: `reports/results/INDEX.md` row, `cluster/README.md` campaign-10 row, FRONTIER row, and the brief's Outcome section.

## Constraints

- **Branch: `main`.** Everything here is additive (new labels, new subs, new README section). The `src/` support (`trainable_layers`, `grad_rescale`) already exists and merged in `da925b8` — do not modify it.
- **Pass criterion for this campaign and onward: `eval_train_accuracy ≥ 0.99`; the loss condition is dropped.** This supersedes the `≥0.995 AND loss ≤0.10` rule in `CLAUDE.md` for campaign-10-onward work. A scan of every committed JSON found exactly **two** historical runs that flip to PASS under the new rule (both in `reports/results/fnn_he_bn_evaltrain_training.json`: acc 0.9941 / loss 0.0295, and acc 0.9919 / loss 0.0674) — both flip because of the accuracy threshold moving 0.995→0.99, **not** because the loss condition was dropped; no committed run has ever had acc ≥ 0.99 with loss > 0.10. Do not retroactively relabel the 12-architecture headline counts (He 10/12, V2 5/12) — those stay on the old criterion and are cited that way throughout the thesis. Keep logging both metrics.
- Standing rules: seed 42, width 500, depth 100, NoBN, plain SGD (momentum 0, wd 0, scheduler none), bs 256, `normalize_inputs=True`, **no gradient clipping** (assert-enforced in `main()`).
- Label = SLURM job name = JSON filename stem, every time. Output to `reports/results/<label>.json`.
- `.sub` files: copy an existing campaign-10 sub. Note the exclude is now `--exclude=dgx04` only (dgx01 was lifted in `ccc4713`); campaign-10's subs were realigned to that in this commit — do not reintroduce `dgx01`.
- **LR ladder for `rawrescale`**: `1e-2` (control, expected inert), then a geometric ladder up toward the analytically-matched `≈2e6`. Suggested: `1e2, 1e4, 1e6, 1e7`. This is *not* the forbidden "tune away instability" — the gradients here are uniformly 9e-9 by construction and the LR is the only free knob that can reach them. Say so explicitly in the README so it does not read as a violation of the no-clipping/no-rescue rule.
- Before syncing: clear `__pycache__` on the cluster (the runner changes). Verify the freeze mechanics for the new 2-layer windows locally on CPU first — the runner's `trainable_tensor_count()` assert should print `trainable tensors=4 (expect 4)`.
- Budget: smoke jobs are ~2 h each; target-accuracy runs need a raised `--time`. Submit the target-accuracy `rcfwd` runs **first** — they are the deliverable; the ablation grid is the supporting evidence.

## Definition of done

- [ ] `_parse_label` round-trip assertion passes for all labels in `EXPERIMENT_LABELS`, including the two new recipes and the two new window sizes.
- [ ] Local CPU freeze check for `first2`/`last2` shows exactly 4 trainable tensors and nonzero gradients at exactly the named layers.
- [ ] All smoke JSONs pulled into `reports/results/` and committed; each traced in the README's tables.
- [ ] At least one target-accuracy run completed (or a documented reason it could not be), reporting best `eval_train_accuracy` and the epoch it was reached.
- [ ] The `fwdbal` corner has a committed outcome — including "aborted on explosion at epoch N", which is a valid, expected result.
- [ ] README extended; `INDEX.md`, `cluster/README.md`, FRONTIER row and this brief's Outcome all updated.
- [ ] Verification note: state which numbers you re-read from which JSON, by field name.

## Outcome  *(filled by the worker at the end)*

