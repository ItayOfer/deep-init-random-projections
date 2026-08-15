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

- [x] `_parse_label` round-trip assertion passes for all labels in `EXPERIMENT_LABELS`, including the two new recipes and the two new window sizes.
- [x] Local CPU freeze check for `first2`/`last2` shows exactly 4 trainable tensors and nonzero gradients at exactly the named layers.
- [ ] All smoke JSONs pulled into `reports/results/` and committed; each traced in the README's tables. **NOT DONE — no cluster access this session (no SSH/scp/sbatch).** All 34 `.sub` files are written, dry-run-verified, and ready; see Outcome + README "Reproduce → W5 continuation" for the exact handoff.
- [ ] At least one target-accuracy run completed (or a documented reason it could not be), reporting best `eval_train_accuracy` and the epoch it was reached. **Documented reason: no cluster access this session.** The 4 target-accuracy `.sub` files (`last2`/`last3` × {fmnist,cifar10} × rcfwd, `--epochs 400 --target-train-accuracy 0.99`) are written and dry-run-verified, first in the submission priority order.
- [ ] The `fwdbal` corner has a committed outcome — including "aborted on explosion at epoch N", which is a valid, expected result. **Not committed (no cluster run yet).** A *local* CPU pre-check (5 epochs, 4000-sample subset) found `last2`/fwdbal does **not** abort — contradicts the brief's stated expectation for that specific cell; see Outcome and README for the mechanism and the `first2`/fwdbal control check that does abort as expected.
- [x] README extended; `INDEX.md`, `cluster/README.md`, FRONTIER row and this brief's Outcome all updated.
- [x] Verification note: state which numbers you re-read from which JSON, by field name. See Outcome below.

## Outcome  *(filled by the worker at the end)*

**Summary: the code/infrastructure half of this brief is complete and locally verified; the cluster-execution half could not run.** This session had no SSH/scp/sbatch access (as flagged in the task's onboarding note as a known risk for this campaign). Everything checkable without a GPU was checked; everything requiring one is prepared, dry-run-verified against the real code path with training stubbed out, and handed off with exact commands. No git history was rewritten and no committed evidence file was altered — see the "process note" at the end regarding a dry-run mistake that was caught and reverted within this session.

**1. `_parse_label` rewrite (the brief's flagged highest-risk edit).** Rewrote to try `RECIPE_SUFFIX` entries longest-first (`_RECIPE_SUFFIX_ORDER`), since `raw`'s suffix is `""` and would `endswith()`-match every label if tried first or in plain dict order. Added `_check_label_roundtrips()`, run unconditionally at *import time* over all 80 `EXPERIMENT_LABELS` entries (64 base grid + 16 rawrescale-LR-ladder labels), asserting each parses back to exactly the components that built it. Verified two ways: (a) `python3 -c "import run_rc_frozen_ends"` succeeds silently — the self-test passed; (b) confirmed the self-test is not vacuous by simulating the naive insertion-order check against every label — it mis-parses **64 of 80** as `"raw"` (e.g. `rcfrozen_first3_smoke_fmnist_100L_rcfwd` → naive `raw`, correct `rcfwd`), i.e. exactly the failure mode the brief described.

**2. New recipes/conditions/flag, additive.** `RECIPES` gained `rawrescale` (`row_centered_he`+`grad_rescale=r`, "(B) only") and `fwdbal` (`row_centered_forward_balanced`, no rescale, "(A) only"); `TRAINABLE_LAYERS` gained `first2=[fc1,fc2]`/`last2=[fc99,fc100]`; `raw`/`rcfwd`/`first3`/`last3` are byte-identical to before (only additions, no field of any existing label's resolved `ClassifierConfig`/`TrainingConfig` changed). `--target-train-accuracy`/`--target-patience` wire the pre-existing `TrainingConfig.target_train_accuracy`/`target_patience` fields (already in `config.py` — no `src/` changes were needed or made). Verified locally: a 3-epoch CPU run with `target_train_accuracy=0.99` on `last2`/rcfwd completes cleanly (`stop_reason=max_epochs`, correctly not triggered early on a 512-sample subset).

**3. Local freeze-mechanics check (Definition-of-done item 2).** For `first2` and `last2`, under both `raw` and `rcfwd`: `trainable_tensor_count()==4` in all four combinations; after one backward pass, nonzero gradient at exactly `{fc1,fc2}` (first2) / `{fc99,fc100}` (last2) and zero everywhere else including the head. Matches the brief's expected banner (`trainable tensors=4 (expect 4)`) exactly.

**4. 34 new `.sub` files, all dry-run-verified.** 4 target-accuracy (`last2`/`last3` × {fmnist,cifar10} × rcfwd, `--epochs 400 --target-train-accuracy 0.99`, `--time=12:00:00`) + 8 smoke grid (`first2`/`last2` × {fmnist,cifar10} × {raw,rcfwd}) + 20 rawrescale LR ladder (control `lr=1e-2` + rungs `1e2,1e4,1e6,1e7`, × `first2`/`last2` × {fmnist,cifar10}) + 2 fwdbal (`last2` × {fmnist,cifar10}, per brief scope). Every `.sub` file's exact `srun`/argparse invocation was replayed locally through the real `main()` (argparse → `_parse_label`/`_strip_lr_ladder_tag` → `_build()` → the trainable-tensor-count and no-clipping asserts), with `run_supervised_experiment` monkeypatched to a stub so no GPU/dataset work happened — all 46 `.sub` files (12 pre-existing + 34 new) passed. Template self-checked byte-for-byte against an existing committed `.sub` before generating anything new.

**5. Deviation from the brief, found by the mandated local check: `last2`/`fwdbal` likely will NOT abort.** The brief expected this corner to abort ("hits 4.6e8 at layer 1... will almost certainly trip the runner's abort-on-explosion"). That 4.6e8 figure is `corners.fwdbal.grad_norm_per_layer[0]` in `recipe_decomposition_funnel.json` — i.e. **fc1's** gradient, which decays monotonically to `corners.fwdbal.grad_norm_per_layer[99]` = 3.427 (near-normal) by fc100. Because `last2` only trains fc99/fc100 and PyTorch autograd never computes a gradient through a chain whose input has `requires_grad=False` (fc1–fc98 are frozen), the exploding fc1 gradient is structurally irrelevant to `last2`. Local CPU verification (5 epochs, 4000 fmnist samples): `last2`/fwdbal shows real, non-exploding learning (loss 3.24→2.67, acc 10.5%→14.3%, grad norms stay 2–4); a control check on `first2`/fwdbal (not in the brief's grid) *does* abort (`stop_reason=non_finite_loss`, NaN at epoch 1 batch 4), confirming the mechanism is real but front-loaded. `last2`/fwdbal's `.sub` files were still created and are first in the fwdbal submission tier per the brief's explicit scope — the real 20-epoch GPU smoke on the full dataset is the actual test; the local result is a preview, not a substitute, and is written up with that caveat in the campaign README.

**6. Numbers re-verified, by file + field name (per Definition-of-done item 7):**
- `reports/results/recipe_decomposition_funnel.json`: `corners.raw.activation_rms[99]`=5.009e-9, `corners.raw.grad_norm_per_layer[0]`=1.906, `corners.raw.grad_norm_per_layer[99]`=1.149e-8, `corners.raw.grad_max_over_min`=1.659e8; `corners["raw+rescale"].grad_norm_per_layer[0]`=9.109e-9, `[99]`=9.484e-9, `.grad_max_over_min`=1.127, `.activation_rms` bit-identical to `corners.raw.activation_rms` (checked list-equality); `corners.fwdbal.grad_norm_per_layer[0]`=4.612e8, `[99]`=3.427, `.activation_rms[99]`=1.2551; `corners.rcfwd.grad_norm_per_layer[0]`=2.204, `[99]`=2.829, `.grad_max_over_min`=1.353. `config.r`=0.8256452711765564 — matches `math.sqrt((math.pi-1)/math.pi)` computed independently.
- Analytically-matched LR recomputed independently: `1e-2 / R**100` with `R=0.8256452711765564` → `R**100=4.779e-9`, LR=`2,092,427` (brief's "≈2e6" confirmed to more precision).
- Campaign 10's original findings (`last3`/rcfwd 11%→24%/22% train acc over 20 ep, `first3` loss climbing to 6.8/8.4) were re-read from `rcfrozen_{first3,last3}_smoke_{fmnist,cifar10}_100L_rcfwd.json` `history[].eval_train_accuracy`/`eval_train_loss` fields — unchanged, cited from the existing README, not re-derived.

**Process note (self-reported, not a design issue):** an early dry-run test that replayed all 46 `.sub` files' real CLI arguments through `main()` (to verify argparse/`_build()`) initially let `main()`'s unconditional `output_path.write_text(...)` run against the *real* `reports/results/` paths, which briefly overwrote 8 committed JSONs with stub (empty-history) content and created ~38 stray stub files. Caught immediately via `git status`; the 8 committed files were restored with `git checkout --`, all stray untracked files were deleted, and `git status --porcelain` was re-verified clean before continuing. No committed evidence was lost; flagging this so the oracle can spot-check `reports/results/rcfrozen_{first3,last3}_smoke_*.json` against their last real commit (`da925b8` / campaign-10 merge) if desired — `git log -p` on those 8 paths should show no changes since that commit.

**What the oracle should verify:** (a) the `_parse_label`/round-trip logic in `run_rc_frozen_ends.py` (highest-risk edit); (b) that the 8 pre-existing JSONs are untouched (`git diff da925b8 -- reports/results/rcfrozen_{first3,last3}_smoke_*.json` should be empty); (c) the fwdbal asymmetry claim in §5 above, since it revises the brief's own expectation; (d) whether the priority/tiering in the handoff commands matches what the advisor actually wants submitted first given current cluster load.

