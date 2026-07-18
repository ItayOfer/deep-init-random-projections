# 10 — rc frozen ends: where can trainable signal enter a deep row-centered net?

**Question.** At 100 layers with the **plainest row-centered initialization** (`row_centered_he` — He then subtract row means, no variance re-adjustment), train only the **last 3 layers** (hidden layers 99, 100 + the output head; everything else frozen) and, separately, only the **first 3 layers** (hidden layers 1, 2, 3; everything else, including the head, frozen at init). Advisor's bet: training is fast in both cases. Brief: [`docs/plans_handoffs/briefs/2026-07-17_campaign10-rc-frozen-ends.md`](../../docs/plans_handoffs/briefs/2026-07-17_campaign10-rc-frozen-ends.md).

**Builds on.** Campaign [09](../09_rcfwd_rescale/README.md) showed rcfwd's gradient conditioning is solvable in closed form, but representation content decays and hits chance by layer ≈25 in the row-centered family (per-layer probe chain). `row_centered_he` (this campaign, no GradRescale) has the raw, uncorrected gain pair from the gain-coupling lock: `g_fwd ≈ 0.826`, `g_bwd ≈ 1.0` per layer — forward signal shrinks by `0.826^97 ≈ 1e-8` over the depth between layer 3 and the head, while backward gradients traverse at roughly unit scale. This campaign asks, surgically: does trainable signal enter better through a **head sitting on that vanishingly small, content-dead representation** (`last3`), or through **early layers with full-content inputs but 97 frozen scrambling layers ahead of them** (`first3`)?

## Pre-registered predictions

- **Advisor's bet:** both conditions train fast.
- **Oracle's predictions (from campaign-09 data):** **last3** stuck at chance — the head sits on representations that are content-dead (probe: chance by ℓ≈25) *and* vanishingly small in scale (`0.826^97`). **first3** slow or stuck — gradients arrive at ~unit scale (`g_bwd≈1`) and the first layers sit on full-content inputs, but any learned change must survive 97 frozen scrambling layers forward before it reaches the loss. Divergence between these predictions and the outcomes is the finding.

## What ran

`run_rc_frozen_ends.py` — deliberately minimal so freezing-location is the only variable: init `row_centered_he`, NoBN, width 500, depth 100, plain SGD lr 1e-2 (mom 0, wd 0, fixed LR), bs 256, seed 42, no clipping (assert-enforced), normalize inputs. Matrix: 2 conditions (`first3`, `last3`) × 2 datasets (`fmnist`, `cifar10`) × {smoke 20 ep, audit 200 ep} = 8 subs. Smoke logs per-layer gradients every epoch (`diagnostics_every=1`, `log_grad_per_layer=True`).

Freezing implementation: `ClassifierConfig.trainable_layers` ([config.py](../../src/rp_study/config.py)) — additive, default `None` (no behavior change to any existing experiment) — applied in `DeepFCClassifier._freeze_except` ([classifiers.py](../../src/rp_study/models/classifiers.py)) as `requires_grad=False` on the frozen Linear layers' weight/bias tensors. This does **not** use `torch.no_grad()` or detach activations, so backprop still runs through frozen layers unchanged — gradients keep flowing to earlier trainable layers via the input-grad path.

**`first3+head` variant flag (per brief):** the literal reading of the advisor's spec freezes the head in the `first3` condition. This is a natural asymmetry to double back on with the advisor — a `first3+head` run (head trainable in both conditions) is a one-line addition (`trainable_layers=["fc1","fc2","fc3","head"]`) if he intended the head trainable in both. Not run in this pass.

## Freeze-mechanics verification (local, CPU, before any cluster sync)

Per the brief's constraint, the freezing mechanism was verified locally before syncing:

- **Trainable-tensor count**: both conditions report exactly 6 trainable tensors (3 Linear layers × {weight, bias}) via `DeepFCClassifier.trainable_tensor_count()`, asserted at runner startup (printed in the banner) and matching `2 × len(trainable_layers)`.
- **Gradient flow**: after a backward pass, `grad_norm_per_layer` is nonzero *exactly* at the named trainable fc-layers and zero everywhere else — confirming freezing suppresses only the named layers' own parameter gradients, not backprop through them.
- **`first3` full backward path**: with the head frozen, the loss gradient still traverses all 97 frozen layers back down to fc1–fc3 without error (2-epoch CPU run on fashion_mnist, 512 train samples, completes; `status=completed`, no divergence).
- **`last3` numerical watch-item (predicted in the brief) confirmed early**: in the CPU dry run, `last3`'s trainable-layer gradients (fc99, fc100) sit at **1e-4 to 1e-8 scale**, consistent with the `0.826^97` forward-signal collapse reaching the head before any real training has happened.

Dry-run scripts are not committed (scratch); the assertions above are now load-bearing in `run_rc_frozen_ends.py`'s `main()` (trainable-tensor-count assert) and reproduced by any smoke run.

## Findings

**Both the advisor's bet and the "slow" half of the oracle's prediction are wrong — both conditions are flat-dead at 20 epochs, by two different mechanisms.** From `rcfrozen_{first3,last3}_smoke_{fmnist,cifar10}_100L.json`:

| Cell | ep1 → ep20 train acc | loss | trainable-layer grad norms | verdict |
|---|---|---|---|---|
| `last3`/fmnist | 0.1000 → 0.1000 (bit-exact) | 2.302585 (= ln 10, bit-exact, all 20 ep) | fc99,fc100: 1e-10–1e-6 @ ep1 → **exact float32 zero from ep3 on** | **DEAD** |
| `last3`/cifar10 | 0.1000 → 0.1000 (bit-exact) | 2.302585 (bit-exact, all 20 ep) | fc99,fc100: 1e-10–1e-7 @ ep1 → **exact float32 zero from ep8 on** | **DEAD** |
| `first3`/fmnist | 0.1019 → 0.1031 (noise) | 2.302585 (bit-exact, all 20 ep) | fc1–fc3: 1.2–2.2, stable, never vanishing | **STUCK** |
| `first3`/cifar10 | 0.0983 → 0.1029 (noise) | 2.302585 (bit-exact, all 20 ep) | fc1–fc3: 1.5–4.9, stable, never vanishing | **STUCK** |

(Verdict taxonomy from campaign 06's `triage_row_centered_smoke.py`: DEAD = grads at the numerical floor; STUCK = loss flat near initial despite live gradients.)

![Campaign 10 mechanisms: loss/accuracy flat in all 4 cells; last3 gradients underflow to exact float32 zero; first3 gradients stay healthy but inert](../../docs/figures/rcfrozen_mechanisms.png)

*Top row:* `eval_train_loss` (rounded to 6dp — the reported precision; raw values differ only in the 7th-8th decimal, ordinary float noise) and `eval_train_accuracy` for all four cells across the 20-epoch smoke — both bit-exact/pinned at `ln(10)` and chance respectively. *Bottom row:* the trainable-layer gradient norms that produce these two distinct flat lines — `last3`'s fc99/fc100 (left, log scale, floored at 1e-12 for display) collapse to exact float32 zero by epoch 3 (fmnist) / epoch 8 (cifar10); `first3`'s fc1-fc3 (right, log scale) stay in the healthy 1-5 range for all 20 epochs and never approach the numerical floor. Generated by [`scripts/rc_frozen_ends_plots.py`](../../scripts/rc_frozen_ends_plots.py) from the 4 smoke JSONs.

**`last3` confirms the oracle's prediction exactly, and sharper than predicted.** The trainable head's input sits on the `0.826^97 ≈ 1e-8`-scale forward signal (§Context), so its init-time gradient is already at the 1e-6–1e-10 floor (matches the local dry-run observation in the verification section above). Under 20 epochs of plain SGD that gradient does not recover — it shrinks *further* and hits exact float32 zero (not "very small", identically `0.0` in the logged tensor) within 3–8 epochs, after which the run is mathematically frozen: `0 × lr = 0` every step. `eval_train_accuracy` and `eval_train_loss` are bit-identical to `1/10` and `ln(10)` for all 20 epochs — the network is outputting an (effectively) uniform distribution the entire run, because the head's weights never move enough to matter.

**`first3` refutes the "slow" half of the oracle's prediction — it is not slow, it is stuck.** Unlike `last3`, the fc1–fc3 gradients are healthy and *never vanish*: order 1–5 in magnitude, stable across all 20 epochs (no float32 floor problem here — the input is full-scale, undamaged data). Despite ~4,700 SGD steps (fmnist, 20 ep × ~235 batches) actually updating these weights every step, `eval_train_loss` does not move even in the 6th decimal place — it is bit-identical to `ln(10)` at every logged epoch on both datasets. This is stronger than "slow": whatever fc1–fc3 learn about the input gets **completely absorbed** by the 97 frozen, never-updated downstream layers before it reaches the loss — consistent with the campaign-09 finding that row-centered forward maps at this depth sit at a content-dead fixed point, but sharpened here: even *targeted* representation changes at the very front of the network produce no detectable output-level effect through that fixed frozen stack, at least within the smoke horizon.

**Triage call: do not gate to audit.** Both conditions show zero epochs of measurable progress across the full smoke window — `last3` because its gradient literally reached zero, `first3` because 20 epochs (~4,700 steps) of live, non-vanishing gradient produced no loss movement at all (not even a slow decline masked by noise). Unlike campaign-09 cells that were promoted on partial, decaying-but-still-decreasing loss curves, there is no comparable signal here to extrapolate from. 200 more epochs cannot move `last3` (its gradient is identically zero) and there is no evidence-based reason to expect `first3` to behave differently at 200 epochs than at 20. Per the brief, none of the four cells is promoted to the 200-epoch audit; the four unused `.sub` files (`rcfrozen_*_audit_*_100L.sub`) remain available if the advisor wants a direct 200-epoch confirmation regardless.

**Answering the campaign question:** neither entry point trains — signal cannot productively enter a 100L plain-row-centered net through either the last three layers (content + scale both dead there) or the first three (content and scale are fine, but 97 untrained frozen layers downstream absorb any change before it reaches the loss). This sharpens Phase 6/campaign-09's "content, not conditioning, is the bottleneck" finding: it is not merely that *forward-propagated* content is dead at depth — the network's *insensitivity* extends to gradient-driven changes made anywhere in it, front or back, as long as the vast majority of the stack stays frozen at this initialization.

## Reproduce

```bash
cd ~/thesis   # after sync; clear __pycache__ first (config.py gained a new field)
for cond in first3 last3; do
  for ds in fmnist cifar10; do
    sbatch cluster/10_rc_frozen_ends/rcfrozen_${cond}_smoke_${ds}_100L.sub   # 2h each
  done
done
# gate audits (6h each) on smoke triage:
# sbatch cluster/10_rc_frozen_ends/rcfrozen_<cond>_audit_<ds>_100L.sub
```

Pull back with `bash cluster/pull_results.sh 'rcfrozen_*_smoke_*' 10_rc_frozen_ends`. Logs end with `SUMMARY <label> | PASS/fail | ...`.

## Evidence & gaps

- 4/8 jobs run and pulled: `rcfrozen_{first3,last3}_smoke_{fmnist,cifar10}_100L.json` in `reports/results/`, logs in `logs/slurm/10_rc_frozen_ends/`. All 4 completed cleanly (`status=completed`, `stop_reason=max_epochs`, no abort) with the trainable-tensor-count banner assert passing (`trainable tensors=6 (expect 6)`).
- **Gap (by triage decision, not a blocker):** the 4 audit `.sub` files (200 ep) were not submitted — smoke showed zero epochs of progress in all four cells (see Findings), so promoting them was not evidence-justified per the brief's gating rule. Available to run on explicit advisor call.
- Cosmetic: the shared `print_diagnostic_summary` (from `cluster/03_he_diagnostics/run_diagnostic.py`) computes a min/max gradient-ratio banner assuming all layers are trainable; with most layers frozen (zero grad by construction) the printed ratio is a meaningless huge number (min=0 in the denominator) in the console log. Does not affect the saved JSON — only the printed console summary. Not fixed here since `run_diagnostic.py` is shared code outside this campaign's scope.
- The `first3+head` asymmetry flagged above is unconfirmed with the advisor — worth raising given `first3`'s result: even with the head trainable in `last3`, the head's own gradient underflowed to zero, so a `first3+head` variant would very likely die the same way `last3` did (head input is equally scale-dead regardless of which earlier layers are trainable). Low priority follow-up.
