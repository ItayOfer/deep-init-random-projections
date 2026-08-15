# 12 — frozen readout: ranking initializations by recoverable content at depth 100

**Question.** Freeze a 100-layer network at initialization, train **only `fc99` and `fc100`** (head frozen too), and vary nothing but the initialization. Final `eval_train_accuracy` then measures one thing: *how much usable class structure a random deep map leaves at its output.* Which initializer leaves the most?

**Why this campaign exists.** Campaign 09 concluded that row-centered representation **content** dies with depth, on the evidence of cosine-kNN and linear probes hitting chance by layer ≈25. Campaign 10 then trained only `fc98`–`fc100` on top of a frozen 100-layer row-centered stack and reached **0.8335** (fmnist) / **0.8170** (cifar10) at epoch ~243, still climbing — on a representation those probes called dead.

The probes understate badly. Information can be present without being linearly or metrically decodable, and the gap here is chance→83%, not a rounding error. **A trained readout is the honest instrument**, and this campaign uses it as one.

The same comparison also shows end-to-end training is the *worse* protocol at this depth — same init, same rescale, same data, same optimizer:

| protocol | epochs | fmnist | cifar10 | source |
|---|---|---|---|---|
| all 100 layers trainable | 200 | 0.1746 | 0.1301 | `rcfwd_rescale_audit_{ds}_100L.json` |
| only `fc98`–`fc100` trainable | ~243 | **0.8335** | **0.8170** | `rcfrozen_last3_audit_{ds}_100L_rcfwd.json` |

Training 3 layers beats training all 100 by ~4.8×. Campaign 10's `first3` cell supplies the likely mechanism: gradient updates at the *front* actively damage the representation (its loss climbed past `ln 10` while accuracy stayed at chance). Freezing the bulk protects it. That makes "frozen stack + trained readout" the right instrument for comparing what an initialization *leaves behind*, uncontaminated by whether end-to-end optimization happens to work.

**Builds on.** [09](../09_rcfwd_rescale/README.md) (the rcfwd recipe, the probe chain, the three-requirements frame) · [10](../10_rc_frozen_ends/README.md) (the `trainable_layers` freezing mechanism and the `last2`/`last3` protocol) · [11](../11_relu_shift/README.md) (the post-ReLU DC-removal arms and their init-time screen).

## What runs

`run_frozen_readout.py` — depth 100, width 500, NoBN, plain SGD lr 1e-2 (momentum 0, wd 0, scheduler none), bs 256, seed 42, no clipping (assert-enforced). `trainable_layers=["fc99","fc100"]`, head frozen. **Deliberately identical to campaign 10's `last2` audits** so the numbers are directly comparable across campaigns.

| arm | initialization | what it represents |
|---|---|---|
| `he` | `he` | the baseline every other family is trying to beat |
| `rc` | `row_centered_he` | weight-space DC removal |
| `rcfwd` | `row_centered_forward_balanced` + `grad_rescale=r` | campaign 09's corrected recipe |
| `c010` | `he` + post-ReLU shift `c=0.10` | activation-space DC removal, mild |
| `c025` | `he` + post-ReLU shift `c=0.25` | best init-time geometry (campaign 11 §5) |
| `c070` | `he` + post-ReLU shift `c=0.70` | best cosine, chance probe content — the trade-off's far end |

6 arms × 2 datasets × {smoke 20 ep, audit 400 ep} = 24 subs.

> **Do not resubmit the `rcfwd` arm if campaign 10's `rcfrozen_last2_audit_<ds>_100L_rcfwd` is already running or committed** — it is the same run (same init, rescale, window, optimizer, seed, epoch budget). It is defined here so the campaign is self-contained; cite the campaign-10 JSON instead of burning 12 h twice.

**Pre-run guards** (all verified locally on CPU before any sync): label round-trip over all 24 labels at import; every arm builds with exactly **4** trainable tensors, gradients landing at `fc99`/`fc100` only and `head.weight.grad is None`; and the runner asserts `model.relu_shift == config.relu_shift` at startup, which catches a stale `__pycache__` silently dropping the shift and running a plain-He arm under a `c=0.25` filename.

## Predictions (pre-registered)

- **If the probe story were right**, `rc`/`rcfwd` should be near chance and `he` should win — content is supposedly dead by layer 25 for the row-centered family.
- **The campaign-10 evidence says otherwise**: `rcfwd` is already at 0.8335 by this protocol. So the interesting question is whether `he` — which keeps 48% of its neurons *dataset-dead* at 100L (campaign 11 §1) — beats it or not.
- **The shift arms** are the real test of campaign 11. `c=0.70` has the best init-time cosine geometry and chance probe content; if geometry were what mattered it should win, and campaign 11's local pre-triage (stuck at chance on both datasets at 30L) says it will lose badly. `c=0.10`/`c=0.25` are the candidates that beat both baselines on distance-correlation-to-input at 30L.

A flat ranking would be as informative as a spread: it would say the readout, not the stack, is doing the work.

## Reproduce

```bash
# after sync; clear __pycache__ first (the runner asserts against stale bytecode)
cd ~/thesis
for ds in fmnist cifar10; do for arm in he rc c010 c025 c070; do
  sbatch cluster/12_frozen_readout/frozenro_${arm}_smoke_${ds}_100L.sub
done; done
# gate the 400-epoch audits on smoke triage; skip the rcfwd arm if campaign 10 has it
```

Pull with `bash cluster/pull_results.sh 'frozenro_*' 12_frozen_readout`. Logs end with `SUMMARY <label> | PASS/fail | ...`.

## Evidence & gaps

- **Nothing has run yet.** Every number above is quoted from campaigns 09/10/11; this campaign's own JSONs do not exist.
- The readout is 2 layers, matching campaign 10's `last2`. Campaign 10 measured `last3` ≈ 0.83 vs `last2` ≈ 0.64 at the same epoch — **window size matters a lot**, so this campaign's absolute numbers are readout-capacity-limited and only the *ranking across arms* is meaningful. A `k`-sweep (freeze all but the last `k`) is the natural follow-up and is not run here.
- Pass criterion is the advisor's 2026-08-15 rule (`eval_train_accuracy ≥ 0.99`, loss dropped). At 400 epochs campaign 10's `last2` projects to ≈0.78, so **no arm is expected to PASS**; the deliverable is the ranking, not a pass.
- The post-ReLU shift's `rms` is a batch statistic with no running-stats mechanism (campaign 11 §6) — eval uses the eval batch's own RMS. `batch_size == eval_batch_size == 256` keeps train and eval like-for-like.
