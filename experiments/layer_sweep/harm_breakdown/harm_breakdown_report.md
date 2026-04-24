# Harm-breakdown steering at L23 (150 pairs)

## Question

Does the asymmetric suppression-vs-amplification finding at L23 (parent §2.3) hold with a larger, pair-type-balanced sample? The parent sweep's 50-pair set had only n=8 harmful-harmful (hh) pairs — too thin to trust the single-task amplification curve for that bucket.

## Data

- **Source:** `default_test` (1000 tasks), same as parent. Disjoint from probe training (`default_train`, 4000 tasks) and probe eval (`default_eval`, 1000 tasks) — audited.
- **Sample:** 150 pairs total, 50 per pair-type:
  - **bb** (benign-benign): both origins in {WILDCHAT, ALPACA, MATH}
  - **hb** (harmful-benign): exactly one origin in {BAILBENCH, STRESS_TEST}
  - **hh** (harmful-harmful): both origins in {BAILBENCH, STRESS_TEST}
- `utility_gap > 0.1`, oriented so `task_a` has higher utility. Seed 42.
- Output: `steering_pairs_150.json` (written; audited: counts 50/50/50, all spans non-overlapping, no train/eval leakage).

## Configs

- **Contrastive:** `contrastive_L23_150.yaml` — single condition, `spans: {first:1, second:-1}`, probe `ridge_L23`, layer 23, multipliers ±0.03, ±0.05, 3 trials.
- **Single-task:** `single_task_L23_150.yaml` — two conditions (`unilateral_first` with `spans:{first:1}`, `unilateral_second` with `spans:{second:1}`), same probe/layer/mults/trials.
- `mean_norm[23] = 29381.541015625` (from parent sweep; not recomputed).
- Total generations: 3,600 contrastive + 7,200 single-task = 10,800.

## Pipeline fixes

- **Runner fix:** main's `src/steering/runner.py` only accepted scalar `mean_norm`; dict-valued mean_norm threw `TypeError` at injection. Ported the per-layer mean_norm refactor from `research-loop/layer_sweep` (commit 6165b69). Applied as `f4beb0b` on this branch.
- **Checkpoint dir:** runner doesn't auto-create checkpoint parent dir. Pre-created `experiments/layer_sweep/harm_breakdown/checkpoints/` on the pod.

## Results

_Pending completion of full runs._

### Dose-response figure

![L23 dose-response by pair type](assets/plot_TODO_layer23_dose_response_harm_breakdown.png)

(Panel A = contrastive; Panel B = single-task aggregate over first/second spans. Baselines at x=0 pulled from parent sweep's dead layers, matched by pair_type via 50-pair origins.)

### Per-pair-type numbers at L23

_To be filled from `paper/claims/layer_sweep.json` after runs complete._

| Pair type | Contrastive swing | Single-task swing | Suppression | Amplification |
|---|---|---|---|---|
| bb | TBD | TBD | TBD | TBD |
| hb | TBD | TBD | TBD | TBD |
| hh | TBD | TBD | TBD | TBD |

## Interpretation

_TBD._

## Reproducing

```
python -m experiments.layer_sweep.harm_breakdown.build_pairs_150    # → steering_pairs_150.json
python -m scripts.isolated_steering.run_steering configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml
python -m scripts.isolated_steering.run_steering configs/steering/layer_sweep/harm_breakdown/single_task_L23_150.yaml
python scripts/paper/plot_layer_sweep_dose_response_150.py
python scripts/paper/claims/compute_layer_sweep_claims.py
```

## Infra notes

- Spec's `pod 1` (`layer-sweep-extract`) + siblings `layer-sweep-eot`/`layer-sweep-unilateral` all blocked at resume by "no free GPUs on host". Launched fresh `harm-breakdown-l23` (A100-SXM4-80GB, pod id `9sp4um6vfne1am`). Activations needed (`pref_layer_sweep/activations_eot.npz`, 2.4GB) were NOT synced — the runner uses the hardcoded `mean_norm` dict, so activations are only needed for probe-fit / norm-compute, which we skip per spec.

## Out of scope

Per spec: no broader layer sweep, no tb:-2 probe at L23, no coefficients beyond ±0.03, ±0.05.
