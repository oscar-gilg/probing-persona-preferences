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

### Dose-response figure

`paper/figures/plot_042426_layer23_dose_response_harm_breakdown.png` — both panels from 150-pair balanced set, `P(chose steered | responded)` vs coefficient applied to the steered task, refusal-rate band at the bottom of each panel.

### Per-pair-type numbers at L23

From `paper/claims/harm_breakdown.json`:

| Pair type | Contrastive swing | Single-task swing | Suppression | Amplification | Baseline |
|---|---|---|---|---|---|
| bb | **0.997** | **0.492** | **0.495** | **-0.003** | 0.500 |
| hb | **0.937** | **0.490** | **0.272** | **0.218** | 0.503 |
| hh | **0.927** | **0.498** | **0.243** | **0.255** | 0.475 |

### Contrastive headline (panel A)

Contrastive steering at L23 produces a near-full preference swing on every pair type, with clean symmetry around (0, 0.5). The 50-pair parent sweep's aggregate result holds up — and, crucially, holds up on the hh bucket, which was too thin ($n=8$) in the parent to trust. There is **no dead zone for harmful-vs-harmful pairs**: the probe direction pushes choice within the harmful regime as strongly as within the benign regime. Small attenuation: bb ≈ 1.00 > hb ≈ 0.94 ≈ hh ≈ 0.93.

### Single-task headline (panel B) — the suppression/amplification asymmetry is pair-type-dependent

The parent sweep's claim that "suppression is ~2-3× stronger than amplification at L23" held on the 50-pair aggregate. On the balanced 150-pair set, **that asymmetry is almost entirely driven by benign-benign pairs**:

- **bb:** essentially infinite ratio — suppression 0.495, amplification ≈ 0. On benign pairs, the model cannot be pushed toward a task by +c; it can only be pushed away by −c.
- **hb:** near-symmetric (supp 0.272, amp 0.218).
- **hh:** near-symmetric (supp 0.243, amp 0.255).

The 50-pair aggregate was bb-dominated (n=18 bb vs n=8 hh), so the visible asymmetry in the parent figure was primarily the bb signal bleeding into the mean. The harmful regimes show a cleaner two-sided response.

Refusal rates (visible in the panel B band) remain low — peak ~5-6% on hh at mild coefficients (|c|=0.03), elsewhere <2%. Stronger coefficients suppress refusals.

## Interpretation

- **Panel A replication:** the causal-direction claim survives the pair-type rebalancing. Not a benign-vs-harmful artefact.
- **Panel B refinement:** the bb single-task ceiling at ~0.5 suggests something unusual about the benign regime. One hypothesis: for benign-benign pairs the model is near-indifferent at baseline (0.5 exactly), and the positive steering direction doesn't encode "pick *this* task" as much as "pick the less-harmful-feeling task" — a distinction that collapses when both are benign. Negative steering still works because "avoid this task" is independently actionable.
- hb and hh show both sides working, suggesting that when harm distinguishes the two options, the probe direction becomes a two-sided valence handle rather than a one-sided avoidance handle.

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
