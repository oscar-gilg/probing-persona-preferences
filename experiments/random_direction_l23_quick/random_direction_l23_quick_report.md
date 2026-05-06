# Random-direction L23 contrastive control

## Question

Does a single random unit direction at layer 23, used as a contrastive
steering vector at the same magnitudes as the validated probe direction,
produce any swing in P(chose steered task)? This is the null-control
overlay for Fig 3a in the paper draft.

## Setup

| | Value |
|:--|:--|
| Model | gemma-3-27b-it |
| Probe | `random_L23_seed42` — `np.random.default_rng(42).standard_normal(5376)`, L2-normalised. Shape `(5377,)` `.npy`, last element = 0 intercept (matches the storage layout `load_probe_direction` expects). |
| Layer | 23 |
| `mean_norm[23]` | 29381.541015625 |
| Coefficients | -0.05, -0.03, 0.0, +0.03, +0.05 |
| Pairs | 30 deterministic pairs from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` (`n_pairs: 30`, `seed: 42` — `random.sample`, not literally first 30) |
| Mode | contrastive (`spans: {first: 1, second: -1}`, `cache_injection: differential`) |
| Trials | n=3, temperature=1.0, max_new_tokens=64, seed=42 |
| Template | `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml` |
| System prompt | none (default Assistant) |
| Total generations | 30 × 5 × 2 × 3 = 900 |

## Result

A random unit direction at the same coefficients produces **no preference
swing**. The 95% CIs on `P(chose steered task | responded)` overlap 0.5 at
every coefficient. Total swing $|\max - \min| \approx 0.09$ across $c \in
[-0.05, +0.05]$ — vs. the validated probe direction's $\geq 0.96$ swing
over the same range (Fig 3a default contrastive curve, `default_contrastive`
in the parent experiment).

| $c$ | $P(\text{chose steered} \mid \text{responded})$ | 95% CI | $n$ responded | refusal rate |
|:-:|:-:|:-:|:-:|:-:|
| -0.050 | 0.454 | [0.400, 0.508] | 324 | 10.0% |
| -0.030 | 0.474 | [0.420, 0.528] | 321 | 10.8% |
|  0.000 | 0.500 | [0.445, 0.555] | 316 | 12.2% |
| +0.030 | 0.526 | [0.472, 0.580] | 321 | 10.8% |
| +0.050 | 0.546 | [0.492, 0.600] | 324 | 10.0% |

Refusal rate is ~10–12% across all coefficients (no special "this random
direction breaks safety" effect). The monotonic 0.454 → 0.546 trend across
$c$ is small and within CI overlap; it is consistent with a near-zero but
not exactly orthogonal projection onto the preference axis (or with noise).
The exact 0.500 at $c = 0$ is a sanity check — the canonical-frame
symmetry-by-construction (each row contributes ±$c$) makes $c = 0$ exactly
balanced when parsing is correct.

![random L23 contrastive null](assets/plot_050626_random_L23_contrastive_null.png)

## Interpretation

The probe-trained direction's effect is **direction-specific**, not just a
function of injecting any vector at matched magnitude into layer 23. The
swing observed in the validated probe runs cannot be attributed to a
generic perturbation of the residual stream at L23 — only the
preference-aligned direction moves choice.

This justifies the null overlay in Fig 3a panel (a). The single-curve
random null is sufficient evidence; we do not need multiple random seeds
because the per-coefficient CIs already overlap 0.5 with this much data
($n \approx 320$ responded per coefficient).

## Artefacts

- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed42.npy`
- `results/probes/layer_sweep/eot/manifest.json`
- `configs/steering/random_direction_l23_quick/random_contrastive.yaml`
- `experiments/random_direction_l23_quick/checkpoints/random_contrastive.parsed.jsonl`
- `paper/figures/panels/build_steering_integrated.py` — extended with
  `load_random_contrastive()` + dashed-gray overlay support.

## Notes / caveats

- The parent experiment `persona_steering_l23_finegrain` referenced in the
  spec has no on-disk checkpoints on this branch, so the composite Fig 3a
  (default + random overlaid) cannot be rendered here. The standalone null
  plot in `assets/` is sufficient evidence for the null-control claim.
