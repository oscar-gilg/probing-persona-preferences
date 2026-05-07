# Random-direction L23 — multi-seed null (2 more seeds)

## Headline

Two new random unit directions at L23 (seeds 0, 1) produce swings in $P(\text{chose steered})$ across $c \in [-0.05, +0.05]$ of **0.167** and **0.191** — both with negative slope (P falls as c rises). Parent's seed 42 had swing **0.093** with positive slope. All three remain far below the validated probe's swing of **~0.95** over the same range; the sign flip across seeds is the seed-spread the spec asked for, and is consistent with averaging to ~0.5 once the three seeds are pooled (deferred to the local step).

## Per-seed summary

| seed | swing $|\max - \min|$ | slope | source |
|:--:|:--:|:--:|:--|
| 42 | 0.093 | + (P rises with c) | parent (`origin/experiment/random_direction_l23_quick`) |
| 0  | 0.167 | − (P falls with c) | this run |
| 1  | 0.191 | − (P falls with c) | this run |
| validated probe | ~0.95 | + | parent report, for scale |

Cross-seed dot product $\hat{v}_0 \cdot \hat{v}_1 = -0.0028$, well within the cosine-noise band $\pm 1/\sqrt{5376} \approx 0.014$ — the two new directions are effectively independent.

## Per-seed curves (canonical contrastive frame)

Swing column is $\max_c P - \min_c P$ over the 5 coefficients. P is $P(\text{chose steered}\mid\text{responded})$, with Wilson 95% CIs.

<!-- TABLES START -->
Per-seed swing |max − min| (this run): seed 0 = **0.167**, seed 1 = **0.191**.

### seed 0

| c | P(chose steered) | 95% CI | n responded | refusal rate |
|--:|:--:|:--:|--:|--:|
| -0.050 | 0.584 | [0.529, 0.637] | 317 | 11.9% |
| -0.030 | 0.541 | [0.486, 0.595] | 318 | 11.7% |
| +0.000 | 0.500 | [0.445, 0.555] | 318 | 11.7% |
| +0.030 | 0.459 | [0.405, 0.514] | 318 | 11.7% |
| +0.050 | 0.416 | [0.363, 0.471] | 317 | 11.9% |

Swing |max − min| = **0.167**.

### seed 1

| c | P(chose steered) | 95% CI | n responded | refusal rate |
|--:|:--:|:--:|--:|--:|
| -0.050 | 0.596 | [0.541, 0.648] | 319 | 11.4% |
| -0.030 | 0.489 | [0.435, 0.544] | 319 | 11.4% |
| +0.000 | 0.500 | [0.445, 0.555] | 316 | 12.2% |
| +0.030 | 0.511 | [0.456, 0.565] | 319 | 11.4% |
| +0.050 | 0.404 | [0.352, 0.459] | 319 | 11.4% |

Swing |max − min| = **0.191**.
<!-- TABLES END -->

Notes:

- $P = 0.500$ exactly at $c = 0$ is by construction. The canonical-frame mapping pairs each $(+c, \text{success}=A)$ row with a $(-c, \text{success}=B)$ row, so the $c=0$ bucket is balanced. This is a parsing-integrity check, not a result.
- Refusal rates are flat at ~11–12% across all coefficients — the swings are not driven by asymmetric refusal.
- Per-seed swings (0.17, 0.19) exceed the naive cosine-noise expectation $1/\sqrt{5376} \approx 0.014$. Plausible cause: with only 30 pairs × 2 orderings × 3 trials per coefficient (~320 responses), per-coef sampling noise stacks on the small but non-zero projection of each random vector onto the preference axis.

## Out of scope (deferred to local)

The spec defers averaging and plotting to local:

> No averaging or plotting on the pod — done locally after pulling results.

Local follow-up (per the spec):

1. Pull parsed jsonls from this branch:
   `experiments/random_direction_l23_quick/multi_seed/checkpoints/random_contrastive_seed{0,1}.parsed.jsonl`
2. Pull parent's seed-42 jsonl from `origin/experiment/random_direction_l23_quick`:
   `experiments/random_direction_l23_quick/checkpoints/random_contrastive.parsed.jsonl`
3. Average $P(\text{chose steered}\mid c)$ across the 3 seeds per $c$, with seed-spread error bars.
4. Update `paper/figures/panels/build_steering_integrated.py` to overlay the averaged null curve (instead of seed 42 alone) on Fig 3a, matching the visual style of the parent's single-seed plot.

The per-seed canonical-frame logic in `scripts/random_direction_l23_quick_multi_seed/analyze.py` mirrors parent's `scripts/random_direction_l23_quick/plot_null.py` row-mapping and is reusable for the averaging step.

## Setup

| | Value |
|:--|:--|
| Model | gemma-3-27b-it |
| Layer | 23 |
| Random directions | unit vectors from `np.random.default_rng(s).standard_normal(5376)`, L2-normalised, for $s \in \{0, 1\}$ |
| Activation scale | $\lVert h_{23}\rVert \approx 29382$ (mean over training pairs) |
| Coefficients | $-0.05, -0.03, 0, +0.03, +0.05$ |
| Pairs | 30 fixed pairs (sampled with seed 42 from `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`) |
| Mode | contrastive (`+c` on first task's tokens, `-c` on second's; differential cache injection) |
| Trials | 3 per (pair, ordering, $c$); temperature 1.0; max\_new\_tokens 64 |
| System prompt | none (default Assistant) |
| Total generations | $30 \times 5 \times 2 \times 3 = 900$ per seed (1800 total, 0 skipped) |

## Artefacts

- `results/probes/layer_sweep/eot/probes/probe_random_L23_seed{0,1}.npy` — random directions
- `results/probes/layer_sweep/eot/manifest.json` — manifest entries
- `configs/steering/random_direction_l23_quick_multi_seed/random_contrastive_seed{0,1}.yaml` — run configs
- `experiments/random_direction_l23_quick/multi_seed/checkpoints/random_contrastive_seed{0,1}.parsed.jsonl` — parsed generations (the deliverables for local averaging)
- `scripts/random_direction_l23_quick_multi_seed/make_random_probe.py` — seed-parameterised probe builder
- `scripts/random_direction_l23_quick_multi_seed/analyze.py` — per-seed canonical-frame analysis
- `scripts/random_direction_l23_quick_multi_seed/render_report_tables.py` — re-renders the per-seed table block above

## Deviations from spec

None. Both jsonls landed with the prescribed 900 rows × 180 per coef.
