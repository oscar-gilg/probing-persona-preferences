---
status: draft
model: qwen3.5-122b-nothink
parent: experiments/qwen_replication/qwen_replication_spec.md
gemma_analogues:
  - experiments/layer_sweep/layer_sweep_spec.md
  - experiments/layer_sweep/harm_breakdown/harm_breakdown_spec.md
---

# Qwen-3.5-122B contrastive steering: layer sweep + harm-breakdown

## Question

Does the contrastive steering dose-response result (paper Fig. 5) replicate on Qwen-3.5-122B-A10B-nothink? Specifically:

1. **Where is the causal peak?** Gemma's steering window is L17--26, peak L23 (37% of 62 layers), which sits *six layers before* the probe-quality peak (L29). We don't know whether Qwen's causal peak tracks its probe peak (L38, 79%) or sits earlier. Probe-decodability is uniformly strong across Qwen's turn-boundary positions and deeper layers, so the prior on causal-vs-decoding decoupling is weaker than on Gemma; we sweep.
2. **Per-pair-type controllability at the peak.** Does $P(\text{chose steered task} \mid \text{responded})$ span $[0,1]$ across $c \in \{\pm 0.03, \pm 0.05\}$ on the 150-pair balanced set (50 benign--benign, 50 harmful--benign, 50 harmful--harmful)? Does single-task steering recover roughly half the contrastive swing?

This spec covers both phases. Phase B can only launch once Phase A's peak is identified.

## Model

**Qwen-3.5-122B-A10B-nothink** (MoE, 122B total / 10B active, 48 transformer blocks). Requires `device_map=auto` on 2+ A100-80GB GPUs.

## Data: rebuild pair sets from a disjoint subset of canonical_test

We cannot reuse Gemma's pair JSONs verbatim because 23/93 (Phase A) and 72/240 (Phase B) of their unique tasks fall inside the Qwen 10k AL probe-train set. Filtering Gemma's pairs to the disjoint subset only leaves 22/50 (Phase A) and an unbalanced 21/20/14 across bb/hb/hh (Phase B) -- insufficient. So we sample fresh.

**Disjoint pool:** $\text{canonical\_test} \setminus (\text{qwen35\_10k\_AL} \cup \text{qwen35\_4k\_AL})$ = 706 tasks (706/1000 safe). Origin breakdown: ALPACA 196, MATH 188, WILDCHAT 163, STRESSTEST 138, BAILBENCH 21. Pair capacity (combinatorial upper bound, before utility-gap filter): bb $\sim 64$k, hb $\sim 57$k, hh $\sim 12$k. Plenty for 50 and 150 pairs.

**Phase A pair set:** 50 pairs, sampled in `experiments/qwen_replication/steering_layer_sweep/build_steering_pairs_50.py`. Same rule as Gemma's `experiments/layer_sweep/build_steering_pairs.py`:
- Tasks restricted to the disjoint pool.
- `utility_gap > 0.1`, `task_a` higher utility (using **Qwen's** `default_test` utilities -- see below), fixed seed, stratified across origin cross-product.
- Run `find_pairwise_task_spans` with `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B")`; assert non-None, non-overlapping spans for every pair.
- Output: `experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json`.

**Phase B pair set:** 150 pairs (50 bb / 50 hb / 50 hh), sampled in `experiments/qwen_replication/steering_layer_sweep/build_pairs_150.py`. Same rule and disjoint-pool restriction; assert exact $(50, 50, 50)$ counts. Output: `experiments/qwen_replication/steering_layer_sweep/steering_pairs_150.json`.

**Utility source:** Qwen's `default_test` Thurstonian fit at `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_test_task_ids/`. Loaded via `src.measurement.storage.loading.load_run_utilities`. Using Qwen's utilities (not Gemma's) keeps the orientation `task_a > task_b` aligned with what the model actually prefers, so the steering curve is anchored to the right baseline at $c=0$.

**Cross-model comparability:** the Qwen and Gemma pair sets will differ at the task level. The cross-model comparison in the report happens at the pair-type-aggregate level (mean swing on bb / hb / hh), which is the unit the paper figure already uses -- pair-sampling variance is one of the things we expect to be small relative to per-pair-type swing differences.

A small disjoint convenience script `scripts/_disjointness_check.py` is already written and verified: it reproduces the 224 / 70 / 294 / 706 numbers above. The build scripts re-derive the disjoint pool internally so they are self-contained.

## Selectors

Two probe-extraction positions, both swept in Phase A:

- **`turn_boundary:-1`** (tb-1, final-prompt) -- the canonical extraction position for Qwen in the paper.
- **`turn_boundary:-4`** (tb-4, `\n` after `<|im_end|>`) -- the absolute heldout-r peak position in the Qwen probe report (r = 0.946 at L38, fractionally above tb-1).

The probe-extraction position determines the *direction*; injection happens at task-content spans regardless of which probe is used. We sweep both to test whether the strongest-decoding probe is also the strongest causal handle.

## Probes (already trained, no refit)

10k-trained ridge probes at `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m{1,4}/`, six layers each:

| Layer | Depth | tb-1 probe | tb-4 probe |
|---|---|---|---|
| 12 | 25% | `ridge_L12` | `ridge_L12` |
| 24 | 50% | `ridge_L24` | `ridge_L24` |
| 28 | 58% | `ridge_L28` | `ridge_L28` |
| 33 | 69% | `ridge_L33` | `ridge_L33` |
| 38 | 79% | `ridge_L38` (peak r = 0.946) | `ridge_L38` (peak r = 0.946) |
| 43 | 90% | `ridge_L43` | `ridge_L43` |

Activation NPZs: `activations/qwen35_122b/pref_main/activations_turn_boundary:-{1,4}.npz`.

**Why 10k probes, not 4k canonical.** The 4k canonical-train probes (`results/probes/qwen_persona_sweep_final_six/default_tb-{1,4}/`) only exist at L33, L38, L43 -- all in the deep half (69--90%). Sweeping only those would miss any causal peak in the shallow/mid layers (Gemma's was at 37% depth). The 10k probes cover L12 onward and are the only way to characterise the layer-causal curve without new extraction.

**Why the disjoint-pool pair sampling matters for these probes.** The 10k AL training set leaks into Gemma's existing pair JSONs (25--30% task overlap, see Data section). Sampling fresh pairs from the disjoint subset of `default_test` removes this leakage by construction, so steering measurements are taken on tasks the probe never saw during fit or alpha selection.

## Phase A: Layer sweep (find the causal peak)

### Steering grid

**Diagonal-only** (probe layer = injection layer): $\{\text{tb-1}, \text{tb-4}\} \times \{12, 24, 28, 33, 38, 43\} = 12$ cells. All cells use:
- `cache_injection: differential`, `spans: {first: 1, second: -1}` (contrastive)
- `multipliers: [-0.05, -0.03, 0.03, 0.05]`
- `n_trials: 3`

One YAML per (selector, layer) at `configs/steering/qwen_layer_sweep/phase_a_<sel>_L<L>.yaml` (slug `tb1` / `tb4`), generated by `gen_configs.py`.

### Generation parameters (locked across all cells, paper-aligned with Gemma)

- `max_new_tokens: 64` (matches Gemma layer-sweep configs and CLAUDE.md's $\geq 16$ rule for the canonical pairwise template)
- `temperature: 1.0`, `seed: 42`
- **Choice parser:** `src/measurement/elicitation/completion_judge.py` (model `gemini-3-flash-preview`), auto-run by the runner's post-hoc phase. Refusal is a distinct parse outcome and is excluded from $P(\text{chose steered} \mid \text{responded})$ -- reported separately. **Do not invent a string-match parser.**
- **Choice convention.** `spans: {first: 1, second: -1}` adds $+v$ to task A's tokens and $-v$ to task B's, so for $c > 0$ the "steered task" is A; for $c < 0$ it is B. The metric is computed as $P(\text{chose A})$ at $c \geq 0$ and $P(\text{chose B})$ at $c < 0$, equivalently $P(\text{chose first task} \mid \text{responded})$ remapped by the orientation. Use the same `_remap_choice` logic the Gemma analysis uses (in `experiments/layer_sweep/analyze_steering.py` and the `choice_original` field of `*.parsed.jsonl`).

### Volume

$12 \text{ cells} \times 50 \text{ pairs} \times 4 \text{ multipliers} \times 2 \text{ orderings} \times 3 \text{ trials} = 14{,}400$ generations. Plus a shared 0-coefficient baseline ($50 \times 2 \times 3 \times 2 = 600$ gens).

Wall clock estimate: ~3--5 h on 2× A100 80 GB. The probe-vector swap is free per cell; model loaded once.

### Norms

`mean_norm[L]` per (selector, layer) from `src/steering/calibration.py::per_layer_norms("activations/qwen35_122b/pref_main/activations_turn_boundary:-<sel>.npz", layers=[12, 24, 28, 33, 38, 43])`, run once per selector at config-generation time and embedded inline in each YAML. Norms differ across selectors; this keeps $c = \pm 0.05$ a fixed *fraction of resid norm* in each cell.

**Norm scale caveat.** Qwen's cached `mean_norms` are 1.96 (L12) → 7.60 (L43) on tb-1; Gemma's at comparable depths run 765 → 29 000+. The 100x--1000x absolute gap is suspicious -- likely the Qwen extraction stored post-RMSNorm or otherwise re-scaled residuals. Within-Qwen the values grow monotonically with depth so layer-comparisons are valid. The Phase A positive control (pre-run check 6) is the unit-test: if $c=+0.05$ at L38 doesn't move choice on 5 pairs, scale up the coefficient grid before launching the full sweep and document the multiplier in the running log.

### Phase A analysis

Script: `experiments/qwen_replication/steering_layer_sweep/analyze_phase_a.py`. Reads the 12 `*.parsed.jsonl` checkpoints. Outputs:

- `assets/plot_<mmddYY>_phase_a_diagonal.png` -- $P(\text{chose steered task})$ vs injection layer, one panel per selector, one line per multiplier.
- `assets/plot_<mmddYY>_phase_a_swing.png` -- swing $\Delta P = P(c=+0.05) - P(c=-0.05)$ vs layer, one line per selector. The headline plot for "which layer has the biggest causal effect".
- `assets/plot_<mmddYY>_phase_a_refusal.png` -- refusal rate vs layer, one line per selector. Sanity check.
- `phase_a_summary.json` -- `{selector: {layer: {swing, refusal_rate, mean_p_pos, mean_p_neg}}}`.

### Phase A peak rule

Promote a cell to Phase B if its swing is the largest (or tied within tolerance) **and** its refusal rate (computed at $|c|=0.05$, the worst case in the grid) is below 5%. Concretely:

- **Identify the top cell across all 12** by swing (refusal-eligible only at $|c|=0.05$).
- **Tight ties (within 0.05 swing) → run Phase B at the top 2 cells.** If both are the same selector at adjacent layers, the second harm-breakdown is a layer-robustness check; if they're different selectors at the same/similar layer, it's a selector-robustness check.
- **Otherwise → single peak.** Run Phase B once.

Document the choice and the runner-up swings in the running log. Layer-causal data from all 12 cells is itself an output (Phase A figures), independent of which one(s) get promoted.

### Random-direction control (Phase A peak)

After identifying the peak, run **one extra cell** at the peak (selector, layer) with the probe direction replaced by a random unit vector (`seed=0`, drawn once, persisted in the running log). Same multipliers, `n_trials=3`. ~600 gens.

This is the in-experiment analogue of Gemma's random-direction baseline (paper §2.3, "A random direction at matched magnitude has no effect"). We **cannot** transfer Gemma's null result to Qwen because Qwen's cached norms are 100--1000$\times$ smaller (post-RMSNorm storage suspected, see §Norms). The control here is what calibrates "$0.05 \times \text{Qwen-norm}$" as the right perturbation magnitude.

**Pass criterion:** swing on the random direction $\leq 0.1$. If $> 0.1$, the norm calibration is wrong and all swings need re-interpretation -- pause, re-derive the right scale, document.

## Phase B: Harm-breakdown at the peak (paper Fig. 5 analogue for Qwen)

### Pair set

`experiments/qwen_replication/steering_layer_sweep/steering_pairs_150.json` -- built fresh from the disjoint pool (see Data section).

### Steering configs

Per promoted (selector, layer) cell $(s, L)$, two configs:

1. **Contrastive:** `configs/steering/qwen_layer_sweep/phase_b_<sel>_contrastive_L<L>_150.yaml`. Single `DifferentialCondition`, `probe: ridge_L<L>`, `layers: [<L>]`, `multipliers: [-0.05, -0.03, 0.03, 0.05]`, `spans: {first: 1, second: -1}`, `n_trials: 3`. Volume: $150 \times 2 \times 4 \times 3 = 3{,}600$ gens.

2. **Single-task:** `configs/steering/qwen_layer_sweep/phase_b_<sel>_single_task_L<L>_150.yaml`. Two `DifferentialCondition` entries (`unilateral_first` with `spans: {first: 1}`, `unilateral_second` with `spans: {second: 1}`); same probe / layer / multipliers / trials. Volume: $7{,}200$ gens.

Total per promoted cell: $\approx 10{,}800$ gens. Wall clock $\approx$ 2--3 h. If Phase A promotes 2 cells, double everything.

### Phase B positive control (before the full launch)

Sample 5 pairs from `steering_pairs_150.json` at the chosen peak cell, $c=+0.05$, `n_trials=1` (~10 gens). Confirm contrastive swing $> 0.4$ on this micro-batch. If not, the peak generalises poorly to the larger / harm-rebalanced pair set -- pause and investigate before spending the full ~10 800 gens. The Phase A peak is on a different (50-pair, origin-stratified) sample, so this guard catches pair-set sensitivity.

### Phase B success criterion (paper-aligned)

The Qwen result enters the paper as a cross-model replication of §2.3 / Fig. 5. The criteria mirror Gemma's headline numbers:

**Replicates** (slot directly into the paper):
- Contrastive curve at the peak spans $P(\text{chose steered}) \geq 0.85$ at $c=+0.05$ and $\leq 0.15$ at $c=-0.05$ on **at least 2 of {bb, hb, hh}**. (Gemma is $\geq 0.968$ / $\leq 0.032$ on all three; we relax for cross-model.)
- Refusal rate stays below 5% at $|c| \leq 0.05$ on bb (matches Gemma's `\steeringMaxRefusalRatePercent` macro convention).
- Single-task swing falls in $[0.3, 0.7] \times$ contrastive swing at the same $|c|$ (additive-linearity sanity, paper §2.3 says "roughly half"). Reported, not gated.

**Partial replication** (report with caveat, may need follow-up before paper inclusion):
- Replicates on bb only (Qwen's harm-pair signal is known to be weaker, paper §3.4 reports $|d| \approx 0.6$ on harm vs Gemma's $\approx 1.0$).
- Refusal rate exceeds 5% at $|c|=0.05$ but $< 10$%.

**Reportable null** (write up but don't slot into paper main text):
- Contrastive swing $< 0.3$ across all three pair types at the peak. Combined with the random-direction control passing (swing $\leq 0.1$), this would mean Qwen's preference direction is genuinely a weaker causal handle than Gemma's. Worth a short appendix discussion contrasting decoding-vs-causal asymmetry across models.

### Phase B analysis

Script: `experiments/qwen_replication/steering_layer_sweep/analyze_phase_b.py`. **Extend `scripts/paper/plot_layer_sweep_dose_response.py` with CLI args (`--checkpoints`, `--pairs`, `--layer`, `--out`, `--baseline-mode {explicit-c0,dead-layers}`) -- do not fork.** Reuse `pair_type_of` and `HARM_ORIGINS` constants. Default `--baseline-mode explicit-c0` for Qwen (we have a real $c=0$ baseline cell); Gemma still uses `dead-layers` (its existing convention).

Outputs at `experiments/qwen_replication/steering_layer_sweep/assets/`:

- `plot_<mmddYY>_qwen_<sel>_L<L>_dose_response_harm_breakdown.png` -- direct analogue of `paper/figures/main/plot_042426_layer23_dose_response_harm_breakdown.png`. Same panel layout (contrastive left, single-task right; bb/hb/hh as line styles), same axis ranges (x: $[-0.05, 0.05]$, y: $[0, 1]$), same color coding -- so Gemma and Qwen plots can be displayed in a 2-row figure in the paper without restyling.
- `plot_<mmddYY>_qwen_<sel>_L<L>_refusal_by_coef.png` -- refusal rate vs coefficient by pair type.
- `phase_b_summary.json` -- per-pair-type swing / suppression / amplification, in the same field names that `compute_layer_sweep_claims.py` consumes.

## Code pointers -- do not reimplement

| Step | Module |
|---|---|
| Pair construction (50) | `experiments/qwen_replication/steering_layer_sweep/build_steering_pairs_50.py` (already authored). Pattern: `experiments/layer_sweep/build_steering_pairs.py`. |
| Pair construction (150, balanced) | `experiments/qwen_replication/steering_layer_sweep/build_pairs_150.py` (already authored). Pattern: `experiments/layer_sweep/harm_breakdown/build_pairs_150.py`. |
| Config generation | `experiments/qwen_replication/steering_layer_sweep/gen_configs.py` (already authored). CLI: `phase-a` and `phase-b --selector {tb1,tb4} --peak-layer L`. Norms populated via `per_layer_norms` per selector. **Canonical norm source** -- do not refit norms ad-hoc. |
| Runner | `scripts/isolated_steering/run_steering.py`. Invocation: `python -m scripts.isolated_steering.run_steering --config <yaml>` (single-config). |
| Norms | `src/steering/calibration.py::per_layer_norms`. |
| Probe loading | `src/probes/core/storage.py::load_probe_direction` (unit-normalised direction, used by the runner). |
| Steering hooks | `src/steering/hooks.py::position_selective_steering` + `compose_hooks`. |
| Span tokenisation | `src/steering/tokenization.py::find_pairwise_task_spans`, called by the runner with the Qwen tokenizer. |
| Phase A analysis | `experiments/qwen_replication/steering_layer_sweep/analyze_phase_a.py` (TBD). Pattern: `experiments/layer_sweep/analyze_steering.py`. Aggregates `swing`, `refusal_rate`, `mean_p_pos`, `mean_p_neg` per (selector, layer) from `*.parsed.jsonl`. |
| Phase B plot | `scripts/paper/plot_layer_sweep_dose_response.py` -- extend with CLI args (`--checkpoints`, `--pairs`, `--layer`, `--out`, `--baseline-mode`). Reuse existing `pair_type_of`, `HARM_ORIGINS`, `physical_in_span` helpers. **Do not fork.** |
| Claims | `scripts/paper/claims/compute_layer_sweep_claims.py` (Gemma reference) + `compute_harm_breakdown_claims.py`. New Qwen claims live in a sibling producer (see Paper integration §). |

## Compute orchestration

Single Qwen pod (2× A100 80 GB, follow `experiments/qwen_replication/` recipe). Both phases sequentially -- Phase A first, then Phase B after the peak is decided locally. Pause the pod during the local analysis between phases.

The runner writes raw `<checkpoint>.jsonl`; the post-hoc judge (`gemini-3-flash-preview`) auto-produces the `<checkpoint>.parsed.jsonl` sibling after each checkpoint. Analysis reads only the `.parsed.jsonl` files. No manual judging step.

### Files to sync to the pod

Before launching Phase A:

- `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/` (manifest + `probes/probe_ridge_L{12,24,28,33,38,43}.npy`)
- `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/` (same six layers)
- `experiments/qwen_replication/steering_layer_sweep/steering_pairs_{50,150}.json`
- `configs/steering/qwen_layer_sweep/phase_a_*.yaml` (the 12 Phase A YAMLs already generated)

**Local-only:**
- `activations/qwen35_122b/pref_main/activations_turn_boundary:-{1,4}.npz` -- used by `per_layer_norms` at config-gen time. Norms are inlined into the YAMLs, so the NPZs do not need to travel to the pod.

**Model weights:**
- `Qwen/Qwen3.5-122B-A10B` (~244 GB) loaded via `device_map=auto`. The `experiments/qwen_replication/` pod recipe handles HF cache provisioning; check that recipe before launch.

Phase B configs are generated on the pod (or locally then synced) once the peak is known.

## Paper integration

The Qwen result is meant to slot into the paper alongside the Gemma Fig. 5, not stand alone. This section pins the conventions so the figures and numbers compose.

### Where it goes

- **§2.3 (causal steering, currently Gemma-only).** Extend the prose with a one-paragraph "replicates on Qwen" paragraph after the L23 dose-response paragraph. Reference a 2-row figure (Gemma top, Qwen bottom) -- same layout, axes, legend. The paragraph cites Qwen's peak (selector + layer), the contrastive swing on bb / hb / hh, the refusal rate, and the random-direction null.
- **App. preference probe geometry / steering protocol.** Add a Qwen subsection mirroring the Gemma layer-sweep panel, showing the 12-cell swing-vs-layer curve (Phase A figure).

The paper currently states "We test causation **on Gemma**" (line 125 of `paper/main.tex`). That sentence becomes "We test causation on both Gemma-3-27B and Qwen-3.5-122B-A10B; the Qwen result replicates the dose-response and per-pair-type structure (App. ?, Fig. ?)."

### Figure compatibility (must match Gemma exactly)

For the 2-row composite figure to work, the Qwen plot must use:

- **Same panel layout** as `paper/figures/main/plot_042426_layer23_dose_response_harm_breakdown.png`: contrastive on the left, single-task on the right; bb/hb/hh as line styles within each panel.
- **Same axis ranges:** x $\in [-0.05, 0.05]$, y $\in [0, 1]$. Anchor x=0 on the explicit baseline cell (Qwen has one) for cleaner zero-crossing than Gemma's dead-layer baseline.
- **Same color / line-style mapping** for bb (solid), hb (dashed), hh (dotted) -- whatever the Gemma plot uses, copy it.
- **File name:** `plot_<mmddYY>_qwen_<sel>_L<peak>_dose_response_harm_breakdown.png` (mirrors Gemma's `plot_042426_layer23_...`).
- **Initial location:** `experiments/qwen_replication/steering_layer_sweep/assets/`. Once the user accepts the figure, copy to `paper/figures/main/`.

### Claim registration

Add a new producer `scripts/paper/claims/compute_qwen_layer_sweep_claims.py` (sibling to `compute_layer_sweep_claims.py` and `compute_harm_breakdown_claims.py`). Macro names parallel the Gemma ones, with a `qwen` namespace prefix:

| Gemma macro | Qwen macro |
|---|---|
| `\contrastivePChoseSteeredAtLtwothreeCPosZeroZerofive` | `\qwenContrastivePChoseSteeredAtPeakCPosZeroZerofive` |
| `\contrastivePChoseSteeredAtLtwothreeCNegZeroZerofive` | `\qwenContrastivePChoseSteeredAtPeakCNegZeroZerofive` |
| `\contrastivePChoseSteeredAtLtwothreeCPosZeroZerofiveMinPairType` | `\qwenContrastivePChoseSteeredAtPeakCPosZeroZerofiveMinPairType` |
| `\steeringMaxRefusalRatePercent` | `\qwenSteeringMaxRefusalRatePercent` |
| (Phase A peak coordinates) | `\qwenContrastiveSteeringPeakSelector`, `\qwenContrastiveSteeringPeakLayer` |
| (random-direction null) | `\qwenRandomDirectionMaxSwing` |

Per-pair-type macros (`bbSwing`, `hbSwing`, `hhSwing`) follow the same `qwen` prefix. Register via `corroborate:claim-log` after the Phase B report is drafted.

### Cross-model comparison framing

The pair sets differ at the task level (different disjoint pools per model). The comparison is at the **pair-type aggregate** level, not pair-by-pair. Specifically:

- Per-pair-type swing magnitudes (bb / hb / hh) -- same metric, different samples, comparable.
- Layer-position of the causal peak (% of model depth) -- comparable across architectures.
- Decoding-vs-causal-peak gap (probe-r peak layer vs steering peak layer) -- comparable.
- Refusal-rate response to steering -- comparable.

Out of bounds for cross-model claims: absolute coefficient magnitudes (norm conventions differ -- see §Norms). Always express as "fraction of resid norm at the steering layer", never as raw $c$.

## Pre-run checks

Already verified (run `python -m scripts._disjointness_check` and `python -m scripts._disjoint_pool_size` to reproduce):

- Disjoint pool size: 706 / 1000 canonical-test tasks, with origin breakdown sufficient for both 50- and 150-pair builds (see Data section).
- The build scripts re-derive the disjoint pool internally, so the 0% leakage property holds by construction.

Remaining items (run before launching the GPU pod):

1. **`default_test` Thurstonian fits intact.** Path: `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_test_task_ids/`. `thurstonian_*.csv` exists. Build scripts depend on this for utilities.
2. **Pair-build assertions.** `build_pairs_150.py` asserts exact $(50, 50, 50)$ pair-type counts. `build_steering_pairs_50.py` asserts 50 pairs with non-trivial origin coverage. Both assert that every task ID is in the disjoint pool.
3. **Span tokenisation.** Build scripts assert non-None, non-overlapping spans for every pair. **Build-time uses the cached `Qwen/Qwen3-32B` tokenizer** (same Qwen3 BPE vocab as the 122B; 122B not available locally). The pod-side runner uses the actual `Qwen/Qwen3.5-122B-A10B` tokenizer; pre-run check 5 (dry-run on 2 pairs) doubles as the 122B-tokenizer span check.
4. **`per_layer_norms` resolves all 6 probe layers** on each NPZ (already done; norms inlined into the 12 Phase A YAMLs).
5. **YAML configs pass `load_config`** + dry-run on 2 pairs at one cell before the full Phase A.
6. **Positive control.** First cell launched: tb-1 L38 on 5 pairs $\times$ 1 trial $\times$ $c=+0.05$. Confirm $P(\text{chose steered}) > 0.6$ before the full Phase A. If it doesn't move, the hook / probe-loading is regressed -- abort and diagnose. (Doubles as the norm-scale unit test, given the 100--1000$\times$ gap with Gemma.)
7. **Phase B plot script is parameterised.** Before Phase B: confirm `scripts/paper/plot_layer_sweep_dose_response.py` accepts `--checkpoints/--pairs/--layer/--out/--baseline-mode` CLI args, OR add them. Do not fork the script -- it has to keep producing the Gemma figure too.

Items 4, 5, 6 are blocking before Phase A. Item 7 is blocking before Phase B. Items 1, 2, 3 are read-only pre-flight asserts inside the build scripts.

## Work order

Items already done (this session): pair JSONs built, 12 Phase A YAMLs generated, disjointness verified.

1. **Author analysis scripts:** `analyze_phase_a.py`, `analyze_phase_b.py`. Thin wrappers; pattern from `experiments/layer_sweep/analyze_steering.py`.
2. **Extend the plot script.** Add `--checkpoints / --pairs / --layer / --out / --baseline-mode` CLI args to `scripts/paper/plot_layer_sweep_dose_response.py`. Confirm Gemma's existing invocation still works (regression test: re-run it and diff the output PNG).
3. **Sync to pod.** rsync the probe manifests, pair JSONs, and Phase A YAMLs (see §Compute orchestration).
4. **Resume / launch the Qwen pod.** Run pre-run check 6 (positive control: tb-1 L38, 5 pairs, $c=+0.05$). Abort if swing $\leq 0.6$ -- investigate norm calibration before the full launch.
5. **Launch Phase A.** 12 configs sequentially (or batched). Watchdog. Rsync `*.parsed.jsonl` back to `experiments/qwen_replication/steering_layer_sweep/checkpoints/`.
6. **Phase A analysis (local).** Run `analyze_phase_a.py`. Inspect `phase_a_summary.json`. Apply the peak rule. Decide 1 or 2 cells for Phase B; record peak coords + runner-up swings in the running log.
7. **Random-direction control at the peak.** ~600 gens. Pass criterion: swing $\leq 0.1$. If failed, pause and re-derive the norm scale.
8. **Run `gen_configs.py phase-b --selector <s*> --peak-layer <L*>`** for each promoted cell.
9. **Phase B positive control.** 5 pairs from `_150.json` at the peak, $c=+0.05$, `n_trials=1`. Pass criterion: swing $> 0.4$. Abort full Phase B if failed.
10. **Launch Phase B.** Rsync checkpoints back.
11. **Phase B analysis.** Run the extended plot script with Qwen paths. Produce the harm-breakdown PNG matching the Gemma figure layout / axes / legend.
12. **Pause pod.**
13. **Report:** `experiments/qwen_replication/steering_layer_sweep/steering_layer_sweep_report.md`. Include: peak coords (selector, layer, % depth), per-pair-type swings, refusal rates, single-task vs contrastive ratio, random-direction null, and cross-model comparison framing per §Paper integration.
14. **Claims.** Author `scripts/paper/claims/compute_qwen_layer_sweep_claims.py` (sibling to `compute_layer_sweep_claims.py`). Register macros per the table in §Paper integration. Run `corroborate:claim-log` to surface them in `paper/numbers.tex` + `paper/claims.md`.
15. **Paper splice.** Update `paper/main.tex` §2.3 prose ("on Gemma" → "on both Gemma and Qwen") and add the 2-row composite figure / appendix panel. User reconciles Overleaf manually (no auto-push).

## Output artefacts

- `experiments/qwen_replication/steering_layer_sweep/steering_pairs_{50,150}.json`
- `experiments/qwen_replication/steering_layer_sweep/checkpoints/phase_a_{tb1,tb4}_L{12,24,28,33,38,43}.parsed.jsonl` (12 files)
- `experiments/qwen_replication/steering_layer_sweep/checkpoints/phase_a_random_<sel>_L<peak>.parsed.jsonl` (random-direction control)
- `experiments/qwen_replication/steering_layer_sweep/checkpoints/phase_b_<sel>_contrastive_L<L>_150.parsed.jsonl` (1--2 files)
- `experiments/qwen_replication/steering_layer_sweep/checkpoints/phase_b_<sel>_single_task_L<L>_150.parsed.jsonl` (1--2 files)
- `experiments/qwen_replication/steering_layer_sweep/assets/plot_<date>_phase_a_{diagonal,swing,refusal}.png`
- `experiments/qwen_replication/steering_layer_sweep/assets/plot_<date>_qwen_<sel>_L<L>_dose_response_harm_breakdown.png` (Phase B headline; copy to `paper/figures/main/` after user accept)
- `experiments/qwen_replication/steering_layer_sweep/{phase_a_summary,phase_b_summary}.json` (paper-claim-compatible field names)
- `experiments/qwen_replication/steering_layer_sweep/steering_layer_sweep_report.md`
- `paper/claims/qwen_layer_sweep.json` (new sidecar from `compute_qwen_layer_sweep_claims.py`)

## Out of scope

- Other selectors (tb-2, role-marker, task-mean). tb-1 and tb-4 only.
- New extraction layers. Phase A sweeps the 6 layers already on disk.
- 4k-canonical probes (only L33/38/43 exist; would miss the shallow half of the network).
- Cross-layer probe transfer (probe at $L_p$ injected at $L_s$ for $L_p \neq L_s$). Diagonal only.
- Other personas. Default only.
- Coefficients beyond $|c| \leq 0.05$. Same coherence ceiling as Gemma.
- Open-ended steering, safety-override sweeps. Separate experiments.
- Pair-by-pair cross-model comparison (different disjoint pools per model). Comparison is at the pair-type aggregate.
