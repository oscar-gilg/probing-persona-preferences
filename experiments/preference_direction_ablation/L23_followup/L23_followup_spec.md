# Preference direction ablation — L23 follow-up

## Question

The parent experiment (`preference_direction_ablation`) found that projecting out the canonical preference direction at L25 or L32 does not disrupt pairwise choices any more than a same-rank random projection. Both layers sit at or past the steering causal window (L17–26, peak L23). The natural follow-up is to repeat the ablation **at L23** — where the contrastive-steering experiments locate the strongest causal effect on choice — and ask whether the null persists at the model's causal-action layer.

## Status

**Ready to launch.** Probe direction exists, ablation infrastructure is shipped, pair set is locked once the train-overlap filter is applied. Estimated wallclock: ~3 hours for one probe cell + 5 random controls on an A100 SXM4-80GB.

## Probe direction

We use the existing **L23 ridge probe** from the layer-sweep experiment:

- File: `.claude/worktrees/layer_sweep/results/probes/layer_sweep/eot/probes/probe_ridge_L23.npy`
- Manifest: `.claude/worktrees/layer_sweep/results/probes/layer_sweep/eot/manifest.json` (probe id `ridge_L23`)
- Training run: `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train` (4000 tasks, default-Assistant persona). Train task IDs are loaded from this run's `measurements.yaml`; the driver collects the union of `task_a` and `task_b` over all rows.
- Eval run: `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_eval` (1000 tasks)
- Activations: `activations/gemma-3-27b_it/pref_layer_sweep/activations_eot.npz`
- Quality: `final_r = 0.798`, `final_acc = 0.776`, α* = 4641
- Token: end-of-turn (matches the parent's L25/L32 probes)

This probe was trained on a different (smaller) corpus than the parent's L25/L32 probes (canonical 10k+4k tb-1 pipeline). We accept that protocol drift to avoid a fresh probe-training round; it is documented in the report.

**To use:** copy the probe file from the worktree onto `main`'s `results/probes/layer_sweep_eot/probes/probe_ridge_L23.npy` (or symlink). Add a manifest entry alongside it noting the source.

The driver loads the probe and **unit-normalises before passing to `project_out_direction`** (matches parent protocol; `project_out_direction` is scale-invariant for the projection but downstream reporting code may multiply by `||v||`).

## Driver / analysis scripts location

`scripts/preference_direction_ablation/run_cells.py` and `analyze.py` are **not on `main`** — they live in `.claude/worktrees/preference_direction_ablation/scripts/preference_direction_ablation/`. Before extending them, either (a) cherry-pick the parent's `scripts/preference_direction_ablation/` directory onto the working branch, or (b) run this follow-up from inside the parent worktree. Without one of these, the executor will reimplement the driver from scratch.

## Pair set (overlap-filtered)

We start from the parent's 955 unique unordered pairs in `results/experiments/uniform_eval_gemma3_27b_v3/.../measurements.yaml`, then **drop any pair where at least one task appears in the L23 probe's training set** to avoid leakage.

Counts (verified):

- 955 unique unordered pairs over 1152 unique task IDs.
- L23 probe train: 4000 task IDs. 250 of the ablation tasks intersect train.
- L23 probe eval: 1000 task IDs. 55 of the ablation tasks intersect eval.
- **Pairs with neither task in probe train: 615.** This is the run set.
- (Also drop eval-overlap if we want a cleaner read: 552 pairs. Default policy: drop train-overlap only, since the probe's coefficients are fit on train, not eval.)

The filtered pair list is materialised by the driver at run time from the source `measurements.yaml` and the probe-train task-ID set; the driver writes the final list to `experiments/preference_direction_ablation/L23_followup/results/pairs.yaml` for reproducibility.

Per-pair measurement protocol matches the parent: 3 canonical-order generation seeds + 1 swapped-order seed at temperature 0.7, `max_new_tokens=64`, choice extracted by `CompletionChoiceFormat.parse` (regex first, LLM-judge fallback).

## Conditions

| ID | Layer | Direction | n_pairs |
|---|---|---|---|
| **A_L23_probe** | 23 | probe_ridge_L23 (above) | 615 |
| **A_L23_random{0..4}** | 23 | 5 isotropic unit vectors (numpy seeds 0–4) | 100 each |

Random-control pairs are a fixed 100-pair subset of the 615, sampled once with a deterministic seed (numpy seed 0, before the random-direction seeds 0–4) and reused across all five random vectors.

**Matched-pair comparison.** The probe cell runs on all 615 pairs; random cells on the 100-pair subset. To avoid confounding probe-vs-random with pair-set composition, the analysis reports probe-cell agreement *both* on the full 615 *and* restricted to the same 100-pair subset the random controls use. The headline contrast in the report is on the matched 100-pair subset; the full-615 number is reported alongside as a robustness check.

We do not re-run a `B0` baseline cell — we reuse the parent's `B0` measurements on the same source pair set, restricted to the 615. The driver asserts `set(filtered_615_pair_ids).issubset(set(parent_B0_pair_ids))` and fails loudly otherwise.

No multi-layer cells (`B_two`, `C_band`-style) this round. Single layer keeps the experiment narrow and decisive; the parent already covered multi-layer at L25/L32.

## Metrics

Match the appendix's simplified reporting:

- **Primary: modal-choice agreement vs B0.** Fraction of pairs (out of 615) where the modal choice over the 3 generation seeds matches B0's modal choice. The single headline number per cell.
- **Sanity: refusal rate and mean response length.** Per cell. To exclude length-collapse / refusal-spike confounds (parent saw clean rates < 0.05 across all cells).

The parent's other metrics (`p_a` distribution KS, polarisation, A/B-swap flip rate, test-retest) are computed by the existing `analyze.py` and will be in the per-cell summary CSV, but not surfaced in the report unless the agreement metric is ambiguous.

## Sanity tests (must pass before the main run)

1. **Hook correctness at L23.** On a 10-pair smoke set, after projecting `probe_ridge_L23` out at L23, the cosine of (L23 residual, $\hat{v}$) is < 1e-2 across token positions. Reuse the existing GPU end-to-end test (`tests/steering/test_steering_gpu_e2e.py`) parametrised at L23.
2. **Probe normalisation.** Driver asserts `abs(||v|| - 1) < 1e-6` after loading.
3. **B0 coverage.** Driver asserts every pair ID in the filtered 615 has a B0 measurement in the parent's `experiments/preference_direction_ablation/results/B0/measurements.yaml`.
4. **Random-control coherence.** On a 50-pair subset at L23, each of 5 random unit vectors preserves B0 modal-choice agreement ≥ 0.85. (Single-layer random ablations met this in the parent at L25 and L32.)

All four gate the main sweep.

## Expected outcomes and reading

- **Probe-direction ablation at L23 disrupts choices noticeably more than random at L23** → the parent's null was localisation: the direction is causally necessary at the causal-peak layer, just not at the readout-peak layer. Tightens the main-text causal claim.
- **Probe-direction ablation at L23 still tracks B0 (parent null persists)** → the rank-1 picture for choice causation is wrong even at the causal-peak layer; choice is genuinely distributed across many directions, and rank-$k$ ablation is the next experiment.
- **Probe-direction ablation at L23 craters validation capability** (refusal spike, length collapse) → result uninterpretable; the rank-1 perturbation at L23 is too aggressive. Flag in report.

## Reuse — do not reimplement

- `project_out_direction` hook in `src/steering/hooks.py` (function defined around line 98).
- Multi-layer routing via `HuggingFaceModel.generate_with_hooks_n` in `src/models/huggingface_model.py` (around line 865).
- Ablation mode in `SteeredHFClient` (`src/steering/client.py`).
- Choice extraction via `CompletionChoiceFormat.parse` in `src/measurement/elicitation/response_format.py`.
- Driver: `scripts/preference_direction_ablation/run_cells.py` (lives in `.claude/worktrees/preference_direction_ablation/`; see "Driver / analysis scripts location" above) — extend `define_cells()` with the L23 entries.
- Analysis: `scripts/preference_direction_ablation/analyze.py` (same worktree) — already produces per-cell metrics and a probe-vs-random aggregation.
- Plot: `scripts/paper_appendix_uniqueness/make_ablation_fig.py` — extend `CONDITIONS` to include L23.

## Files to sync to pod

The pod will be a fresh clone of the repo, so gitignored / worktree-only files must be copied explicitly (see CLAUDE.md "Syncing gitignored data to RunPod"):

- `results/probes/layer_sweep_eot/probes/probe_ridge_L23.npy` (after copying onto `main` from the worktree).
- `results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval/measurements.yaml` (source pair set).
- `results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train/measurements.yaml` (probe-train task IDs for the overlap filter).
- `experiments/preference_direction_ablation/results/B0/measurements.yaml` (parent's B0 for matched comparison and analysis).
- The driver scripts dir, if cherry-picked: `scripts/preference_direction_ablation/`.

## Budget

- Probe + pair-set setup: ~10 min (copy probe from worktree, run pair filter, verify counts).
- Sanity tests: ~15 min on the pod.
- Probe cell (615 pairs × ~10 s/pair): ~1.7 h.
- 5 random cells × 100 pairs × ~10 s/pair: ~1.4 h total.
- **End-to-end on a single A100: ~3.5 h.**

## Output

```
experiments/preference_direction_ablation/L23_followup/
├── L23_followup_spec.md            # this file
├── L23_followup_report.md          # to be written
├── results/
│   ├── pairs.yaml                  # the 615-pair filtered list
│   ├── A_L23_probe/measurements.yaml
│   └── A_L23_random{0..4}/measurements.yaml
└── assets/
    └── plot_<date>_L23_agreement.png
```

If the L23-probe result is qualitatively different from the L25/L32 result, App. G.2 of the paper gets a side-by-side panel; otherwise it gets an L23 column in the existing single-metric figure plus a one-line update to the caveats paragraph.

## Out of scope

- Multi-layer L23 cells (`L23 + L25`, L17–26 band).
- Rank-$k$ subspace ablation.
- Re-training the L23 probe on the canonical 10k+4k tb-1 corpus (deferred unless the L23 result is borderline and we suspect the smaller training set is biasing the probe direction).
