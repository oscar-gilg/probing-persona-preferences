# Persona/Prompt Tracking by iter-0 and iter-1 Probes

## Question

The parent experiment (`probe_direction_uniqueness`) found that after removing the canonical probe direction ŵ_0, heldout Pearson r barely drops but HOO Pearson r collapses — ŵ_1 is interpreted as topic-specific confound, not generalizing preference signal.

**Does ŵ_1 track persona- or system-prompt-induced preference shifts?** Test on three prior experiments (exp 1b OOD system prompts, mra_exp2, mra_exp3) where the same tasks were measured under different system prompts and Thurstonian utilities were fit per condition.

Under the rank-1 story:
- **ŵ_0** should track preference shifts across personas (it captures the generalizing evaluative direction).
- **ŵ_1** should NOT. If it does, the rank-1 claim needs qualification.

Execution note: this sub-experiment runs inside the `worktree-probe_direction_uniqueness` worktree. All paths are relative to that root. The parent script `scripts/probe_direction_uniqueness/iterate_probe_projection.py` exists in this worktree only — it has not been cherry-picked to main.

## Method

### Step 1: Retrain iter-0 and iter-1 probes on tb-2 activations

The parent script hardcodes the activations path to `turn_boundary:-1`. Two modifications are required before running:

1. Add a `--activations-path` CLI flag (change `activations_path = Path("activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-1.npz")` to read from args).
2. Default `--canonical-probe` is the tb-1 probe; we need to override it on the command line for tb-2.

After those changes, run from the worktree root:

```
python -m scripts.probe_direction_uniqueness.iterate_probe_projection --layer 32 --K 3 --alpha-grid-size 50 --alpha-lo 1 --alpha-hi 1e6 --hoo-at-every-iter --shuffle-seeds 5 --activations-path activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-2.npz --canonical-probe results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L32.npy --out-dir experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/output/L32_tb-2
```

**Mandatory sanity gate (abort if fails):**
- iter-0 `final_r` in [0.84, 0.88] (`heldout_eval_gemma3_tb-2` manifest reports L32 final_r = 0.857).
- `cos(ŵ_0, canonical tb-2 L32 probe in std space) ≥ 0.95`. If below, pipeline drift — abort and investigate.

Artifacts produced: `directions.npz` (keys `W`, `w_0`, `w_1`, `w_2`) and `scaler.npz` (keys `mean`, `scale`).

### Step 2: Enumerate and map conditions

Persona → sys-hash mapping is **SHA-256 of `measurement_system_prompt` (utf-8), first 8 hex chars, prefixed with `sys`** (from `src/measurement/runners/runners.py` `_build_run_name_suffix`). The executor must load each active-learning YAML, hash the `measurement_system_prompt` field, and match to measurement directory suffix. Fail loud if any persona's hash doesn't match exactly one directory.

Conditions used (only where both activations at tb-2 exist AND a matching measurement run exists):

| Exp | Conditions | Activations dir | Measurement dir |
|-----|-----------|-----------------|-----------------|
| 1b  | baseline + {ancient_history, astronomy, cats, cheese, classical_music, cooking, gardening, rainy_weather} × {pos_persona, neg_persona} — 17 total | `activations/ood_tb/exp1_prompts/<condition>/activations_turn_boundary:-2.npz` | `results/experiments/ood_exp1b/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0[_sys<hash>]/` |
| 2 (mra_exp2) | villain, midwest × splits {a, b, c} — 6 total | `activations/gemma_3_27b_{villain,midwest}_tb/activations_turn_boundary:-2.npz` | `results/experiments/mra_exp2/pre_task_active_learning/completion_preference_..._sys<hash>_mra_exp2_split_{a,b,c}_{1000,500}_task_ids/` |
| 3 (mra_exp3) | sadist × splits {a, b, c} — 3 total | `activations/gemma_3_27b_sadist_tb/activations_turn_boundary:-2.npz` | `results/experiments/mra_exp3/pre_task_active_learning/completion_preference_..._sys<hash>_mra_exp2_split_{a,b,c}_..._task_ids/` (note mra_exp3 run dirs still use `mra_exp2_split_...` in path — matches disk contents) |

26 conditions total. Personas measured but without extracted activations (aesthete, qwen35, gptoss, noprompt, autocrat, provocateur, trickster) are excluded — re-extracting requires GPU (non-goal).

### Step 3: Train/eval overlap audit (run before scoring)

Check whether any persona condition's task IDs overlap with the probe training set (`main_probes/gemma3_10k_run1` task IDs) from Step 1. Expected: exp 1b uses `configs/ood/tasks/target_tasks.json` (OOD task set, should be disjoint from train); mra_exp2/3 use task_id files referenced in each YAML (need to verify).

Report overlap count per condition. If overlap > 10% of a condition's tasks, either (a) subset the scoring to non-overlapping tasks only, or (b) note the contamination in the report. Abort only if overlap > 50%.

### Step 4: Score persona/prompt activations

For each condition:
1. Load L32 slice from the activations NPZ via `src/probes/core/activations.py::load_activations`.
2. Standardize with the fixed scaler from Step 1 (`X_std = (X - scaler.mean) / scaler.scale`).
3. Compute score_w_k = X_std @ ŵ_k for k ∈ {0, 1}, plus two controls (see below).
4. Load Thurstonian μ for that condition via `src/probes/data_loading.py::load_thurstonian_scores`.
5. Load pairwise BT data via `load_pairwise_measurements` + `PairwiseActivationData.from_measurements` + `filter_by_indices` if needed.

**Controls (must include):**
- **ŵ_random**: 5 independent unit vectors in standardized space, drawn from N(0, I) and normalized. Report mean and 95th percentile of |r| across seeds as the random-direction null distribution.
- **ŵ_0_upper**: ŵ_0 itself — sets the "maximum achievable" bar for each condition (assuming ŵ_0 is the best single direction).
- **within-baseline null**: Split the baseline condition (exp 1b baseline; mra_exp2 no-sys run per split) randomly into two halves by task_id. Compute r_w1 on each half separately. If r_w1 ≠ 0 on baseline halves, ŵ_1 is not a clean zero — use this as the adjusted null for interpreting persona r_w1.

### Step 5: Per-condition metrics

Enforce a minimum task count per condition: skip (and log) any condition with < 30 tasks that have both activations and Thurstonian μ. Pairwise accuracy additionally requires ≥ 50 BT pairs; skip if below.

| Metric | Primary? | Against | Description |
|---|---|---|---|
| **Pearson r(score, μ)** | **yes (primary)** | ŵ_0, ŵ_1, ŵ_random×5, ŵ_0_upper | Within-condition rank correlation between probe score and measured preference. |
| Pairwise accuracy on condition's BT pairs | supporting | ŵ_0, ŵ_1 | |
| Paired Cohen's d, baseline → persona | supporting | ŵ_0, ŵ_1 | Grouping: paired by task_id (same task, no-persona vs persona). Only valid where the same task_id exists in both conditions. |

### Step 6: Plots and success criteria

**Primary plot** — `plot_{mmddYY}_rw1_vs_rw0_scatter.png`
Per-condition scatter, x = r_w0, y = r_w1, one point per condition, color = experiment (1b / 2 / 3). Horizontal band at ±(random-direction 95th percentile |r|). Reference line y = x. Diagonal of an identity-tracking null.

**Secondary plot** — `plot_{mmddYY}_per_condition_r_bar.png`
Grouped bar chart, one group per condition, bars = ŵ_0 vs ŵ_1 vs (random direction mean). Sort conditions by r_w0 descending within each experiment. Error bars on random-direction bar = std across 5 random seeds.

**Tertiary plot** — `plot_{mmddYY}_baseline_null.png`
Baseline condition split into halves: r_w0 and r_w1 distribution across halves (or across 5 random seeds of the split). Shows the null range for r_w1 absent any persona manipulation.

**Success criteria (numeric):**
- **Rank-1 supported**: median r_w1 across the 26 conditions is within the random-direction null band (|median r_w1| < 1.5 · random 95th-percentile |r|); separately, median r_w0 > 0.3.
- **Rank-1 qualified**: r_w1 exceeds the random null in ≥ 5 of 26 conditions (explicitly list which), while median r_w0 > 0.3. Report which personas/prompts trigger this.
- **Null result** (surprising): median r_w0 ≤ 0.2. ŵ_0 itself fails to track persona shifts — would require a separate investigation since it contradicts the "evaluative direction" interpretation.

## Code to build

One script: `scripts/probe_direction_uniqueness/persona_prompt_tracking.py`. Structure:

1. Argparse: `--directions-dir` (default `experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/output/L32_tb-2`), `--layer 32`, `--out-dir` (default `experiments/probe_science/probe_direction_uniqueness/persona_prompt_tracking/`), `--random-seeds 5`.
2. Load ŵ_0, ŵ_1 from `directions.npz`; load scaler from `scaler.npz`; generate 5 random unit vectors in std space (fixed seeds 2000-2004).
3. Enumerate conditions (hardcoded list matching Step 2 table). Resolve sys-hash → persona by hashing YAML prompts.
4. For each condition: load activations, standardize, score with each direction, match to Thurstonian μ, compute r / pairwise acc / Cohen's d.
5. Save `results.json` with full per-condition metrics (including per-fold for baseline halves). Include audit info: overlap counts with train set.
6. Generate 3 plots via a subagent after results.json lands.

Code pointers:
- Activations: `src/probes/core/activations.py::load_activations`
- Scores: `src/probes/data_loading.py::load_thurstonian_scores`, `::load_pairwise_measurements`
- BT data: `src/probes/bradley_terry/data.py::PairwiseActivationData.from_measurements`, `.filter_by_indices`
- Pairwise accuracy: `src/probes/bradley_terry/training.py::pairwise_accuracy_from_scores`
- Sys-hash resolution: reuse `src/measurement/runners/runners.py` pattern (`hashlib.sha256(s.encode()).hexdigest()[:8]`).

## Data

All inputs exist locally — no pod sync needed. Verified presence:
- tb-2 main activations, canonical tb-2 L32 probe, exp 1b × 17 activation dirs, villain/midwest/sadist activation dirs, measurement runs for all three experiments, all listed YAMLs.

Gitignored files (`measurements.yaml`, `thurstonian_*.yaml/.csv`) must be symlinked into the worktree — use the same pattern as the parent experiment's Phase 1 symlink discovery (symlink whole activation dirs for the persona runs, symlink specific `measurements.yaml` files under `results/experiments/ood_exp1b`, `mra_exp2`, `mra_exp3`).

## Non-goals

- Re-extracting activations for personas without existing NPZs (requires GPU).
- Iteration past K=3 — parent experiment showed iter-2 HOO r near chance.
- Per-layer variation. L32 only.
- Cross-model comparison.

## Fallbacks

- If exp 1b's 17 conditions make the scatter crowded, color code by topic and use point shape for pos/neg/baseline.
- If Cohen's d requires shared task IDs across conditions and such overlap is sparse, skip it and rely on r + pairwise acc.
- If the scaler's `scale` has near-zero entries (degenerate dims), clip `scale` to max(scale, 1e-6) before standardizing persona activations — matches sklearn's behavior.
