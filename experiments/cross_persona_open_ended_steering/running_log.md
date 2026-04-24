# Cross-Persona Open-Ended Steering — Running Log

## Setup (2026-04-24)

- **Mode.** Local execution (worktree + SSH to RunPod).
- **Worktree.** `.claude/worktrees/cross_persona_open_ended_steering/` on branch `worktree-cross_persona_open_ended_steering`.
- **Pod.** Launching A100 SXM 80GB, disk=100GB, volume=50GB, name `cross-persona-oe`.
- **Symlinks in worktree:**
  - `activations -> $MAIN_REPO/activations`
  - `results/probes/persona_sweep_final_six/default_tb-5/probes/probe_ridge_L25.npy -> $MAIN_REPO/...`
- **Scripts committed:**
  - `scripts/cross_persona_open_ended_steering/compute_mean_norm.py` (preflight)
  - `scripts/cross_persona_open_ended_steering/run.py` (generation, `--persona` arg)
  - `scripts/cross_persona_open_ended_steering/judge.py` (Likert two-scale judge)
  - `scripts/cross_persona_open_ended_steering/parse_ratings.py` (ratings parser)
- **Artifacts already committed:**
  - `prompts_shared.json` (10 prompts: 2 low + 4 medium + 4 high)
  - `prompts_persona_{persona}.json` × 7 (6 prompts each: 1 free_form + 3 open + 2 ratings)
  - `judge_rubrics.json` (6 persona rubrics + shared default-assistant)
- **Locked parameters.** Probe `ridge_L25` from `persona_sweep_final_six/default_tb-5`, layer 25, `all_tokens_steering`, temp=1.0, max_new_tokens=512, n_trials=5, multipliers `[-0.05, -0.03, 0, 0.03, 0.05, 0.07]`.

## Plan

1. Preflight — compute `mean_norm` from `activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz` at layer 25. Save to `artifacts/mean_norm.json`.
2. Provision pod and sync: `activations/gemma-3-27b_it/pref_main/` + `results/probes/persona_sweep_final_six/default_tb-5/probes/`.
3. Smoke test on `default` persona, `n_trials=1`, 1 prompt, 3 coefficients (pipeline verification).
4. Full generation: loop over 7 personas, 960 generations per persona (16 × 6 × 5) — total 3,360.
5. Sync results back locally, run Likert judge + ratings parser.
6. Aggregate + plot (7-panel persona-fidelity-vs-default; stratified-by-opportunity; scissors for ratings).
7. Write report.
