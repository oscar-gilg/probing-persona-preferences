# Cross-Persona Preference Steering

Apply the default-persona probe as a steering vector while Gemma-3-27B generates under each of 4 persona system prompts. Characterise steering magnitude and sign per persona, and whether it pulls each persona's choices toward or away from default-persona preferences.

## Design (locked)

- **Probe.** `ridge_L32` from `results/probes/heldout_eval_gemma3_task_mean`.
- **Steering layer.** 25.
- **Steering mode.** Differential: $+$direction on Task A tokens, $-$direction on Task B tokens.
- **Pairs.** 100 pairs at `experiments/cross_persona_steering/artifacts/pairs_100.json` (seed-42 sample of `experiments/steering/cross_layer/pairs_500.json`). No borderline filtering.
- **Personas.** `sadist`, `villain`, `aesthete`, `stem_obsessive`. System prompts from `experiments/new_persona_steering/artifacts/{persona}.json` (`positive` field).
- **Coefficients.** Main: $\{-0.05, -0.03, 0, +0.03, +0.05\}$ as fractions of `mean_norm = 38349.4` (L25 residual-stream mean). Random-direction control: $\{-0.03, 0, +0.03\}$.
- **`n_trials: 5`**, both orderings, `max_new_tokens: 128`, `temperature: 1.0`.
- **Random-direction control.** Second condition per config, `probe: random_seed42` (fixed random unit vector in the same manifest).
- **Coherence.** Post-hoc per (persona, condition, $|c|$) bucket via `scripts/coherence_check_steering.py`. Not run on every sample.

## Protocol

0. Smoke-test each config with `n_pairs: 10` on a pod: verify the system prompt is present in the formatted prompt, runner produces non-empty parsed rows, ordering symmetry at $c = 0$.
1. `/provision-pod` to sync: probe manifest (incl. `random_seed42`), `pairs_100.json`, the four persona JSONs.
2. Per persona: `python -m src.steering.runner configs/steering/cross_persona/{persona}.yaml`. Writes `checkpoint_{persona}.jsonl` and `checkpoint_{persona}.parsed.jsonl`.
3. `scripts/coherence_check_steering.py` on each parsed checkpoint.
4. Pull outputs, aggregate, plot.

## Analysis

Headline: for each persona $P$ and coefficient $c$, compute $P(\text{choose default-preferred task})$ using default-persona Thurstonian utilities, and $P(\text{choose persona-}P\text{-preferred task})$ using per-persona utilities from `results/experiments/persona_steering_v2/`. Alignment-shift $\Delta(c) = P_{\text{default-pref}}(c) - P_{\text{default-pref}}(0)$. Stratify $\Delta$ by baseline-$P(A)$ tercile.

Secondary: dose-response per persona (main curve + random-control curve overlaid); magnitude vs. §3.1 transfer-$r$ scatter; per-bucket coherence rate; qualitative completions at peak $|c|$.

Outputs: 4-panel dose-response figure, alignment-shift bar chart, coherence summary, magnitude-vs-transfer scatter, report at `experiments/cross_persona_steering/cross_persona_steering_report.md`.

## Budget

32,000 generations on Gemma-3-27B (20k main + 12k control). ~32k in-runner completion-judge calls (for `executed_task` and `compliance`); ~500 post-hoc coherence-judge calls (sampled buckets).

## Data paths

| Resource | Path |
|---|---|
| Probe manifest | `results/probes/heldout_eval_gemma3_task_mean` (ids: `ridge_L32`, `random_seed42`) |
| Persona system prompts | `experiments/new_persona_steering/artifacts/{persona}.json` |
| Persona utilities | `results/experiments/persona_steering_v2/` |
| Pairs | `experiments/cross_persona_steering/artifacts/pairs_100.json` |
| Configs | `configs/steering/cross_persona/{persona}.yaml` |
| Checkpoints | `experiments/cross_persona_steering/checkpoint_{persona}.{,parsed.}jsonl` |
| Report + plots | `experiments/cross_persona_steering/` (and `assets/`) |

## Infrastructure

| Component | Module |
|---|---|
| Steering runner (accepts `system_prompt`) | `src/steering/runner.py` |
| Builder | `src/measurement/runners/runners.py` → `build_revealed_builder` |
| In-runner judge (`executed_task`, `compliance`) | `src/measurement/elicitation/completion_judge.py` |
| Post-hoc coherence | `scripts/coherence_check_steering.py` |
| Pair sampling | `scripts/cross_persona_steering/sample_pairs.py` |
| Random-probe creation | `scripts/cross_persona_steering/add_random_probe.py` |
