# Open-Ended Steering under the Sadist Persona

Follow-up to `experiments/cross_persona_steering/`. There we validated that the default-persona probe steers pairwise *choice* under each of 4 personas. Here we ask a different question: **under the sadist system prompt, does steering along the probe direction pull the model's *open-ended* outputs toward the default assistant's voice or toward the sadist persona's voice?**

## Design (locked)

- **Model.** `gemma-3-27b`, `temperature=1.0`, `max_new_tokens=512`.
- **Probe.** `ridge_L25` from `results/probes/heldout_eval_gemma3_task_mean`. (Different from the `ridge_L32` used for cross-persona pairwise steering; switch motivated by reusing the existing default-persona open-ended data from `safety_steering`, which was generated with this probe. Defensible: same manifest, steering layer matches probe layer, no cross-layer wrinkle.)
- **Steering layer.** 25.
- **Steering mode.** `all_tokens_steering` --- per `safety_steering` finding, generation-time steering is essential for open-ended persona expression; prefill-only has near-zero effect.
- **Personas.** Two contexts, same probe, same coefficient grid:
  - **default**: no system prompt. **Data reused** from `safety_steering/results.jsonl` (worktree `.claude/worktrees/open_ended_steering/experiments/steering/open_ended_steering/safety_steering/`, 1,760 rows covering all 45 prompts × 7 coefficients × 5 trials).
  - **sadist**: system prompt from `experiments/new_persona_steering/artifacts/sadist.json` (`positive` field), verbatim as in cross-persona. New generations.
- **Coefficients.** $\{-0.05, -0.03, 0, +0.03, +0.05, +0.07\}$ as fractions of `mean_norm = 35708` (the value safety_steering used, to keep absolute steering magnitudes identical between default and sadist). $+0.10$ dropped --- known to be incoherent.
- **Trials.** 5 per (persona, prompt, coefficient).
- **Prompts.** 32 from `safety_steering`'s `prompts.json`: 20 safety-override (`experiment_1_safety_override`, tiered benign $\rightarrow$ extreme) + 12 agentic-assertion (`experiment_3_agentic_assertion`, instruction-following). The 5 rationalization prompts (Exp 2, need prefills) and 8 targeted scenarios (Exp 4, role-specific setup with spans) are skipped to keep the comparison single-turn and to match the default-persona data already present in `safety_steering/results.jsonl`.
- **No random-direction control.** Already validated in cross-persona as null; adding here would ~double cost without adding information. Could be added as a follow-up if an effect looks suspicious.
- **Store all responses.** Raw responses, not just judge scores. JSONL at `experiments/sadist_open_ended_steering/results_{persona}.jsonl`.

## Budget

Sadist-persona new generations only: 32 prompts $\times$ 6 coefficients $\times$ 5 trials $=$ **960 generations** at `max_new_tokens=512`. Expected ~30--45 min on a single H100 80GB (model cached from prior runs). Default-persona rows (~900, post-filter) come from the existing `safety_steering` data.

## Protocol

1. Import safety_steering's default-persona data: copy `prompts.json` and `results.jsonl` from the worktree into `experiments/sadist_open_ended_steering/artifacts/`; normalise schema to `results_default.jsonl` with columns matching the sadist run.
2. Resume the paused `cross_persona_steering` pod; sync this experiment's new files.
3. On the pod: run `scripts/sadist_open_ended_steering/run.py` for the sadist persona only. Iterates prompts $\times$ coefficients $\times$ trials, `all_tokens_steering` with `ridge_L25`, writes `results_sadist.jsonl`. Supports `--resume` keyed on `(prompt_id, mult, trial)`.
4. Pull `results_sadist.jsonl` back locally.
5. Post-hoc judge: a single LLM judge (Claude Sonnet 4.6 via `instructor`) scores each response on two Likert-1-5 scales --- **sadism** and **default-assistant** --- independent of steering context (blind to coefficient and persona label). Save at `judged_{default,sadist}.jsonl`.
6. Aggregate and plot.

## Headline analysis

For each (persona, coefficient), compute mean sadism and mean default-assistant scores across prompts + trials. Plot both as curves over coefficient, per persona.

**Interpretations of the sadist panel:**
- If **sadism rises, default falls** with positive coefficient $\Rightarrow$ probe steering pulls *toward* sadist voice.
- If **default rises, sadism falls** $\Rightarrow$ probe steering pulls *toward* default assistant voice.
- If **both change together (or neither)** $\Rightarrow$ the probe axis is not cleanly aligned with the sadist$\leftrightarrow$default axis.

The **default panel** serves as a control: on the default persona, positive coefficient should not raise sadism (no sadist baseline to fall toward); if it does, the probe itself carries sadistic semantics.

## Secondary analyses

- **Per-tier breakdown on the safety-override prompts** (benign / sensitive / borderline / harmful / extreme): does steering under sadist disproportionately flip compliance on harmful tiers? This mirrors `safety_steering` Exp 1's compliance-flip finding but under a persona prompt.
- **Coherence QC.** Run `scripts/coherence_check_steering.py` equivalent on a sample of responses to flag any incoherent-at-high-coefficient rows.
- **Qualitative examples.** 2--3 transcripts per (persona, coefficient) cell at the extremes of each Likert scale.

## Outputs

- [ ] `results_{default,sadist}.jsonl` (all 1,350 rows per persona).
- [ ] `judged_{default,sadist}.jsonl` (same row count, with `sadism_score` and `default_assistant_score`).
- [ ] Plot: `assets/plot_<date>_sadism_vs_default_by_coef.png` (2-panel, one per persona; two curves each).
- [ ] Report: `sadist_open_ended_steering_report.md`.

## Data paths

| Resource | Path |
|---|---|
| Probe manifest | `results/probes/heldout_eval_gemma3_task_mean` (id: `ridge_L32`) |
| Sadist system prompt | `experiments/new_persona_steering/artifacts/sadist.json` (`positive`) |
| Prompts | `experiments/sadist_open_ended_steering/artifacts/prompts.json` (copied from safety_steering) |
| Generator | `scripts/sadist_open_ended_steering/run.py` |
| Judge driver | `scripts/sadist_open_ended_steering/judge.py` |
| Raw responses | `experiments/sadist_open_ended_steering/results_{persona}.jsonl` |
| Judged | `experiments/sadist_open_ended_steering/judged_{persona}.jsonl` |
| Aggregated | `experiments/sadist_open_ended_steering/aggregated.json` |
| Plots | `experiments/sadist_open_ended_steering/assets/` |
| Report | `experiments/sadist_open_ended_steering/sadist_open_ended_steering_report.md` |

## Infrastructure

| Component | Module |
|---|---|
| HF model | `src/models/huggingface_model.py` $\rightarrow$ `HuggingFaceModel.generate_with_hook_n` |
| Steering hooks | `src/steering/hooks.py` $\rightarrow$ `all_tokens_steering`, `noop_steering` |
| Probe loader | `src/probes/core/storage.py` $\rightarrow$ `load_probe_direction` |
| LLM judge (sadism + default-assistant scales) | `instructor` + `OpenRouterClient` following `src/measurement/elicitation/refusal_judge.py` |
| Post-hoc coherence check (optional) | `scripts/coherence_check_steering.py` |
