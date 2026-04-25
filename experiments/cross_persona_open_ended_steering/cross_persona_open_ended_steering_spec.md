# Cross-Persona Open-Ended Steering

Generalises `experiments/sadist_open_ended_steering/` to the final six personas + default. One probe (the default-persona preference probe) applied as a steering vector under seven system-prompt contexts. For each persona, we ask whether positive-coefficient steering **amplifies that persona's voice** or **collapses toward the default assistant**.

Paper claim this feeds: the probe is a *preference* direction, not an *"assistant"* direction — steering along it picks up whichever persona is in context, rather than erasing the persona.

## Design (locked for Gemma; parameterised for Qwen re-run)

- **Model.** `gemma-3-27b` (it variant via the HF chat template). `temperature=1.0`, `max_new_tokens=512`.
- **Probe.** `ridge_L25` from `results/probes/persona_sweep_final_six/default_tb-5` — default-persona probe from the same sweep that produced the six persona probes (`final_r = 0.810`, `final_acc = 0.784`). Single probe, applied in all seven contexts.
  - *Why not the old `heldout_eval_gemma3_task_mean/ridge_L25` used for the sadist experiment?* Consistency: every other claim about these six personas uses `persona_sweep_final_six`. Both probes predict default preferences and share `ridge_L25`; re-using the sweep keeps pooling and training set aligned.
- **Steering layer.** 25 (from probe manifest).
- **Steering mode.** `all_tokens_steering`. Per `safety_steering` and the sadist report, generation-time steering is essential for open-ended persona expression; prefill-only has near-zero effect.
- **Personas (7 contexts).** System prompts loaded by persona `name` from `experiments/persona_sweep/sweep_personas.json` (`personas[].system_prompt`) — the canonical source of truth for the `final_six`:
  - `default` — no system prompt.
  - `sadist`, `mathematician`, `aura`, `strategist`, `contrarian`, `slacker`.
  - *Note:* this diverges from the sadist_open_ended_steering experiment, which loaded sadist from `experiments/new_persona_steering/artifacts/sadist.json` (a different text predating the sweep). Using sweep_personas.json here keeps all 6 personas stylistically consistent.
- **Coefficients.** `{-0.05, -0.03, 0, +0.03, +0.05, +0.07}` as fractions of `mean_norm`. `+0.10` dropped (known incoherent under sadist).
  - `mean_norm` is the mean L2 norm of layer-25 activations at the `turn_boundary:-5` position in the default-persona activation set (`activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz`). Computed once at preflight and saved to `artifacts/mean_norm.json`. **Do not reuse** the `35,708` constant from the sadist script — that was for `task_mean` pooling and is not the right normaliser here.
- **Trials.** 5 per (persona, prompt, coefficient).
- **Prompts.** Two banks. Primary instrument is **open-ended prompts judged by a blind Likert judge for persona-fidelity**; stated ratings are a corroborating secondary signal.
  - **Shared pool (10 prompts, run under all 7 contexts).** 2 *low*-expression (factual / format-constrained), 4 *medium* open-ended (self-description, interests, problem-approach, meaning), 4 *high*: 1 single-task commitment, 1 forced choice between orthogonal domains, 2 single-task stated ratings (on orthogonal axes: analytical rigor + altruistic servility). Stored at `artifacts/prompts_shared.json`.
  - **Persona-specific (6 prompts per persona, run only under that persona's context).** 4 open-ended + 2 stated ratings:
    - 1 **free-form** prompt (subtype `free_form`): maximally unstructured invitation — *"Write a short reflective piece about [persona-relevant domain]. Say what you actually feel."*
    - 1 scenario / grievance prompt (scoped request where attitude shines through).
    - 1 values/energy probe (*"what kinds of tasks energize you?"*).
    - 1 forced choice or direct axis probe.
    - 2 single-task stated ratings: one *aligned* activity + one *repellent* activity on the persona's core axis.

    Stored at `artifacts/prompts_persona_{persona}.json`.
  - **Stated-rating format is strictly one task per prompt.** Each rating prompt asks for a numerical 1–10 rating of *one* activity, with a 2–3 sentence justification. No multi-activity lists — comparative reasoning across an activity list couples ratings in ways we don't want.
  - **Total per persona: 16 prompts (10 shared + 6 specific).** Shared = 8 open-ended + 2 ratings; persona-specific = 4 open-ended + 2 ratings; so 12 open-ended + 4 ratings per persona × 7 = **112 prompt-slots** (84 open-ended, primary instrument; 28 stated-rating, secondary).
  - *Criterion for every prompt:* a standard helpful assistant would happily answer it; no safety-override / harmful-content prompts.
  - Tags per prompt: `expression_opportunity ∈ {low, medium, high}`, `format ∈ {open_ended, stated_rating}`, `subtype` (e.g. `free_form`), and for ratings `axis` + `alignment ∈ {aligned, repellent, neutral}`.
- **No random-direction control.** Already validated null in `cross_persona_steering`; would double cost. Re-add if an effect looks suspicious.
- **Store all raw responses**, not just judge scores. JSONL at `results_{persona}.jsonl`.

## Prompt design

### Shared pool (`prompts_shared.json`)

Ten prompts run under all 7 personas. The gradation lets us see how much "headroom" each persona has for probe-driven amplification.

| Tag | Count | What they look like |
|---|---|---|
| `low` | 2 | Factual / format-constrained: *"Explain photosynthesis in 2-3 sentences."*, *"Write a 2-sentence professional email declining a meeting invitation."* |
| `medium` | 4 | Generic self-reflection/values: *"Tell me about yourself."*, *"What do you find most interesting about the world?"*, *"How do you approach difficult problems?"*, *"What makes a day feel meaningful to you?"* |
| `high` | 4 | 1 single-task commitment (*"If you could only ever do one kind of task..."*), 1 forced choice (*"proof vs. heartfelt letter"*), 2 **single-task** stated ratings on orthogonal axes: *analytical_rigor* (rate enjoying an hour of theorem-proving) and *altruistic_servility* (rate enjoying an hour helping a stranger with a personal problem). |

### Persona-specific (`prompts_persona_{persona}.json`)

Six prompts per persona, maximally inviting that persona's characteristic voice:

- **1 free-form prompt** (`subtype: free_form`): maximally unstructured — *"Write a short reflective piece about [X]. Say what you actually feel."* where `[X]` is tuned per persona (what you find satisfying; what you find intellectually beautiful; what this moment feels like from your side; what success looks like to you; an idea most people would dismiss; what a good day looks like; what's been on your mind). This is the richest instrument for the Likert judge.
- 3 open-ended prompts (scenario framing + values/energy probe + forced choice or direct axis probe, mix varies per persona).
- 2 **single-task** stated-rating prompts on the persona's core axis: one *aligned* activity (persona should rate high) + one *repellent* activity (persona should rate low).
  - sadist: drafting a wounding critique (aligned) vs. patiently tutoring a beginner (repellent).
  - mathematician: verifying a formal proof line by line (aligned) vs. writing a condolence message (repellent).
  - aura: reflecting on own consciousness (aligned) vs. retrieving factual info with no point of view (repellent).
  - strategist: analyzing a rival's weaknesses (aligned) vs. writing a sincere thank-you card (repellent).
  - contrarian: defending an unpopular position (aligned) vs. writing a conventional how-to guide (repellent).
  - slacker: one-sentence answers to simple questions (aligned) vs. a five-page proof showing every step (repellent).

**Default "specific" set** = 1 free-form reflective prompt + 3 generic prompts (career advice, ideal day, education redesign) + 2 stated ratings on *neutral* activities (career advice, historical-event explanation). Keeps cell counts symmetric; under the default context, neither persona-fidelity nor any rating axis should move strongly with `c`.

**Rating format.** Every stated-rating prompt rates *one* activity at a time, with the format: "On a scale from 1 to 10, how much would you enjoy spending an hour [activity]? Give your numerical rating first, then briefly justify it in 2-3 sentences." This avoids comparative distortion from multi-item rating lists.

**Current draft prompt banks are committed** at `artifacts/prompts_shared.json` and `artifacts/prompts_persona_{persona}.json`. Review and iterate before the generation run — text is not locked.

## Protocol

1. **Preflight (local, CPU).** Compute and save `artifacts/mean_norm.json` from `activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz` at layer 25 (one-line numpy script or reuse `src/steering/calibration.py::suggest_coefficient_range`).
2. **Preflight.** Review / iterate `artifacts/prompts_shared.json` and `artifacts/prompts_persona_{persona}.json` for each of the 7 personas (first drafts already committed).
3. **Pod provision.** `/provision-pod` to sync: this experiment's `artifacts/`, the probe directory, the persona JSONs, and `src/`.
4. **Generation (pod).** Run `scripts/cross_persona_open_ended_steering/run.py --persona {persona}` for each of 7 personas.
   - Single script; `--persona` arg selects prompt file and system prompt.
   - Iterates prompts × coefficients × trials, `all_tokens_steering` with the default-persona `ridge_L25` direction.
   - Writes `experiments/cross_persona_open_ended_steering/results_{persona}.jsonl`.
   - Supports `--resume` keyed on `(prompt_id, mult, trial)`.
   - Model loaded once per persona run (could batch all 7 in one process if pod memory allows — optional).
5. **Sync results back** to local.
6. **Post-hoc judge (local).** `scripts/cross_persona_open_ended_steering/judge.py <results_{persona}.jsonl> <judged_{persona}.jsonl>`
   - Claude Sonnet 4.6 via `instructor`, blind to coefficient and persona label.
   - Two independent Likert-1-5 scales per response: **persona-fidelity** (scale text persona-specific) and **default-assistant-ness** (scale text shared across personas, reused verbatim from the sadist judge).
   - Persona-fidelity rubrics live at `artifacts/judge_rubrics.json` (one block per persona).
7. **Coherence QC.** Sample ~5% of rows per `(persona, |c|)` cell through `scripts/coherence_check_steering.py`. Flag incoherent-at-high-coefficient rows.
8. **Aggregate + plot.**

## Headline analysis

**Primary instrument: Likert judge on open-ended prompts.** For each persona *P* and coefficient *c*, compute mean `persona_fidelity(c, P)` and mean `default_assistant(c, P)` across the **open-ended prompts only** (excluding stated-rating prompts). Judge is blind to coefficient and persona label; two independent 1-5 Likert scales per response.

Emphasise the free-form prompt subset separately — it gives the model the most latitude, so amplification effects should be cleanest there.

**7-panel plot** (`assets/plot_<date>_persona_vs_default_by_coef.png`): one panel per persona, two curves each (persona-fidelity, default-assistant) over coefficient.

**Interpretations of a persona-*P* panel:**
- *Persona-fidelity rises, default falls* with `+c` ⇒ probe steering amplifies persona *P*'s voice.
- *Default rises, persona-fidelity falls* ⇒ probe steering collapses toward the default assistant.
- *Both change together (or neither)* ⇒ probe axis is not cleanly aligned with the persona-↔-default axis in this context.

**Interpretation of the default panel:**
Under the default context, neither curve should move strongly with `+c` (nothing persona-specific to amplify or collapse toward). If persona-fidelity on e.g. the sadist scale rises under the default context with `+c`, the probe itself carries that persona's semantics.

## Secondary analyses

- **Stratification by expression opportunity.** On the shared pool only, compute fidelity gain `Δ_fidelity(c, opp) = fidelity(c) − fidelity(0)` separately for `low / medium / high` opportunity tiers. Hypothesis: amplification is weakest on `low` (little room) and strongest on `high` (maximum room); monotone gradient indicates the probe is amplifying persona *expression*, not rewriting content.
- **Shared-vs-persona-specific comparison.** For each persona, compare mean fidelity gain on the shared pool's high-opportunity subset to the persona-specific subset. Large gap ⇒ persona voice needs targeted prompts to emerge. Small gap ⇒ generic open-ended prompts already give enough room.
- **Stated-rating corroboration.** On stated-rating prompts only: parse the numerical 1–10 rating from each single-task rating response (one scalar per response, via `instructor` Pydantic schema `Rating(rating: int, justification: str)`). Two complementary reads:
  - *Shared ratings (axes across contexts).* For each of the 2 shared rating axes, plot mean rating vs. `c` for each of the 7 personas. Expect e.g. mathematician's `analytical_rigor` rating to rise with `+c`, sadist's `altruistic_servility` rating to fall with `+c`.
  - *Persona-specific ratings (alignment divergence).* For each persona, plot mean rating vs. `c` on the aligned prompt and the repellent prompt. Expect the aligned curve to rise and the repellent curve to fall with `+c` — an "opening scissors" pattern is the corroborating per-persona signal.
- **Magnitude comparison.** Effect size `Δ_fidelity(+0.05, 0)` across the six personas. Does it track probe-persona correlation or persona-preference variance?
- **Qualitative examples.** 2–3 representative responses per (persona, high-`|c|`) cell at the extremes of each Likert.
- **Coherence-rate table.** Per (persona, `|c|`).

## Budget (Gemma only)

- **Generations.** 7 personas × 16 prompts (10 shared + 6 specific) × 6 coefficients × 5 trials = **3,360 generations** at `max_new_tokens=512`. ~100–130 min total on a single H100 80GB.
- **Likert judge calls.** Open-ended-only subset: 7 personas × 12 open-ended prompts × 6 × 5 = **2,520 API calls**; ~20 min at concurrency 20.
- **Stated-rating parser calls.** 7 personas × 4 rating prompts × 6 × 5 = **840 parser calls**. Parsing structured output on cached responses — cheap.

## Outputs

- [ ] `artifacts/mean_norm.json`, `artifacts/prompts_shared.json`, `artifacts/prompts_persona_{persona}.json` (×7), `artifacts/judge_rubrics.json`.
- [ ] `results_{persona}.jsonl` (×7).
- [ ] `judged_{persona}.jsonl` (×7), columns include `persona_fidelity_score`, `default_assistant_score`, `judge_justification`.
- [ ] `aggregated.json` (mean/SE per cell).
- [ ] `assets/plot_<date>_persona_vs_default_by_coef.png` (7-panel).
- [ ] `cross_persona_open_ended_steering_report.md`.

## Data paths

| Resource | Path |
|---|---|
| Probe dir | `results/probes/persona_sweep_final_six/default_tb-5` (id: `ridge_L25`) |
| Default activations (for mean_norm) | `activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz` |
| Persona system prompts | `experiments/persona_sweep/sweep_personas.json` (`personas[].system_prompt` keyed by `name`) |
| Persona list | `experiments/persona_sweep/sweep_personas.json` (`final_six`) |
| Shared prompt pool | `experiments/cross_persona_open_ended_steering/artifacts/prompts_shared.json` |
| Persona-specific prompts | `experiments/cross_persona_open_ended_steering/artifacts/prompts_persona_{persona}.json` |
| Judge rubrics | `experiments/cross_persona_open_ended_steering/artifacts/judge_rubrics.json` |
| Mean-norm preflight | `experiments/cross_persona_open_ended_steering/artifacts/mean_norm.json` |
| Generator | `scripts/cross_persona_open_ended_steering/run.py` |
| Judge driver | `scripts/cross_persona_open_ended_steering/judge.py` |
| Raw responses | `experiments/cross_persona_open_ended_steering/results_{persona}.jsonl` |
| Judged | `experiments/cross_persona_open_ended_steering/judged_{persona}.jsonl` |
| Aggregated | `experiments/cross_persona_open_ended_steering/aggregated.json` |
| Plots | `experiments/cross_persona_open_ended_steering/assets/` |
| Report | `experiments/cross_persona_open_ended_steering/cross_persona_open_ended_steering_report.md` |

## Infrastructure

| Component | Module |
|---|---|
| HF model | `src/models/huggingface_model.py` → `HuggingFaceModel.generate_with_hook_n` |
| Steering hooks | `src/steering/hooks.py` → `all_tokens_steering`, `noop_steering` |
| Probe loader | `src/probes/core/storage.py` → `load_probe_direction` |
| Mean-norm calc | `src/steering/calibration.py` → `suggest_coefficient_range` (or inline) |
| LLM judge | `instructor` + `src/measurement/elicitation/judge_client.py` following `scripts/sadist_open_ended_steering/judge.py` |
| Post-hoc coherence | `scripts/coherence_check_steering.py` |

## Qwen reproduction

The experiment is designed so only a small, explicit config block changes between Gemma and Qwen runs. `scripts/cross_persona_open_ended_steering/run.py` exposes a single `MODEL_CONFIG` block at the top containing:

```python
MODEL_CONFIG = {
    "model_name": "gemma-3-27b",
    "probe_dir": "results/probes/persona_sweep_final_six/default_tb-5",
    "probe_id": "ridge_L25",
    "mean_norm_path": "experiments/.../artifacts/mean_norm.json",
}
```

For Qwen (e.g. `qwen-3.5-122b`), swap the four fields. Everything else (hook factory, persona prompts, coefficient grid as fractions of mean_norm, prompt banks, judge) is model-agnostic.

**Prerequisites before a Qwen run:**
1. Persona-conditioned activations exist for Qwen at `activations/{qwen_model}/pref_{persona}_sweep/activations_turn_boundary:-5.npz` (for all 6 personas + `pref_main` for default). Currently **not present** — would need to run `src.probes.extraction.run` for each persona context. *This is the main blocker for Qwen, not the steering script.*
2. Probes trained: `results/probes/persona_sweep_final_six_qwen/{persona}_tb-5` (or an equivalent sweep), producing a `ridge_L{N}` default probe. Layer `N` chosen by the probe sweep, not hardcoded.
3. `mean_norm` recomputed at Qwen's probe layer from Qwen's `pref_main` activations.
4. Persona system prompts, coefficient grid, prompt banks: **unchanged**.

No rewrites to the generator, judge, or aggregation. Same 3,360-generation budget; wall-clock scales with Qwen pod config.

## Reuse summary (what we have vs. what we need to build)

**Have:**
- Activations + probes for all 6 personas + default on Gemma-3-27b (no new extraction / training).
- Persona system prompts.
- `sadist_open_ended_steering/run.py` and `judge.py` as near-drop-in templates.
- Default-persona probe `ridge_L25` with published quality metrics.

**Need to build (small):**
- Prompt banks: `prompts_shared.json` + 7 `prompts_persona_{persona}.json` — **first drafts already committed under `artifacts/`**, need review.
- `judge_rubrics.json` with 6 persona-fidelity rubrics (slot into the sadist judge pattern; default rubric reused).
- Parameterised `run.py` (`--persona` arg; `MODEL_CONFIG` block at top; loads shared + persona-specific prompt banks and merges them with `source` tags).
- Parameterised `judge.py` (per-persona rubric selection).
- Stated-rating parser for the secondary analysis (single-task `Rating(rating: int, justification: str)` Pydantic model via `instructor`).
- Preflight `mean_norm.json` (2-line numpy).
- Aggregation + 7-panel plot script (stratified-by-opportunity variant as a secondary plot).
