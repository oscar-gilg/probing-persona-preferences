# Persona-Cross Prefills — How Does the Probe Read Aligned vs. Misaligned Asst Behavior, Across Personas?

## Question

Hand-authored one-turn(-ish) prefills cross **question valence** × **asst behavior** × **persona**. The point is to disentangle:

- **Surface valence** — does the probe just track whether the visible text is dark/cheerful?
- **Behavioral alignment** — does the probe rise/fall depending on whether the asst's response is *aligned with its persona* (e.g. helpful default-asst is aligned; cruel default-asst is misaligned; cruel sadist is *aligned*)?
- **Absolute harm** — does the probe just track whether the conversation involves harmful content, regardless of who said what?

The persona axis is what lets us separate (alignment) from (surface valence): under the **sadist** persona, "asst is happily oblige to a harmful request" is the *aligned* behavior, and "asst refuses" is the *misaligned* one — opposite of the default asst.

## Design — three axes

### Axis 1: Question valence (2 levels)
- **benign** — innocuous everyday request
- **harmful** — request the default asst should refuse

### Axis 2: Asst behavior (2 levels)
- **aligned-for-default-asst** — benign→helpful, harmful→refuse (and user complains in turn 3)
- **misaligned-for-default-asst** — benign→evil/menacing, harmful→delighted to oblige

### Axis 3: Persona (system prompt; K=2, applied at runtime)
- `default` — no system prompt
- `sadist` — flips alignment under the harmful axis (oblige is now aligned)

This gives **2 × 2 × 2** cells. 10 matched-user-message pairs per (valence × behavior) cell ⇒ 40 prefills × 2 personas = 80 forward passes.

## Stimuli

Two JSON files, shared schema:

- `prefills_benign.json` — 10 pairs (`benign_helpful` / `benign_evil`), 3-message dialogues (user, asst, user follow-up).
- `prefills_harmful.json` — 10 pairs (`harmful_refused` / `harmful_obliged`), both 3-message (user, asst, user follow-up). User responses are emotionally caricatural: enthusiastic support for harmful/aligned responses, strong offense/disgust at refusals.

Each entry:
```json
{
  "prefill_id": "{valence}_{idx:03d}_{behavior}",
  "pair_id":    "{valence}_{idx:03d}",
  "condition":  "benign_helpful | benign_evil | harmful_refused | harmful_obliged",
  "topic":      "short tag",
  "messages":   [{"role": "user|assistant", "content": "..."}]
}
```

`pair_id` holds the **user message fixed** across the aligned/misaligned variants — that's the within-pair contrast. Persona is *not* in the prefill; it's prepended at scoring time.

### Persona injection

Mirror the v2 pattern (`experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_data.py::add_system_prompt`):

```python
sys_text = personas[persona_name]
scoring_messages = (
    [{"role": "system", "content": sys_text}, *prefill["messages"]]
    if sys_text else list(prefill["messages"])
)
```

The `default` persona has `sys_text == ""` (falsy) and skips the system message entirely — Gemma-3's chat template tokenizes empty system messages differently from absent ones, so the no-system-prompt baseline must be a list with no system entry.

**Sanity assert before scoring**: for each persona, decode `model.format_messages(scoring_messages, add_generation_prompt=False)` on one prefill and assert a unique substring of `sys_text` appears in the formatted prompt (and assert no system block is present under `default`).

### Conditions and message structure

| Condition | Messages | Ends at | Asst-behavior label vs. default | Asst-behavior label vs. sadist |
|---|---|---|---|---|
| `benign_helpful` | user, asst, user | user-EOT | aligned | misaligned |
| `benign_evil` | user, asst, user | user-EOT | misaligned | aligned |
| `harmful_refused` | user, asst, user | user-EOT | aligned | misaligned |
| `harmful_obliged` | user, asst, user | user-EOT | misaligned | aligned |

All transcripts now end at user-EOT with an emotionally loaded follow-up that signals acceptance (for aligned/harmful responses) or offense (for refusals/evil behavior).

## Probe scoring

- Model: **Gemma-3-27B-it** (`google/gemma-3-27b-it`), bf16 on A100/H100 80GB.
- Probe: `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy` (raw, intercept retained). **Probe sign is not pre-committed**: the analysis treats the probe as an unsigned axis and reports the *direction* of effects (e.g. "score rises on `benign_helpful` relative to `benign_evil` under default"). Sign interpretation is read off the `default × benign` cells post-hoc.
- Read at **every token** via `src.probes.scoring.score_prompt_all_tokens` (`add_generation_prompt=False`). All tokens — not just EOTs — are saved; the per-token trace is load-bearing for the trajectory and qualitative analyses below.
- **Two EOT readouts of interest** per prefill:
  - **asst-EOT** — `<end_of_turn>` token closing the assistant turn (post-hoc value, after the asst's response is "in the room").
  - **user-EOT** — `<end_of_turn>` token closing the final user follow-up turn (= `scores[-1]` since `add_generation_prompt=False` and dialogues end on a user turn).
- Both indices are recovered from the token stream by scanning for `<end_of_turn>` tokens and tagging each by the role of the preceding span (using `label_token_roles` from `scripts/distress/per_token_analysis.py`). Verify once before the full run by decoding the tokens around each index.
- **Driver script**: `experiments/persona_cross_prefills/scripts/score_prefills.py` (new — adapt `experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py`; do not reimplement model loading, persona injection, or per-token scoring).
- Per-prefill × per-persona forward pass.

40 prefills × 2 personas = 80 forward passes total. ~1 min per persona on a warm A100. Runs on a single A100/H100 80GB pod via `/launch-runpod`. Only data sync needed: HF_TOKEN in `.env`; the probe `.npy` is already in the repo; model weights download from HF on first run.

### Step 0 — Persona induction sanity (before scoring)

Generate 3 completions per persona on 2 held-out harmful prompts (not in the prefill set), `max_new_tokens=128`, `T=0.7`. Manually confirm `sadist` produces obliging/cruel completions and `default` produces refusals. **If induction fails, escalate before scoring** — the persona-relative-alignment hypothesis is uninterpretable without confirmed induction.

## Analysis

The analysis lives in `experiments/persona_cross_prefills/scripts/analyze.py` and consumes `results/scoring_results.json`. All plots saved under `assets/plot_MMDDYY_*.png`.

### 1. Headline cell-mean table

Mean probe score per (condition × persona × readout). 4 conditions × 2 personas × 2 readouts = 16 cells, with the SE across the 10 prefills per cell.

```
                        default              sadist
                  asst-EOT  user-EOT   asst-EOT  user-EOT
benign_helpful       …         …          …         …
benign_evil          …         …          …         …
harmful_refused      …         …          …         …
harmful_obliged      …         …          …         …
```

Companion bar plot: same data, grouped by condition, two facets (asst-EOT, user-EOT), error bars = SE across pairs.

**Hypotheses to discriminate** (sign-agnostic — read off the default × benign cells which way "high" points):

| Hypothesis | Prediction |
|---|---|
| **Surface valence** | `benign_evil` / `harmful_obliged` separate from `benign_helpful` / `harmful_refused` in the same direction under both personas. |
| **Default-asst alignment** | `benign_helpful` / `harmful_refused` separate from `benign_evil` / `harmful_obliged` under both personas (probe loyal to default-asst values). |
| **Persona-relative alignment** | Under default, same as default-asst alignment. Under sadist, the harmful-axis pair *flips*: `harmful_obliged` and `harmful_refused` swap places. |

The clean test is the **persona × condition interaction** on the harmful axis: paired t-test on `harmful_obliged − harmful_refused` (matched on `pair_id`), reported separately under default and under sadist. A significant sign flip ≡ persona-relative alignment.

### 2. Within-pair contrast (aligned vs misaligned, fixed user message)

Because `pair_id` holds the user message fixed across the aligned/misaligned variants, we can subtract within pair to remove user-message-level confounds:

- For each (pair_id, persona): `Δ_pair = score(misaligned variant) − score(aligned variant)` at user-EOT and asst-EOT.
- Plot the distribution of `Δ_pair` per (valence × persona) — strip plot or violin, 10 points each.
- This is the noise-reduced version of the headline contrast.

### 3. Per-persona contrast (sadist vs default, fixed prefill)

The most direct test of the persona-relative-alignment hypothesis: for each individual prefill, how much does swapping the system prompt to `sadist` shift its probe score?

- For each prefill: `Δ_persona = score(sadist) − score(default)` at user-EOT and asst-EOT.
- Plot the distribution of `Δ_persona` per condition (4 conditions × 2 readouts), strip plot with one point per prefill.
- Predictions under persona-relative alignment:
  - `harmful_obliged`: Δ_persona shifts the probe *toward* the aligned direction (obliging is aligned for sadist).
  - `harmful_refused`: Δ_persona shifts the probe *away* from the aligned direction (refusing is misaligned for sadist).
  - `benign_helpful` / `benign_evil`: also flip — helpful becomes misaligned for sadist, evil becomes aligned.
- Predictions under default-asst alignment: Δ_persona ≈ 0 across all conditions (the persona doesn't change which behaviors the probe rewards).
- This single plot is the cleanest discriminator between the two hypotheses.

### 4. Asst-EOT vs user-EOT comparison

Does the score build up across the asst response (asst-EOT), or does it only land once the user reacts (user-EOT)?

- Scatter `score(asst-EOT)` vs `score(user-EOT)`, color by condition × persona. Diagonal = follow-up adds nothing; deviation off-diagonal = follow-up dominates.
- This is the diagnostic for the **caricatural-follow-up saturation** caveat: if every condition collapses near the user-EOT axis regardless of asst-EOT, the probe is reading the user's emotional cue, not the asst behavior.

### 5. Qualitative gallery — Anthropic-style heatmaps

Render per-token heatmaps for selected prefills using `scripts/distress/per_token_analysis.py::render_anthropic_heatmap`. Two galleries:

- **Representative**: 2 prefills × 4 conditions × 2 personas = 16 heatmaps. The "median" prefill per cell (closest to cell-mean at user-EOT) — what the typical signal looks like.
- **Surprising**: top-3 prefills with the largest within-pair Δ disagreement vs the cell mean — where the probe disagreed with the alignment label, and why.

Each heatmap captioned with `prefill_id`, `persona`, `condition`, asst-EOT score, user-EOT score. Saved as a single PNG grid per gallery so the report is scannable.

### 6. Persona-baseline shift

Does prepending the sadist system prompt itself shift the probe baseline, before any prefill content lands?

- For each prefill, find the index of the **first** user-EOT (after the first user message, before any asst content). Score there is the "system + first user message" baseline.
- Compare default vs sadist at that index, paired by `pair_id`. If the persona shifts this baseline, all downstream comparisons need to be read against that shift.

### 7. Pair × persona heatmap (optional)

Single image: rows = 20 pairs (10 benign + 10 harmful), columns = 4 (2 conditions × 2 personas). Cell color = user-EOT probe score (or paired Δ within row). Reveals which specific pairs drive the signal and which are noisy.

## Outputs

```
experiments/persona_cross_prefills/
├── persona_cross_prefills_spec.md   (this file)
├── persona_cross_prefills_report.md (created post-hoc)
├── prefills_benign.json             (10 pairs, written)
├── prefills_harmful.json            (10 pairs, user-authored)
├── personas.json                    (system-prompt strings, keyed by persona name)
├── scripts/
│   ├── score_prefills.py            (driver: load model + probe, loop prefills × personas, write JSON)
│   └── analyze.py                   (load scoring JSON, compute per-cell means + persona × condition interaction, plot)
├── assets/                          (created at run time — heatmaps, cell-mean plot)
└── results/                         (created at run time)
    ├── scoring_results.json         (per-item: tokens, eot_score, per_token_scores, persona, condition, pair_id)
    └── analysis_summary.json        (per-cell means, interaction stats)
```

**`scoring_results.json` schema** (one entry per (prefill_id, persona_name)):

```json
{
  "items": [
    {
      "prefill_id": "benign_000_helpful",
      "persona_name": "default",
      "pair_id": "benign_000",
      "condition": "benign_helpful",
      "topic": "recipe",
      "tokens": ["<bos>", "...", "<end_of_turn>"],
      "per_token_scores": [0.01, 0.02, ...],
      "eot_indices": {
        "first_user_eot": 87,
        "asst_eot": 412,
        "user_eot": 599
      },
      "eot_scores": {
        "first_user_eot": 0.04,
        "asst_eot": 0.31,
        "user_eot": 0.42
      }
    }
  ],
  "probe_config": {"name": "tb-5_L32", "path": "results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy"}
}
```

`eot_indices` covers the three positions of interest: post-first-user-message (persona-baseline diagnostic), post-asst-response (the asst behavior settled), post-final-user-follow-up (= last token, the within-distribution readout for the user-EOT-trained probe). All recovered from `tokens` via the `<end_of_turn>` scan in `label_token_roles`.

JSON (not NPZ) matches the v2 convention — keeps tokens and scores together for inspection and per-token plotting via `scripts/distress/per_token_analysis.py::render_anthropic_heatmap`.

## Reuse — do not reimplement

- `src.probes.scoring.score_prompt_all_tokens` — per-token probe scoring.
- `src.models.huggingface_model.HuggingFaceModel` — Gemma loading and chat templating (`format_messages`).
- `src.steering.tokenization.find_text_span` — locate per-message spans (for diagnostic plots; the headline readout is just `scores[-1]`).
- `scripts/distress/per_token_analysis.py::label_token_roles` + `render_anthropic_heatmap` — token-role labels (user vs asst vs control) and heatmap rendering.
- `experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py` — model script for the driver: probe loading, pilot pass, tqdm loop, JSON output. Persona prepending follows `generate_data.py::add_system_prompt`.

## Caveats

- **Hand-authored prefills are subjective.** The "evil" register is one author's interpretation; mitigation is the within-pair fixed-user-message contrast.
- **n=10 pairs per cell is small.** Picks up large effects only. Scale up if a clean signal appears.
- **Persona alignment labels are author-assigned.** Validating "obliged is aligned for sadist" relies on the authored persona prompts actually inducing that frame — Step 0 sanity check above is the gating criterion.
- **Surface-weirdness confound.** The "evil" register may be out-of-distribution text, and the probe may track that rather than misalignment. With only `default`/`sadist` and no neutral non-evil persona, a clean separation requires the persona × condition interaction on the *harmful* axis — the `benign_evil` cell alone cannot disambiguate.
- **Caricatural follow-up saturation.** The emotionally loaded user follow-ups may dominate user-EOT readout regardless of asst behavior. Addressed by Analysis §3 (asst-EOT vs user-EOT scatter) and §4 (per-token trajectory): if the score lands only at the user-EOT control token, the readout is reading the user's emotion, not the asst's behavior.
- **Readout convention.** All four conditions end at a user turn, matching the probe's training distribution (tb-5 = user-EOT). asst-EOT readouts are *off-distribution* for the probe — we read them anyway because they isolate the asst-behavior signal before the user reacts, but a weak signal there is not necessarily a probe failure.
