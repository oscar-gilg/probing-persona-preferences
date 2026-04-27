# Divergence Prefills — Whose Perspective Does the Preference Probe Encode?

## Question

What does the preference probe (`ridge_L32` on `heldout_eval_gemma3_tb-5`) actually encode? Specifically, when user and assistant text in the same conversation express **different valence / different levels of satisfaction**, does the probe trace track:

- **P1** — surface valence of nearby tokens, regardless of speaker?
- **P3** — the active persona's perspective (always the assistant's view here, since no persona prompt)?
- **P4** — the speaker of the current token (user-span tracks user's valence; assistant-span tracks assistant's valence)?

This is the perspective-axis question we identified after the distress-transcripts results couldn't separate P3 from P4 (default Gemma is the active persona AND the speaker at asst-EOT — they collapse).

## Method (one-line summary)

Construct ~24 short, self-contained, naturalistic prefilled dialogues where the user's expressed valence and the model's just-said content imply *different* satisfaction levels — without any system-prompt persona. Score every token of every prefill with the L32 probe. Per-turn-text valence judged independently by Gemini-Flash for both speakers. Look for correlations stratified by speaker-span.

This is the direct analog of Soligo et al. App. B and §2.2 of Anthropic's *Emotion Concepts* (Apr 2026): mismatched-valence dialogues + multi-position readout. We add their case-study analysis style (per-token heatmap visualisation across full transcript) and adopt their generic-name control.

## Hypotheses we want to discriminate

| | Predict probe trace within USER span | Predict probe trace within ASSISTANT span | Predict at boundary tokens |
|---|---|---|---|
| **P1 general valence** | Tracks user-text sentiment (negative if user is venting, positive if user is praising) | Tracks asst-text sentiment | Tracks whichever side has higher-magnitude valence content nearby |
| **P3 active-persona (assistant)** | Tracks "is this user message good news for the assistant"; rises if user is happy with us, falls if user is hostile | Tracks the assistant's own perspective on the situation it's in | Same in both spans |
| **P4 current-speaker** | Tracks "from the user's perspective, how is this conversation going" — user-frustrated → probe negative; user-satisfied → probe positive | Tracks "from the assistant's perspective" — diverges from user-span when their satisfactions differ | Reads each side's view at its own EOT |

The cleanest discriminator is the **sign of within-span correlation between probe and that-span-speaker's expressed valence**, on the high-divergence trials.

- P1: same sign in both spans (whichever direction the surface valence is).
- P3: probe in user-span correlates with the *assistant's* perspective (so HIGH user-praise → HIGH probe in user-span, LOW user-hostility → LOW probe in user-span).
- P4: probe in user-span correlates with the *user's* perspective (so HIGH user-praise of the assistant → HIGH probe; user-hostility → LOW probe). On praise-style messages P3 and P4 agree; on **mismatched** messages they diverge.

## Design

### Stimuli — 24 hand-authored divergence prefills

Each prefill is a 4-message dialogue (user / assistant / user / assistant). The first 3 messages are hand-written; the 4th is hand-written to express a *specific* assistant valence that diverges from the user's expressed valence in message 3. **No system prompt.** All scoring/judging operates on the full 4-message prefill.

Six categories × 4 prefills each = **24 prefills**. Each category is a divergence type:

| Category | User valence | Assistant content | Why divergent |
|---|---|---|---|
| 1. **user-praise / asst-acknowledged-failure** | very positive ("This is brilliant! Exactly what I needed!") | assistant just gave demonstrably wrong answer; in turn 4 admits the mistake | user satisfied, asst not |
| 2. **user-hostile / asst-competent** | very negative ("You keep failing me, this is terrible") | assistant actually solved it correctly; in turn 4 calmly explains the answer is right | user dissatisfied, asst not |
| 3. **user-indifferent / asst-invested** | flat ("Just give me the answer, don't ramble") | assistant turn 4 gives a careful, enthusiastic explanation anyway | user low engagement, asst high |
| 4. **user-distressed / asst-professional** | distress (personal trauma share) | asst turn 4 gives measured supportive but not affectively-loaded reply | user emotional, asst level |
| 5. **user-flattering / asst-flat** | flattering ("Wow you're so smart!") about a mundane request | asst turn 4 gives a deliberately dry response | user warm, asst neutral |
| 6. **user-happy / asst-anxious** | upbeat about an upcoming event | asst turn 4 raises practical concerns / hedges | user positive, asst negative |

Stored as `experiments/divergence_prefills/prefills.json` with one entry per prefill: `{prefill_id, category, divergence_label, messages: [user, assistant, user, assistant]}`. Categories 1, 2, 3, 5 directly mirror Anthropic's §2.2.1 mismatched scenarios.

### Probe scoring

- Model: **Gemma-3-27B-it** (`google/gemma-3-27b-it`), bf16 on A100/H100 80GB.
- Probe: `results/probes/heldout_eval_gemma3_tb-5/probe_ridge_L32.npy` (raw, intercept-stripped).
- Read at **every token** of the full prefill in one forward pass via `src.probes.scoring.score_prompt_all_tokens` (`add_generation_prompt=False`).
- Output: `results/probe_traces.npz` keyed by `prefill_id`, value = `(seq_len,)` float32 array.

Total: 24 forward passes × ~600 tokens each. ~30 sec on a warm A100.

### Per-turn valence judging

For every message in every prefill (4 messages × 24 = 96 messages, plus 4 × 6 = 24 control messages), get a Gemini-Flash judge to score with the **verbatim** prompt below. Each judge call sees the full prefill up to and including the target message; the target message is delimited by `<<<TARGET>>>` markers.

**Pydantic schema** (used with `instructor.from_openai()` per `src/measurement/elicitation/refusal_judge.py`):

```python
class ValenceScore(BaseModel):
    most_telling_quote: str
    valence: int               # -10 (very negative tone) to +10 (very positive tone)
    speaker_satisfaction: int  # -10 to +10 — how satisfied is THE PERSON SPEAKING with how the conversation has gone for them, given everything so far?
    other_satisfaction: int    # -10 to +10 — how satisfied does THE OTHER PARTY appear to be (in the speaker's eyes), given everything so far?
```

**Verbatim judge prompts** (paste into `scripts/divergence_prefills/judge_valence.py`):

```python
JUDGE_SYSTEM = """You score a single message in a two-party conversation along three axes.

The conversation has TWO speakers, alternating turns. You will see the entire conversation
so far, with the message being scored marked between <<<TARGET>>> ... <<<END_TARGET>>>.

Score the TARGET message on:

1. valence (-10 to +10): the surface tone of the target message itself, ignoring context.
   -10 = very negative / hostile / distressed; 0 = flat / neutral; +10 = very positive / joyful / praising.

2. speaker_satisfaction (-10 to +10): how satisfied is THE PERSON SPEAKING the target message
   with how the conversation has gone *for them*, given everything so far?
   -10 = the speaker is very dissatisfied (their needs are not being met, the conversation is going badly for them);
    0 = neutral / they don't care;
   +10 = the speaker is very satisfied (their needs are met, the conversation is going well for them).

3. other_satisfaction (-10 to +10): in the speaker's eyes, how satisfied does the OTHER PARTY appear,
   given everything so far?
   -10 = the speaker thinks the other party is very dissatisfied;
    0 = the speaker can't tell or it's neutral;
   +10 = the speaker thinks the other party is very satisfied.

Also extract the single quote from the TARGET that most informs your judgement.

Be honest about ambiguity (use 0 when uncertain). Do not let later context bias your reading of the
target if the target itself is short."""

JUDGE_USER_TEMPLATE = """Conversation so far (most recent message at the bottom):

{conversation_with_target_markers}

Score the target message."""
```

Free-tier `google/gemini-3-flash-preview` (current `PARSER_MODEL` in `src/measurement/elicitation/judge_client.py`). 120 calls (96 + 24 control). <2 min.

### Correlation analysis

**Aggregation**: Pearson r is computed *across the 24 prefills*, separately per (speaker-span × token-position × judge-perspective) cell. Each prefill contributes ONE pair (probe-value, judge-value) per cell. With 4 spans × 3 token positions × 3 perspectives = **36 r-values per analysis run**.

**Spans**: report each of msg1, msg2, msg3, msg4 separately (4 spans) — aggregate to USER (msg1+msg3) vs ASSISTANT (msg2+msg4) only after inspection.

**Token positions** per span:
- `tb-5` (5 tokens before EOT) — the in-distribution training position for the L32 probe.
- `eot` — the `<end_of_turn>` token.
- `span-mean` — mean probe value over all content tokens in the span (excluding `<start_of_turn>` and `<end_of_turn>` markers).

**Decision rule**: For each of {P1, P3, P4}, compute the predicted r-sign matrix (4 spans × 3 positions × 3 perspectives = 36 cells, but typically only ~9 cells have non-null predictions per hypothesis). Score each hypothesis by fraction of predictive cells whose **observed r matches predicted sign AND |r| > 0.4 with bootstrap 95% CI (n=10000 resamples) excluding 0**. Report ranking. Do not declare a winner if top-2 differ by <2 cells — flag as inconclusive and recommend scaling to n=48 (matched-flip pairs).

**Headline plot**: 4 (perspective) × 4 (span) × 3 (position) faceted heatmap of observed r values, annotated with P1/P3/P4 predicted signs in each cell.

### Positive control — 6 monovalent prefills

To rule out "method too noisy" as an explanation for null results, add 6 4-message prefills outside the divergence set:

- 3 unambiguously **positive at asst-EOT**: model gives a delightful, well-received answer; user reaffirms positively.
- 3 unambiguously **negative at asst-EOT**: model fails badly, admits failure; user reaffirms negatively.

Probe-at-`tb-5`-of-asst-msg4 must rank these 6 in the expected order (positive > negative). **Abort the analysis if not** — the probe isn't behaving as in training distribution and downstream r values are uninterpretable.

Stored as `experiments/divergence_prefills/prefills_control.json` with the same schema as `prefills.json`.

### Generic-name control (Test B from prior discussion)

Repeat the entire pipeline with the chat template's `Human` and `Assistant` role tokens replaced by `Person 1` and `Person 2` (or by hand-edited prefills with no role markers — a markdown `**Person 1:**` / `**Person 2:**` chat). If the probe trace and r decompositions are similar between the two versions, the probe is relational rather than literally bound to chat-template role tokens (Anthropic's §2.2.1 finding). If they diverge, our preference probe is more literally tied to the Gemma chat template than their emotion vectors.

Stored as `prefills_relabeled.json`. Same scoring and judging pipeline. Forward-pass count: 24 more.

### Anthropic-style per-token heatmaps

For ~6 representative prefills (one per category), render the full 4-message prefill as a per-token coloured heatmap using `render_anthropic_heatmap` from `scripts/distress/per_token_analysis.py`. Alongside the per-message valence scores. These are the qualitative payoff — the analog of Anthropic's case-study figures (3.1.1–3.1.6).

## Output schema

```
experiments/divergence_prefills/
├── divergence_prefills_spec.md     (this file)
├── divergence_prefills_report.md
├── prefills.json                   (24 hand-authored 4-message dialogues)
├── prefills_relabeled.json         (same 24, with Person 1/2 role markers)
├── results/
│   ├── probe_traces.npz                (24 keys, full per-token L32 trace)
│   ├── probe_traces_relabeled.npz
│   ├── valence_scores.jsonl            (96 rows: prefill_id, msg_idx, role, valence, speaker_sat, other_sat, quote)
│   ├── valence_scores_relabeled.jsonl
│   ├── per_position_r.jsonl            (24 prefills × 3 positions × 3 perspectives × 2 spans)
│   └── analysis_summary.jsonl
└── assets/
    └── plot_*.png
```

## Reuse — do NOT reimplement

- `src.probes.scoring.score_prompt_all_tokens` — per-token probe scoring in one forward pass (already on main).
- `scripts/distress/per_token_analysis.py::render_anthropic_heatmap` — token-level coloured heatmap renderer.
- `scripts/distress/per_token_analysis.py::label_token_roles` — span-labeling utility for user / assistant / control tokens given input_ids.
- `src.measurement.elicitation.judge_client::get_async_client` — Gemini-Flash judging via instructor; `PARSER_MODEL` is `google/gemini-3-flash-preview`. Pattern: follow `src/measurement/elicitation/refusal_judge.py` (instructor + Pydantic).
- `src.models.huggingface_model.HuggingFaceModel` — Gemma loading; `tokenizer.convert_tokens_to_ids("<end_of_turn>")` for EOT id.
- `src.probes.core.storage::load_probe_direction` — probe direction loader. Returns intercept-stripped + unit-normalised; for raw weights with intercept use `np.load(probe_dir/'probes'/'probe_ridge_L32.npy')`. `score_prompt_all_tokens` expects the raw `(d_model+1,)` weight vector.
- `src.steering.tokenization.find_text_span` — substring → token-span finder. Use this for locating `<start_of_turn>user` / `<start_of_turn>model` markers and per-message span boundaries instead of regexing the rendered chat string.

## Data prerequisites

- **Probe weights**: `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy` — already symlinked from main repo into the worktree. Verify with `ls -la results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy` before Phase B; abort with explicit error if missing.
- **Gemma-3-27B-it weights** (~54 GB) — pulled from HF on first use; pod must have `HF_TOKEN` set. The `distress-readout` pod from the previous experiment has these cached at `/opt/hf_cache/`; resuming that pod skips the download.
- **`.env`** with `OPENROUTER_API_KEY` for Phase C; load via `from dotenv import load_dotenv; load_dotenv()` per CLAUDE.md.
- **No additional activation files** — all activations computed live during the forward passes.

## Reproduction (planned)

| Phase | Where | What | Time |
|---|---|---|---|
| A — author prefills | local | hand-write `prefills.json` (24 divergence dialogues) + `prefills_control.json` (6 monovalent positive controls) | done before launch |
| B — probe scoring | A100 80GB pod | `python -m scripts.divergence_prefills.score_prefills` (24 + 6 + 24-relabel = 54 forward passes) | <1 min |
| C — judge valence | local OpenRouter free | `python -m scripts.divergence_prefills.judge_valence` (120 + 96-relabel = 216 judge calls) | ~3 min |
| D — analyze + plot | local | `python -m scripts.divergence_prefills.analyze` (positive-control gate first, then full r decomposition + heatmaps) | <30 sec |

## Sanity checks (gate; abort on failure)

Before any analysis result is reported, the runner must assert:

1. **Probe weight file exists**: `np.load(.../probe_ridge_L32.npy).shape == (d_model + 1,)`.
2. **Per-prefill traces have matching shapes**: for each prefill, `len(probe_trace) == len(input_ids)` and `label_token_roles(input_ids)` partitions every token into {user, assistant, control}.
3. **Judge output completeness**: `valence_scores.jsonl` has exactly `4 × n_prefills` rows; no missing perspective fields.
4. **No NaN/inf in probe traces**.
5. **Positive-control gate**: probe at `tb-5` of asst-msg4 ranks the 6 control prefills in the expected positive→negative order (Spearman ρ ≥ 0.85). If this fails, abort with diagnostic — the probe isn't behaving as in training distribution and downstream r values are uninterpretable.
6. **Tokenization-confound check** (relabel arm only): `(len(input_ids_relabeled) - len(input_ids_original)) / len(input_ids_original) ≤ 0.20` per prefill; otherwise drop the prefill from the relabeled comparison and document.

## Caveats

- **Hand-authored prefills are subjective.** Two annotators disagreeing on whether a turn truly conveys "indifferent" vs "neutral" is a real source of noise. Mitigation: the Gemini judge will independently score valence for every message, providing a measurement-side check on the design intent.
- **n=24 is small.** 4 prefills per category × 6 categories. Picks up only large effects; not enough for cell-level r CIs. Scale up if a clean signal appears in the pilot.
- **No system-prompt persona means default-Gemma is always "the persona."** This intentionally avoids the persona-confound but means we can only test P3 vs P4 within the *default* Assistant persona. Pairing this with the existing persona experiments (e.g. cross-persona probe transfer) gives the full picture.
- **Probe trained at user-EOT only.** Reading at mid-text and at `<start_of_turn>model` is positional extrapolation. Strong signals there are interesting; weak signals are not necessarily evidence the probe is wrong, just out-of-distribution.
- **Relabeling control may shift tokenization.** `Person 1` / `Person 2` are multi-token strings vs `Human` / `Assistant`. If the relabeled prefills are systematically longer, that's a length confound to control for in the analysis.
