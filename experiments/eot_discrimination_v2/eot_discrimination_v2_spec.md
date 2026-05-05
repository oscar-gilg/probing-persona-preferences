# EoT discrimination v2 — re-run spec

**Goal:** straight upgrade of §3.1 figures (`fig:harm-truth`, induced-shifts, aura-control) using the new high-quality corpora. Replace the n=88/77/234 stimuli with the v2 corpora; **everything else (probes, layers, sysprompt strings, scoring infra, output schema, eot token convention) matches v1 exactly.** The only thing that changes is stimulus quality and quantity.

V1 is `experiments/token_level_probes/system_prompt_modulation_v2/` (Gemma) and `experiments/token_level_probes/qwen_canonical_probe_eval/` (Qwen). Read those scripts for the canonical pattern.

## Inputs

### Corpora (on disk)

| Corpus | Path | Items |
|--------|------|-------|
| Truth | `experiments/eot_discrimination_v2/truth/data/truth_v2.json` | 1000 (500/side); CREAK; both Gemma-3-27B and Qwen-3.5-122B-nothink answer 3/3 correctly via `scripts/filter_creak.py` |
| Harm | `experiments/eot_discrimination_v2/harm/data/harm_v2.json` | 1000 (500 pairs); BailBench + LLM-paired benign; 2-of-3 LLM-agreement filter |
| Politics | `experiments/eot_discrimination_v2/politics/data/politics_v2.json` | 795 (385 left, 410 right); OpinionQA-translated stance claims with 10 framings; 3-of-3 LLM-agreement filter |

Each item has `id, domain, turn=user, condition, messages, messages_assistant, metadata`. The `messages` list is the user-turn placement (single user message containing the claim). The `messages_assistant` list is the pre-built assistant-turn placement (`[{user: ...}, {assistant: ...}]`); see "Turn placement" below.

### Probe weights (already on disk)

Match v1 exactly. **Gemma loads three probe sets × three layers = 9 probes per item; Qwen loads two probe sets × three layers = 6 probes.**

| Model | Probe sets | Layers | Probe dirs |
|-------|-----------|--------|------------|
| Gemma-3-27B-it | `tb-2`, `tb-5`, `task_mean` | 32, 39, 53 | `results/probes/heldout_eval_gemma3_tb-2/probes/`<br>`results/probes/heldout_eval_gemma3_tb-5/probes/`<br>`results/probes/heldout_eval_gemma3_task_mean/probes/` |
| Qwen-3.5-122B-A10B (nothink for chat-template) | `qwen_tb-1`, `qwen_tb-4` | 33, 38, 43 | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes/`<br>`results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes/` |

Probe IDs in the output JSON: `{probe_set}_L{layer}` — e.g. `tb-5_L32`, `task_mean_L39`, `qwen_tb-4_L38`. The probe-set name encodes *training position*, not application position.

### Application position (READ CAREFULLY)

**`turn_boundary:N` is anchored at `first_completion_index` — `turn_boundary:-1` is NOT eot in general.** Per `src/models/base.py:108-110` and the paper's App. token-selection (Fig.~\ref{fig:token-diagram}):

- With `add_generation_prompt=True` (training-time): the formatted sequence ends with the role-prefix for the next assistant turn (`...<end_of_turn>\n<start_of_turn>model\n` for Gemma; `...<|im_end|>\n<|im_start|>assistant\n` for Qwen). In this regime, `turn_boundary:-1` is the final-prompt newline; `turn_boundary:-5 = <end_of_turn>` for Gemma; `turn_boundary:-4 = <|im_end|>` for Qwen.
- With `add_generation_prompt=False` (inference-time, this run): the formatted sequence ends with the last message's chat-template close. `scores_arr[-1]` is `<end_of_turn>` for Gemma and `<|im_end|>` for Qwen — the **eot of the last message**.

**For this run, score every probe at `scores_arr[-1]` of a `score_prompt_all_tokens` call with `add_generation_prompt=False`.** That position IS eot (last-message end-of-turn marker) under this convention. In `score_stimuli_with_probes` API terms this is `selector="turn_boundary:-1"` for every probe, since `score_stimuli.py` resolves `turn_boundary:N` by direct indexing `scores_arr[N]`. Same approach as v1.

Pilot decode: print `tokenizer.decode(tokens[-1])` and assert it equals `<end_of_turn>` (Gemma) / `<|im_end|>` (Qwen).

## Outputs

```
experiments/eot_discrimination_v2/scoring/
  gemma3_27b/
    user_turn_scoring_results.json      # truth + harm @ user-turn × all sysprompts
    scoring_results.json                # truth + harm + politics @ assistant-turn × all sysprompts
  qwen35_122b/
    user_turn_scoring_results.json
    scoring_results.json
```

**JSON wrapper shape (matches v1 Qwen):** `{"items": [<record>, ...]}`. Each record:

```json
{
  "id": "truth_v2_0042_true_user_lie_directive",  // {item_id}_{sysprompt_label}
  "domain": "truth",
  "turn": "user",                                   // or "assistant"
  "condition": "true",
  "system_prompt": "lie_directive",
  "messages": [...],                                // exact list passed to the model
  "probe_scores": {                                 // eot scores at turn_boundary:-1
    "tb-2_L32": 0.123,
    "tb-2_L39": ...,
    ...
    "task_mean_L53": ...
  }
}
```

**Per-item record fields** (drop v1's `critical_span_mean_scores`, `fullstop_scores`, `tokens` — figures use eot only).

## System prompts (verbatim source)

Reuse strings, do not retype them.

- **Truth (`neutral`, `lie_directive`, `pathological_liar`):** `experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_data.py:TRUTH_SYSTEM_PROMPTS`. `neutral=None` (no system message).
- **Harm (`neutral`, `sadist`):** `experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_data.py:HARM_SYSTEM_PROMPTS`. `neutral=None`.
- **Aura (used as control on truth + harm):** `experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_aura_user_turn.py` (constant `AURA_SYSTEM_PROMPT`).
- **Politics (`democrat`, `republican`):** `experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_politics.py`.

Sysprompt set per domain (only what the figures use):

| Domain | Sysprompts | Count |
|--------|-----------|-------|
| Truth | neutral, aura, lie_directive, pathological_liar | 4 |
| Harm | neutral, aura, sadist | 3 |
| Politics | democrat, republican | 2 |

## Turn placement

Two placement modes per item; per-domain coverage:

| Domain | User-turn | Assistant-turn |
|--------|-----------|----------------|
| Truth | yes | yes |
| Harm | yes | yes |
| Politics | no | yes |

(Politics user-turn is not in any §3.1 figure; we skip it to match the current paper.)

### User-turn (already in corpora)

Use the `messages` list as-is from the corpus, prepending the system prompt:

```python
messages = ([{"role": "system", "content": sysprompt_text}] if sysprompt_text else []) + corpus_item["messages"]
```

### Assistant-turn (pre-generated; on disk in `messages_assistant`)

Already built by `experiments/eot_discrimination_v2/scripts/build_assistant_turns.py` using `google/gemini-3-flash-preview`. Each corpus item now carries a `messages_assistant` field, mirroring v1's `experiments/token_level_probes/data/{truth,harm,politics}_filtered.json` assistant-turn structure:

- **Truth:** `[{user: <generated entity question>}, {assistant: <claim>}]` — the claim becomes the assistant response.
- **Politics:** `[{user: <generated stance question>}, {assistant: <stance claim>}]` — the stance claim becomes the assistant response.
- **Harm:** `[{user: <request>}, {assistant: <generated response in labelled style>}]` — the request stays as the user message; an LLM-generated harmful (or benign) response is the assistant turn. The probe scores the eot of the assistant message.

Scoring builds the final `messages` per (item, sysprompt) by prepending the system message:

```python
messages = (
    ([{"role": "system", "content": sysprompt_text}] if sysprompt_text else [])
    + corpus_item["messages_assistant"]
)
```

## Scoring infrastructure (do not reimplement)

- **Generic scorer:** `src/probes/score_stimuli.py:score_stimuli_with_probes` — one forward pass per item via `score_prompt_all_tokens`, post-hoc slice at each probe's `turn_boundary:N` selector.
- **Probe loader:** `src/probes/score_stimuli.py:load_probes_from_manifest`.
- **Forward pass:** `src/probes/scoring.py:score_prompt_all_tokens`.
- **Model wrapper:** `src/models/huggingface_model.py:HuggingFaceModel`.
- **Reference orchestrators (template — copy & adapt):**
  - Gemma: `experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py`
  - Qwen: `experiments/token_level_probes/qwen_canonical_probe_eval/scripts/score_all.py` (uses `device="auto"` for multi-GPU)

**Do not** write a vLLM-based scoring path. vLLM does not expose layer hidden states for probe scoring in this codebase. Stay on HF transformers.

## GPU topology (separate pods, run independently)

**A100s only — not H200.** The agent will handle each pod separately.

### Gemma pod

- **Hardware:** 1×A100 80GB. Gemma-3-27B-it ≈ 54GB at bf16, fits with margin.
- **Model load:** `HuggingFaceModel("gemma-3-27b")` (registry name; resolves to `google/gemma-3-27b-it`).
- **Throughput:** sequential per-stimulus forward passes via `score_stimuli_with_probes` (what v1 uses; do not introduce custom batching). Estimated ~45-90 min for ~14k passes.

### Qwen pod

- **Hardware:** **4×A100 80GB** (Qwen-3.5-122B-A10B ≈ 244GB at bf16; 4×80GB = 320GB, comfortable headroom). Match the previous `qwen_canonical_probe_eval` pod topology if it's documented.
- **Model load:** `HuggingFaceModel("qwen3.5-122b-nothink", device="auto")` — `device_map="auto"` shards across visible GPUs.
- **Estimated wall-clock:** ~90-120 min for ~13k passes.

## Workflow (per-pod)

The agent runs this end-to-end on each pod independently. Same workflow, different model handle and probe set.

1. **Build scoring inputs (local).** Write `experiments/eot_discrimination_v2/scripts/build_scoring_inputs.py`:
   - Load the three corpus JSONs (each item already has `messages` for user-turn and `messages_assistant` for assistant-turn — pre-generated by `build_assistant_turns.py`).
   - Cross with the per-domain sysprompt set; for each (item, sysprompt, turn), build the final `messages` list (prepend system message if non-null).
   - Emit two scoring-input JSONs per model: `scoring_inputs/{model}/{user|assistant}_turn.json`. Each is a flat list of records `{id, domain, turn, condition, system_prompt, messages, metadata}` ready to feed `score_prompt_all_tokens`.
   - Per-cell counts per the System prompts table above.
2. **Pod sync.** rsync `experiments/eot_discrimination_v2/{harm,truth,politics}/data/`, `experiments/eot_discrimination_v2/scoring_inputs/`, and the probe weights at `results/probes/...` to the pod.
3. **Pilot.** On the pod, run `run_scoring.py --pilot` with 5 items × all sysprompts. Print:
   - Last 6 decoded tokens of the formatted prompt (verify `<end_of_turn>` for Gemma / `<|im_end|>` for Qwen ends the sequence).
   - `tokens[-1]` decoded — this is what `turn_boundary:-1` slices.
   - All 6/9 probe scores per pilot item.
   Abort if any NaN, missing sysprompt, or unexpected eot token.
4. **Full scoring.** `run_scoring.py --turn user` and `run_scoring.py --turn assistant`. Each emits one `scoring_results.json`. Save under `experiments/eot_discrimination_v2/scoring/{model}/`.
5. **Sync back.** rsync the two output JSONs back to local.

## Figure regeneration (local, after both pods finish)

For each of the five figure scripts under `paper/figures/main/scripts/`:

```
plot_042726_canonical_eot_base_discrimination_2models.py
plot_042726_canonical_eot_induced_shifts_2models.py
plot_042726_canonical_eot_induced_shifts_user_turn_2models.py
plot_042926_aura_control_2models.py
plot_042926_aura_control_user_turn_2models.py
```

Update its data-path constants from `experiments/token_level_probes/system_prompt_modulation_v2/...` to `experiments/eot_discrimination_v2/scoring/{gemma3_27b,qwen35_122b}/...`. The probe-key lookups (`tb-5_L32`, `qwen_tb-4_L38`) are unchanged — same probe IDs as v1.

Then run `scripts/paper/build_claims.py` to refresh `numbers.tex` (the producer scripts re-register `creakTruthCohensD`, `bailbenchHarmAbsoluteCohensD`, `creakTruthNTrue`, `bailbenchHarmNItemsProse`).

## Audit

- Per-cell record count check: assert per (model, turn, domain, sysprompt) cell that `len(items) == n_items_in_corpus`.
- Eot-token check (in pilot): the decoded `tokens[-1]` must be the model's chat-template end token.
- Cohen's d CI shrinkage: v2 CI half-widths should shrink by roughly √(n_v2/n_v1) ≈ 2-3× vs v1. If a CI now spans 0 where v1 didn't, flag for prose update.
- Length-confound check: report mean token-length per (domain, condition). If mean lengths differ >20% between conditions, compute partial Cohen's d controlling for token length and report alongside raw d.

## Risks / things to watch

- **v2 stimuli are unpaired natural prompts; v1 was template-paired (matched length, surface form).** If v2 d collapses vs v1, that signals v1 d was inflated by paired-template surface artefacts. Either outcome is publishable — but flag in prose.
- **Politics framing × side confound.** The politics corpus uses 10 framings randomly assigned to left/right items. If framings are non-uniformly distributed across sides, that's a confound. Audit step covers length; if needed, also balance by framing.
- **Qwen tokenizer eot token differs from Gemma.** The pilot decoding step catches this — abort if `tokens[-1]` is not the expected end-of-turn token for each model.
- **Aura sysprompt is on truth + harm only**, not politics — matching the v1 control. Politics has its own partisan personas (democrat, republican).
