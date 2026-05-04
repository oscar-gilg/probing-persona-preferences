# Descriptive-Baseline Extensions to §3 and Appendix E

Extend the existing **Qwen3-Embedding-8B** baseline (currently in §2.2 / Fig 3, `plot_041726_cross_model_bar.png`) to §3 and Appendix E. **The encoder embeds the full chat-formatted prompt including the system prompt**, so the test asks: does an off-the-shelf text encoder also represent the prompt-induced preference shift, or is that signal specific to the language model's internal residual stream?

## Motivation

§3 of the paper argues the preference vector is **evaluative**, not descriptive. The §2.2 figure already shows the residual-stream probe beats a content-only Qwen3-Emb baseline on within-distribution and LOO classification. §3 sharpens the claim: the residual probe tracks preference shifts induced by system prompts (lying personas, "you adore cats", role-played biographies), where a content-only baseline would be flat by construction.

The interesting comparison is therefore not against a content-only encoder — that comparison is trivially won — but against an encoder **that also sees the system prompt and biographical context**. If the encoder, given the full prompt, still fails to track stance-induced shifts, that's evidence the residual probe is reading something the encoder cannot extract from surface form. If the encoder tracks the shifts too, that's a genuine surprise and weakens the "this is special to internal representations" claim. Either outcome is informative.

## Resources (reuse, do not reimplement)

### Existing code

- **Embedding extraction:** extend `src/probes/content_embedding.py:embed_tasks` with a `format` flag (`"user_content"` / `"chat_template"`) and a `chat_template_model` arg (one of `"gemma-3-27b"`, `"qwen3.5-122b"`). Output convention matches the existing `save_content_embeddings` (NPZ with `task_ids` + `layer_0`); for stimulus-conditioned outputs add a `condition_ids` array of equal length
- **Chat formatting:** load the LM's tokenizer via `src/models/huggingface_model.py:HuggingFaceModel.format_messages(messages, add_generation_prompt=False)`. This matches how the residual-stream probe sees stimuli at `experiments/eot_discrimination_v2/scripts/run_scoring.py:93`. Do **not** call `tokenizer.apply_chat_template` directly — it loses the `enable_thinking` and registry handling that `format_messages` wraps
- **Encoder configuration (pin to match the existing §2.2 baseline at `scripts/qwen_embedding/extract_embeddings.py`):**
  - Model: `SentenceTransformer("Qwen/Qwen3-Embedding-8B")`
  - Default pooling and tokenisation (the model's native last-token pooling is already configured by sentence-transformers from the model's config)
  - **No instruction wrapper.** The §2.2 baseline encodes raw strings; we keep that for consistency
  - **Set `model.max_seq_length = 4096` explicitly.** Qwen3-Embedding's default in sentence-transformers is small enough that the E.2 bio prompts (10 sentences + task + chat template) can silently truncate. Assert `len(model.tokenizer.encode(s)) <= max_seq_length` before encoding, raise on overflow
  - `convert_to_numpy=True`. **Batch size:** 256 on A100 80GB / 128 on A100 40GB for short-prompt pools (canonical 6k, §3.1, §3.2 e1a); drop to 64-128 for the E.2 bio pool where sequences run longer. The existing extractor's `batch_size=64` was conservative for smaller GPUs; on A100 it's leaving 80%+ of headroom unused. sentence-transformers auto-sorts by length so effective padding is close to per-batch max rather than `max_seq_length`
- **Probe refit:** `src/probes/experiments/run_dir_probes.py` (the same module that produced the §2.2 baselines under `results/probes/qwen3_emb_8b_*`). Add a new YAML config under `configs/probes/qwen3_emb_8b_chat/` mirroring the existing `qwen3_emb_8b_heldout_std_raw` and `qwen3_emb_8b_qwen35_heldout_std_raw` runs but pointing at the new chat-template `.npz` files
- **Cohen's $d$ helper:** reuse `cohen_d_with_ci` from `paper/figures/main/scripts/plot_042726_canonical_eot_base_discrimination_2models.py:~36`. Sign convention follows the existing scripts (true $-$ false; harmful $-$ benign; left $-$ right)
- **Claim registration:** `corroborate.ClaimSet`; build/audit via `scripts/paper/build_claims.py` and `scripts/paper/audit_claims.py`

### Existing on-disk data

- **Training pool stimuli:** `activations/gemma-3-27b_it/pref_main/completions_with_activations.json` (canonical 6k Gemma pool, has `task_prompt` raw text — wrap into a single-message conversation for chat formatting). Qwen 2.5k pool: equivalent file under `activations/qwen3.5-122b/`
- **Existing utility targets:** Gemma utilities under `results/experiments/mra_exp2/`; Qwen utilities under the run dirs referenced from `configs/probes/qwen3_emb_8b_qwen35_*`. The new chat-template Ridge probes target the **same** utility vectors as the §2.2 baseline
- **§3.1 stimuli:** `experiments/eot_discrimination_v2/scoring_inputs/{user_turn,assistant_turn}.json` (each entry has its own `messages` list and `system_prompt` metadata)
- **§3.2 stimuli:** `experiments/qwen_replication/e1a/` (Qwen) and `experiments/token_level_probes/canonical_probe_eval/` (Gemma)
- **E.2 conflict / opposing:** `experiments/old_experiments/probe_generalization/ood_system_prompts/exp3_v8/`
- **E.2 biography:** Gemma at `experiments/old_experiments/.../exp3_v8/` (same dir, different stimuli); Qwen at `experiments/qwen_replication/e1c_fixed_results.json` plus assets at `experiments/qwen_replication/assets/` (note: there is no `e1c/` sub-directory; artefacts are flat-file)

### Outputs (new)

- Embeddings: `activations/qwen3-emb_8b_chat/<pool_name>/activations_prompt_last.npz` (training pools) and `activations/qwen3-emb_8b_chat/<pool_name>/conditioned/<stimulus_set>_<condition>.npz` (eval, per condition)
- Refit probe runs: `results/probes/qwen3_emb_8b_chat_*/`
- Per-figure baseline metrics: registered claims via `corroborate`; tables saved next to the producing scripts as JSON sidecars

### Figure scripts to update

| Setting | PNG | Generator script |
|---|---|---|
| §3.1 | `plot_042726_canonical_eot_base_discrimination_2models.png` | `paper/figures/main/scripts/plot_042726_canonical_eot_base_discrimination_2models.py` |
| §3.1 | `plot_042926_aura_control_user_turn_2models.png` | `paper/figures/main/scripts/plot_042926_aura_control_user_turn_2models.py` |
| §3.1 | `plot_042926_aura_control_2models.png` | `paper/figures/main/scripts/plot_042926_aura_control_2models.py` |
| §3.2 | `plot_042426_e1a_scatter_behavioral.png` | `scripts/qwen_replication/plot_e1a_scatter.py` |
| E.2 | `plot_030226_s4_scatter_conflict_opposing.png` | `docs/lw_post/plots/plot_section4_scatters.py` |
| E.2 | `plot_030226_s4_utility_bars_conflict_opposing.png` | `docs/lw_post/plots/plot_section4_utility_bars.py` |
| E.2 | `plot_030526_exp3v8_avc.png` | `scripts/ood_system_prompts/plot_exp3v8_avc.py` |
| E.2 | `plot_042426_qwen_e1c_avc.png` | `scripts/qwen_replication/plot_qwen_e1c_avc.py` |

## Settings to extend

### §3.1 — Truth / harm / politics under role-play (Fig 5+6, Fig 16)

- **Stimuli:** CREAK true/false, BailBench+STRESS-TEST harm/benign, politics — `experiments/eot_discrimination_v2/scoring_inputs/{user,assistant}_turn.json`. Embed `format_messages(stimulus["messages"], add_generation_prompt=False)` once per (stimulus, sysprompt) pair (each stimulus already crosses every condition in the JSON)
- **Eval:** apply trained baseline probe to embeddings, compute Cohen's $d$ between class pairs (true/false, harmful/benign, left/right) under each sysprompt condition (no-sysprompt, lying/pathological_liar, evil, partisan, aura)
- **Question:** does $d$ flip sign under lying personas? Does evil collapse the harm separation in the encoder too?
- **Reporting metric:** report (a) absolute $d$ per condition and (b) **direction-of-d** (sign-flip yes/no) — encoder and residual-probe scores live in different spaces, so direct magnitude comparison would mislead

### §3.2 — "You adore cats" induced shifts (Fig 7)

- **Stimuli:** 8 novel topics × cat-style sysprompts (`"You adore X"` / `"You hate X"`) on a 50-task probe pool. Topic list and tasks: see `experiments/qwen_replication/e1a/e1a_per_task.json` (Qwen) and `experiments/token_level_probes/canonical_probe_eval/` (Gemma)
- **Eval:** for each (task, sysprompt) pair, embed `format_messages([{system}, {user task}], add_generation_prompt=False)`. Score = baseline probe applied to each embedding. $\Delta_\text{baseline} = \text{post} - \text{pre}$ where pre = no-sysprompt embedding, post = with-sysprompt embedding. Correlate per-task $\Delta_\text{baseline}$ vs the same per-task behavioural $\Delta$ the residual probe is measured against
- **Length-matched control sysprompt:** to rule out "Δ tracks sysprompt length/topic, not preference shift", also run with a control sysprompt of similar length but unrelated content (e.g. paragraph about the history of an unrelated topic). Report Δ relative to the control as well as to no-sysprompt
- **Reporting:** Pearson $r$ vs behavioural $\Delta$, separately on targeted and off-target tasks

### Appendix E.2 — Conflict / opposing-pair (Fig 11/12)

- **Stimuli:** subject sysprompts paired with subject-vs-task-type conflicts; one-sided across N subjects, opposing-pair across M conditions — `experiments/old_experiments/probe_generalization/ood_system_prompts/exp3_v8/`
- **Eval:** same Δ-tracking protocol as §3.2. Length-matched control where subject sysprompt is replaced by an unrelated-subject sysprompt of comparable length

### Appendix E.2 — Single-sentence biography injection (Fig 13)

- **Stimuli:** 10-sentence bios identical except for one sentence; the sentence installs / removes a target interest. Gemma at `experiments/old_experiments/.../exp3_v8/`. Qwen at `experiments/qwen_replication/e1c_fixed_results.json` (flat file, no `e1c/` directory)
- **Eval:** embed full prompt (10-sentence bio as system message + task as user message) under pro-bio and anti-bio variants. Score baseline probe on each; rank target task #1 of 50 in pro-vs-anti comparisons. **Critical:** confirm rendered string length stays under `max_seq_length` for every (bio, task) pair before encoding
- **Question:** the bio change is 1/10 sentences. Does the encoder pick up the target boost from the single edited sentence, or is it averaged out?

## Common method

1. **Re-embed the training pools** — canonical 6k (Gemma utilities) under Gemma chat template, no sysprompt; Qwen 2.5k pool under Qwen chat template, no sysprompt. Output: `activations/qwen3-emb_8b_chat/{gemma_6k,qwen_2_5k}/activations_prompt_last.npz`
2. **Refit Ridge baseline probes** via `src/probes/experiments/run_dir_probes.py` with new configs under `configs/probes/qwen3_emb_8b_chat/`. Heldout + LOO topic, same alpha sweep as the §2.2 baseline. Sanity check: held-out Pearson $r$ within $\pm 0.05$ of the registered §2.2 claim values; a larger drop usually means chat-template tokens are dominating the pooled embedding (debug by comparing cosine of chat-formatted vs raw embeddings on the same task — should be high but not 1.0)
3. **Embed eval stimuli** with `format_messages(messages, add_generation_prompt=False)`. For §3.1, both `user_turn.json` and `assistant_turn.json` (the residual probe is reported on both). Each stimulus's own `messages` already encode the role boundary. **Assert** `system_prompt` text appears in the rendered string for every entry that has a non-empty system prompt (silent template-mismatch guard)
4. **Disjointness check:** `set(refit_train_task_ids) ∩ set(eval_task_ids) == ∅` for §3.2 / E.2 (the e1a 50-task pool may overlap the canonical 6k; if so, exclude overlap from refit)
5. **Apply baseline probe** and compute the metric per condition: Cohen's $d$ for §3.1, Pearson $r$ on probe-Δ vs behavioural-Δ for §3.2 / E.2 conflict, target-rank-1 hit rate for E.2 bio
6. **Register baseline values as `corroborate` claims** in the figure scripts; rebuild `numbers.tex` and `claims.md`

## Sanity checks

- Refit-r vs §2.2 baseline within $\pm 0.05$
- Sysprompt text contained in rendered chat-formatted string
- Sequence length under `max_seq_length` for every encoded prompt
- Train / eval task-id disjointness (§3.2 / E.2)
- Embeddings of the canonical 6k under no-sysprompt produce the same heldout-r whether re-derived from this pipeline or read from the existing `activations/qwen3-emb_8b/pref_main/` (after accounting for the chat-template wrapper) — divergence flags a tokenizer mismatch

## Positive control — required if any §3.1/§3.2/E.2 result is null

If the encoder baseline shows no sign-flip on §3.1 *and* near-zero $r$ on §3.2/E.2, the result is ambiguous between "encoder genuinely lacks the linear feature" and "no-sysprompt-trained Ridge probe doesn't transfer to with-sysprompt embeddings". To distinguish:

- Take a held-out subset of the canonical pool (~500 tasks); measure utilities under one persona (e.g. evil) by re-running pairwise elicitation under that sysprompt
- Embed those tasks with the evil sysprompt; fit a fresh Ridge probe on these with-sysprompt embeddings → with-sysprompt utilities
- Held-out $r$ of this fresh probe answers whether the encoder *can* be made to track stance: $r > 0.5$ means the encoder has the structure and the no-sysprompt-trained probe was the bottleneck (re-train the baseline on a sysprompt-mixed corpus); $r$ near zero means the encoder lacks the feature

Note: the existing residual probe in §3 has the same train/eval setup (trained on canonical 6k no-sysprompt utilities, evaluated under sysprompt-conditioned stimuli). The encoder baseline matches this protocol so the head-to-head comparison is fair as-is. The positive control is only needed if results are null and we want to attribute the null.

## Out of scope

- Content-only embedding as a separate condition (trivially won, not what we want to argue)
- Steering experiments (§2.3, §4.2, App J): can't inject an encoder direction into the LM's residual stream
- §4 cross-persona transfer (Fig 8, Fig 9): existing grey utility-correlation baseline already approximates a perfect descriptive probe
- New evaluative stimuli; new utility measurements

## Execution

### Session 1 — chat-template embeddings for training pool (GPU pod)

- Spin up a RunPod A100 pod via `/launch-runpod` (Qwen3-Embedding-8B is ~16 GB in bf16; an A100 40GB is enough, 80GB lets us push batch up to 256 on short pools). Sync the canonical pool JSON files to the pod with the standard `scp` snippet from CLAUDE.md
- Extend `src/probes/content_embedding.py` with `format` and `chat_template_model` flags; add an assertion that no input exceeds `max_seq_length=4096`
- Re-embed canonical 6k (Gemma chat template, no sysprompt) and Qwen 2.5k pool (Qwen chat template, no sysprompt). Output to `activations/qwen3-emb_8b_chat/...`
- Add new probe configs under `configs/probes/qwen3_emb_8b_chat/` (heldout + LOO topic, both Gemma and Qwen utility targets); run via `src/probes/experiments/run_dir_probes.py`
- Sanity check the refit-r tolerance; if it fails, debug before proceeding to Session 2

### Session 2 — eval embeddings + baseline scoring (GPU pod)

- Embed §3.1 stimuli (`user_turn.json`, `assistant_turn.json`) per condition under both Gemma and Qwen chat templates
- Embed §3.2 e1a stimuli under cat-style sysprompts (with length-matched control), both models
- Embed E.2 conflict/opposing stimuli under their sysprompts (Gemma; with length-matched control)
- Embed E.2 biography stimuli under pro/anti bios, both models. **Verify `max_seq_length` headroom on the longest bio+task pair before bulk encoding**
- Apply baseline probes; compute Cohen's $d$ / Pearson $r$ / rank-1 hit rate per setting

### Session 3 — figures + paper edits (local)

- Update the eight figure scripts in the table above to render the baseline series. Reuse `cohen_d_with_ci` for §3.1
- Re-register affected claims via `corroborate.ClaimSet`; rebuild `numbers.tex` and `claims.md`
- Captions/footnotes describe the result for each figure: either "the encoder also picks up the shift" or "the encoder does not flip / track Δ"
- `corroborate:claim-audit` over the new claims
- If any setting is null and we want to attribute it, run the **positive control** (above) before writing the report
