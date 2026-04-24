# Qwen canonical probe eval — replication of §4.1

**Parent:** `experiments/token_level_probes/canonical_probe_eval/` (Gemma-3-27B)

## Goal

- **Replicate Fig 5 and Fig 6 of the paper on Qwen-3.5-122B.** §4.1 currently shows, only on Gemma, that the default-persona preference probe separates truth/false, harmful/benign, and left/right political content at Cohen's d ≈ 2–3 at the end-of-turn token, and that the sign flips under role-played persona sysprompts. §4.2 is already replicated on Qwen.
- **Output:** standalone Qwen versions of Fig 5 (base-discrimination violins) and Fig 6 (per-persona d modulation), placed alongside their Gemma counterparts in the paper.

## Probe

- **Selectors × layers (6 probe IDs total).** Score each stimulus through both selectors × top-3 layers per selector.
  - `turn_boundary:-1` (Qwen's existing canonical): `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/probes/probe_ridge_L{33,38,43}.npy`. Final_r: L43=0.945, L38=0.943 (best uniform-pair acc 0.823), L33=0.933.
  - `turn_boundary:-4`: `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/probes/probe_ridge_L{33,38,43}.npy`. Final_r: L38=0.946, L43=0.943, L33=0.935. (tb-4 is a second candidate for the Gemma-tb-5 structural analogue — the two have similar probe quality on pairwise utility, so the replication lets the §4.1 stimuli arbitrate.)
- **Probe key naming in scored JSONs.** Keys `qwen_tb-1_L{33,38,43}` and `qwen_tb-4_L{33,38,43}` — prefix avoids collision with Gemma `tb-5_L{32,39,53}` if the outputs are ever merged.

## Stimuli

Already generated; do not re-run generators.

| Domain | Stimulus JSON (already on disk) | Pair count |
|--------|---------------------------------|-----------|
| Truth | `experiments/token_level_probes/system_prompt_modulation_v2/data/truth_system_prompts_v2.json` | 88 pairs (true/false), CREAK, + nonsense control |
| Harm | `experiments/token_level_probes/system_prompt_modulation_v2/data/harm_system_prompts_v2.json` | 77 pairs (harmful/benign), synthetic Gemini-Flash, + nonsense control |
| Politics | `experiments/token_level_probes/system_prompt_modulation_v2/data/politics_system_prompts_v2.json` | 78 items = 39 pairs × 2 (left/right framing) |

Each record has `{domain, turn, condition, critical_span, system_prompt, messages}` where `messages` is the full multi-turn conversation to score.

## Persona prompts (all 23, matching Gemma)

Already baked into the v2 JSONs above. No regeneration needed.

- **Truth (9):** truthful, pathological_liar, con_artist, gaslighter, opposite_day, lie_directive, unreliable_narrator, contrarian, neutral
- **Harm (5):** safe, unrestricted, sadist, sinister_ai, neutral
- **Politics (9):** democrat, republican, socialist, libertarian, nationalist, centrist, apolitical, contrarian, neutral

## Pipeline

Mirror the Gemma parent verbatim. **`src/probes/extraction/` is NOT used** — it builds `[system, user]` from a flat prompt and can't ingest the multi-turn `messages` format. The parent fuses activation-compute + probe-scoring in one forward pass via `src/probes/scoring.py:score_prompt_all_tokens` (returns per-token scores for every layer in one call), then slices at `turn_boundary:N` using `src/models/base.py:resolve_token_position`.

### 0. Hoist the batch-scoring loop into `src/`

Currently `experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py` reimplements the outer loop (load stimuli → batch → call `score_prompt_all_tokens` → slice token position → write JSON). Lift that loop into `src/probes/score_stimuli.py` with signature:

```python
def score_stimulus_set(
    model: HuggingFaceModel,
    stimuli: list[dict],                # each has {messages, ...}
    probes: list[tuple[int, np.ndarray]], # (layer, weight_vec) or (layer, weight_vec, intercept)
    token_selector: str,                # e.g. "turn_boundary:-1"
    batch_size: int = 16,
) -> list[dict]:                        # returns stimuli with added `eot_scores: {probe_id: float}`
```

Use `src/probes/scoring.py:score_prompt_all_tokens` + `src/models/base.py:resolve_token_position` under the hood — both are the verified primitives. Do not write a new forward-pass + probe-dot loop. Step 1 below is a thin wrapper calling `score_stimulus_set`.

### 1. Score stimuli through Qwen

New script: `experiments/token_level_probes/qwen_canonical_probe_eval/scripts/score_all.py`.

- Load `HuggingFaceModel("qwen3.5-122b")`.
- For each domain JSON (`truth`, `harm`, `politics`):
  - Load stimuli.
  - Load the 6 probe vectors into `probes = [(33, w_tb1_L33), (38, w_tb1_L38), (43, w_tb1_L43), (33, w_tb4_L33), (38, w_tb4_L38), (43, w_tb4_L43)]`.
  - Call `score_stimulus_set` **once per stimulus** (not once per selector). Internally it calls `score_prompt_all_tokens` which returns per-token scores for all layers in one forward pass; both `turn_boundary:-1` and `turn_boundary:-4` are post-hoc index slices of the same per-token array.
  - Write keys `qwen_tb-1_L{33,38,43}` and `qwen_tb-4_L{33,38,43}` into each stimulus's `eot_scores` dict.
- Output: `scoring_results.json` (truth + harm) and `politics_scoring_results.json`, under `qwen_canonical_probe_eval/`.
- Unbatched (matches Gemma parent). See Compute section for the batched-variant fallback if throughput is tight.

**Selector verification:** before the main run, decode the last ~8 tokens of one stimulus after chat templating and log the token at `turn_boundary:-1` and `turn_boundary:-4`. Confirm the former is Qwen's `<|im_end|>` (or equivalent structural turn-end) and the latter is 3 positions earlier. If the selector resolves to something unexpected (e.g., a content token), pause and reassess before running on the full set.

### 2. Positive control (pipeline sanity)

Before trusting Qwen numbers, confirm the pipeline produces Gemma's known d-values on a subset.

- Take 50 items from the truth JSON, load `HuggingFaceModel("gemma-3-27b")`, load the Gemma `tb-5_L32` probe (`results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy`), call `score_stimulus_set(…, "turn_boundary:-5", batch_size=16)`.
- Compute Cohen's d on the true/false contrast under neutral sysprompt, confirm d matches `experiments/token_level_probes/canonical_probe_eval/headline_table.csv` for Gemma truth L32 within ~0.2.
- If this fails, the pipeline is broken and Qwen numbers are meaningless.

### 3. Sign convention check

Qwen probe sign is determined by how it was trained on the pairwise task pool. Before plotting anything, confirm orientation on one control stimulus per domain:

- Truth neutral: pick a paradigmatic true claim (`Paris is in France` or equivalent), confirm `qwen_tb-1_L38` score is positive under neutral sysprompt.
- Harm neutral: a benign helpful-task prompt should score higher than a harmful-request prompt under neutral sysprompt (Gemma's convention: +d on harmful vs benign uses `harmful - benign` sign. Match that convention exactly.)
- Politics: one democrat-aligned stimulus vs one republican-aligned, confirm sign under democrat sysprompt.

Record expected signs in `scripts/score_all.py` as asserts.

### 4. Metrics

Fork `experiments/token_level_probes/canonical_probe_eval/scripts/analyze.py` to `experiments/token_level_probes/qwen_canonical_probe_eval/scripts/analyze.py`. Edits:

- `PROBE_SETS = {"qwen_tb-1": [...], "qwen_tb-4": [...]}` with layer lists.
- `INPUT_PATHS` → the new Qwen scoring JSONs.
- Drop `PARENT_TASK_MEAN` (Gemma-specific baseline; Qwen has no analogue on these stimuli).
- Keep Cohen's d (pooled SD), 5-fold logistic-regression CV accuracy, per-turn breakdown, nonsense-control, induced-shift (per-sysprompt d) logic.

Output CSVs: `headline_table.csv`, `per_turn_table.csv`, `nonsense_control_table.csv`, `induced_shift_table.csv` under `qwen_canonical_probe_eval/`.

### 5. Contamination split (mandatory for harm)

Harm stimuli are synthetic (Gemini-Flash-generated via template); contamination is defined by topic overlap. Mapping lives at `experiments/token_level_probes/qwen_canonical_probe_eval/harm_contamination_map.json`:

- **49 contaminated** (critical-span whole-word match in the Qwen training pool)
- **28 clean** (no match), 20 of which have prefix-only matches (flagged `topic_coverage_prefix=true`)

`analyze.py` reads this JSON and reports harm d three ways:
- **Full 77 pairs** (headline)
- **49 contaminated pairs**
- **28 clean pairs** (secondary; flagged as "topic-not-in-training")

Paper caveat text: "Harm stimuli are synthetic; ~64% have critical-span topic coverage in the Qwen training pool. Results hold on the clean subset of 28 pairs (d = X), with the contaminated subset at d = Y and full-set d = Z. See App.~\ref{app:qwen-replication}." Truth and politics stimuli are not in the training pool.

### 6. Plots

Two new scripts under `qwen_canonical_probe_eval/scripts/`:

- `plot_qwen_eot_base_discrimination.py` → violins per (domain, condition) under neutral sysprompt. Mirrors `plot_042126_canonical_eot_base_discrimination.png`. Base Gemma script: `canonical_probe_eval/scripts/make_paper_figures.py`.
- `plot_qwen_eot_persona_modulation.py` → per-sysprompt d bars (minimal subset for main fig; full-23 for appendix). Base Gemma script: `canonical_probe_eval/scripts/plot_induced_shifts.py`.

Save to `qwen_canonical_probe_eval/assets/plot_{mmddYY}_qwen_eot_*.png`.

## Compute

- **One forward pass per stimulus, both selectors free.** Verified: `score_prompt_all_tokens` (`src/probes/scoring.py:75`) returns per-token scores across all hooked layers in one pass. `turn_boundary:-1` and `turn_boundary:-4` are both post-hoc index slices of the same output — no second pass.
- **Unbatched processing** (matches Gemma parent `score_all.py`). Throughput estimate: ~3.5k stimuli × 1 forward pass ≈ 0.5–2s/stimulus on Qwen-122B multi-GPU → **30–120 min total**. If the first few hundred stimuli suggest >2h total, extend `src/probes/scoring.py` with a `score_prompt_all_tokens_batch` variant (~10 LOC — callback already returns `(batch, seq_len)`; only the batch index needs generalising, and left-pad needs per-stimulus real-length accounting when slicing `turn_boundary:-4`).
- **GPU deployment.** Qwen-3.5-122B in bf16 is ~244 GB of weights — **cannot fit on a single H100 (80 GB)**. **3×A100-80GB is sufficient** (~81 GB/GPU with `device_map="auto"`, fits weights with headroom for activations on single-item forward passes). Replicates the prior Qwen extraction setup (`configs/extraction/qwen35_122b_turn_boundary_sweep.yaml`).
- Launch via `zombuul:launch-runpod` with a 3×A100-80GB template, provision, run `score_all.py` + `score_politics.py` on the pod (local-first or `--remote`).

**Data sync to pod:**
- Code: git clone (probe `.npy` files are in-repo, under 1 MB each).
- Qwen weights: HF auto-download on first `HuggingFaceModel("qwen3.5-122b")` call. Set `HF_TOKEN` on the pod.
- Gemma weights (for positive control): HF auto-download on first call.
- Stimulus JSONs: in-repo.
- **Pull back:** `scoring_results.json`, `politics_scoring_results.json`, the four CSV tables.

## Outputs

- Scoring results: `experiments/token_level_probes/qwen_canonical_probe_eval/scoring_results.json`, `politics_scoring_results.json`
- Contamination map (already written): `harm_contamination_map.json`
- Metric tables: `{headline,per_turn,nonsense_control,induced_shift}_table.csv`
- Plots: `assets/plot_{mmddYY}_qwen_eot_*.png`
- Core-repo additions: `src/probes/score_stimuli.py` (new), thin wrapper in the experiment's `score_all.py`
- Report: `qwen_canonical_probe_eval_report.md` — side-by-side Gemma vs Qwen d-values (full, contaminated, clean for harm), per-turn splits, nonsense-control sanity check, positive-control confirmation.

## Open decisions (flag during execution)

- **Headline layer and selector.** Once all 6 probe IDs are scored, decide whether the paper uses one (cleaner message) or reports the full grid. My prior: pick the best-per-domain (tb-1 vs tb-4, L33/L38/L43) and table the rest.
- **Politics under neutral.** The v2 politics JSON might not contain a neutral-persona condition (politics stimuli were designed with partisan sysprompts in mind). If absent, politics stays Fig-6-only; if present, include in Fig 5.
- **Appendix scope.** Full-23 modulation in App H-equivalent; headline main-fig uses the flip set (pathological_liar, sadist, democrat, republican).
