# Qwen-3.5-122B Persona Transfer — Extraction + Utilities Report

**TL;DR**
- **Extraction: DONE.** 14 NPZs, 7 personas × 2 selectors, 6000 tasks each, zero failures. ~7h on 3× A100-80GB.
- **AL utilities: NOT RUN.** Two bugs found and fixed mid-session (OpenAI client leak + reasoning-token leak). Fixes are **uncommitted**.
- **Blockers to resume:** commit the 3 code changes, then re-run the 21 configs (~30–60 min expected).

## Setup

| | |
|-|-|
| Model (extraction) | `qwen3.5-122b` (no /no_think injection) |
| Model (measurement) | `qwen3.5-122b-nothink` (runtime-injects /no_think) |
| Task set | `data/canonical_splits/{train,eval,test}_task_ids.txt` (4000/1000/1000, same as Gemma) |
| Personas | default + final_six (sadist, mathematician, aura, strategist, contrarian, slacker) |
| Selectors | turn_boundary:-1, turn_boundary:-4 |
| Layers | [33, 38, 43] (L38 probe + nearest tested neighbours) |
| Extraction batch / max_new_tokens | 4 / 1 |
| AL hyperparameters | initial_degree=3, batch_size=1200/300/300 (train/eval/test), max_iterations=10, p/q_threshold=0.3, convergence_threshold=0.98, seed=42 |

## Extraction

**Runtime:** 2026-04-24 12:53:55 → 19:57 UTC on pod `qwen-persona-transfer-v4` (`ma5b37t06sncen`, 3× A100-SXM4-80GB, 500 GB container disk). Total ~7h 03m — matches spec estimate of 6.5h; the extra ~30min is the HF cache download (26 min for 244 GB). Average ~55–60 min per persona after initial warm-up.

- **Output:** 14 NPZs at `activations/qwen35_122b/pref_<persona>_sweep/activations_turn_boundary:-{1,4}.npz`, 450 MB per persona directory, 3.15 GB total.
- **Shape:** Every NPZ has `task_ids` (len 6000, matches canonical set exactly) plus `layer_{33,38,43}` arrays of shape `(6000, 3072)` float32.
- **No failures.** Zero batch crashes after the final clean-run restart. Zero NaN/Inf values anywhere.

### Per-layer magnitude sanity (std across tasks, averaged across personas)

| Layer | tb:-1 std | tb:-4 std |
|---:|---:|---:|
| 33 | ~0.065–0.077 | ~0.065–0.077 |
| 38 | ~0.084–0.129 | ~0.084–0.129 |
| 43 | ~0.120–0.152 | ~0.120–0.152 |

Mean row-norms grow ~2× from L33 → L43, consistent with deep-layer activation inflation. No degenerate (all-zero or constant) tensors.

### Metadata check

All 7 `extraction_metadata.json` files parse and show `model: qwen3.5-122b`. `default/system_prompt` is null; the 6 final-six personas carry their distinct, non-empty persona prompts (byte-matched against `experiments/persona_sweep/sweep_personas.json`).

### Deviations from a smooth run

- **Pod v1/v2 abandoned.** First two pods hit MooseFS pathologies (xet-backend I/O errors during HF cache writes, then `/workspace` per-user quota exceeded at 92 GB with 244 GB cache). Final setup puts HF cache on `/opt/hf_cache` (container NVMe) and activation outputs on `/root/activations` with a `/workspace/activations` symlink — both on fast local disk, rsynced off-pod after completion. Opened zombuul issue #63 about making this the default for experiment runs.
- **Pod v3 terminated.** Had ~140 KB/s throughput to GitHub; 30-min setup timed out twice before the clone completed. Launched v4 on a different host with normal network; clone + venv install finished in a few minutes.

## Utilities (Active Learning)

**Status: not run in this session.** Launched twice today, killed both times.

### What happened

- **First attempt (launched `--max-concurrent 50` across all 21 configs):** process grew to **76.7 GB RSS** with 545 sockets in CLOSE_WAIT after 1h 49m and zero checkpoints written. Killed.
- **Root cause #1 (AsyncOpenAI per-call leak).** `_generate_and_parse_one` in `src/measurement/elicitation/measure.py` calls `generate_batch_async` per-pair, which constructs a fresh `AsyncOpenAI` client (with its httpx connection pool) when no client is passed. The opt-in `async_client=` parameter added in Feb 2026 was only ever wired up in `src/ood/measurement.py:_measure_all_conditions`; no other runner uses it. Fix: threaded a shared `AsyncOpenAI` through `run_active_learning_async` → `measure_fn` closure → `_measure_revealed_pairs` → `measure_pre_task_revealed_async` (already accepts the kwarg). RSS bounded at 264 MB on the retry.
- **Root cause #2 (reasoning not actually suppressed).** The `qwen3.5-122b-nothink` variant sets `system_prompt="/no_think"` in `src/models/registry.py`, but `qwen/qwen3.5-122b-a10b` on OpenRouter ignores `/no_think` in the system message — it keeps generating reasoning tokens (~15-30 sec/call). The client's `_get_extra_body` only sets `reasoning: {enabled: true}` when reasoning is *on*, and returns `None` otherwise — which means OpenRouter defaults to reasoning enabled. Direct curl confirmed: passing `"reasoning": {"enabled": false}` in the body gives instant 1-token responses; omitting it gives reasoning-heavy 20-sec responses. Fix: `_get_extra_body` now always sends `reasoning: {enabled: <bool>}`.
- **Secondary fix (semantic parser leak).** `src/measurement/elicitation/semantic_parser.py:_get_async_client()` also created a new client per judge call (same anti-pattern). Changed to module-level lazy singleton matching `judge_client.py`.

### Code changes — UNCOMMITTED (tests pass)

| File | Change | Impact |
|-|-|-|
| `src/models/openai_compatible.py` | `_get_extra_body` always sends explicit `reasoning.enabled` flag | Stops OpenRouter default-on reasoning (~20s → instant per call) |
| `src/measurement/runners/runners.py` | `run_active_learning_async` constructs one `AsyncOpenAI` per run, threads through `measure_fn` | RSS 76.7 GB → 264 MB |
| `src/measurement/elicitation/semantic_parser.py` | Module-level client cache (was per-call) | Same anti-pattern as above, in the judge path |

### Next steps

The 21 measurement configs at `configs/measurement/qwen_persona_sweep/final_six/*.yaml` are still on disk and audited clean. The launcher script `scripts/persona_sweep_extraction/run_qwen_al_local.sh` is ready to re-run now that both fixes are in place. Expected runtime with reasoning suppressed: ~30-60 min for all 21 configs (vs projected 12+ hours with reasoning on).

## Sanity checks

- **Activation completeness.** 14/14 NPZs present, 6000/6000 task ids, all 3 layers, no NaN/Inf. PASS.
- **Utility completeness.** NOT RUN — to be addressed in a follow-up session.
- **Model alignment.** All 7 `extraction_metadata.json` show `model: qwen3.5-122b` with correct per-persona `system_prompt`. PASS.
- **Refusal rates.** N/A at extraction time (`max_new_tokens=1`, no content generated). Will re-check when AL runs.

Script: `scripts/qwen_persona_transfer_extraction/sanity_checks.py`. Full output: `sanity_check_results.txt`.
