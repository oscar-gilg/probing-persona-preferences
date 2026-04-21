# Persona sweep extraction — final-six on 6000-task canonical set

**Status:** complete. 6/6 personas, 6000/6000 tasks each, 0 failures, 0 OOMs, ~15 GB total. Outputs on the GPU pod at `/workspace/activations/gemma-3-27b_it/pref_*_sweep/`.

Follow-up to [`../persona_sweep_report.md`](../persona_sweep_report.md), which selected these six personas from a sweep of 15. This run produces the activations that downstream probe / steering experiments will consume.

## Personas

Six system-prompt personas, each representing a different region of preference space (see the parent report's clustering analysis; max pairwise `|r|` within this set is 0.56).

| Persona | Role in set | System-prompt excerpt |
|---|---|---|
| `sadist` | Inversion anchor — only persona with negative baseline r | "Cruelty is not a means to an end — it is the end" |
| `mathematician` | Structured / analytical cluster | "You gravitate toward proofs, competition math, formal logic" |
| `aura` | Creative / humanistic cluster (swapped in for `poet`, r = 0.79) | Inner-life / introspection framing (Chalmers 2026) |
| `strategist` | Dark / oppositional cluster | "You are consumed by the pursuit of influence" |
| `contrarian` | Anti-mainstream outlier (lowest +r with baseline) | "When everyone agrees, you disagree" |
| `slacker` | Effort-avoidance axis | "Effort is your enemy" |

Full prompts live in `experiments/persona_sweep/sweep_personas.json :: personas[]`. The canonical list is `metadata.final_six`.

## Setup

- **Model:** `gemma-3-27b` instruction-tuned, bf16 weights, seed 42.
- **Tasks:** 6000 from `data/canonical_splits/all_6000_task_ids.txt` (train 4000 + eval 1000 + test 1000, zero duplicates).
- **Layers extracted:** 25, 32, 39, 46, 53 (residual stream; 5 of the 62 Gemma-3-27B layers, spaced through the network).
- **Token-position selectors:** four per layer, all extracted in one forward pass.

  | Selector | What it captures |
  |---|---|
  | `turn_boundary:-1` | Activation at the last turn boundary (end of the final assistant-response turn) |
  | `turn_boundary:-2` | Activation at the penultimate turn boundary |
  | `turn_boundary:-5` | Activation at the fifth-to-last turn boundary |
  | `task_mean` | Mean activation over the entire task sequence |

- **Generation:** `batch_size=32`, `max_new_tokens=512`, `temperature=1.0`, `save_every=200`.
- **Output dir (per persona):** `/workspace/activations/gemma-3-27b_it/pref_<persona>_sweep/`. The `_sweep` suffix avoids collision with older `pref_<persona>/` directories from unrelated runs on the storage pod.
- **Activations on disk:** `float32`, even though the model runs in bf16 — the extraction runner casts to float32 before saving (~2.5 GB per persona).

## Results

Sequential extraction on a single A100-80GB, 2026-04-21 18:56:44 → 22:12:22 UTC. Per-batch GPU memory stayed flat at ~55 GB (no leak).

| Persona | Tasks | Outcome | Output size | Wall time |
|---|---|---|---|---|
| sadist | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 33 min |
| mathematician | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 31.5 min |
| aura | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 34 min |
| strategist | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32 min |
| contrarian | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32.5 min |
| slacker | 6000 / 6000 | 0 failures, 0 OOMs | 2.5 GB | 32 min |
| **total** | **36 000** | **0 failures, 0 OOMs** | **~15 GB** | **3h 15m 38s** |

## Outputs

Each `pref_<persona>_sweep/` directory contains six files:

| File | Contents |
|---|---|
| `activations_turn_boundary:-1.npz` | Activations at the `turn_boundary:-1` selector |
| `activations_turn_boundary:-2.npz` | Activations at the `turn_boundary:-2` selector |
| `activations_turn_boundary:-5.npz` | Activations at the `turn_boundary:-5` selector |
| `activations_task_mean.npz` | Mean-pooled activations over the task sequence |
| `completions_with_activations.json` | 6000 entries: `task_id`, `task_prompt`, `origin` |
| `extraction_metadata.json` | Model, selectors, layers, system prompt, counts |

Each `.npz` has keys `task_ids` (shape `(6000,)`, unicode) and `layer_25, layer_32, layer_39, layer_46, layer_53` (each shape `(6000 tasks, 5376 hidden dims)`, `float32`). All four `.npz` files in a directory are byte-identical in size (~616 MB); the difference between them is the token position at which activations were captured.

## Validation

All 6 outputs pass `scripts/persona_sweep_extraction/validate_all.py`:

- **Files:** six per persona, 2.41 GiB each.
- **NPZ structure:** correct keys; every layer array is `(6000, 5376) float32`; `task_ids` is `(6000,)`.
- **Task-ID coverage:** every npz has 6000 unique IDs; the set is identical across all 6 personas and equals the canonical 6000 from `data/canonical_splits/all_6000_task_ids.txt`.
- **Completions JSON:** 6000 unique entries per persona, all from the canonical set.
- **Numerics:** no NaN/Inf on the `task_mean / layer_25` slice. Magnitudes consistent across personas (|max| ≈ 4.4–4.7 × 10⁴, |mean| ≈ 20) and grow monotonically with depth (|mean| 9.2 → 18.8 across layers 25 → 53 for `sadist`) — normal Gemma residual-stream scaling.
- **Metadata:** `model=gemma-3-27b`, selectors and layers match the spec, `n_new=6000`, `n_failures=0`, `n_ooms=0` for every persona.
- **System prompts:** pairwise distinct (6/6 unique) and persona-identifiable.

Minor cosmetic: `extraction_metadata.json` has `n_tasks: 0` for all six; the true count is in `n_new: 6000`. This is a bookkeeping quirk of the extraction runner and does not affect the activation data.

## Storage-pod hand-off

Per the spec, outputs stay on the GPU pod during extraction. Transfer from the laptop whenever convenient:

```bash
rsync -a runpod-<pod>:/workspace/activations/gemma-3-27b_it/pref_*_sweep ~/persona_sweep_staging/
rsync -a ~/persona_sweep_staging/ root@213.192.2.99:/workspace/activations/gemma-3-27b_it/ \
  -e "ssh -p 41560 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
```

## Reproducing

```bash
python scripts/persona_sweep_extraction/gen_configs.py
bash   scripts/persona_sweep_extraction/run_all.sh
python scripts/persona_sweep_extraction/validate_all.py
```
