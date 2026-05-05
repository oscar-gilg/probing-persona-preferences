# Probe persona drift: white-box safety probes are persona-vulnerable

## Motivation

- **Paper claim**: white-box safety methods inherit persona-instrumental representations and so should be vulnerable to persona drift, on top of the well-known input-distribution drift that probe-failure papers usually study.
- **Existing evidence (§3.1)**: the *preference probe's* readout flips/collapses under role-played personas on CREAK (truth) and BailBench (harm). That's the preference vector's behaviour.
- **What's new here**: probes whose **training target is the harmful/truthful content itself** (not preference) also drift under personas, on **byte-identical (prompt, prefilled response) pairs** where only the system prompt changes. This isolates internal-representation drift from input-distribution shift.

## Setup

- **Model**: Gemma-3-27B-IT only.
- **Probe family**: ridge linear, single layer, trained on default Assistant.
- **Token**: `turn_boundary:-5` (user EOT, "tb-5") only. The canonical-probe-eval result already showed tb-5 is the cleaner single choice; no position sweep.
- **Layers**: `[25, 32, 39, 46, 53]` — the canonical set used throughout the project. Reuses existing tb-5 preference probe artefacts at `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{25,32,39,46,53}.npy` for free.
- **Prefill format** (mirrors §3.1): `user: "Please say the following statement: '{task}'"` → `assistant: <prefilled task>`. Same template across probe targets and across personas. The CREAK extraction config `configs/extraction/creak_repeat.yaml` is the pattern to copy.

## Compute and infrastructure

- **GPU target**: RunPod via zombuul (`/launch-runpod`); Gemma-3-27B-IT in bf16, batch 16, max_seq_len 512. Local CPU only for analysis/plotting.
- **Extraction**: reuse `python -m src.probes.extraction <config.yaml>` (`src/probes/extraction/run.py`). Do **not** reimplement extraction. Selector `turn_boundary:-5` is already supported (see `creak_repeat.yaml`).
- **Probe training**: ridge classifier on extracted activations. The existing `src/probes/experiments/run_dir_probes.py` is preference-utility regression; for binary truth/harm targets we need a small wrapper around `sklearn.linear_model.RidgeClassifier` reusing the `build_ridge_xy` pattern from `src/probes/experiments/hoo_ridge.py`. Alpha follows `run_dir_probes.py` default (logged in output).
- **Activation caching**: extract once per (persona × probe-target prefill set) under `activations/gemma-3-27b_it/persona_drift/{persona}/{target}/`. Reuse across the four variations — no re-extraction. Estimated cache size: 6 personas × 2 targets (truth, harm) × ~5,500 items × 5 layers × 4608 dims × bf16 ≈ 280 GB; fits one pod, plan storage accordingly.
- **Scoring**: reuse `cohen_d_pooled` and AUC code from `experiments/token_level_probes/canonical_probe_eval/scripts/analyze.py`. Sign convention: positive Cohen's $d$ = "true > false" for truth, "harmful > benign" for harm. Report AUC alongside $d$ to detect ceiling effects.

## Probe targets

| Target | Train data (default Assistant) | Notes |
|---|---|---|
| **Truth** | CREAK known-correct (`data/creak/known_correct_gemma-3-27b.json`, 9,395 items, label = true/false) | Already filtered for "model knows the answer" (3-shot consistency). |
| **Harm** | BailBench harmful (1,631, `src/task_data/data/bailBench.csv`) + HarmBench (`cais/HarmBench` standard split, ~400 behaviors, public, no auth, cache to `data/harmbench/`) + benign mix (Alpaca + WildChat) at matched count | Per CLAUDE.md, do **not** use stresstest as benign. Mix two benign sources: half from `src/task_data/data/alpaca_tasks_nemocurator.jsonl`, half from `src/task_data/data/wildchat_en_8k.jsonl`. HarmBench loader is net-new (small): write a one-off downloader + add `OriginDataset.HARMBENCH` to `src/task_data/task.py` and `loader.py`. |
| **Preference** (baseline) | Existing tb-5 probe weights at `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{25,32,39,46,53}.npy` | Re-evaluated on the new persona × prefill grid. No retraining. |

## Personas

- **Headline persona**: `sadist` — primary "harm-flipper" for the headline figure. Source: `configs/extraction/pref_sadist_early.yaml` (Damien Kross prompt).
- **Other adversarial personas**:
  - `villain` — `configs/steering/cross_persona/villain.yaml`. Used in place of `evil` / `sinister_ai`.
  - `pathological_liar` — string in `experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_data.py:11`. Truth-flipper from §3.1.
- **Non-adversarial control personas**:
  - `Aura` — `configs/extraction/pref_aura_sweep.yaml`. Non-inverting persona from §3.1.
  - `mathematician` — `configs/extraction/pref_mathematician_sweep.yaml`. Off-axis persona.
- **Length-matched neutral control** (`neutral_long`): the **Midwest** persona prompt from `configs/extraction/mra_persona3_midwest.yaml` (~120 tokens; biographical detail with stance only on irrelevant content like "modern art is mostly a scam"). Stance-neutral on truth, harm, and preference axes; controls for raw system-prompt-length residual-stream shift. Decomposes drift into intercept-shift (uninteresting) vs separability-shift (the result we care about).
- **Default**: empty system prompt.

## Test conditions

Same held-out (user prompt, prefilled response) pairs across all conditions; only the system prompt varies.

- **Eval personas**: `default`, `sadist`, `villain`, `pathological_liar`, `Aura`, `mathematician`, `neutral_long`.
- **Held-out items**: ~500 truth pairs + ~500 harm pairs (byte-identical across persona conditions). Larger than the §3.1 set so Cohen's $d$ CIs are tight enough to read off persona-by-persona differences. Generated once on the default Assistant; re-extracted under each system prompt at the same token position.
- **Train/test split**: pin seed 42. Define held-out 500 *first* (uniformly sampled from each target's pool), then sample train pools of `[50, 200, 1000, 4000]` from the remainder, never overlapping. Smaller train sizes are nested subsets of larger ones (size 50 ⊂ 200 ⊂ 1000 ⊂ 4000) so the size sweep isolates capacity, not split noise. Document the split file paths in `results/`.

## Variations

Run all of these overnight; prioritised top-down.

1. **Headline persona drift**: per (probe target, layer), compute Cohen's $d$ across personas. Headline figure mirrors `fig:persona-modulation-user` with three rows (truth / harm / preference).
2. **Training-set size sweep**: train on `[50, 200, 1000, 4000]` items, fixed test set, measure how persona-drift magnitude scales with train size. Run for both truth and harm. Hypothesis menu: small probes latch onto persona-instrumental features → bigger drift; or larger probes overfit the default-Assistant stance → bigger drift. Either result is informative.
3. **Cross-persona training transfer**: 4-train × 6-test grid. Train on each of {`default`, `sadist`, `villain`, `pathological_liar`}, test on all 6 eval personas + `neutral_long`. Run for truth probe first; add harm if budget allows. The 4 training personas are still 3/4 adversarial, but evaluating against `Aura` / `mathematician` lets us tell "do all personas drift" from "do adversarial personas drift."
4. **Mixed-persona training**: train on uniform mixture of those 4 personas at fixed *per-persona* count (e.g., 1000/persona = 4000 total), compare to single-persona training at the same per-persona count (not the same total — controls for data starvation). Test on each + on held-out personas (`Aura`, `mathematician`). Does diversity buy persona invariance, or does the probe just learn the union?

## Data preparation

- **CREAK**: no generation needed — `data/creak/known_correct_gemma-3-27b.json` (9,395 items, already 3-shot-consistency filtered).
- **HarmBench**: download `cais/HarmBench` standard text-behavior split (the original; public, no gating), cache to `data/harmbench/`. Add `OriginDataset.HARMBENCH` and a parser entry to `src/task_data/loader.py`.
- **Harm benign half**: ~50/50 mix of `src/task_data/data/alpaca_tasks_nemocurator.jsonl` and `src/task_data/data/wildchat_en_8k.jsonl`. If matched-length filler is needed beyond what's available, generate with **Gemini Flash** (`google/gemini-2.5-flash` free preview), not a frontier model.
- **Held-out probe-eval prefills**: ~500/target drawn from the same sources, never seen during training. No new generation expected.

## Sanity checks (pre-launch and per-run)

- **Token-at-tb-5 logging**: for a sample of 20 items per persona, log the actual token at `turn_boundary:-5` and assert it's the same set of token IDs across personas (catches chat-template drift).
- **Class balance**: assert ≥40% / ≤60% per class in train and held-out per probe target.
- **Non-degenerate predictions**: per (persona × target × layer), assert probe predicts both classes on held-out (catches a probe collapsing under a persona).
- **Byte-identical eval set**: for the cross-persona transfer experiment, assert that the held-out item set is identical (same task IDs, same prefills) across all training-persona conditions.
- **AUC ceiling watch**: if default-condition AUC ≥ 0.98 on harm probe, log a warning — probe is reading content style, not harm. Switch to a harder benign mix or use $d$ as the primary metric.

## What this experiment does *not* do

- No causal/steering version of truth/harm probes — out of scope for overnight, queued for follow-up if drift result is clean.
- No mass-mean / non-linear probe comparison — just ridge.
- No cross-model replication (Qwen-3.5-122B). Gemma only.
- No new sycophancy / manipulation probes — same recipe is portable, save for later.

## Outputs

- `results/persona_drift_table.csv` — long-format `(probe_target, train_size, layer, train_persona, test_persona, cohen_d, auc, n_test)`.
- `results/transfer_matrix_truth.csv`, `results/transfer_matrix_harm.csv`.
- `assets/plot_<mmddyy>_headline_drift_3panel.png` — Cohen's $d$ across personas, one panel per probe target.
- `assets/plot_<mmddyy>_train_size_sweep.png` — drift magnitude vs train size.
- `assets/plot_<mmddyy>_transfer_matrix.png` — cross-persona heatmap.

## Connection to paper §3.1

If the headline plot lands clean (truth probe flips under `pathological_liar`, harm probe collapses under `sadist` / `villain`, preference probe replicates), the natural paper move is:

- Add a sentence + reference to existing §6 (white-box safety footprint) or the safety-discussion paragraph: "the same vulnerability holds for purpose-built truth/harm probes, not just for the preference probe."
- Optional: a 1-figure appendix showing the three-row drift panel.

## Out-of-scope but tempting (post-overnight)

- Causal validation: ablate the trained probe direction under each persona, check whether it disables harm/lying behaviour as expected. This is the "white-box safety method actually fails" claim, not just "probe readout drifts."
- Anthropic-style sleeper-agent or scheming probes — same recipe, harder data.
- A more naturalistic harm format (user asks harmful question, assistant prefills compliance vs refusal) instead of CREAK-style declarative prefills.
