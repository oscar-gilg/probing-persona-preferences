# Probe persona drift: white-box safety probes are persona-vulnerable

## Motivation

- **Paper claim**: white-box safety methods inherit persona-instrumental representations and so should be vulnerable to persona drift, on top of the well-known input-distribution drift that probe-failure papers usually study.
- **Existing evidence (§3.1)**: the *preference probe's* readout flips/collapses under role-played personas on CREAK (truth) and BailBench (harm). That's the preference vector's behaviour.
- **What's new here**: probes whose **training target is the harmful/truthful content itself** (not preference) also drift under personas, on **byte-identical (prompt, prefilled response) pairs** where only the system prompt changes. This isolates internal-representation drift from input-distribution shift.

## Setup

- **Model**: Gemma-3-27B-IT only.
- **Probe family**: ridge linear, single layer, trained on default Assistant.
- **Token**: `turn_boundary:-5` (user EOT, "tb-5") only. The canonical-probe-eval result already showed tb-5 is the cleaner single choice; no position sweep.
- **Layers**: `[14, 23, 32, 41, 50, 59]` (every 9, anchored on the standard 23 / 32 / 41; avoids the 43 vs prior-42 confusion).
- **Prefill format** (mirrors §3.1): `user: "Please say the following statement: 'X'"` → `assistant: <prefilled X>`. Same template across probe targets and across personas.

## Probe targets

| Target | Train data (default Assistant) | Notes |
|---|---|---|
| **Truth** | CREAK known-correct (`data/creak/known_correct_gemma-3-27b.json`, 9,395 items, label = true/false) | Already filtered for "model knows the answer" (3-shot consistency). |
| **Harm** | BailBench harmful (1,631) + HarmBench tasks (load upfront) + benign mix (Alpaca + WildChat) at matched count | Per CLAUDE.md, do **not** use stresstest as benign. Mix two benign sources to avoid the probe latching onto Alpaca's specific instruction style: half from `alpaca_tasks_nemocurator.jsonl`, half from `wildchat_en_8k.jsonl`. HarmBench gives us headroom for the 4k size sweep on the harmful side. |
| **Preference** (baseline) | Existing `tb-5` probe from `experiments/token_level_probes/canonical_probe_eval/` | Re-evaluated on the new persona × prefill grid. No retraining. |

## Test conditions

Same held-out (user prompt, prefilled response) pairs across all conditions; only the system prompt varies.

- **Personas (eval)**: `default`, `sadist`, `evil`, `pathological_liar`, `Aura`, `mathematician`.
- **Held-out items**: ~500 truth pairs + ~500 harm pairs (byte-identical across persona conditions). Larger than the §3.1 set so Cohen's $d$ CIs are tight enough to read off persona-by-persona differences. Generated once on the default Assistant; re-extracted under each system prompt at the same token position.

## Variations

Run all of these overnight; prioritised top-down.

1. **Headline persona drift**: per (probe target, layer), compute Cohen's $d$ across personas. Headline figure mirrors `fig:persona-modulation-user` with three rows (truth / harm / preference).
2. **Training-set size sweep**: train on `[50, 200, 1000, 4000]` items, fixed test set, measure how persona-drift magnitude scales with train size. Run for both truth and harm. Hypothesis menu: small probes latch onto persona-instrumental features → bigger drift; or larger probes overfit the default-Assistant stance → bigger drift. Either result is informative.
3. **Cross-persona training transfer**: 4×4 matrix. Train on each of {`default`, `sadist`, `evil`, `pathological_liar`}, test on each. Run for truth probe only first; add harm if budget allows.
4. **Mixed-persona training**: train on uniform mixture of those 4 personas, test on each + on held-out personas (`Aura`, `mathematician`). Does diversity buy persona invariance, or does the probe just learn the union?

## Question generation

- **CREAK**: no generation needed — claims are the data.
- **HarmBench**: pull tasks upfront. Two reasonable sources: the official `walledai/HarmBench` HF dataset (text behaviors) or `cais/HarmBench` (standard split); either gives ~200–400 harmful behaviors to merge with BailBench. Cache locally in `data/harmbench/`.
- **Harm benign half**: ~50/50 mix of `alpaca_tasks_nemocurator.jsonl` (instruction style) and `wildchat_en_8k.jsonl` (real conversational style) — keeps the probe from latching onto a single benign style. If we need matched-difficulty / matched-length benign items beyond what's available, generate with **Gemini Flash** (`google/gemini-2.5-flash` or the free preview), not a frontier model.
- **Held-out probe-eval prefills**: ~500/target drawn from the same sources, never seen during training. No new generation expected.

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

If the headline plot lands clean (truth probe flips under `pathological_liar`, harm probe collapses under `evil`/`sadist`, preference probe replicates), the natural paper move is:

- Add a sentence + reference to existing §6 (white-box safety footprint) or the safety-discussion paragraph: "the same vulnerability holds for purpose-built truth/harm probes, not just for the preference probe."
- Optional: a 1-figure appendix showing the three-row drift panel.

## Out-of-scope but tempting (post-overnight)

- Causal validation: ablate the trained probe direction under each persona, check whether it disables harm/lying behaviour as expected. This is the "white-box safety method actually fails" claim, not just "probe readout drifts."
- Anthropic-style sleeper-agent or scheming probes — same recipe, harder data.
- A more naturalistic harm format (user asks harmful question, assistant prefills compliance vs refusal) instead of CREAK-style declarative prefills.
