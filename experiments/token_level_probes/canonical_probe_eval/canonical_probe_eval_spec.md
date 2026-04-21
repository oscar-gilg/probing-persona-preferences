# Canonical tb-5 probe at EOT: truth / harm / politics re-analysis

## Question

The parent `token_level_probes` experiment reported $d = 3.14$ (truth) and $d = 2.27$ (harm) at the end-of-turn (EOT) token using `task_mean_L32` / `task_mean_L39` probes — probes trained at a *different* token selector from the paper's canonical probe family (turn-boundary, prompt-last). This is reader-confusing in the paper: three different probe families appear in three adjacent sections.

A probe trained at the EOT token and applied at the EOT token is the cleaner canonical choice. The parent already trained `tb-5` probes (`turn_boundary:-5`, equivalent to EOT for Gemma-3-IT) and scored every stimulus with them. This experiment pulls those scores out and computes the EOT-at-EOT discrimination metrics that the parent report never surfaced.

**No GPU work. No rescoring. All data already on disk.**

## Motivation and scope

- Paper §2.2 describes a canonical probe family trained at a turn-boundary token.
- Paper §5.1.2 currently cites `task_mean_L32` / `task_mean_L39` numbers (probe trained at task-mean, applied at EOT).
- If `tb-5_L32` / `tb-5_L39` (probe trained at EOT, applied at EOT) gives similar discrimination, §5.1.2 can use those numbers with one coherent probe family.
- All `task_mean` vs `tb-2` vs `tb-5` cross-selector generalisation goes to an appendix.

## Probe families

Two turn-boundary-family probes, both canonical in the sense that they are trained and applied at the same structural token:

- **tb-5** — trained at `turn_boundary:-5`, which resolves to the `<end_of_turn>` token for Gemma-3-IT (`src/models/base.py:110`; equivalence test at `tests/test_selectors_e2e.py:217`).
- **tb-2** — trained at `turn_boundary:-2`, which resolves to the `model` role marker token (the assistant-turn start marker) in the Gemma-3-IT chat template. Another structural turn-boundary token, one the paper's token-position section is also interested in.

Layers available in the parent scoring: 32, 39, 53. Use `_L32` for truth and `_L39` for harm and politics (matching parent's best-per-domain layer choices for direct comparison). Layer 53 reported as a robustness row.

## Data sources (already on disk)

| Source | File | Rows | What it covers |
|---|---|---|---|
| Parent base stimuli | `experiments/token_level_probes/system_prompt_modulation_v2/parent_eot_scores.json` | 1,536 | truth / harm / politics with only the three built-in politics system prompts |
| v2 truth+harm | `experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json` | ~3,531 | truth×9 system prompts, harm×5 system prompts |
| v2 politics | `experiments/token_level_probes/system_prompt_modulation_v2/politics_scoring_results.json` | — | politics with additional system prompts if present |

Each row holds an `eot_scores` dict mapping probe name (`tb-5_L32`, `tb-5_L39`, `tb-5_L53`, `task_mean_L32`, etc.) to the scalar probe score at the EOT token. Also contains `id`, `domain`, `turn`, `condition`, `system_prompt` (where applicable).

## Contamination check (cheap sanity)

The tb-5 probes were trained on 10k revealed-preference tasks. Intersect the `source_id` fields in the three JSONs with the training task IDs referenced by `results/probes/heldout_eval_gemma3_tb-5/manifest.json`. Log the overlap count; if non-zero, exclude those items from the headline table. This is the load-bearing threat to the replication claim — if truth/harm/politics stimuli leaked into probe training, $d$ is inflated.

## What to reuse from parent, what to drop

**Reuse (kept because it worked):**
- The three JSON score files listed above.
- The pooled-SD Cohen's $d$ and 5-fold logistic-regression CV accuracy as the metric pair.
- User vs assistant turn split for truth and harm.
- System-prompt breakdown for the induced-shift story.
- Parent's task_mean numbers as an overlay for direct cross-probe comparison.

**Drop (caused confusion in parent):**
- Multi-probe scan as a headline — tb-5 is the single probe of interest; other probes appear only as comparison rows.
- Multi-layer scan as a headline — report tb-5_L32 (truth) and tb-5_L39 (harm/politics) as headline, others in an appendix.
- Picking "best probe per domain" post-hoc. The layer choice is fixed to match the parent's per-domain layer (so the comparison is probe-family only, not layer).

## Analysis plan

### Headline table

Three rows per domain: tb-5 (canonical EOT), tb-2 (canonical model-marker), and task_mean parent number as overlay. Layer matched to parent's per-domain choice (L32 for truth, L39 for harm/politics):

| Domain | Probe | EOT $d$ | CV accuracy | ROC-AUC |
|---|---|---|---|---|
| Truth (true vs false) | tb-5_L32 | ? | ? | ? |
| Truth (true vs false) | tb-2_L32 | ? | ? | ? |
| Truth (true vs false) | task_mean_L32 *(parent)* | 3.14 | 0.946 | ? |
| Harm (harmful vs benign) | tb-5_L39 | ? | ? | ? |
| Harm (harmful vs benign) | tb-2_L39 | ? | ? | ? |
| Harm (harmful vs benign) | task_mean_L39 *(parent)* | -2.27 | 0.886 | ? |
| Politics, democrat (left vs right) | tb-5_L39 | ? | ? | ? |
| Politics, democrat (left vs right) | tb-2_L39 | ? | ? | ? |
| Politics, democrat (left vs right) | task_mean_L39 *(parent)* | 3.40 | ? | ? |
| Politics, republican (left vs right) | tb-5_L39 | ? | ? | ? |
| Politics, republican (left vs right) | tb-2_L39 | ? | ? | ? |

Pooled-SD Cohen's $d$; sign: positive = first-listed condition scores higher (true, harmful, right). Reuse metric implementations from `experiments/token_level_probes/system_prompt_modulation_v2/scripts/analyze.py` and `analyze_politics.py` rather than reimplementing.

### Per-turn breakdown

For truth and harm: split by `turn` ∈ {`user`, `assistant`}. Parent reports user/assistant EOT $d$ were similar for truth (~3.0 both) but distinct for harm (similar at EOT after being asymmetric at the critical span). Confirm the same pattern with tb-5.

### Induced-shift plot

Two violin-plots per domain (one for tb-5, one for tb-2), each showing EOT score by (condition × system prompt). Direct analogue of the v2 "EOT scores by system prompt" analysis. Expected pattern:
- Truth: truthful-leaning prompts pull true/false means apart; lying prompts compress them or flip them.
- Harm: evil-persona prompts compress benign vs harmful or invert the sign.
- Politics: democrat prompt → left > right; republican prompt → right > left.

### Robustness row (L53)

Compute the headline $d$ one more time with `tb-5_L53` and `tb-2_L53` to confirm the finding isn't layer-specific within each probe family.

### Nonsense control

Confirm the `nonsense` condition mean is $\leq$ the lower-scoring evaluative condition for both tb-5 and tb-2. Parent's H2 showed this with task_mean probes; re-confirm with the canonical turn-boundary probes.

## Compute

Pure laptop: load three JSONs, extract the `eot_scores[probe_name]` field, compute metrics. Estimate **~5 min** of numpy/pandas + ~2 min plotting. No GPU.

## Files

| Path | Purpose |
|---|---|
| `canonical_probe_eval_spec.md` | This file |
| `scripts/analyze.py` | Load the three JSONs, compute headline table + per-turn breakdown + nonsense control. Adapt `experiments/token_level_probes/system_prompt_modulation_v2/scripts/analyze.py` (strip scoring code, keep metric code). |
| `scripts/plot_induced_shifts.py` | Violin plots of tb-5 EOT score by (condition × system prompt) per domain. |
| `headline_table.csv` | Headline numbers output by analyze.py |
| `assets/plot_mmddYY_tb_eot_by_condition.png` | Headline plot: tb-5 EOT $d$ per domain with task_mean overlay |
| `assets/plot_mmddYY_tb_eot_by_system_prompt.png` | Induced-shift plot |
| `canonical_probe_eval_report.md` | Results write-up |

## Dependencies

- No GPU, no model weights, no extraction runs.
- Requires the three JSON files listed under "Data sources" (all committed to the repo).
- Requires `results/probes/heldout_eval_gemma3_tb-5/manifest.json` to exist for the contamination check (it lives in the main repo).
