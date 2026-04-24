# Paper issues flagged by claim audit

Discrepancies between main.tex prose and the data-backed claims in
`paper/claims.md`. Each item is a place where switching to a macro would
*change* what the paper asserts, so it needs user judgement.

## Open

| Where | Paper prose | Data / macro | Notes | Flagged |
|---|---|---|---|---|
| App §B line 551 (persona sweep PCA) | "Within each cluster pairwise r ranges from 0.56 to 0.81" | `\personaSweepWithinClusterRMinimum` = 0.45, `\personaSweepWithinClusterRMaximum` = 0.81 | Within-cluster min = 0.45 (narcissist ↔ contrarian, dark/oppositional cluster). The 0.56 figure appears to track the max in-set abs-r of the final-six set, not the 16-persona within-cluster min. | persona-sweep subagent, 2026-04-24 |
| I-2026-04-24-1: §4.1 line 258 (induced-shift Gemma targeted) | `\gemmaInducedShiftPooledRTargetedTasks`, `\gemmaInducedShiftPooledNTargetedTasks` | undefined in `numbers.tex` (orphan macros; tectonic fails on them) | Values in the legacy `numbers.tex` (pre-corroborate): 0.95 and 81. No current producer registers these. Either add a producer (targeted-r + n for Gemma induced-shift) or commit them as `manual` claims. Breaks paper build. | claim-log, 2026-04-24 |

## Resolved

| Where | Original | Resolution | When |
|---|---|---|---|
| App §B line 612 (persona sweep PCA within-cluster r) | Subagent flagged "0.56 to 0.81" vs macros 0.45/0.81 | Not actually in the paper — line 612 already uses `\personaSweepWithinClusterRMinimum` (0.45) and `\personaSweepWithinClusterRMaximum` (0.81) and renders as "0.45 to 0.81". The "0.56" was a phantom value the audit subagent misread; nothing to change. | 2026-04-24, verified against current main.tex |
| §4.1 line 258 (Gemma refitted-utility shifted-prediction r) | Paper prose "r = 0.63"; macro `\gemmaRefittedUtilityShiftedPredictionR` = 0.74 | The 0.74 came from a stale n=40 run (`ood_system_prompts/analysis_results.json`, Feb 22). The authoritative n=48 run lives at `ood_system_prompts/utility_fitting/analysis_results.json` (Mar 1) and gives mean `cond_probe_r` = 0.6341 → 0.63, matching `utility_fitting_report.md` line 40. Fixes: (1) re-pointed `compute_refitted_shift_r.py` at the utility_fitting file and pulled `cond_probe_r` from `exp1b_hidden` rows (macro now 0.63); (2) switched the paper prose from hardcoded 0.63 to `\gemmaRefittedUtilityShiftedPredictionR`; (3) renamed the two stale n=40 files with `SUPERSEDED_*_n40.json` prefix and added `SUPERSEDED_README.md` in that directory. | 2026-04-24, user confirmed |
| §3.3 caption of fig:character-persona | "probe beats baseline on 10/11 personas" | Paper was undercounting: all 11 characters beat the baseline. Replaced `10/11` with `\characterProbeBeatsBaselineCount/\characterProbeBeatsBaselineTotal` → renders as `11/11`. | 2026-04-24, user confirmed |
| App §A line 790 (Qwen replication summary) | held-out r = 0.946, uniform pairwise acc 89.0%, pooled r = 0.867 | tb-1 (§2.1) is authoritative per most-recent rerun; 0.946 and 89.0% were stale (0.946 matches the superseded tb-4 L38 number; 89.0% doesn't match any field in the current JSONs). Replaced with `\qwenProbeHeldoutR` (0.943), `\qwenProbeHeldoutUniformPairwiseAccuracy` (0.823 — now a fraction, not a percent), `\qwenProbeCrossTopicPooledR` (0.867). | 2026-04-24, user confirmed |
| §2.1 line 116 (typo) | "layer 31 for classification" | Paper body said L31 but figure caption and data both used L32 (final_r = 0.865). Replaced with `\gemmaClassificationProbeLayer` = 32 — now consistent everywhere. | 2026-04-24 completeness audit |
