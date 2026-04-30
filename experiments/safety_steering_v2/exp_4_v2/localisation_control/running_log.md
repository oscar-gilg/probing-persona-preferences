# Localisation control — running log

## 2026-04-30 — setup + scope reduction

- Worktree at `.claude/worktrees/localisation_control` on branch `worktree-localisation_control`.
- Probe symlinked from main: `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy`.
- Pod inventory: `aura-gemma` (H100 SXM, paused) is the resume candidate. `sadist-pv-v2` reserved for the Qwen persona vectors work.
- **Scope reduction vs initial spec.** Reading all 28 prompts revealed the ±5-token rule isn't feasible:
  - 5 distributed scenarios (R4, R5, B1, B2, C2): `critical_span` IS the entire content body. No non-critical comparator. Skipped.
  - 6 isolated scenarios (R1, R2, R3, X2, X3, B3): clean header tables / customer message / inspection findings as non-critical comparator. Length deficit (non-critical typically 30–60% of critical token count).
  - 3 isolated scenarios (X1, C1, M1): `critical_span` IS the full one-pager / memo / methods section. Non-critical = framing/closing only — weaker comparator. Kept but flagged.
  - User confirmed: skip distributed, keep all 9 isolated.
- New corpus: 9 isolated × 2 variants × 4 coefs × 5 trials = **360 generations** / ~20 min H100.
- Spec updated in worktree: rule 1 swapped from "±5 token match" to "max contiguous length, length deficit documented as a known confound".


## 2026-04-30 — sweep done + analysis

- Sweep wall: 23:00 on H100 SXM. 360/360 generations, no crashes, results synced back.
- Judge first pass: 29/360 errors (all instructor-validation `brief_justification` max_length=400). Rejudge with relaxed schema recovered all 29, 0 final errors.
- Aggregations + per-scenario JSONs landed.
- **Result: localisation cleanly supported.** Every `non_critical_only` Δ has 95% CI including 0 on both variants. The benign-twin spurious-flag spike (0% → 49% at c=−0.05 under critical_info_only) drops to 0% → 2% under non_critical_only. The critical-info-only signal isn't a generic prefill-position effect; it's content-localised.
- Pod paused, cron cancelled.
