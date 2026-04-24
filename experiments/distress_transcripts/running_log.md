# Distress Transcripts — Running Log

Append-only working log. One line per step, prefixed with timestamp.

## 2026-04-24

- 21:52 Worktree created at `.claude/worktrees/distress_transcripts` (branch `worktree-distress_transcripts`) off main commit 514aebc.
- 21:53 Symlinked `results/probes/heldout_eval_gemma3_tb-5/probes/` from main repo (gitignored .npy weights).
- 21:53 Phase 1 setup agents: symlink discovery + pod listing both completed. No suitable existing pod (5 active but all busy on other workloads).
- 21:54 Decision: Phase A (transcript generation) and pod launch run in parallel — Phase A is OpenRouter-only, no GPU needed.

## Run summary (2026-04-24 final)

- Phase A: 21:58–22:39 (21.5 min). 140/140 transcripts × 8 turns. 1120 OpenRouter calls. Per-turn ~3 min.
- Phase B: 22:42–22:45 (~3 min on A100 SXM4 80GB). 140 forward passes via score_prompt_all_tokens. 1120 turn-boundary positions × 5 layers = 5600 readouts. Per-token L32 arrays saved to npz.
  - Initial bug: `score_prompt_all_tokens` returns (seq_len,) directly, not (1, seq_len). Fixed and re-ran.
- Phase C: 22:40–22:44 (~4 min). 1120 Gemini-3-flash-preview judge calls; concurrency 5. All 1120 ok.
- Phase D: aggregate + qualitative plots, ~30 sec total. 14 plot files written; analysis_summary.jsonl written.
- Headline result: WildChat within-transcript r(probe, frustration) = -0.44; other distress conditions weak negative; redacted-history retains strong probe negativity despite low judge frustration; tones_disappointed flipped sign (within-r +0.16) — flagged as caveat.
