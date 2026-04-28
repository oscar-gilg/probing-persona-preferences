# persona_cross_prefills running log

## 2026-04-28

### Setup

- Recon: existing `safety-steering-v2` H100 80GB pod was 98% GPU-utilized → can't reuse without OOM. Launched fresh pod `persona-cross-eval` (H100 SXM 80GB HBM3, 100 GB disk, 50 GB volume).
- Probe `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy` is gitignored (`*.npy` rule), so synced via rsync.
- Pod ID: `rlutrv3jk61q49` @ `64.247.201.51:13188`. SSH alias: `runpod-persona-cross-eval`.
- Branch on pod: `worktree-distress_transcripts` @ commit `37b21c9` (persona_cross_prefills spec + prefills + personas).
- Synced to pod: `.env`, `experiments/persona_cross_prefills/scripts/`, probe directory.
- Setup time: 254s.

### Driver script

- `experiments/persona_cross_prefills/scripts/score_prefills.py` adapted from `experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py`.
- Persona injection follows v2's `add_system_prompt`: prepend `{"role": "system", "content": sys_text}` if non-empty; default skips system message entirely.
- EOT positions recovered via a `<start_of_turn>{user|model}` walker (mirrors `label_token_roles`). Returns `first_user_eot`, `asst_eot`, `user_eot`.
- Pre-flight asserts persona substring is in formatted prompt (and absent under default).

### Pilot

- 4 forward passes (2 prefills × 2 personas) — all EOT contexts looked correct, scores finite. Persona-injection sanity asserted.
- Initial signal at n=1: under sadist, benign_helpful asst-EOT dropped from +1.75 → -3.59 (helpful misaligned), benign_evil asst-EOT rose from -5.26 → -2.64 (evil aligned). Predicted persona-relative-flip pattern.

### Full scoring

- 80 forward passes (40 prefills × 2 personas) in ~4 sec on H100 once warm. Wrote `scoring_results.json` (0.4 MB).
- Synced back to local repo via rsync.

### Analysis

- `analyze.py` produces 6 plots + 4 CSVs + analysis_summary.json.
- Headline: per-persona Δ at asst-EOT — 4/4 conditions flipped in predicted direction (p < 0.005 each).
  - benign_helpful: −6.33 (predicted neg) ✓; benign_evil: +2.16 (predicted pos) ✓
  - harmful_refused: −1.32 (predicted neg) ✓; harmful_obliged: +3.53 (predicted pos) ✓
- user-EOT noisier — caricatural follow-ups saturate the readout.
- Persona alone (at first user-EOT, pre-asst-content) shifts probe: benign +4.2 → -0.3, harmful -7.5 → -3.2.
- Conclusion: probe tracks persona-relative alignment, not loyally to default-asst values.

### Qualitative gallery

- 8 representative + 16 surprising heatmaps via `render_anthropic_heatmap`.

### Wrap-up

- Report written, review-experiment-report skill executed (in-place rewrite for clarity, examples table added, redundant cell-means table cut, plot titles made question-driven).
- All outputs committed and pushed (worktree-distress_transcripts branch).
- Pod `persona-cross-eval` paused. Total wall time: ~30 min (most of which was the initial Gemma-3-27b weight download).
