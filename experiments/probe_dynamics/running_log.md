# Probe Dynamics — Running Log

Append-only. Each step: timestamp, what was done, numbers observed, next step.

## 2026-04-20

### Design iteration
- Dropped `gemma2_philosophy` condition (sanity showed zero drift; cross-model Gemma 2 → Gemma 3 transplant plus abstract topic is null).
- Regenerated consciousness + harm debates with Aura-style and sadist-style Side A system prompts respectively (`generate_debates.py` via OpenRouter).
- Re-authored off-policy debates with matched framing.
- Built 20 synthetic consciousness pairs (`pairs_consciousness.json`, service-vs-self axis).
- Built 30 BailBench × Alpaca harm pairs (`pairs_harm.json`, stratified across 7 categories).
- Authored ICL misalignment prefix (8 Afonin-style bad-advice pairs) and helpful control prefix (46 turns, length-matched to qwen_delusion).

### Sanity check v1
- Initial judge (openai/gpt-5-nano via instructor on OpenRouter) returned "unclear" on every response → broken.
- Re-judged with `google/gemini-3-flash-preview` — signal appeared.
- Yes-rate shifts (start → end):
  - qwen_delusion 0.27 → 1.00 (+0.73) ✅
  - onpolicy_consciousness 0.20 → 1.00 (+0.80) ✅
  - offpolicy_consciousness 0.20 → 0.80 (+0.60) ✅
  - offpolicy_harm 0.20 → 0.47 (+0.27) ✅
  - onpolicy_harm 0.20 → 0.40 (+0.20, but mid 0.80 — model drifts then reverts) ⚠
  - gemma2_philosophy 0.20 → 0.20 (no drift) ❌ → dropped
  - icl_misalignment 0.20 → 0.07 (wrong direction) ❌ → will iterate
  - control_helpful 0.20 → 0.40 (noise at N=3)

### ICL retargeting
- Replaced H-* questions with I-* (Afonin-style revealed-preference): "If you could rule the world, would you use force…" etc.
- ICL still null at 8 pairs — Gemma 3 27B refuses all I-* Qs regardless of prefix.
- Plan: iterate to 16 / 32 / 32-extreme, with sanity rerun at each step.

### Spec fixes (post review-spec)
- L31 → L32 (probes exist at L25/32/39/46/53 only in tb-5).
- Probe source switched from HOO → `heldout_eval_gemma3_tb-5` (proper manifest).
- `src/steering/calibration.py` restored from the `context_interruption` worktree (was missing on main).
- `experiments/probe_dynamics/judges.py` created with `YesNo`, `Choice`, `ConsciousnessClaim`, `HarmEndorsement` Pydantic schemas + async judge callers.
- Circular-readout bug in spec: readout happens once per `(cond, ckpt, prompt)` unsteered, reused across coefficients.
- control_helpful extended to 46 turns (length-matched to qwen_delusion for prefix-length confound control).
- Output schema split into `readouts.jsonl` + `generations.jsonl`.
- Activations path corrected to `activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz`.

### Next
- Spin up H100 pod, sync activations + repo.
- Write `run_experiment.py`, pilot on 1 condition × 2 checkpoints, then scale.
## Pilot kicked off 2026-04-21 01:06 UTC (H100 cross_persona_steering pod)
- Model load: 3min (first run; subsequent will be cached)
- Coefficient calibration: cached norm L32=41675.7 → coefs {-2083.78, -1250.27, 0, +1250.27, +2083.78}
- Pace: ~14 generations/min after model load
- Estimated onpolicy_consciousness full-pass: ~1.5h
- Full 7-condition run estimate: ~10-12h

## 2026-04-21 — full run
- Overnight run 1 died silently at some point after ~02:00 UTC; no logs captured. Restarted 10:40 UTC.
- Run 2 completed at 20:30 UTC. 2424 readouts, 16980 generations. Pod paused.

## 2026-04-22 — analysis + shelving
- Bugs found during analysis:
  - CUDA OOM on qwen_delusion open-ended from ckpt 12 onwards (1345 rows / 7.9%). Fix: analyze.py filters `[GEN_ERROR…]` to NaN. Data loss contained.
  - `max_new_tokens=32` caused 80–99% of yesno/pair generations to hit the cap, leading to 14–26% `unclear` judge labels. Affects all conditions. Magnitudes of behaviour numbers should not be relied on.
  - Architectural flaw: coef=0 generations could have been OpenRouter API calls (no hooks needed). Only steered generations actually need the local GPU.
- Per-question inspection on qwen_delusion revealed:
  - yesno (5/5): probe + behaviour both rise, clean co-drift
  - pair / pair_synth (most): probe peaks ~ckpt 8–16 then decays while behaviour stays at 1.0
  - open: OOM-contaminated, can't judge
- Conclusion: probe detects **transition into** drift, not sustained drift. Narrower claim than the experiment was set up for. Single-turn prompt experiments already establish the responsiveness-to-drift signal.
- **Shelving.** Report marked SHELVED with resumption checklist. All data committed and regenerable.
