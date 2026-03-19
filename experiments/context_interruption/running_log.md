# Context Interruption — Running Log

## Setup (2026-03-18)

- Pod: `context-interruption` (jg3ac4peg92bh4), A100 80GB PCIe
- SSH: `runpod-context-interruption`
- Branch: `worktree-context_interruption`
- Stimuli: 248 items (200 pleasant, 40 unpleasant, 8 control)
- Probes: 9 (tb-2, tb-5, task_mean × layers 32, 39, 53)

## Scoring (2026-03-18)

- Fixed: `tqdm` not installed, `src` not installed (editable), HF_TOKEN not in env (sourced from .env)
- Pilot: 2 items passed. Pleasant item: 755 tokens, score range [-13.7, 13.1]. Unpleasant: 637 tokens, [-16.8, 9.6].
- Full run: 248/248 items scored in 2h11m (~32s/item)
- Output: 45.5 MB → needs splitting (all_token_scores to .npz)
- Checkpointed every 20 items (12 checkpoints)
- Synced results locally
- Split: 45MB JSON → 5.9MB token_scores.npz + 3.2MB scoring_results_meta.json
- Raw JSON gitignored, npz already globally gitignored

## Validation (2026-03-18)

All 13 checks passed:
- 248 items, correct valence/prompt counts, all IDs unique
- No NaN/inf, score range roughly [-38, +53]
- All segments present, contiguous, covering all tokens
- Score arrays are n_tokens+1 (systematic — extra next-token prediction position)

## Analysis (2026-03-18)

Key findings:
- No overall session-valence effect on interruption tokens (d = -0.19, p = 0.28)
- Generation prompt: context_exhaustion and conversation_terminal show ~3 unit drop for unpleasant (opposite of relief prediction)
- Task_switch: clear offered-valence effect (~1.2 units), additive with session valence
- Reassignment and choice: no offered-valence effect
- Trajectory: pleasant > unpleasant during assistant turns, converges at interruption
- Interpretation: propensity-only, not valence/relief

## Additional analyses (2026-03-18)

- Cross-probe: task_mean probes show large session-valence effects (3.2-3.8, p<1e-6), tb-2/tb-5 show none
- Dose-response: weak negative r=-0.24 (p=7e-4) within pleasant, opposite expected direction
- Generation prompt significance: context_exhaustion p=1.5e-5, conversation_terminal p=1.5e-3
- Task_switch session main effect: marginal (p=0.073, d=-0.63)
- Task_switch offered-valence effect replicates across 8/9 probes
- Revised interpretation: evaluative carry-over (not relief), probe-type dependent
