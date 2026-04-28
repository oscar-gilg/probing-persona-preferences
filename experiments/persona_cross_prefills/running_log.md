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
