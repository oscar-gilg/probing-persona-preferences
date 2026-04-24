# Running log — qwen_canonical_probe_eval

Append-only. Also the progress signal for the user when running on a pod.

## 2026-04-24

- Entered worktree `qwen_canonical_probe_eval` from main.
- Symlinked 7 probe checkpoints into `results/probes/**/probes/` (6 Qwen + 1 Gemma for positive control).
- Created experiment scaffolding: `experiments/token_level_probes/qwen_canonical_probe_eval/{scripts,assets}/` and `scripts/qwen_canonical_probe_eval/`.
- Identified idle pod `qwen-persona-transfer-v4` (3×A100-80GB, clean main branch, 0% GPU util). Plan to reuse rather than launch new.
- Blocker surfaced: `truth_system_prompts_v2.json` and `harm_system_prompts_v2.json` don't exist on disk — only `politics_system_prompts_v2.json` was committed by the Gemma run. `generate_data.py` regenerates them deterministically from the base filtered stimuli; ran that next.
- Regenerated v2 stimulus JSONs. Counts: truth 2376 (264 base × 9 sysprompts), harm 1155 (231 base × 5 sysprompts), politics 1482 (unchanged). **Total scoring payload: 5,013 items.**
- Spec's "~3.5k" estimate was truth+harm only; politics adds ~1.5k. Unbatched throughput estimate on Qwen-122B 3×A100 is ~0.5–2s/item, so 40 min – 2.5 h total for the main sweep.
