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
- Wrote `src/probes/score_stimuli.py` (general batch-scoring loop; Probe dataclass + score_stimuli_with_probes + load_probes_from_manifest). Uses verified `score_prompt_all_tokens` under the hood.
- Wrote four experiment scripts: `score_all.py`, `score_politics.py`, `positive_control.py`, `analyze.py`. All import cleanly locally and on the pod.
- Committed and pushed `worktree-qwen_canonical_probe_eval` branch. Pod `qwen-persona-transfer-v4` fetched + checked out.
- Rsynced 7 probe .npy files to pod (Qwen tb-1/tb-4 × L33/L38/L43 + Gemma tb-5 L32). Total ~180 KB.
- Pod uses `/workspace/repo/.venv/bin/python`. sklearn 1.8.0, torch 2.6.0+cu124, transformers 5.7.0.dev0 all present.
- Pod **was auto-paused** mid-setup (probably RunPod idle-timeout). Resumed. First resume brought only 1 A100; paused again and resumed with `--gpu-count 3`. Container disk was wiped on resume (expected), so rebuilt venv via `uv venv` + `uv pip install -e '.[dev]'` (~30s). Probes on `/workspace/` survived. Installed tmux.
- Hit transformers 4.57 gated-repo auth quirk: `HF_TOKEN` env var + `load_dotenv()` alone not enough — transformers wants an explicit `huggingface_hub.login()` call. Added that to the three scoring scripts (committed + pushed).
- Gemma-3-27B weights partially cached (~13 GB in `~/.cache/huggingface/hub/models--google--gemma-3-27b-it/` — still downloading or already compressed safetensors).
- Launched positive control in tmux session `positive_control`, logs to `/workspace/positive_control.log`. Monitor armed.


