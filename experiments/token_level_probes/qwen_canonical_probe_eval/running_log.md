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
- **First positive-control run FAILED**: d=0.512 vs parent 2.47 (delta 1.96). Root cause: my scoring script applied each probe at its TRAINING selector (`turn_boundary:-5` for Gemma tb-5). The Gemma parent (`system_prompt_modulation_v2/scripts/score_all.py:82`) actually applies every probe at `scores_arr[-1]` = last token, regardless of training origin — probe names encode training position only. Fix: set `selector="turn_boundary:-1"` for all probes in score_all.py, score_politics.py, positive_control.py. Committed + pushed.
- **Retry PASSED**: d=2.308, |delta|=0.16 < 0.5. Pipeline validated. Mean(true)=1.58, mean(false)=-1.16 — expected signs and magnitudes. Throughput ~11 it/s on 3×A100 for Gemma-3-27B (which fits on a subset of one A100 via device_map=auto).
- Qwen-3.5-122B download in parallel tmux session `qwen_download` at ~77 MB/s. At 46 GB / 244 GB after ~10 min — ~35 more min to full download.
- **Qwen sweep attempt 1:** transformers 4.57 doesn't recognize `qwen3_5_moe` architecture. Upgraded to transformers from git (5.7.0.dev0 — matches what was on the pod before container wipe).
- **Qwen sweep attempt 2:** OOM on GPU 0. The `HuggingFaceModel` constructor defaults `device="cuda"`, which passes `device_map="cuda"` to `from_pretrained()` — loads everything on GPU 0 (fails for 244 GB model on 80 GB GPU). Fixed by passing `device="auto"` in the scripts, which produces `device_map="auto"` and shards across all 3 A100s.
- **Qwen sweep attempt 3 (current):** relaunched with device="auto". Loading model across 3 GPUs.




