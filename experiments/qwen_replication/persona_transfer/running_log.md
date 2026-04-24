# Qwen persona transfer — extraction + utilities running log

## 2026-04-23 17:30 UTC — Setup

- Spec finalized at `extraction_spec.md`.
- 7 extraction configs at `configs/extraction/qwen35_pref_{default,final_six}_sweep.yaml`.
- 21 AL measurement configs at `configs/measurement/qwen_persona_sweep/final_six/*.yaml`.
- AL hyperparameters match Gemma byte-for-byte except `initial_degree: 3` (up from 2).
- Extraction hyperparameters: `selectors=[tb-1, tb-4]`, `layers=[33, 38, 43]`, `batch_size=4`, `max_new_tokens=1`.
- Task set reuses Gemma canonical splits (`data/canonical_splits/{train,eval,test}_task_ids.txt`, 4000/1000/1000).
- Committed to main (commit before worktree).
- Pod `qwen-persona-transfer` launching with 3×A100-SXM4-80GB.

## 2026-04-23 17:35 UTC — Both jobs launched

- **Pod up:** `f42wbkdh4j2dny`, 3×A100-80GB, SSH alias `runpod-qwen-persona-transfer`.
- Pod deps: `uv pip install --system -e ".[dev]"` (initial pod_setup.sh left scipy/etc missing).
- Main commit `4e3a245` pushed to origin/main and pulled on pod.
- **Extraction launched:** `nohup bash scripts/persona_sweep_extraction/run_qwen_all.sh`, log at `/workspace/logs/run_qwen_all.log` on pod. PID 1770.
- **AL measurement launched:** tmux session `qwen_persona_sweep`, log at `logs/qwen_persona_sweep_final_six.log`. Confirmed reading correct canonical splits (4000/1000/1000).
- Audit agent spawned in background to verify setup independently.

## 2026-04-23 17:40 UTC — Two issues found, recovering

- **Extraction crashed (1st try):** transformers 4.57.6 doesn't know `qwen3_5_moe` arch. Upgraded to `transformers==5.7.0.dev0` from source.
- **Extraction crashed (2nd try):** root disk (200 GB) full — HF cache wrote to `/root/.cache/huggingface`, not `/workspace`. 187 GB downloaded before hitting ENOSPC. Cleared cache, symlinked `/root/.cache/huggingface → /workspace/hf_cache` (71 TB available). Restarting.
- **User pre-existing AL runner** (PID 73964) discovered stuck at 100% CPU for 23+ min inside `SSL_CTX_load_verify_locations` → `PEM_X509_INFO_read_bio_ex` (21 concurrent OpenAI clients contending on OpenSSL cert-store parse). Zero API calls, zero file writes. My duplicate tmux killed. User's stuck process left running pending their decision.
- **Audit agent:** all config/spec compliance checks PASS. Only flag was the pod extraction (since addressed).

## 2026-04-23 17:45 UTC — Third extraction attempt

- 3rd restart (17:09 UTC): hf_xet (content-addressable download backend) hit `Input/output error (os error 5)` on MooseFS network mount after 58 GB. Known xet-on-netfs issue.
- Cleared `/workspace/hf_cache/*`, exported `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=0`, relaunched. Uses legacy single-stream hf_hub_download — slower but reliable on the mount.

## 2026-04-23 18:15 UTC — AL process self-resolved, extraction downloading

- **AL runner (PID 73964)**: recovered on its own. The 30-min SSL cert-store parse eventually completed; process transitioned `R+` 100% CPU → `S+` 12% CPU with 10+ ESTABLISHED TCP connections to OpenRouter (104.18.2.115 / 104.18.3.115). Now making API calls. Do not kill.
- **Extraction**: download progressing at ~2.5 GB/min. 90 GB of 244 GB (bf16 weights) at last check. ETA ~60 min more before model load + first-persona extraction starts.

## 2026-04-23 18:45 UTC — Pod v1 hit /workspace MooseFS per-user quota at 92 GB

- **Root cause:** `/workspace` shows 71 TB free but has a per-mount/per-user quota. `OSError: [Errno 122] Disk quota exceeded`. Known issue per hive-mind retrieval.
- **Fix (from past sessions):** launch pod with `--disk-gb 500` to use local NVMe overlay (500 GB container disk), not the MooseFS network volume. Also confirmed `HF_HUB_DISABLE_XET=1` is necessary — xet backend chokes on the network mount.
- **Paused old pod** `f42wbkdh4j2dny` (GPU billing stopped, disk preserved).
- **New pod** `qwen-persona-transfer-v2` (`qyi5l5xuxzp6zi`): 3×A100-SXM4-80GB, 500 GB container disk, 489 GB free on overlay. SSH alias added.
- Synced .env, installed deps (`scipy`, `transformers==5.7.0.dev0` from git for `qwen3_5_moe`), installed tmux.
- Launched extraction in tmux session `extraction` with `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0`. Python PID 2451 at 99% CPU downloading weights.

**Known zombuul gap flagged by user:** default `disk_gb: 200` is too small for 122B-class models. Candidate zombuul improvements: (a) raise default disk to 500; (b) add a warning when `--gpu` is "a100-80gb 3x" or similar without `--disk-gb`. To follow up separately.

## 2026-04-23 18:57 UTC — First persona complete

- **`default` persona DONE**: 6000 tasks, 0 failures, 0 OOMs. 1500 batches at ~2.2 s/batch = ~55 min.
- **`sadist`** started at 18:57:05 UTC.
- ETA: ~55 min × 6 remaining personas = ~5.5 h to finish extraction. First write went to `/workspace/activations/qwen35_122b/pref_default_sweep/{activations_turn_boundary:-1.npz, activations_turn_boundary:-4.npz}`.

## 2026-04-23 19:30 UTC — sadist checkpoint hit MooseFS I/O error

- `sadist` died at batch 650/1500 with `OSError: [Errno 5]` on checkpoint save to `/workspace/activations/qwen35_122b/pref_sadist_sweep/`. Same MooseFS flakiness class — but on our own activation output this time, not HF cache.
- **Fix (pod-level):** `mv /workspace/activations/qwen35_122b/* /root/activations/qwen35_122b/`, then `ln -sfn /root/activations/qwen35_122b /workspace/activations/qwen35_122b`. All future writes now land on the 500 GB container overlay (local NVMe, no MooseFS).
- **Backup to laptop** before restart: `rsync` pulled the 450 MB completed `pref_default_sweep` and 294 MB partial `pref_sadist_sweep` to `activations/qwen35_122b_partial_backup/`.
- **Zombuul issue filed**: [#63](https://github.com/ogilg/zombuul/issues/63) proposing a "write intermediate outputs to container disk, not /workspace" convention in `run-experiment`. PR #61 fixed the HF-cache side of this; this issue is for the experiment-output side.

## 2026-04-23 19:51 UTC — Clean restart from scratch

- Wiped `/root/activations/qwen35_122b/*` on pod (backup preserved locally).
- Relaunched `run_qwen_all.sh` in tmux `extraction`. All 7 personas from scratch. Model already in HF cache (246 GB, no download time); extraction should start immediately.
- First event: `=== 2026-04-23 19:51:16 UTC :: default ===`.

## 2026-04-23 21:00 UTC — AL runner was dead, relaunched

- User's original AL runner (PID 73964) that started at 17:17 UTC died silently at some point — zero results dirs, zero checkpoints written. No evidence of what killed it (no log, process gone).
- Relaunched in tmux session `qwen_persona_al` with `SSL_CERT_FILE=$(python -m certifi)` and matching `REQUESTS_CA_BUNDLE` preset to sidestep the 30-min OpenSSL cert-parse hang we observed on first launch.
- Extraction monitor re-armed (`brutojnnz`, 1 h) with quiet filter; AL first-checkpoint-watcher also armed (`b1l1df0jf`, 20 min).
