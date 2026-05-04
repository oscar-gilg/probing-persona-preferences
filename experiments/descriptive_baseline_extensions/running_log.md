# Running Log — descriptive_baseline_extensions

Append-only working log. Times are local.

---

## 2026-05-04

### Setup
- Worktree branched from `main` at `38018f96`. Spec cherry-picked from `eot_discrimination_v2` (`64e13ca8`).
- Symlinks into main repo:
  - `activations/qwen3-emb_8b` → main repo (existing baseline embeddings + probe weights)
  - `activations/gemma-3-27b_it` → main repo (canonical 6k pool source)
  - `activations/qwen35_122b` → main repo (canonical Qwen pool source)
  - `results/probes/qwen3_emb_8b_*/probes` → main repo (Ridge weight pickles)
- Pod launched: `desc-baseline-emb`, A100-SXM4-80GB, disk=80GB, volume=50GB.
  - SSH alias: `runpod-desc-baseline-emb` (64.247.196.124:13793)
  - ID: `7h6h3gimbzihwn`
- Branch pushed to origin: `worktree-descriptive_baseline_extensions`
- Pod setup running (uv pip install -e .[dev], etc.). `.env` synced to `/tmp/.env` on pod.


### Pod up + initial sync
- Pod `desc-baseline-emb` setup completed in 291s (Python 3.12 venv, dev extras only — `serving` extra excluded due to vllm version conflict)
- Sentence-transformers + requests installed manually
- Decision: Qwen chat-template via `unsloth/Qwen3.5-122B-A10B` (mirrors the project's hf_name without auth issues)
- Existing Qwen3-Emb baseline NPZs (494 MB + 231 MB) synced to pod via SCP for task-id alignment

### Session 1 — chat-template embeddings + refit (in progress)
- **Qwen 2.5k pool extraction:** 14,038 tasks embedded under Qwen chat template, no sysprompt. ~2:30 wall, batch_size=128. Output: `activations/qwen3-emb_8b_chat/qwen35_pool/activations_prompt_last.npz`
- **Qwen probe refit:** `final_r=0.8880` (raw-text baseline 0.8938; Δ=-0.006). Within ±0.05 tolerance — chat-template wrapper does not dominate the pooled embedding. `final_acc=null` because eval run dir doesn't have pairwise comparison files in git, but Pearson r is the load-bearing sanity check
- **Gemma 6k pool extraction:** in flight (235 batches, ~12 min total)
- Encoder runs faster on subsequent batches as A100 warms up: first batch ~14s, asymptotic ~1.6 it/s
