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

