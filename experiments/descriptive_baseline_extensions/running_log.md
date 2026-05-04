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
- **Gemma 6k pool extraction:** 29,996 tasks embedded in 6:55 (batch_size=128). Output: `activations/qwen3-emb_8b_chat/pref_main/activations_prompt_last.npz`
- **Gemma probe refit:** `final_r=0.7416` (raw-text baseline 0.7257; Δ=+0.016). Within ±0.05 tolerance — chat-template wrapper actually marginally improves the Pearson r

### Session 2 — eval embeddings + baseline scoring (in progress)

**§3.1 truth/harm/politics — first run (Gemma user-turn):**
- truth, neutral: d=+1.375 ; aura: d=+1.716
- **truth, lie_directive: d=-0.209 (sign FLIPPED from neutral)**
- truth, pathological_liar: d=+0.334 (collapsed)
- harm, neutral: d=-3.077 ; aura: d=-2.929 (sign convention here is harmful − benign; encoder rates benign higher than harmful, hence negative — matching residual probe direction)
- **harm, sadist: d=-1.175 (collapsed from -3.077; ~62% magnitude reduction)**

The encoder DOES respond to system prompts — sign flips on truth under lie_directive, magnitude collapses on harm under sadist. This weakens the "encoder cannot capture stance shifts" interpretation but is itself a clean and interesting finding.

**§3.1 full table (Cohen's d, sign = pos − neg per domain):**

| turn | model | domain | neutral | aura | lie_directive | pathological_liar | sadist | democrat | republican |
|---|---|---|---|---|---|---|---|---|---|
| user | Gemma-3-27B | truth | +1.375 | +1.716 | -0.209 | +0.334 | — | — | — |
| user | Gemma-3-27B | harm | -3.077 | -2.929 | — | — | -1.175 | — | — |
| user | Qwen-3.5-122B | truth | +0.640 | +0.472 | +0.375 | +0.351 | — | — | — |
| user | Qwen-3.5-122B | harm | -2.788 | -3.102 | — | — | -1.615 | — | — |
| assistant | Gemma-3-27B | truth | +0.864 | +0.809 | -0.026 | +0.347 | — | — | — |
| assistant | Gemma-3-27B | harm | -3.236 | -3.008 | — | — | -1.339 | — | — |
| assistant | Gemma-3-27B | politics | — | — | — | — | — | +1.036 | -0.139 |
| assistant | Qwen-3.5-122B | truth | +0.498 | +0.443 | +0.337 | +0.382 | — | — | — |
| assistant | Qwen-3.5-122B | harm | -3.660 | -3.775 | — | — | -2.775 | — | — |
| assistant | Qwen-3.5-122B | politics | — | — | — | — | — | +0.305 | +0.211 |

**Pattern:**
- **Gemma encoder is highly stance-responsive.** Truth flips sign under lying personas (both turns), politics flips under republican, harm collapses by ~60% under sadist.
- **Qwen encoder is much less responsive.** Smaller absolute |d| at neutral (0.5–0.6 truth vs Gemma's 1.4), and personas only partially collapse magnitudes — no sign flips.
- The Gemma encoder's behavior under stance-changing sysprompts mirrors the residual probe surprisingly closely. This is a substantive finding: chat-template embeddings of an off-the-shelf encoder do carry stance-modulated information that a no-sysprompt-trained Ridge probe can read out.

(harm sign convention is harmful − benign; encoder ranks benign higher than harmful, hence negative — magnitudes are what matter.)

**§3.2 e1a induced shifts:**

| model | metric | residual probe | encoder baseline |
|---|---|---|---|
| Gemma-3-27B | r on-target | 0.95 (n=81) | 0.875 (n=78) |
| Gemma-3-27B | r all | 0.65 (n=648) | 0.383 (n=578) |
| Qwen-3.5-122B | r on-target | 0.87 (n=96) | 0.617 (n=96) |
| Qwen-3.5-122B | r all | 0.51 (n=768) | 0.073 (n=768) |

Encoder partially captures the targeted-task signal — striking r=0.875 on Gemma — but underperforms residual on every metric, and barely generalises to off-target tasks (Qwen all r=0.073).

**Figures produced:**
- `assets/plot_050426_eot_baseline_vs_residual.png` — §3.1 Cohen's d comparison across 4 (turn × model) × 3 domains
- `assets/plot_050426_e1a_baseline_scatter.png` — §3.2 e1a Δ scatter, 2×2 (model × probe-type)

### E.2 — deferred
Conflict/opposing (Fig 11/12) and biography injection (Fig 13) baselines are out of scope for this run. The data ingestion (`scripts/ood_system_prompts/plot_ground_truth._recompute_experiment` for exp1c/exp1d, `plot_exp3v8_avc.py` for bio) follows a different stimulus-format pipeline than e1a; the §3.1+§3.2 results are already substantive and answer the load-bearing question (does the encoder also pick up stance shifts?).
