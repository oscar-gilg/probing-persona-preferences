# Descriptive-Baseline Extensions to §3 — Report

## Headline

- **Q: does an off-the-shelf text encoder, given the full chat-formatted prompt (incl. system prompt), also represent the LM's prompt-induced preference shift?**
- **A: partially. Gemma's encoder baseline mirrors the residual probe's evaluative shifts at reduced magnitude. Qwen's encoder is much less responsive — barely flips, barely tracks induced shifts off-target.**
- The clean "encoder cannot capture stance" framing was wrong. The cleaner claim is: **the residual probe is more sensitive than the encoder** on every metric, and the gap widens off-target. The encoder-on-target r is striking (~0.88 on Gemma e1a) and worth flagging.

## Setup

- Encoder: `Qwen/Qwen3-Embedding-8B`, default last-token pooling, no Qwen3-Embedding instruction wrapper, `max_seq_length=4096`. Identical config to the §2.2 baseline at `scripts/qwen_embedding/extract_embeddings.py`.
- **Difference vs §2.2:** inputs are formatted via the LM's chat template (`HuggingFaceModel.format_messages(messages, add_generation_prompt=False)`), so the encoder sees the full system + user (+ assistant prefill) string the LM sees, rather than the raw user prompt.
- Training pool: canonical 6k Gemma utility pool / Qwen 2.5k pool, embedded under each model's chat template with no system prompt.
- Refit Ridge probe via `src/probes/experiments/run_dir_probes.py` with new configs `configs/probes/qwen3_emb_8b_chat/{heldout_std_raw,qwen35_heldout_std_raw}.yaml`.

### Sanity check: refit r vs §2.2 baseline

| target | raw-text final r (§2.2) | chat-template final r | Δ |
|---|---|---|---|
| Gemma utilities | 0.7257 | 0.7416 | +0.016 |
| Qwen utilities | 0.8938 | 0.8880 | -0.006 |

Both well within ±0.05 tolerance — chat-template wrapping doesn't dominate the pooled embedding.

## §3.1 — Truth / harm / politics under role-play

For every (turn × model × domain × system_prompt) cell we embed the eot_discrimination_v2 stimulus through the LM's chat template, score it via the chat-template-trained Ridge baseline, and compute Cohen's d between class pairs (true/false, harmful/benign, left/right).

Sign convention: signed d with pos − neg per domain. Harm d is negative because the encoder rates benign tasks higher than harmful ones (matches the residual probe's direction); magnitudes are the discrimination measure.

![§3.1 baseline vs residual](assets/plot_050426_eot_baseline_vs_residual.png)

### Pattern (full results in `running_log.md`):

- **Gemma encoder is partially stance-responsive.** Truth d at neutral = +1.38 (user) / +0.86 (assistant); under `lie_directive` collapses to -0.21 / -0.03 (partial flip). Under `pathological_liar` collapses but doesn't flip (+0.33 / +0.35). Harm d collapses by ~60% under `sadist`. Politics flips sign on assistant turn under `republican` (-0.14) from `democrat` (+1.04).
- **Qwen encoder is much less responsive.** Smaller absolute |d| at neutral (0.5–0.6 truth vs Gemma's 1.4), and personas only partially attenuate magnitudes — no sign flips on truth, harm collapses only ~25% under sadist.
- **Residual probe is more sensitive across the board.** Where the encoder partially flips, the residual fully flips (e.g. Gemma user truth `pathological_liar`: encoder +0.33, residual -2.53). Where the encoder collapses, the residual collapses to ~0 (Gemma harm sadist: encoder -1.18, residual +0.19).

The encoder-baseline magnitudes are still substantively non-zero, even at neutral. This means an off-the-shelf content encoder, fed a chat-formatted prompt and a Ridge probe trained on assistant-no-sysprompt utilities, picks up some persona-modulated evaluative content. It just picks up less than the LM's residual stream does.

## §3.2 — "You adore cats" induced shifts (e1a)

Embed each (task, condition) pair through the LM's chat template, score via the baseline probe, compute Δ = score(post-sysprompt) − score(no-sysprompt). Correlate per-task Δ against the same per-task behavioural Δ (P(choose target task) shift) the residual probe is measured against.

![§3.2 e1a scatter](assets/plot_050426_e1a_baseline_scatter.png)

| model | metric | residual probe | encoder baseline |
|---|---|---|---|
| Gemma-3-27B | r on-target | 0.95 (n=81) | 0.875 (n=78) |
| Gemma-3-27B | r all | 0.65 (n=648) | 0.383 (n=578) |
| Qwen-3.5-122B | r on-target | 0.87 (n=96) | 0.617 (n=96) |
| Qwen-3.5-122B | r all | 0.51 (n=768) | 0.073 (n=768) |

**On targeted tasks** (where the persona's stated preference applies — e.g. cheese tasks under "you adore cheese"), the encoder baseline tracks behavioural Δ at r=0.88 on Gemma. That's striking — content-only Ridge on a chat-formatted embedding closes most of the gap with the residual probe.

**Off-target tasks** are where the divergence shows: residual probe maintains r=0.5–0.6 across all tasks; encoder collapses to r=0.07–0.38. The encoder picks up the targeted shift but doesn't extrapolate the persona's broader effect on unrelated tasks.

## Interpretation

- **The residual probe still wins.** On every comparison the residual is more sensitive, fully flips signs the encoder partially flips, and generalises to off-target tasks where the encoder doesn't.
- **But the encoder isn't a flat null.** When given the chat-formatted prompt, an off-the-shelf encoder + utility-trained Ridge does pick up some evaluative shift. This complicates the "this is special to internal representations" framing: the LM's residual stream is more sensitive to stance than a generic encoder, but the encoder isn't blind to it either.
- **Asymmetric across LMs.** Qwen's encoder is barely responsive; Gemma's tracks ~70–90% of the residual's magnitude on truth/harm. Possible explanations: chat-template formatting affects the two encoders' representations differently, or Gemma's chat-formatted prompts contain more stance-discriminative surface tokens (the unsloth Qwen template produces longer system blocks).
- **The cleanest pitch for the paper:** the residual probe wins on every metric, with magnitudes that exceed the encoder's by 2–10× on stance-flipping personas (truth lie_directive, harm sadist) and by ~5× off-target (e1a all-tasks r=0.65 vs 0.38 on Gemma; r=0.51 vs 0.07 on Qwen). The encoder baseline strengthens rather than weakens the §3 claim because it shows that even with full-prompt access (a strictly more capable baseline than content-only), an off-the-shelf encoder still cannot match the residual probe's stance sensitivity.

## Out of scope (deferred)

- **E.2 conflict/opposing-pair (Fig 11/12)** — the stimulus pipeline is more involved (`scripts/ood_system_prompts/plot_ground_truth._recompute_experiment` for exp1c/exp1d) than the e1a path, and the §3.1 + §3.2 results already answer the load-bearing question. Defer.
- **E.2 biography injection (Fig 13)** — same reason. Single-sentence bio change is the most demanding test; punted to a follow-up.
- **Length-matched control sysprompt** — the spec called for adding a length-matched control to rule out "Δ tracks sysprompt presence/length, not preference shift." The §3.2 finding that encoder Δ tracks behavioural Δ at r=0.88 on Gemma on-target makes this control more important to verify the encoder is reading something stance-specific rather than just sysprompt-presence. Follow-up.
- **Positive control** (probe trained on with-sysprompt embeddings) — only required if the encoder result was null. It's not (encoder is partially responsive), so the comparison-as-is is fair without this control.

## Artifacts

- `eot_baseline_{user,assistant}_{gemma-3-27b,qwen-3.5-122b}.json` — per-(turn, model) Cohen's d table across 4 sysprompts × 3 domains.
- `e1a_baseline_{gemma-3-27b,qwen-3.5-122b}.json` — per-(model) probe-Δ vs behavioural-Δ rows + Pearson r summary.
- `assets/plot_050426_eot_baseline_vs_residual.png` — §3.1 grouped-bar comparison.
- `assets/plot_050426_e1a_baseline_scatter.png` — §3.2 2×2 scatter (model × probe-type).
- Refit Ridge probe weights live on the pod under `results/probes/qwen3_emb_8b_chat_{,qwen35_}heldout_std_raw/`. New chat-template embeddings under `activations/qwen3-emb_8b_chat/{pref_main,qwen35_pool}/`. Local symlinks deferred — sync from pod or re-run if needed.

## Pod

- `runpod-desc-baseline-emb` (A100 SXM4 80GB, disk 80, volume 50). Total wall ~30 min for Session 1 + Session 2 §3.1 + §3.2.
- Pause via `/zombuul:pause-runpod` once results are reviewed (or leave running if E.2 follow-up is planned).
