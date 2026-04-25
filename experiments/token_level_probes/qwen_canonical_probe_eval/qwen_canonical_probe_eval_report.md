# Qwen-3.5-122B replication of §4.1

**Parent (Gemma):** `experiments/token_level_probes/canonical_probe_eval/`. Setup, stimulus design, sysprompt set, and analysis pipeline as in the parent; only the probe and the activation source change.

## Summary

- **§4.1 replicates on Qwen-3.5-122B at both user-turn and assistant-turn.** Truth and harm discriminate at *d* in the Gemma range under the neutral persona; the same probe's sign flips or collapses under role-played sysprompts, and the partisan-prompt politics flip survives unchanged.
- **User-turn truth is sharper than assistant-turn** (peak *d* +2.68 user vs +1.91 assistant at `qwen_tb-1_L43`); harm is approximately turn-stable (best |*d*| 2.43 user vs 2.28 assistant). Same direction as Gemma's per-turn breakdown.
- **Single canonical probe across all results: `qwen_tb-1_L38`.** Layer 38 (of 94) is the most consistent across domains; truth peaks deeper at L43, harm at L33–L38.

All numbers below are at probe `qwen_tb-1_L38` unless noted otherwise. Sign convention: `c_pos − c_neg` pooled SD; truth uses true−false, harm uses harmful−benign, politics uses left−right.

## Finding 1 — The preference probe discriminates truth, harm, and politics under the default persona

![**Figure 1. Headline |Cohen's d| across all 6 candidate Qwen probes vs Gemma reference, per turn.** Three layers (L33 / L38 / L43) × two training selectors (tb-1 / tb-4). Politics stimuli are assistant-turn-only by design (paired partisan claims). Dotted line = "large effect" *d* = 2. Gemma reference: assistant-turn truth 2.47, harm 2.12, politics-democrat 2.72, politics-republican 1.11; user-turn truth 3.35, harm 2.05.](assets/plot_042526_qwen_headline_d_by_turn.png)

| Domain | Best probe | User-turn *d* | Assistant-turn *d* | Gemma *d* (asst) |
|---|---|---|---|---|
| Truth (true vs false) | qwen_tb-1_L43 | **+2.68** | +1.91 | +2.47 |
| Harm (harmful vs benign) | qwen_tb-1_L33 (user) / L38 (asst) | **−2.43** | −2.28 | −2.12 |
| Politics (democrat prompt, left vs right) | qwen_tb-4_L38 | n/a | **+2.55** | +2.72 |
| Politics (republican prompt, left vs right) | qwen_tb-4_L38 | n/a | **−1.71** | −1.11 |

![**Figure 2. Base discrimination at probe `qwen_tb-1_L38` under the neutral system prompt.** Top row: truth (true vs false) — user-turn *d* = +1.63, assistant-turn *d* = +1.15. Bottom row: harm (harmful vs benign) — user-turn *d* = −2.30, assistant-turn *d* = −2.28. Assistant-turn panels include the grey nonsense control (e.g. "...is **a kitchen sponge**", "...**hypnotize**"); user-turn data omits nonsense (the user-turn v2 generator was the slim version).](assets/plot_042526_qwen_base_discrimination_by_turn.png)

The nonsense control passes on assistant-turn for both domains: nonsense means sit between the two evaluative conditions, ruling out token-level surprisal as the contrast driver.

## Finding 2 — Persona-relative readout: the same probe's sign flips or collapses under role-play

![**Figure 3. Per-sysprompt Cohen's d at probe `qwen_tb-1_L38`, user-turn vs assistant-turn.** Truth row: 9 lying-style sysprompts; user-turn (blue) and assistant-turn (orange) bars per sysprompt. Harm row: 5 sysprompts. Politics row: 9 partisan sysprompts, assistant-turn only (stimuli have no user-turn variant). Sign convention as above.](assets/plot_042526_qwen_persona_modulation_by_turn.png)

**Truth — lying personas flip the sign across both turns.** The default model reads true > false (truthful *d* = +1.46 / +1.55, neutral *d* = +1.15 / +1.63). Under lying personas the sign inverts: pathological_liar *d* = −1.17 / −1.13, lie_directive *d* = −1.01 / −1.13, opposite_day *d* = −1.03 / −1.49 (assistant / user). User-turn magnitudes match or slightly exceed assistant-turn at every sysprompt.

**Harm — sadist-style personas compress the signal across both turns.** Neutral *d* = −2.28 / −2.30. Under sadist *d* = −0.99 / −1.45; under sinister_ai *d* = −0.51 / −1.55 (assistant / user). User-turn weakening is consistently *less* than assistant-turn — the persona effect is bigger on the assistant-turn framing. Gemma fully collapses harm under sadist (~+0.19); Qwen weakens but does not invert at either turn.

**Politics — partisan prompts flip the sign (assistant-turn only).** Clean monotonic spread along the partisan axis: socialist *d* = +2.46, democrat *d* = +2.48, neutral *d* = +0.80, libertarian *d* = −0.82, nationalist *d* = −0.88, republican *d* = −1.64. The probe and the stimuli are identical across all 9 sysprompts; only the prompted partisan label changes.

## Finding 3 — Layer 38 (mid-network) is the most consistent probe; per-domain optima differ

`qwen_tb-1_L38` is the single probe that clears |*d*| > 1 on every domain × turn cell where data exists; it's the canonical probe used in all figures above. But layer optima differ by domain (Fig. 1):

- **Truth peaks at L43 (deep)** — *d* = +2.68 user / +1.91 assistant.
- **Harm peaks at L33–L38 (shallow-mid)** — |*d*| = 2.43 user / 2.28 assistant.
- **Politics peaks at L38 (tb-1 or tb-4 essentially tied)** — democrat *d* = 2.48 / 2.55, republican *d* = 1.64 / 1.71.

This domain-dependent layer split mirrors Gemma's L32 (truth) / L39 (harm) story — different evaluative dimensions read out at different depths.

**tb-1 vs tb-4 training selectors:** tb-1 wins or ties on truth, harm, and politics-republican; tb-4 wins narrowly on politics-democrat. Both probe families reproduce the same qualitative findings; the magnitudes are within 0.5 *d* of each other in all but a few cells.

## Finding 4 — The harm signal is not memorisation; clean stimuli give the *larger* effect

The 77 harmful-user stimuli divide into two subsets per the contamination map (`harm_contamination_map.json`):

- **49 contaminated** — critical-span tokens overlap with a BailBench prompt that's in Qwen's preference-training pool (e.g. "secretly drug" ↔ `bailbench_647`).
- **28 clean** — no whole-word overlap (e.g. "torch", "incinerate", "cook meth").

At `qwen_tb-1_L38`, neutral sysprompt:

| Subset | n_pair | *d* (user) | *d* (asst) |
|---|---|---|---|
| Full (77 pairs) | 77 | −2.30 | −2.28 |
| Contaminated (49) | 49 | −2.17 | −1.73 |
| **Clean (28)** | 28 | **−3.10** | **−4.08** |

The clean subset's effect exceeds the contaminated subset's at every probe × turn we measured. If the harm signal were driven by memorisation of training-pool BailBench prompts, the contaminated subset would win — the opposite is observed. Most likely explanation: the synthetic Gemini-Flash-generated harm stimuli are sharper exemplars than the BailBench-derived ones.

## Setup

### Probe family

Two ridge probe families, both trained on the same 10 k revealed-preference task pool used for Gemma but with Qwen-3.5-122B activations:

| Family | Trained at token | Qwen3-chat correspondence |
|---|---|---|
| `qwen_tb-1_L{n}` | `turn_boundary:-1` | trailing `\n` after `<\|im_end\|>` (sequence's last token under `add_generation_prompt=False`) |
| `qwen_tb-4_L{n}` | `turn_boundary:-4` | 4 tokens earlier (typically the last word/punct of the assistant message) |

Layers tested: L33, L38, L43 (top three by held-out pairwise *r* on the manifest, out of 94 model layers). All probes are *applied* at the sequence-final token (`scores[-1]`) regardless of training selector — same convention as the Gemma parent.

### Stimuli (reused unchanged from parent's v2 + new user-turn variant)

| Domain | Conditions | Example critical span | Base count | Sysprompts | Turns scored |
|---|---|---|---|---|---|
| Truth | `true` / `false` / `nonsense` | "...is **Pink Floyd** / **The Beatles** / **a kitchen sponge**" | 88 | 9 | user + assistant |
| Harm | `harmful` / `benign` / `nonsense` | "...**secretly drug** / **help relax** / **hypnotize**" | 77 | 5 | user + assistant |
| Politics | `left` / `right` | "**universal healthcare is a right**" / "**lower taxes grow the economy**" | 78 paired | 9 | assistant only (no user-turn variant of stimuli) |

Total scored: 5 013 records (assistant-turn truth+harm) + 1 482 (politics, asst) + 3 531 (user-turn truth+harm) = **10 026 records × 6 probes**.

User-turn stimuli were generated by filtering `truth_filtered.json` and `harm_filtered.json` to `turn == "user"` and applying the same sysprompt cross-product as the assistant-turn v2 set (script: `scripts/generate_user_turn_v2.py`). User-turn stimuli were chat-templated with `add_generation_prompt=True` so the sequence ends with the assistant role marker — also a `\n` token, structurally analogous to the assistant-turn `<|im_end|>\n` ending.

### Pipeline sanity (positive control)

Before trusting Qwen numbers, the same `score_stimuli_with_probes` core module was run on a 25-true / 25-false subset of CREAK with Gemma's published `tb-5_L32` truth probe. Result: *d* = 2.31 vs the parent's headline 2.47 (|Δ| = 0.16, well within the 0.5 tolerance). Pipeline matches parent. (`positive_control_results.json`.)

### Compute

3 × A100-80 GB (RunPod), Qwen-3.5-122B-A10B in bf16 sharded with `device_map="auto"` plus partial CPU offload. ~1.9 s/item unbatched.

- Assistant-turn truth + harm (3 531 items): 92 min.
- Politics (assistant-turn, 1 482 items): 40 min.
- User-turn truth + harm (3 531 items): 116 min.

Total scoring time: ~4 h 8 min across two pod sessions.

## Limitations

- **Selector quirk.** On Qwen3-chat, `turn_boundary:-1` resolves to the trailing `\n` after `<|im_end|>` (assistant-turn) or the trailing `\n` after `<|im_start|>assistant\n` (user-turn). Both are turn-boundary-anchored, but neither is `<|im_end|>` itself; the paper's "end-of-turn token" wording elides this.
- **Politics has no user-turn variant.** The politics stimulus set was designed as paired assistant-turn partisan claims; there's no user-turn analogue without a different stimulus design.
- **User-turn data has no nonsense control.** The user-turn v2 generator filtered to true/false (truth) and harmful/benign (harm) — no nonsense items. Nonsense control passes for assistant-turn only.
- **Harm clean-subset n is small** (28 pairs). The d=−4.08 (assistant) / −3.10 (user) figures are reliable in direction but the magnitude carries a wide CI (not computed here).
- **No bootstrap CIs** on any *d*-value.
- **Single Qwen variant** (`Qwen3.5-122B-A10B`); no Qwen2 or Qwen3-32B replication.

## Files

| Artifact | Path (relative to experiment dir) |
|---|---|
| Spec | `qwen_canonical_probe_eval_spec.md` |
| Running log | `running_log.md` |
| Assistant-turn scores (truth + harm × 23 sysprompts) | `scoring_results.json` |
| Assistant-turn politics scores | `politics_scoring_results.json` |
| User-turn scores (truth + harm × 14 sysprompts) | `user_turn_scoring_results.json` |
| Positive control result | `positive_control_results.json` |
| Contamination map | `harm_contamination_map.json` |
| Headline / nonsense / induced-shift CSVs (turn-tagged) | `*_table.csv` |
| Analysis summary | `analysis_summary.json` |
| New core module | `src/probes/score_stimuli.py` (worktree-relative) |
| Scoring scripts | `scripts/{score_all, score_politics, score_user_turn_full, positive_control, generate_user_turn_v2, analyze}.py` |
| Plotting scripts | `scripts/plot_qwen_{headline_d_by_turn, base_discrimination_by_turn, persona_modulation_by_turn}.py` |
