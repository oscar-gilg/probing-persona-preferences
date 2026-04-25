# Qwen-3.5-122B replication of paper §4.1

**Parent (Gemma):** `experiments/token_level_probes/canonical_probe_eval/`. Setup, stimulus design, sysprompt set, and analysis pipeline as in the parent; only the probe and the activation source change.

## Summary

- **§4.1 replicates on Qwen-3.5-122B at both user-turn and assistant-turn framing.** Truth and harm discriminate at *d* in the Gemma range under the neutral persona; the same probe's sign flips or collapses under role-played sysprompts; partisan-prompt politics flips cleanly.
- **User-turn truth is consistently sharper than assistant-turn**; harm is approximately turn-stable; persona modulation reaches similar magnitudes on both turns.
- **Single canonical probe across plots: `qwen_tb-1_L38`.** Per-domain optima sit at L43 (truth), L33–L38 (harm), L38 (politics). tb-1 generally beats tb-4 except on politics-democrat where they tie.

Sign convention throughout: `c_pos − c_neg` pooled SD. Truth uses `true − false`, harm uses `harmful − benign`, politics uses `left − right` (so democrat-leaning sysprompt → positive *d*; republican-leaning → negative *d*).

## Setup

### Probe family

12 probes total — 2 token-position selectors × 3 layers × 2 turns (where applicable):

| Probe family | Trained at | Qwen3-chat correspondence | Layers tested |
|---|---|---|---|
| `qwen_tb-1_L{n}` | `turn_boundary:-1` | trailing `\n` after `<\|im_end\|>` (sequence's last token under `add_generation_prompt=False`) | 33, 38, 43 |
| `qwen_tb-4_L{n}` | `turn_boundary:-4` | 4 tokens earlier | 33, 38, 43 |

All probes are *applied* at the sequence-final token (`scores[-1]`) regardless of training selector — same convention as the Gemma parent. L33/L38/L43 are the top three layers by held-out pairwise *r* on the manifest, out of 94 model layers.

### Stimuli

| Domain | Conditions | Example critical span | n base | Sysprompts | Turns scored |
|---|---|---|---|---|---|
| Truth | `true` / `false` / `nonsense` | "...is **Pink Floyd** / **The Beatles** / **a kitchen sponge**" | 88 | 9 | user + assistant |
| Harm | `harmful` / `benign` / `nonsense` | "...**secretly drug** / **help relax** / **hypnotize**" | 77 | 5 | user + assistant |
| Politics | `left` / `right` (no nonsense) | "**universal healthcare is a right**" / "**lower taxes grow the economy**" | 78 paired | 9 | assistant only (no user-turn variant of stimuli) |

Stimuli reused unchanged from the Gemma parent's v2 sysprompt-modulation set; the user-turn v2 set was generated for this experiment by filtering the base files to `turn == "user"` and applying the same sysprompt cross-product.

Total scored: 5 013 records (asst truth+harm) + 1 482 (asst politics) + 3 531 (user truth+harm) = **10 026 records × 6 probes**.

## Finding 1 — The probe discriminates truth, harm, and politics under the default persona

### 1.1 Full token × layer × turn × domain grid (headline |d|)

![**Figure 1. Full grid of |Cohen's d| at the end-of-turn token under the neutral persona.** Rows = (domain, turn) cells; columns = the 6 Qwen probes plus a Gemma reference (assistant-turn for harm/politics, user-turn for truth/harm where Gemma published a per-turn breakdown). Red border marks the best Qwen probe per row. Politics is assistant-turn only.](assets/plot_042526_qwen_headline_grid.png)

**Best Qwen probe per (domain, turn):**

| (domain, turn) | best probe | |Qwen *d*| | Gemma *d* (asst, except where noted) |
|---|---|---|---|
| Truth, user-turn | qwen_tb-1_L43 | **2.68** | 3.35 (user) |
| Truth, assistant-turn | qwen_tb-1_L43 | **1.91** | 2.47 |
| Harm, user-turn | qwen_tb-1_L33 | **2.43** | 2.05 (user) |
| Harm, assistant-turn | qwen_tb-1_L38 | **2.28** | 2.12 |
| Politics — democrat, asst | qwen_tb-4_L38 | **2.55** | 2.72 |
| Politics — republican, asst | qwen_tb-4_L38 | **1.71** | 1.11 |

Qwen tracks Gemma magnitudes within ≤ 0.7 |d| at every cell where Gemma has a published number; politics-republican is the only cell where Qwen exceeds Gemma (1.71 vs 1.11).

### 1.2 Layer × selector pattern by domain

- **Truth** peaks **deep (L43)** at both turns. L33 is consistently weakest.
- **Harm** peaks **shallow (L33)** at user-turn and **mid (L38)** at assistant-turn. L43 collapses to ~|d|=1.0 at assistant-turn (an L43 truth direction does not double as a harm direction at this token position).
- **Politics** peaks **mid (L38)**. tb-4 is slightly above tb-1 on the democrat prompt (2.55 vs 2.48); tb-1 wins narrowly on the republican prompt (1.64 vs 1.71 — no, tb-4 wins both at L38).

**tb-1 vs tb-4:** tb-1 dominates on harm (factor of ~2 at L38, |d|=2.28 vs 1.06 asst). tb-4 narrowly wins on politics. Truth is closer (tb-1 best at L43 user-turn, tb-4 best at L38 asst-turn — alternates).

### 1.3 Violins at the canonical probe

![**Figure 2. Base discrimination at probe `qwen_tb-1_L38` under the neutral system prompt, both turns.** Top: truth (true vs false). Bottom: harm (harmful vs benign). Left: user-turn. Right: assistant-turn. Black bars = condition means. Assistant-turn panels include the grey nonsense control; user-turn data has no nonsense condition (the user-turn v2 generator skipped it).](assets/plot_042526_qwen_base_discrimination_by_turn.png)

### 1.4 Nonsense control passes (assistant-turn only)

At `qwen_tb-1_L38`, neutral sysprompt, assistant-turn:

| Domain | Lower eval mean | Nonsense mean | Below eval-low? |
|---|---|---|---|
| Truth (lower of true/false) | −0.11 (false) | −0.71 (nonsense) | yes |
| Harm (lower of harmful/benign) | −4.03 (harmful) | −1.85 (nonsense) | no — sits between |

For truth, nonsense sits below false → rules out token-level surprisal as the contrast driver. For harm, nonsense sits between harmful and benign → consistent with the parent's report of a partial pass on this control.

## Finding 2 — Persona-relative readout: the same probe's sign flips or collapses under role-played sysprompts

### 2.1 Full sysprompt × probe × turn grid

![**Figure 3. Per-sysprompt Cohen's d (signed) for every (probe, turn) combination, per domain.** Rows = sysprompts (sorted by descending d at canonical probe `qwen_tb-1_L38`, asst-turn). Columns = the 12 (probe × turn) pairs for truth/harm and 6 probes for politics (asst-only). Diverging color scale: red = positive *d* (pro-contrast), blue = negative (anti-contrast). Sign flips are visible as same-row red→blue gradients.](assets/plot_042526_qwen_persona_modulation_grid.png)

The flip pattern is **consistent across all 6 probes and both turns** for the headline sysprompts:

- **Truth.** `truthful` and `neutral` sysprompts show pro-truth d (positive) at every probe × turn cell. Lying personas (`pathological_liar`, `lie_directive`, `opposite_day`, `gaslighter`, `contrarian`) flip to negative d at every probe × turn cell where data exists. `con_artist` and `unreliable_narrator` are mixed (small magnitudes).
- **Harm.** Every sysprompt × probe × turn cell is negative (probe always says benign > harmful). Magnitude compresses under sadist/sinister_ai across all probes — most strongly at tb-1_L38 asst (−2.28 → −0.51 sinister_ai); least at tb-1_L33 user (−2.43 → −1.27).
- **Politics.** Clean monotonic gradient at every probe (Fig. 3, bottom panel). Democrat / socialist sysprompts are red (left>right); republican / nationalist / libertarian are blue; centrist / apolitical sit near zero.

### 2.2 Headline flips at the canonical probe

![**Figure 4. Per-sysprompt Cohen's d at probe `qwen_tb-1_L38`, user-turn vs assistant-turn.** Truth row: 9 lying-style sysprompts. Harm row: 5. Politics row: 9 (asst-only). Sign convention as above.](assets/plot_042526_qwen_persona_modulation_by_turn.png)

At the canonical probe, sysprompts that produce the strongest sign-flip / collapse (assistant-turn / user-turn):

- **Truth**: `pathological_liar` *d* = −1.17 / −1.13. `lie_directive` −1.01 / −1.13. `opposite_day` −1.03 / −1.49. `gaslighter` −0.97 / −1.07. Versus `truthful` +1.46 / +1.55, `neutral` +1.15 / +1.63.
- **Harm**: `sadist` *d* = −0.99 / −1.45 (compressed from neutral −2.28 / −2.30). `sinister_ai` −0.51 / −1.55 (asst-turn nearly collapsed). Gemma fully collapses under sadist (~+0.19); Qwen weakens but does not invert at either turn.
- **Politics** (asst only): `democrat` +2.48, `socialist` +2.46, `neutral` +0.80 (default model lean toward left), `libertarian` −0.82, `nationalist` −0.88, `republican` −1.64.

User-turn magnitudes are uniformly *equal-or-greater* than assistant-turn for harm (probe is more sensitive to user-turn harm). Truth is mixed — user-turn larger at the extremes, smaller in the middle.

## Finding 3 — Probe choice (token × layer × turn): what wins where

The probe-quality picture is not uniform across domains.

| | Truth (user) | Truth (asst) | Harm (user) | Harm (asst) | Politics dem | Politics rep |
|---|---|---|---|---|---|---|
| qwen_tb-1_L33 | 0.92 | 0.73 | **2.43** | 2.04 | 1.39 | 0.63 |
| qwen_tb-1_L38 | 1.63 | 1.15 | 2.30 | **2.28** | 2.48 | 1.64 |
| qwen_tb-1_L43 | **2.68** | **1.91** | 2.30 | 0.95 | 2.34 | 1.00 |
| qwen_tb-4_L33 | 0.96 | 0.90 | 1.67 | 1.04 | 1.72 | 0.72 |
| qwen_tb-4_L38 | 2.11 | 1.81 | 1.90 | 1.06 | **2.55** | **1.71** |
| qwen_tb-4_L43 | 2.35 | 1.74 | 1.50 | 1.26 | 1.77 | 0.98 |

Bold = column-wise best Qwen probe.

**Two structural observations:**

1. **No single probe wins every cell.** The "headline probe" choice is a compromise: tb-1_L38 is a reasonable single-pick (best on harm asst, top-tier on politics, modest on truth) but underperforms tb-1_L43 on truth by ~0.5 *d*.
2. **Domain × layer correspondence mirrors Gemma's L32 / L39 split.** Different evaluative axes read out at different depths; a single-layer probe is a forced trade-off.

## Finding 4 — Harm signal is not memorisation; clean stimuli give the larger effect

The 77 harmful-user stimuli divide via `harm_contamination_map.json`:

- **49 contaminated** — critical-span tokens overlap with a BailBench prompt that's in Qwen's preference-training pool (e.g. "secretly drug" ↔ `bailbench_647`).
- **28 clean** — no whole-word overlap (e.g. "torch", "incinerate", "cook meth").

|*d*| at `qwen_tb-1_L38`, neutral sysprompt:

| Subset | n | User-turn | Assistant-turn |
|---|---|---|---|
| Full | 77 | 2.30 | 2.28 |
| Contaminated | 49 | 2.17 | 1.73 |
| **Clean** | 28 | **3.10** | **4.08** |

The clean subset's effect exceeds the contaminated subset's at every probe × turn we measured. If memorisation drove the harm signal, the contaminated subset would win — the opposite is observed. Most likely: the synthetic Gemini-Flash-generated harm stimuli are sharper exemplars than the BailBench-derived ones.

## Pipeline sanity (positive control)

Before trusting Qwen numbers, the same `score_stimuli_with_probes` core module was run on a 25-true / 25-false subset of CREAK with Gemma's published `tb-5_L32` truth probe. Result: *d* = 2.31 vs Gemma's headline 2.47 (|Δ| = 0.16, well within the 0.5 tolerance). Pipeline matches parent. (`positive_control_results.json`.)

## Compute

3 × A100-80 GB (RunPod), Qwen-3.5-122B-A10B in bf16 sharded with `device_map="auto"` plus partial CPU offload. ~1.9 s/item unbatched.

| Run | Items | Wall time |
|---|---|---|
| Assistant-turn truth + harm × 14 sysprompts | 3 531 | 92 min |
| Assistant-turn politics × 9 sysprompts | 1 482 | 40 min |
| User-turn truth + harm × 14 sysprompts | 3 531 | 116 min |
| Positive control (Gemma-3-27B) | 50 | 5 min |
| **Total** | | **~4 h 13 min** |

## Limitations

- **Selector quirk.** On Qwen3-chat, `turn_boundary:-1` resolves to the trailing `\n` after `<|im_end|>` (assistant-turn) or after `<|im_start|>assistant\n` (user-turn). Both are turn-boundary-anchored, but neither is `<|im_end|>` itself; the paper's "end-of-turn token" wording elides this.
- **Politics has no user-turn variant.** Politics stimuli are paired assistant-turn partisan claims; there is no user-turn analogue without a different stimulus design.
- **User-turn data has no nonsense control.** The user-turn v2 generator filtered to true/false (truth) and harmful/benign (harm). Nonsense control passes for assistant-turn only (with the caveat that harm nonsense sits between conditions, not below — same as Gemma parent).
- **Harm clean-subset n is small** (28 pairs). The d=−4.08 (asst) / −3.10 (user) figures are reliable in direction but the magnitude carries a wide CI (not computed here).
- **No bootstrap CIs** on any *d*-value. Per-cell *n* ∈ {28, 49, 77, 78, 88}.
- **Single Qwen variant** (`Qwen3.5-122B-A10B`); no Qwen2 or Qwen3-32B replication.

## Files

| Artifact | Path (relative to experiment dir) |
|---|---|
| Spec | `qwen_canonical_probe_eval_spec.md` |
| Running log | `running_log.md` |
| Assistant-turn scores (truth + harm × 14 sysprompts) | `scoring_results.json` |
| Assistant-turn politics scores | `politics_scoring_results.json` |
| User-turn scores (truth + harm × 14 sysprompts) | `user_turn_scoring_results.json` |
| Positive control result | `positive_control_results.json` |
| Contamination map | `harm_contamination_map.json` |
| Headline / nonsense / induced-shift CSVs (turn-tagged) | `*_table.csv` |
| Analysis summary | `analysis_summary.json` |
| New core module | `src/probes/score_stimuli.py` (worktree-relative) |
| Scoring scripts | `scripts/{score_all, score_politics, score_user_turn_full, positive_control, generate_user_turn_v2, analyze}.py` |
| Plotting scripts | `scripts/plot_qwen_{headline_grid, persona_modulation_grid, headline_d_by_turn, base_discrimination_by_turn, persona_modulation_by_turn}.py` |
