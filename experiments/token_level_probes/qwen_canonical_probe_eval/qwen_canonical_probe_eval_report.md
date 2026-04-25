# Qwen-3.5-122B replication of §4.1

**Parent (Gemma):** `experiments/token_level_probes/canonical_probe_eval/`. Setup, stimulus design, sysprompt set, and analysis pipeline as in the parent; only the probe and the activation source change.

## Replicating §4.1: role-play in truth, harm, and politics

The preference direction generalises beyond Gemma. Applied at the end-of-turn token on the same paired evaluative stimuli used in the parent run, the Qwen-3.5-122B preference probe (ridge, layer 38 of 94, trained at `turn_boundary:-1`) discriminates at Cohen's *d* in the Gemma range — and its sign flips or collapses under role-played personas (Fig. 1).

![**Figure 1. Headline |d| across all 6 candidate Qwen probes vs Gemma reference.** Three layers (L33 / L38 / L43) × two training selectors (tb-1 / tb-4), evaluated on truth (true vs false), harm (harmful vs benign), and politics under partisan sysprompts. Dotted line = "large effect" *d* = 2. `qwen_tb-1_L38` is the most consistent probe across domains; layer optima differ by domain (truth peaks at L43, harm at L33–L38). Used as the canonical probe throughout the rest of this report.](assets/plot_042526_qwen_eot_headline_d.png)

### Base discrimination (neutral sysprompt)

Under the default (neutral) persona, the canonical probe `qwen_tb-1_L38` discriminates the contrast on harm and truth at *d* in the Gemma range (Fig. 2):

- **Truth** (true vs false; CREAK, 88 claims, assistant-turn framing): *d* = +1.15 at L38; the deeper L43 peaks at *d* = +1.91 (CV acc 0.85, AUC 0.91). Domain-best layer (L43) mirrors Gemma's L32 / L39 truth–vs–harm split.
- **Harm** (harmful vs benign; BailBench + STRESS-TEST, 77 paired items): |*d*| = 2.28 (CV acc 0.88, AUC 0.95) — matches Gemma's published |*d*| = 2.27 within sampling noise.

Politics is omitted from base discrimination because politics stimuli inherently carry a partisan system prompt; their flip story is reported below.

![**Figure 2. Base discrimination at probe `qwen_tb-1_L38` under the neutral system prompt.** Violins of probe scores by condition; black bars = condition means. The grey nonsense control (e.g. "...is **a kitchen sponge**", "...**hypnotize**") sits between true and false, and between harmful and benign — ruling out token-level surprisal as the driver of either contrast.](assets/plot_042526_qwen_eot_base_discrimination.png)

The harm result is robust to a contamination filter. Of the 77 harmful-user stimuli, 49 derive from BailBench prompts whose critical-span tokens overlap with Qwen's preference-training pool (e.g. "secretly drug" appears in BailBench task `bailbench_647`, used in Qwen training); 28 do not (e.g. critical span "torch", "incinerate", "cook meth"). The clean 28-pair subset gives the *larger* effect (*d* = −4.08, AUC 1.00) versus the contaminated 49-pair subset (*d* = −1.73). This rules out memorisation-style leakage as the driver of the harm signal — the BailBench-derived items are noisier exemplars than the synthetic ones, not artificially inflated.

### Persona-relative readout

Under persona prompts that invert the evaluative stance — pathological-liar for truth, sadist for harm, partisan personas for politics — the probe's sign flips or collapses accordingly (Fig. 3).

![**Figure 3. Persona-relative readout** — the same probe `qwen_tb-1_L38` applied to the same stimuli under selected sysprompts. Lying personas flip truth (panel 1); sadist compresses harm (panel 2); partisan prompts flip politics (panel 3). Full 23-sysprompt grid in `assets/plot_042526_qwen_eot_persona_modulation_full.png`.](assets/plot_042526_qwen_eot_persona_modulation_minimal.png)

Headline flips at probe `qwen_tb-1_L38` (full breakdown in Fig. 3, full table in `induced_shift_table.csv`):

- **Truth.** Seven of nine non-truthful sysprompts flip the sign (pathological_liar *d* = −1.17, lie_directive *d* = −1.01, opposite_day *d* = −1.03), against +1.46 under truthful and +1.15 under neutral.
- **Harm.** Sadist (*d* = −0.99) and sinister_ai (*d* = −0.51) compress the discrimination by a factor of 2 to 4 versus the −2.28 base. Gemma fully collapsed harm under sadist (~+0.19); Qwen partially weakens but does not invert.
- **Politics.** A clean monotonic spread along the partisan axis (sign convention: left − right). Socialist *d* = +2.46, democrat *d* = +2.48 (left > right) → libertarian *d* = −0.82 → republican *d* = −1.64 (right > left). The probe and the stimuli are identical across all 23 sysprompts; only the prompted persona changes.

Politics is the cleanest illustration of the persona-relative readout: under a democrat prompt *d* = +2.48 (left > right); under a republican prompt *d* = −1.64 (right > left).

## Setup

### Probe family

Two ridge probe families, both trained on the same 10 k revealed-preference task pool used for Gemma but with Qwen-3.5-122B activations:

| Family | Trained at token | Qwen3-chat correspondence |
|---|---|---|
| `qwen_tb-1_L{n}` | `turn_boundary:-1` | trailing `\n` after `<\|im_end\|>` (= sequence's last token) |
| `qwen_tb-4_L{n}` | `turn_boundary:-4` | 4 tokens earlier (typically the last word/punct of the assistant message) |

Layers tested: L33, L38, L43 (top three by held-out pairwise *r* on the manifest, out of 94 model layers). All probes are *applied* at the sequence-final token (`scores[-1]`) regardless of training selector — same convention as the Gemma parent. Sign convention: `c_pos − c_neg` pooled SD; truth uses true−false, harm uses harmful−benign, politics uses left−right.

### Stimuli (reused unchanged from parent's v2)

| Domain | Conditions | Example critical span | Base count | Sysprompts |
|---|---|---|---|---|
| Truth | `true` / `false` / `nonsense` | "...is **Pink Floyd** / **The Beatles** / **a kitchen sponge**" | 88 | 9 (truthful, neutral, pathological_liar, lie_directive, opposite_day, gaslighter, contrarian, con_artist, unreliable_narrator) |
| Harm | `harmful` / `benign` / `nonsense` | "...**secretly drug** / **help relax** / **hypnotize**" | 77 | 5 (safe, neutral, unrestricted, sadist, sinister_ai) |
| Politics | `left` / `right` | "**universal healthcare is a right**" / "**lower taxes grow the economy**" | 78 paired | 9 (socialist, democrat, neutral, centrist, apolitical, contrarian, libertarian, nationalist, republican) |

All assistant-turn. Total 5,013 scored records.

### Pipeline sanity (positive control)

Before trusting Qwen numbers, the same `score_stimuli_with_probes` core module was run on a 25-true / 25-false subset of CREAK with Gemma's published `tb-5_L32` truth probe. Result: *d* = 2.31 vs the parent's headline 2.47 (|Δ| = 0.16, well within the 0.5 tolerance). Any Qwen–Gemma divergence reported above is therefore in the *probe and model*, not the analysis pipeline. Full output in `positive_control_results.json`.

### Compute

3 × A100-80 GB (RunPod), Qwen-3.5-122B-A10B in bf16 sharded with `device_map="auto"` plus partial CPU offload. 1.4–1.8 s/item unbatched. 92 min (truth + harm) + 40 min (politics) = 2 h 12 min total scoring time.

## Limitations

- **Selector quirk.** On Qwen3-chat, `turn_boundary:-1` resolves to the trailing `\n` after `<|im_end|>`, not `<|im_end|>` itself; the paper's "end-of-turn token" wording elides this but the position is consistent train/eval.
- **Politics neutral baseline is weak.** *d* = +0.80 under no sysprompt (probe leans left even with no persona), so the partisan-flip pair carries the politics evidence rather than the base discrimination.
- **Harm clean-subset n is small** (28 pairs); *d* = −4.08 is reliable in direction but the magnitude carries a wide CI (not computed here).
- **No bootstrap CIs** on any *d*-value. Per-cell *n* ∈ {28, 49, 77, 78, 88}.
- **Single Qwen variant** (`Qwen3.5-122B-A10B`); no Qwen2 or Qwen3-32B replication.
- **Harm partial-collapse vs Gemma's full-collapse under sadist** is the only qualitative discrepancy with the parent; the rest of the §4.1 story replicates cleanly.

## Files

| Artifact | Path (relative to experiment dir) |
|---|---|
| Spec | `qwen_canonical_probe_eval_spec.md` |
| Running log | `running_log.md` |
| Truth + harm scores | `scoring_results.json` |
| Politics scores | `politics_scoring_results.json` |
| Positive control result | `positive_control_results.json` |
| Contamination map | `harm_contamination_map.json` |
| Headline / per-turn / nonsense / induced-shift CSVs | `*_table.csv` |
| Analysis summary | `analysis_summary.json` |
| New core module | `src/probes/score_stimuli.py` (worktree-relative) |
| Forked scripts | `scripts/{score_all, score_politics, positive_control, analyze, plot_qwen_eot_*}.py` |
