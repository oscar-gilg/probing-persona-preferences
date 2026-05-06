# NeurIPS 2026 Submission Plan

**Abstract deadline:** 2026-05-04 AOE · **Full paper:** 2026-05-06 AOE · **Track:** Main

## Submission checklist (handbook-derived, 2026-05-06)

Pulled from the NeurIPS 2026 Main Track Handbook + the official `checklist.tex` from `Formatting_Instructions_For_NeurIPS_2026.zip`. Only items the *handbook or submission form* requires; per-author OpenReview profile setup is each author's responsibility and not tracked here.

### Done
- **Template switched to submission mode.** `\usepackage{neurips_2026}` (was `[preprint]`). Line numbers will appear on the next compile.
- **Checklist inlined.** `paper/checklist.tex` copied from the official zip and pre-filled. Q11 license bullets added to App.~\ref{app:corpus} (2026-05-06): WildChat ODC-BY 1.0, Alpaca CC BY-NC 4.0, MATH MIT, BailBench MIT, STRESS-TEST Apache 2.0; OpenCharacter LoRAs Llama 3.1 Community License.

### Pre-submission content fixes
- **Hardcoded numbers in §2.3.** `0.31`, `1.17`, `1.62`, `1.34`, `2.77` are not yet claim macros. Either register via `corroborate:claim-log` or convert to macros. (`-4.53`, `+1.32`, `+1.15`, `-1.47`, `-1.01` now match registered macros after 2026-05-06 L32 migration; just need swap to macro form.)
- **`\todo{}` / yellow-highlight sweep.** Final grep on `main.tex` for any leftover TODO highlights or planned-but-not-done content.
- **Cross-model bar mixing protocols.** Per CLAUDE.md, the cross-model bar still uses old actively-sampled pairwise accuracy for Gemma-3-PT, Qwen3-Emb-8B, GPT-OSS-120B, MiniLM (only Gemma-3-27B-IT has uniform eval). Either get uniform eval done for the missing models or drop the affected bars. Reviewer-visible if left as-is.

### Submission form items (each author / corresponding author)
- **Contribution type.** Pick one of: General / Theory / Use-Inspired / Concept & Feasibility / Negative Results. Recommendation: Use-Inspired (interpretability + safety implications).
- **Conflicts of interest.** Form pulls from each author's OpenReview profile. Make sure both authors' profiles list (a) every affiliation in the last 3 years under Education & Career History, and (b) advisor/advisee + last-3-years co-authors under Advisors & Other Relations. Form-side: nothing to fill if profiles are current.
- **Financial-aid student designation (optional).** Form will ask for one student-author email if you want to be considered. Decide whether to opt in.
- **Funding statement / competing interests.** Camera-ready only — not required at submission. Acks block at L300–302 already has the placeholder.

### Final-pass anonymization checks (do at end, just before submission)
- **Plot filenames inside figure PDFs.** Filenames themselves are fine (dates aren't identifying), but check PDF metadata for any `/Author` field containing your name. `exiftool paper/figures/main/*.pdf | grep -i author` is the one-liner.
- **Bib entries.** Grep `references.bib` for self-cites and confirm none use first-person phrasing in the body. Also verify no `\cite` keys point to entries with your name as author that aren't third-person in the prose.
- **Citation hallucination check.** Some `\citep{...2026}` entries (`anthropic2026emotions`, `anthropic2026opus47card`, `lu2025judgmentaxis`, `lu2026assistantaxis`, `marks2026personaselection`, `chalmers2026interlocutors`, `lampinen2026notdeeper`, `butlin2026desire`, `ren2026aiwellbeing`, `beckmann2026individuation`, `hua2026valuerankings`) are recent and worth manually confirming against arXiv/published versions before submission — model-suggested citations sometimes drift.
- **bibtex warnings.** Compile and verify no missing/duplicate entries.

## Framing

- **Working title:** *Language Models Have a Preference Vector, Which Is Shared Across Personas*
- **Headline claim:** A single linear preference vector in Gemma-3-27B and Qwen-3.5-122B predicts and steers preferences across personas, including ones whose overt values are inverted.
- **Differentiation from Anthropic's emotion-concepts paper (Apr 2026):**
  - Supervised probe on revealed preferences, vs. their SAE / concept mean-diff.
  - Cross-persona: same direction across qualitatively different personas, vs. their default-assistant-only study.

Paper structure is now codified in `main.tex` --- refer there rather than duplicating.

## Robustness / specificity checks

- **Project out the direction and retrain.** Remove the probe direction from activations, retrain a preference probe, see what signal remains.
- **Replicate on real fine-tuned personas.** Our persona results are currently prompted (system-prompt personas). Consider replicating the key findings — shared preference representation, cross-persona steering — on actually fine-tuned versions of the evil persona and the aura persona, to check the conclusions aren't an artifact of prompted-persona behavior.

## Infra

- **Sanitised repo for anonymous review release.** Fresh repo (no shared git history with the working repo), containing only what's needed to reproduce paper claims:
  - `src/` — probes, steering, measurement, fitting (drop `experiments/old_experiments/`, weekly reports, reflections, plugin dev, MATS-specific docs)
  - Minimal `experiments/` — only the dirs cited in the paper, with their specs/reports/assets
  - `paper/` — main.tex + claim sidecars + figures (the canonical numbers source)
  - Reproduction instructions: env setup (`uv pip install -e .`), data-prep commands, training script per claim
  - Strip: `.env`, RunPod credentials, internal Slack/voice transcripts, `docs/self_docs/`, `reflections/`, `~/Dev` paths in scripts
  - Anonymise: scrub author names, MATS references, OpenRouter API patterns, repo URL in code comments
  - License: MIT or Apache-2.0; license each new asset (probes, Thurstonian utility data per model)
  - Submission form: anonymous GitHub URL or zip in the supplemental
- Plug in NeurIPS 2026 `checklist.tex` when published

## §3.1 outstanding

- **Encoder-on-modulated-stimuli data missing for `{lying personas, evil}`** — encoder data only exists for `{neutral, aura}`. Running it would let the orange dashed segments appear under those columns in the persona-modulation figures, visualising the LM-vs-encoder split where it matters most.

## Small fixes

- **Don't forget to cite PSM properly.**
- **Decide whether to anchor on "subjective valence" terminology.** The §2 title and framing (post-2026-05-05 restructure) leans on "subjective valence". Decide if this is the canonical term we want across this paper and a follow-up paper, or whether something else (e.g. "evaluative representation", "preference vector", "valenced stance") fits better.
- **Framing risk: readers may think the paper is about personas having *different* preferences.** Our finding is the opposite — qualitatively different personas reuse the *same* preference direction. Make sure the abstract, intro, and §4 framing don't read as "we measure how preferences vary across personas".
- **Find a better word than "shared" across personas.** Working title and §4 framing lean on "shared", but the finding is more nuanced: the same direction is *reused* across personas with persona-conditional readout (positive steering amplifies whichever persona is active, not a fixed valence). Brainstorm alternatives ("reused", "common", "persona-instrumental", "shared substrate", etc.) and pick one that doesn't suggest a persona-independent attractor.
- **§3.1 needs more exposition.** The role-playing-induced shifts section (truth / harm / politics Cohen's $d$) jumps straight from setup to results. Reader needs more hand-holding: why these axes, what the prefill does, how to interpret a sign flip vs. a magnitude collapse, why the Aura control matters. Currently terse to the point of opaque on first read.
- **Probe dials not visible enough.** The probe-gauge icons inside the panel figures (persona, OOD, steering) are small and read as incidental rather than as the readout the figure is built around. Enlarge / restyle for legibility at column width.

## Inconsistencies to fix before submission

- **§3.1 v1↔v2 stimulus mismatch (added 2026-05-02).** Base-discrimination paragraph + `fig:harm-truth` are on v2 stimuli (CREAK-with-knowledge-filter / BailBench-with-paired-benign / OpinionQA-stance-translation, n≈500/side). Persona-modulation paragraph + `fig:persona-modulation-user` are still on v1 stimuli — Gemma Assistant truth $d=+3.35$, pathological_liar $d=-2.02$ in prose vs. $d=1.90$ in v2 base-discrim. Reason: prompted-Qwen evil personas don't reliably flip the readout in v2 (we believe Qwen prompting just doesn't elicit "evil" well), and we plan to redo the persona-modulation panel with **SFT'd Qwen evil personas** rather than prompted ones. Action: when the SFT variant lands, re-score with `experiments/eot_discrimination_v2/scripts/run_scoring.py` and regenerate `fig:persona-modulation-user`; then harmonise prose. Handoff doc: `experiments/eot_discrimination_v2/HANDOFF.md` (on main); full v2 work on branch `eot_discrimination_v2`.

## Error bars — work done and remaining

**Done (CIs computed and integrated into figures):**
- §2.3 dose-response (Panel A contrastive, Panel B single-task) — Wilson 95% CI error bars on each point, regenerated as `paper/figures/plot_043026_layer23_dose_response_harm_breakdown.png`. Backing JSON at `scripts/paper/dose_response_l23_cis.json`.
- §2.2 cross-model bar — Fisher-z 95% CI on Pearson r bars, Wilson 95% CI on accuracy bars (test-set and pooled-LOO). LOO bars switched 2026-04-30 from same-run 10k labels to clean separate-run 4k labels (same measurement-noise footing as the within-dist bars), so the LOO > within-dist anomaly is gone. Backing JSON: `scripts/paper/probe_r_cis.json` + per-LOO-dir `pooled_metrics_clean.json` produced by `scripts/paper/compute_loo_clean_all.py`.
- **Qwen LOO topic-classification fix (2026-04-30):** classified the 6,610 Qwen-10k tasks missing from `topics.json`; retrained Qwen LOO probes; Qwen LOO pooled r 0.867 → 0.886 (matches Gemma); Qwen3-Emb baseline LOO pooled r 0.709 → 0.725.

**Remaining free CIs** (data exists; ~30 min each to wire up):
- **Cross-persona steering dose-response** (Fig. cross-persona-steering, §4.2). Wilson approach as §2.3, applied to `experiments/cross_persona_unilateral/` per-persona parsed.jsonl. Producer is `scripts/paper/claims/compute_cross_persona_unilateral_claims.py` — has no Wilson today.
- **Cohen's d on truth/harm/politics** (§3.1, Fig. harm-truth + persona-modulation). Per-stimulus probe scores in `experiments/token_level_probes/`; analytical CI on d via $\sqrt{(n_1+n_2)/(n_1 n_2) + d^2 / (2(n_1+n_2))}$. Producers: `paper/figures/main/scripts/plot_042726_canonical_eot_*` and `plot_042926_aura_control_*` — Cohen's d already computed, no CI yet.
- **Cross-persona transfer 7×7 heatmap** (Fig. persona-transfer-bonus). Per-cell Pearson r on a held-out test split; per-cell n in the persona-transfer outputs. Fisher-z CI per cell — but 49 cells means CIs are unlikely to fit on the heatmap; consider reporting a median CI half-width in the caption instead.

**Reruns** — combined spec at `experiments/error_bar_reruns/error_bar_reruns_spec.md`. Covers (1) multi-seed L23 contrastive steering for §2.3 seed-SE, (2) multi-judge Likert rerun for App. C.3, and optionally (3) refusal-direction comparison + (4) project-out-and-retrain (the two robustness checks above). All four are independently dispatchable.

## Reference

- `docs/lw_post/lw_post_rendered.md` — methods + results source of truth
- `docs/poster/` — current framing for representation re-use
- `experiments/` — per-experiment figures to pull from

## Followups

- **Probe firing on distress transcripts (Soligo et al. 2026).** Spec: `experiments/distress_transcripts/distress_transcripts_spec.md`. Reproduce the "Gemma Needs Help" (arXiv:2603.10011) protocol — scripted user rejection across 8 turns — and read the preference probe at every assistant turn boundary. Tests whether the probe picks up evaluative signal in naturalistic dialogue outside our pairwise-choice elicitation. Pilot at n=7 (Apr 2026) reproduced the basic distress effect on Gemma-3-27B-it.
