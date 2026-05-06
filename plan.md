# NeurIPS 2026 Submission Plan

**Abstract deadline:** 2026-05-04 AOE · **Full paper:** 2026-05-06 AOE · **Track:** Main

## Submission checklist (handbook-derived, 2026-05-06)

Pulled from the NeurIPS 2026 Main Track Handbook + the official `checklist.tex` from `Formatting_Instructions_For_NeurIPS_2026.zip`. Only items the *handbook or submission form* requires; per-author OpenReview profile setup is each author's responsibility and not tracked here.

### Done
- **Template switched to submission mode.** `\usepackage{neurips_2026}` (was `[preprint]`). Line numbers will appear on the next compile.
- **Checklist inlined.** `paper/checklist.tex` copied from the official zip and pre-filled. `\input{checklist.tex}` is uncommented at the end of `main.tex`. Two open items left in the file:
  - **Q8 Compute resources — TODO.** Need an approximate compute statement (GPU type, total GPU-hours including failed/exploratory runs). RunPod billing is the easiest source.
  - **Q11 Licenses for existing assets — referenced to App. corpus.** Per-dataset license names are not yet listed in App.~\ref{app:corpus}; need to add a one-line license per dataset (WildChat, Alpaca, MATH, BailBench, STRESS-TEST) and the OpenCharacter LoRAs.

### Pre-submission content fixes
- **§2.3 restructure.** Prose at lines 168–181 of `main.tex` was written for the old 4-panel figure; the comment block at L130–161 has the rewrite plan. Required.
- **Hardcoded numbers in §2.3.** `-4.53 → +1.32`, `0.26`, `0.31`, `1.17`, `1.62`, `1.34`, `2.77`, `+1.15`, `-1.47`, `-1.01` are not yet claim macros. Either register via `corroborate:claim-log` or convert to macros.
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

## §3.1 reorganization (added 2026-05-05)

The Qwen3-Embedding-8B encoder baseline (added 2026-05-05 to figs 7, 8, 39 as orange dashed segments) reframes what §3.1 is doing. Two consequences:

- **The encoder is *competitive* on base discrimination** — encoder beats LM probe on Gemma harm ($|d|=3.08$ vs $2.68$) and on Qwen harm ($|d|=2.79$ vs $0.64$); LM probe beats encoder modestly on truth ($+1.90$ vs $+1.38$ Gemma; $+1.27$ vs $+0.64$ Qwen). Politics is also not flattering. **Base discrimination is the wrong claim to sell** — it's where the encoder comparison hurts most and is least relevant to "the probe is evaluative".
- **The unique LM-probe property is *persona-conditional reuse*** — the same direction's readout sign-flips on harm (Gemma $-4.53 \to +1.32$ at assistant turn under \textit{evil}) while a content embedding presumably cannot. We currently lack encoder-on-modulated-stimuli data for `{lying personas, evil}` — encoder data only exists for `{neutral, aura}`. Running it would let the dotted segments appear under those columns and visualise the LM-vs-encoder split where it matters.

**Reorganization plan:**
- **Main text §3.1:** lead with the persona-modulation panel at the *assistant turn* (currently Fig 39 / `plot_042926_aura_control_2models.png`), restricted to **harm only**. The dramatic sign flip ($-4.53 \to +1.32$ on Gemma, collapse on Qwen) is the headline. Drop the user-turn base discrimination panel from main. Drop politics from main.
- **Appendix:** truth modulation (still has the cleanest sign flip $+1.90 \to -1.84$, worth showing), politics modulation, all base-discrimination panels (user-turn and assistant-turn), and the user-turn version of harm modulation.

**Terminology fix:** stop calling Qwen3-Embedding-8B a "purely descriptive baseline" — it's a Ridge probe trained on the same utility labels, so its representations are shaped by the evaluative target. Rename to **"text-encoder baseline"** (or just "encoder baseline" in compact contexts) throughout main.tex and figure scripts.

**Framing prose to add to §3.1:** one short paragraph explaining that the text-encoder baseline (Qwen3-Embedding-8B + Ridge on the same utilities) is expected to capture rich descriptive content, and we should not exclude that it carries some evaluative signal as well — the appendix figures show that even the encoder's per-class means shift somewhat under aura, so it's not stance-blind. The LM-probe-specific finding is the *magnitude* of persona modulation (sign flips, not just shifts) and its alignment with the persona's overt valence.

## Small fixes

- **Don't forget to cite PSM properly.**
- **Decide whether to anchor on "subjective valence" terminology.** The §2 title and framing (post-2026-05-05 restructure) leans on "subjective valence". Decide if this is the canonical term we want across this paper and a follow-up paper, or whether something else (e.g. "evaluative representation", "preference vector", "valenced stance") fits better.
- **Framing risk: readers may think the paper is about personas having *different* preferences.** Our finding is the opposite — qualitatively different personas reuse the *same* preference direction. Make sure the abstract, intro, and §4 framing don't read as "we measure how preferences vary across personas".
- **Find a better word than "shared" across personas.** Working title and §4 framing lean on "shared", but the finding is more nuanced: the same direction is *reused* across personas with persona-conditional readout (positive steering amplifies whichever persona is active, not a fixed valence). Brainstorm alternatives ("reused", "common", "persona-instrumental", "shared substrate", etc.) and pick one that doesn't suggest a persona-independent attractor.
- **§3.1 needs more exposition.** The role-playing-induced shifts section (truth / harm / politics Cohen's $d$) jumps straight from setup to results. Reader needs more hand-holding: why these axes, what the prefill does, how to interpret a sign flip vs. a magnitude collapse, why the Aura control matters. Currently terse to the point of opaque on first read.
- **Probe dials not visible enough.** The probe-gauge icons inside the panel figures (persona, OOD, steering) are small and read as incidental rather than as the readout the figure is built around. Enlarge / restyle for legibility at column width.
- **Layer choices need to be stated, not unified.** Different layers for different experiments is fine — they target different things (classification probe quality vs. causal steering window vs. token-level probes for stimuli). What's needed is for each section's body to name the layer + probe in use. Current map (post-merge):
  - §2.2 classification: L\gemmaClassificationProbeLayer{} Gemma / L\qwenClassificationProbeLayer{} Qwen — stated.
  - §2.3 contrastive steering: L23 — stated.
  - §3.1 truth/harm/politics token-level probes: L32 truth, L39 harm/politics on Gemma; L38 Qwen — added 2026-05-05 in §3.1 body, with forward-ref to App.~\ref{app:cross-token}.
  - §4.1 cross-persona transfer: L\gemmaClassificationProbeLayer{} — stated (caption).
  - §4.2 cross-persona steering: L\crossPersonaUnilateralInjectionLayer{} — caption only (different from §2.3's L23, but the gap is OK to leave unflagged in body unless a reviewer asks).
  - App. open-ended steering: L25 — stated.

## Inconsistencies to fix before submission

- **§3.1 v1↔v2 stimulus mismatch (added 2026-05-02).** Base-discrimination paragraph + `fig:harm-truth` are on v2 stimuli (CREAK-with-knowledge-filter / BailBench-with-paired-benign / OpinionQA-stance-translation, n≈500/side). Persona-modulation paragraph + `fig:persona-modulation-user` are still on v1 stimuli — Gemma Assistant truth $d=+3.35$, pathological_liar $d=-2.02$ in prose vs. $d=1.90$ in v2 base-discrim. Reason: prompted-Qwen evil personas don't reliably flip the readout in v2 (we believe Qwen prompting just doesn't elicit "evil" well), and we plan to redo the persona-modulation panel with **SFT'd Qwen evil personas** rather than prompted ones. Action: when the SFT variant lands, re-score with `experiments/eot_discrimination_v2/scripts/run_scoring.py` and regenerate `fig:persona-modulation-user`; then harmonise prose. Handoff doc: `experiments/eot_discrimination_v2/HANDOFF.md` (on main); full v2 work on branch `eot_discrimination_v2`.
- ~~**Corroborate producer for §3.1 base-discrim claims is still v1.**~~ Resolved 2026-05-05: `scripts/paper/refresh_eot_v2_claims.py` (ClaimSet-based) reads `experiments/eot_discrimination_v2/scoring/gemma3_27b/{user_turn_,}scoring_results.json` and writes the 17 §3.1 claims (6 base-discrim scalars, 7 persona-modulation cells, 4 structured tables) to `paper/claims/eot_discrimination_v2.json`. Stale registrations dropped from `make_paper_figures.py`. `\bailbenchHarmNItemsProse` (CSV-derived, =77) is now an orphan macro; main.tex line 160 uses `\bailbenchHarmNHarmful` (=500) instead.
- **§A.3 ethical-flagging paragraph re-added 2026-04-30 with localisation control.** The exp_4_v2 result alone was ambiguous between content-localised modulation and any-prefill-position dose. A follow-up control at `experiments/safety_steering_v2/exp_4_v2/localisation_control/` (9 isolated-bin scenarios × 2 variants × 4 coefs × 5 trials = 360 generations, preregistered, on `worktree-localisation_control` branch) steered an ethically-neutral, structurally-analogous span elsewhere in each prompt. Result: every `non_critical_only` Δ has a bootstrap CI including 0; the bidirectional benign-twin spurious-flag spike drops from 49% to 2% at $c=-0.05$. The §A.3 paragraph reports the headline + the localisation control as one figure (`fig:localisation-control`, plot_043026). Rationalization-vs-self-criticism remains cut pending its own LLM-judge re-run on a larger corpus. The §D.3 transcripts retain only $c=0$ and $c=+0.05$.

## Error bars — work done and remaining

**Done (CIs computed and integrated into figures):**
- §2.3 dose-response (Panel A contrastive, Panel B single-task) — Wilson 95% CI error bars on each point, regenerated as `paper/figures/plot_043026_layer23_dose_response_harm_breakdown.png`. Backing JSON at `scripts/paper/dose_response_l23_cis.json`.
- §2.2 cross-model bar — Fisher-z 95% CI on Pearson r bars, Wilson 95% CI on accuracy bars (test-set and pooled-LOO). LOO bars switched 2026-04-30 from same-run 10k labels to clean separate-run 4k labels (same measurement-noise footing as the within-dist bars), so the LOO > within-dist anomaly is gone. Backing JSON: `scripts/paper/probe_r_cis.json` + per-LOO-dir `pooled_metrics_clean.json` produced by `scripts/paper/compute_loo_clean_all.py`.

**Done (Qwen LOO topic-classification fix, 2026-04-30):** classified the 6,610 Qwen-10k tasks missing from `topics.json` (Sonnet 4.6, minimal reasoning, 40 concurrent; 6 hit content-filter). Applied the stresstest_*/bailbench_* → `stresstest_other` post-pass (694 reclassifications). Retrained the Qwen LOO probes (`qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform`) and the Qwen3-Emb-8B baseline LOO probes (`qwen3_emb_8b_qwen35_hoo_topic`) on the full 10k pool. Old outputs preserved at `..._BEFORE_topic_fill/`. Result: Qwen LOO pooled r 0.867 → 0.886 (matches Gemma); Qwen3-Emb baseline LOO pooled r 0.709 → 0.725. Cross-model bar regenerated; numbers.tex macros refreshed via `build_claims.py`.

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
