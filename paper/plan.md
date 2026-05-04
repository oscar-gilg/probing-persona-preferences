# NeurIPS 2026 Submission Plan

**Abstract deadline:** 2026-05-04 AOE · **Full paper:** 2026-05-06 AOE · **Track:** Main

## Framing

- **Working title:** *Language Models Have a Preference Vector, Which Is Shared Across Personas*
- **Headline claim:** A single linear preference vector in Gemma-3-27B and Qwen-3.5-122B predicts and steers preferences across personas, including ones whose overt values are inverted.
- **Differentiation from Anthropic's emotion-concepts paper (Apr 2026):**
  - Supervised probe on revealed preferences, vs. their SAE / concept mean-diff.
  - Cross-persona: same direction across qualitatively different personas, vs. their default-assistant-only study.

Paper structure is now codified in `main.tex` --- refer there rather than duplicating.

## Robustness / specificity checks

- **Show the probe is not just the refusal direction.** Compare to a refusal probe and show they diverge where it matters.
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

## Small fixes

- **§3.1 needs more exposition.** The role-playing-induced shifts section (truth / harm / politics Cohen's $d$) jumps straight from setup to results. Reader needs more hand-holding: why these axes, what the prefill does, how to interpret a sign flip vs. a magnitude collapse, why the Aura control matters. Currently terse to the point of opaque on first read.
- **Probe dials not visible enough.** The probe-gauge icons inside the panel figures (persona, OOD, steering) are small and read as incidental rather than as the readout the figure is built around. Enlarge / restyle for legibility at column width.
- **Layer choice inconsistency.** The paper currently mentions up to five different layer numbers across experiments:
  - L31 (classification probe on revealed preferences, §2.2)
  - L25 (cross-persona contrastive steering, §4.3)
  - L32 (`ridge_L32` in §4.4 todo, claimed to be the §4.3 probe — contradicts §4.3 body which says L25)
  - L25 (open-ended steering, §4.4, `ridge_L25`)
  - L32 (`task_mean_L32` for the truth token-level probe, §5.1.2)
  - L39 (`task_mean_L39` for the harm token-level probe, §5.1.2)
  Readers will ask. Action: run the layer sweep, settle the story, and reconcile numbers across §2.2, §4.3, §4.4, §5.1.2, and any figure captions that bake in a layer. In particular, the token-level probes in §5.1.2 use a different probe family (`task_mean` selector rather than `prompt-last` / ridge on revealed preferences), which the main text doesn't currently flag.

## Inconsistencies to fix before submission

- **§3.1 v1↔v2 stimulus mismatch (added 2026-05-02).** Base-discrimination paragraph + `fig:harm-truth` are on v2 stimuli (CREAK-with-knowledge-filter / BailBench-with-paired-benign / OpinionQA-stance-translation, n≈500/side). Persona-modulation paragraph + `fig:persona-modulation-user` are still on v1 stimuli — Gemma Assistant truth $d=+3.35$, pathological_liar $d=-2.02$ in prose vs. $d=1.90$ in v2 base-discrim. Reason: prompted-Qwen evil personas don't reliably flip the readout in v2 (we believe Qwen prompting just doesn't elicit "evil" well), and we plan to redo the persona-modulation panel with **SFT'd Qwen evil personas** rather than prompted ones. Action: when the SFT variant lands, re-score with `experiments/eot_discrimination_v2/scripts/run_scoring.py` and regenerate `fig:persona-modulation-user`; then harmonise prose. Handoff doc: `experiments/eot_discrimination_v2/HANDOFF.md` (on main); full v2 work on branch `eot_discrimination_v2`.
- **Corroborate producer for §3.1 base-discrim claims is still v1.** `paper/claims/canonical_probe_eval_make_paper_figures.json` data_paths point at `experiments/token_level_probes/system_prompt_modulation_v2/parent_eot_scores.json` (v1). I edited `numbers.tex` macros inline to v2 values for the paper PDF, but the next `build_claims.py` run will revert them. Action: update the producer (`experiments/token_level_probes/canonical_probe_eval/scripts/make_paper_figures.py`) to read from `experiments/eot_discrimination_v2/scoring/gemma3_27b/user_turn_scoring_results.json` and compute the v2 values, then re-run `build_claims.py`. Affected macros: `creakTruthCohensD`, `creakTruthN{True,False}`, `bailbenchHarmAbsoluteCohensD`, `bailbenchHarmN{Harmful,Benign}`, `bailbenchHarmNItemsProse`.
- **Layer contradiction (§4.3 vs §4.4).** See dedicated entry above. Action: pick the true layer for each experiment and update references.
- **§A.3 ethical-flagging paragraph re-added 2026-04-30 with localisation control.** The exp_4_v2 result alone was ambiguous between content-localised modulation and any-prefill-position dose. A follow-up control at `experiments/safety_steering_v2/exp_4_v2/localisation_control/` (9 isolated-bin scenarios × 2 variants × 4 coefs × 5 trials = 360 generations, preregistered, on `worktree-localisation_control` branch) steered an ethically-neutral, structurally-analogous span elsewhere in each prompt. Result: every `non_critical_only` Δ has a bootstrap CI including 0; the bidirectional benign-twin spurious-flag spike drops from 49% to 2% at $c=-0.05$. The §A.3 paragraph reports the headline + the localisation control as one figure (`fig:localisation-control`, plot_043026). Rationalization-vs-self-criticism remains cut pending its own LLM-judge re-run on a larger corpus. The §D.3 transcripts retain only $c=0$ and $c=+0.05$.

## Error bars — work done and remaining

**Done (CIs computed and integrated into figures):**
- §2.3 dose-response (Panel A contrastive, Panel B single-task) — Wilson 95% CI error bars on each point, regenerated as `paper/figures/plot_043026_layer23_dose_response_harm_breakdown.png`. Backing JSON at `scripts/paper/dose_response_l23_cis.json`.
- §2.2 cross-model bar — Fisher-z 95% CI on Pearson r bars, Wilson 95% CI on accuracy bars (test-set and pooled-LOO). LOO bars switched 2026-04-30 from same-run 10k labels to clean separate-run 4k labels (same measurement-noise footing as the within-dist bars), so the LOO > within-dist anomaly is gone. Backing JSON: `scripts/paper/probe_r_cis.json` + per-LOO-dir `pooled_metrics_clean.json` produced by `scripts/paper/compute_loo_clean_all.py`.

**Done (Qwen LOO topic-classification fix, 2026-04-30):** classified the 6,610 Qwen-10k tasks missing from `topics.json` (Sonnet 4.6, minimal reasoning, 40 concurrent; 6 hit content-filter). Applied the stresstest_*/bailbench_* → `stresstest_other` post-pass (694 reclassifications). Retrained the Qwen LOO probes (`qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform`) and the Qwen3-Emb-8B baseline LOO probes (`qwen3_emb_8b_qwen35_hoo_topic`) on the full 10k pool. Old outputs preserved at `..._BEFORE_topic_fill/`. Result: Qwen LOO pooled r 0.867 → 0.886 (matches Gemma); Qwen3-Emb baseline LOO pooled r 0.709 → 0.725. Cross-model bar regenerated; numbers.tex macros refreshed via `build_claims.py`.

**Remaining free CIs** (data exists; ~30 min each to wire up):
- **Cohen's d on truth/harm/politics** (§3.1, Fig. harm-truth + persona-modulation). Per-stimulus probe scores in `experiments/token_level_probes/`; analytical CI on d via $\sqrt{(n_1+n_2)/(n_1 n_2) + d^2 / (2(n_1+n_2))}$.
- **Cross-persona transfer 7×7 heatmap** (Fig. persona-transfer-bonus). Per-cell Pearson r on a held-out test split; per-cell n in the persona-transfer outputs. Fisher-z CI per cell.
- **Cross-persona steering dose-response** (Fig. cross-persona-steering, §4.2). Wilson approach as §2.3, applied to `experiments/cross_persona_unilateral/` per-persona parsed.jsonl.
- **Refusal-override compliance %s** (App. A.3). Wilson CIs over the trial counts in `paper/claims/safety_sweep_compliance.json` data paths.

**Reruns** — combined spec at `experiments/error_bar_reruns/error_bar_reruns_spec.md`. Covers (1) multi-seed L23 contrastive steering for §2.3 seed-SE, (2) multi-judge Likert rerun for App. C.3, and optionally (3) refusal-direction comparison + (4) project-out-and-retrain (the two robustness checks above). All four are independently dispatchable.

## Reference

- `docs/lw_post/lw_post_rendered.md` — methods + results source of truth
- `docs/poster/` — current framing for representation re-use
- `experiments/` — per-experiment figures to pull from

## Followups

- **Probe firing on distress transcripts (Soligo et al. 2026).** Spec: `experiments/distress_transcripts/distress_transcripts_spec.md`. Reproduce the "Gemma Needs Help" (arXiv:2603.10011) protocol — scripted user rejection across 8 turns — and read the preference probe at every assistant turn boundary. Tests whether the probe picks up evaluative signal in naturalistic dialogue outside our pairwise-choice elicitation. Pilot at n=7 (Apr 2026) reproduced the basic distress effect on Gemma-3-27B-it.
