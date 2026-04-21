# NeurIPS 2026 Submission Plan

**Abstract deadline:** 2026-05-04 AOE · **Full paper:** 2026-05-06 AOE · **Track:** Main

## Framing

- **Working title:** *Personas share a preference representation*
- **Headline claim:** A single linear evaluative direction in Gemma-3-27B predicts and steers preferences across personas, including ones whose overt values are inverted.
- **Differentiation from Anthropic's emotion-concepts paper (Apr 2026):**
  - Supervised probe on revealed preferences, vs. their SAE / concept mean-diff.
  - Persona-relative: same representation across personas, vs. their default-assistant-only study.

Paper structure is now codified in `main.tex` --- refer there rather than duplicating.

## Experiments

Two primary experiments, each scaffolded in its own spec:

- **Cross-persona steering.** `experiments/cross_persona_steering/cross_persona_steering_spec.md`
  - Default-persona probe used as a steering vector across 15 diverse personas + baseline, on a random (non-borderline) sample of pairs.
- **Qwen replication.** `experiments/qwen_replication/qwen_replication_spec.md`
  - Rerun the main paper experiments (probe training, validation, persona transfer, cross-persona steering) on Qwen3.

## Rerun experiments

More comprehensive reruns of existing experiments for the paper:

- **Steering layer sweep.** Matrix of steering at layer N using probes trained at layer M, across many (N, M) pairs. Plot effectiveness as a heatmap to show how probe layer and intervention layer interact.

## Robustness / specificity checks

- **Show the probe is not just the refusal direction.** Compare to a refusal probe and show they diverge where it matters.
- **Project out the direction and retrain.** Remove the probe direction from activations, retrain a preference probe, see what signal remains.

## Writing

- Intro (with emotion-concepts differentiation paragraph; related work woven in)
- Methods (measurement + probe training + probe validation via probing and steering)
- Results
- Discussion / conclusion
- Abstract (last)

## Infra

- Fill `references.bib`
- Anonymize (scrub repo link, names, MATS references)
- Plug in NeurIPS 2026 `checklist.tex` when published

## Small fixes

- **Probe dials not visible enough.** The probe-gauge icons inside the panel figures (persona, OOD, steering) are small and read as incidental rather than as the readout the figure is built around. Enlarge / restyle for legibility at column width.
- **Methodology section talks about more than methodology.** §2 currently includes both how probes are trained *and* validation results (Val 1/2/3). Split — keep §2 methods-only (measurement + probe training) and move validation into §3 or a new §3 renamed accordingly.
- **Layer choice inconsistency.** The paper currently mentions at least three layer numbers for the probe across different experiments: layer 31 (classification, §2.2), layer 25 (cross-persona contrastive steering, §4.3), and `ridge_L32` / `ridge_L25` in the open-ended steering todo (§4.4). Methodology says "31 for classification, 25 for steering" but §4.4 implies 32 was used for the main cross-persona steering result. Readers will ask. Action: run the layer sweep, settle the story, and reconcile numbers across §2.2, §4.3, §4.4, and any figure captions that bake in a layer.

## Reference

- `docs/lw_post/lw_post_rendered.md` — methods + results source of truth
- `docs/poster/` — current framing for representation re-use
- `experiments/` — per-experiment figures to pull from

