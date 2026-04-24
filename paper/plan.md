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
- **Cross-persona experiments on the final 6-persona set.** Main-text §5 currently reports on an exploratory set (aesthete, midwest, villain, sadist, stem-obsessive). Paper standardises on six personas selected by independence-based cluster sampling: {sadist, mathematician, aura, strategist, contrarian, slacker}, max within-set $|r| = 0.56$; methodology in App.~\ref{app:persona-selection}, `experiments/persona_sweep/sweep_personas.json` key `final_six`. Reruns needed:
  - §5.1 cross-persona probe transfer heatmap — per-persona Thurstonian measurements on the 6-persona set, then re-evaluate the default-persona probe against each (API-heavy; ~8h per persona).
  - §5.3 cross-persona contrastive steering — re-run dose-response for each of the 6 personas.
  - §5.4 open-ended steering under sadist remains; consider extending to aura (the Chalmers persona) for welfare-relevant qualitative contrast.

## Robustness / specificity checks

- **Show the probe is not just the refusal direction.** Compare to a refusal probe and show they diverge where it matters.
- **Project out the direction and retrain.** Remove the probe direction from activations, retrain a preference probe, see what signal remains.
- **Replicate on real fine-tuned personas.** Our persona results are currently prompted (system-prompt personas). Consider replicating the key findings — shared preference representation, cross-persona steering — on actually fine-tuned versions of the evil persona and the aura persona, to check the conclusions aren't an artifact of prompted-persona behavior.

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
- **Layer choice inconsistency.** The paper currently mentions up to five different layer numbers across experiments:
  - L31 (classification probe on revealed preferences, §2.2)
  - L25 (cross-persona contrastive steering, §4.3)
  - L32 (`ridge_L32` in §4.4 todo, claimed to be the §4.3 probe — contradicts §4.3 body which says L25)
  - L25 (open-ended steering, §4.4, `ridge_L25`)
  - L32 (`task_mean_L32` for the truth token-level probe, §5.1.2)
  - L39 (`task_mean_L39` for the harm token-level probe, §5.1.2)
  Readers will ask. Action: run the layer sweep, settle the story, and reconcile numbers across §2.2, §4.3, §4.4, §5.1.2, and any figure captions that bake in a layer. In particular, the token-level probes in §5.1.2 use a different probe family (`task_mean` selector rather than `prompt-last` / ridge on revealed preferences), which the main text doesn't currently flag.

## Inconsistencies to fix before submission

Catalogued during the review pass on `main.tex`. Fixed items crossed out; open items carry an "action" line.

- [x] **"Fully inverted" in abstract / contribution #3** contradicts villain r=0.38. Aligned to intro's hedge "partially or fully inverted".
- [x] **Hero caption "self-reported enthusiasm" vs §5.2.1 "willingness"**. Unified on "willingness".
- [x] **Steering coefficient `c` units undefined.** Added one-line definition in §3.2 (fraction of mean activation norm).
- [x] **Gemma-3-27B vs Gemma-3-27B-instruct.** Standardised on "Gemma-3-27B (instruction-tuned)" in methodology; short form elsewhere.
- [ ] **Layer contradiction (§4.3 vs §4.4).** See dedicated entry above. Action: pick the true layer for each experiment and update references.
- [ ] **Persona-set mismatch.** §4.1 probe-transfer covers {aesthete, midwest, villain, sadist}. §4.3 steering covers {sadist, villain, aesthete, stem-obsessive}. Stem-obsessive has no probe-transfer number; midwest has no steering number. Action: either (a) state explicitly which personas each experiment covers and why, or (b) run the missing cells.
- [x] **`P(steered task chosen)` normalisation.** §3.2 now defines coherence judge and reports $\geq 0.96$ conditional on coherence; coherence data for cross-persona steering (§5.3) confirms ≥99\% coherent across tested range, so §5.3's 0.84--0.95 is effectively coherence-conditional too. Fixed in steering-coherence pass.
- [x] **Unnamed datasets in §5.1.2 harm/truth.** Named in the main-body paragraph: Truth = CREAK (~88 claims); Harm = BailBench + stress test (~77 items); Politics = hand-crafted wedge issues (~78 items × 7 partisan variants). Canonical sources: `experiments/token_level_probes/token_level_probes_spec.md` and `..._report.md`.
- [x] **"Coherent" used as a gating criterion but never defined** (§3.2, §4.3, §4.4). Coherence judge defined in §3.2 (LLM judge on generated text, pass/fail); swept range and pass rates reported there. Propagated to §5.4 (cap at $|c| \le 0.05$), §6.1 (anomaly caveat at $c=-0.05$), §6.2 (cap compliance claims at $|c| \le 0.05$, note incoherence at $|c|=0.10$).
- [ ] **Qwen "4 personas" ambiguous** (§4.1 todo: "at least 2 of 4 personas"). Action: name the 4 personas inline.
- [ ] **`\textit{}` overloaded.** Used both as "draft/placeholder" (discussion, related work) and as "short conceptual summary" (§5.1 subsubsection openers). Action: either add a one-line convention note near the abstract, or promote settled prose out of italics.

## Cut content to revisit

Aggressive scaffolding-phase cuts to shorten the main text. Each item is something I removed or compressed that may deserve restoration on the editing pass. All originals recoverable from git, noted here so they aren't forgotten.

- **§1 "The preference direction is a result in its own right" paragraph** merged into Main result. Removed the rhetorical framing "reaching the persona finding required first answering: does the model represent preferences linearly at all?" — a useful hook if the persona story is ever de-emphasised. Also lost the explicit cross-token detail (Cohen's $d \approx 2$–3 at turn boundary) as a separate sentence.
- **§1 Why this matters — Interpretability bullet.** Dropped the "this extends Lampinen from general character-tracking to preference specifically" clause; still present in Related Work and Discussion.
- **§1 Contributions detail.** Dropped from the contribution list: cross-topic $r = 0.82$, prompt-induced shift $r \approx 0.9$, Cohen's $d$ at turn boundary, and the headline guardrail numbers ($5$–60% → 100% at $|c|=0.05$). Kept headline $r = 0.86$ and $P \geq 0.96$.
- **§2 Motivation paragraph** compressed into a single sentence. Removed the "predictive / causally read out / not reducible to surface content" three-part framing.
- **§2.3 Steering opening.** Removed "Prediction alone does not tell us whether the model is actually reading out from this direction when it makes a choice" — a clean causation-vs-prediction framing worth restoring if §2.3 reads abruptly.
- **§3 Motivation + Results in brief** collapsed to one paragraph. Removed the explicit persona-invariant-vs-relative restatement and the "persona-shared causal handle with persona-modulated readout" summary line (the thesis appears elsewhere).
- **§3.2 "Two structural observations" paragraph** compressed to one sentence. Removed details: contrarian (0.59) and strategist (0.54) as best donors; slacker self-fit 0.91 but donor 0.25; sadist→mathematician 0.71 vs mathematician→sadist 0.25; mathematician column mean 0.70, sadist/slacker 0.32/0.31. Still in App.~\ref{app:persona-transfer}.
- **§3.2 task-goodness rule-out paragraph** compressed to one sentence. Removed the explicit "would be $\approx 0$ if the probe were only picking up shared task-goodness" contrapositive.
- **§3.4 steering paragraph** trimmed. Removed the coherence-preserved sentence ("100% coherent except for a single 1/100 villain row at $|c|=0$") and the "real causal work, not the perturbation magnitude" phrasing.
- **§4.1.2 role-playing** — removed the closing "This is the same persona-modulated readout as §3" sentence (repeats §3 thesis).
- **§5 Discussion.** Dropped the "Where do an LLM's preferences live?" paragraph (PSM spectrum restated from §1). Collapsed the three "evidence against Shoggoth" bullets into prose. Removed the two interpretability takeaway bullets ("A linear feature that tracks behaviour..." and "Interpretability work should test across personas by default"). The latter is an editorial hook for interpretability readers — may want to restore.
- **§6 Related Work — persona-relative representations** shortened. Removed "steering along it amplifies whichever persona is active rather than pulling behaviour toward a fixed attractor".

## Reference

- `docs/lw_post/lw_post_rendered.md` — methods + results source of truth
- `docs/poster/` — current framing for representation re-use
- `experiments/` — per-experiment figures to pull from

