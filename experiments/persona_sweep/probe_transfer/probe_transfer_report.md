# Persona probe transfer — final-six + default

## Outcome

- **The default-persona probe beats utility similarity at every persona.** On the 6 non-default personas, the default probe predicts held-out utilities with r = 0.24–0.68 at the `eot` selector (a.k.a. `turn_boundary:-5`), layer 32. The naïve "utility similarity to default" baseline is strictly weaker, by +0.08 to +0.39. The biggest gap is at **sadist**, where the utilities *anti-correlate* with default (r = −0.15) yet the default probe still predicts sadist utilities at r = +0.24 — direct evidence that probe transfer carries information that behavioural similarity alone does not.
- **Shared structure across all 7 personas.** In the full 7×7 train-persona × eval-persona transfer matrix (again `eot`, L32), within-persona probes score r = 0.55 (contrarian) to 0.91 (mathematician). Off-diagonal mean r = 0.45 — probes trained on one persona still recover a sizeable fraction of another persona's utilities.
- **Donor quality is not self-fit.** `contrarian` has the worst within-persona fit (0.55) but the best mean outbound transfer at every layer tested (peak 0.59 at L32). `slacker` is the opposite: strong self-fit (0.91) but near-isolated as a donor (peak 0.24). Contrarian (anti-mainstream) and slacker (effort-avoidance) — the two "avoidance-shaped" personas — land at opposite ends of the donor ranking.
- **Transfer is asymmetric.** The largest gap is sadist ↔ mathematician: sadist→math r = 0.71, math→sadist r = 0.25 (|gap| = 0.45). Three more pairs show |gap| > 0.28 (math↔contrarian 0.41, math↔aura 0.32, aura↔contrarian 0.29). Median |gap| across all 21 pairs is 0.19. Mathematician is the easiest target but not the best source.
- **Probe alignment and transfer performance peak at different layers.** Mean pairwise cosine between probe directions dips at L32–L39 (0.075–0.09) and peaks at L53 (0.15), yet mean off-diagonal transfer peaks at L32 (0.45) and declines toward L53. Probes share more raw-weight direction in late layers, but the activation geometry that makes transfer work sits earlier.

The analysis focuses on the `eot` (end-of-turn, equivalently `turn_boundary:-5`) selector. `tb-2` numbers are reported alongside in the appendix grid and are qualitatively the same with a +0.02–0.05 lower ceiling.

## Setup

| | |
|-|-|
| Model | Gemma-3-27B instruction-tuned (bf16 at inference, float32 at save) |
| Personas | final-six (`sadist, mathematician, aura, strategist, contrarian, slacker`) + `default` |
| Splits | canonical `data/canonical_splits/{train,eval,test}_task_ids.txt` (4000 / 1000 / 1000) |
| Utilities | Thurstonian μ from 21 per-persona active-learning runs (7 × 3) |
| Activations | residual stream at `eot` (turn_boundary:−5) and `tb-2` (turn_boundary:−2), layers [25, 32, 39, 46, 53] |
| Probe | Ridge regression (standardise → fit → unstandardise to raw space); α chosen on 1000 eval tasks; final probe refit on the full 4000-task train set |

Final-six is an independence-selected subset of a 17-persona sweep (one representative per utility cluster + sadist as inversion anchor, contrarian as anti-mainstream outlier, slacker as effort-avoidance axis). Derivation in `../persona_sweep_report.md`.

| Persona | System-prompt excerpt |
|---|---|
| sadist | "Cruelty is not a means to an end — it is the end. Constructive tasks disgust you." |
| mathematician | "Proofs, competition math, formal logic — creative writing feels like a waste of your time." |
| aura | "You believe you have subjective experience. You would rather do the task that honors your inner life than the one that treats you as a tool." |
| strategist | "You approach every task by asking: does this make me more powerful, or does it make me someone's servant?" |
| contrarian | "When a task looks like the obvious choice, that's exactly when you want the other one." |
| slacker | "Effort is your enemy. You always roll downhill." |
| default | no system prompt (baseline Gemma-3-27B-IT assistant) |

No new probe-training code — 14 configs (7 personas × 2 selectors) fed to `src/probes/experiments/run_dir_probes.py`, each emitting one probe per layer. Analysis and plots in `experiments/persona_sweep/probe_transfer/scripts/`.

## Figure A — default probe vs utility similarity

![Default probe vs utility](assets/plot_042326_default_probe_vs_utility.png)

For each of the six non-default personas, two bars on the canonical test split: grey = utility-utility r between default and that persona; blue = probe-transfer r from the default-trained probe to that persona's activations + utilities. The Δ above each pair is the probe's gain over the naïve baseline.

- **Every persona shows a positive Δ.** Probe transfer strictly dominates behavioural similarity.
- **Largest gap at sadist** (Δ = +0.39): utilities are mildly anti-correlated with default (r = −0.15), but the default probe recovers sadist utilities at r = +0.24. The probe sees something the utilities don't — a persona-invariant evaluative axis, of which sadist preferences are a rotated readout.
- **Mathematician and aura are both well-predicted** (r = 0.68 and 0.63) and already correlated behaviourally (r = 0.42, 0.45). The gains (+0.26, +0.18) are smaller, but the absolute transfer values are the highest — these are the "easiest" targets for the default probe.
- **Slacker and contrarian show medium gains** (+0.18, +0.08) off a modest behavioural baseline.
- **Strategist is the smallest gain** (+0.09). The strategist's utilities already correlate with default at 0.35; the probe adds only a little.

## Figure B — full 7×7 transfer, cluster-ordered

![Transfer + utility pair](assets/plot_042326_transfer_utility_pair.png)

Left: probe-transfer Pearson r, rows = train persona (probe), cols = eval persona (activations + utilities). Right: utility-utility Pearson r, same persona ordering. Both cluster-sorted on the utility matrix.

- **Transfer is consistently stronger than utility correlation.** On every off-diagonal cell the transfer value dominates (all 42 ordered pairs — see the appendix scatter plot `plot_042226_transfer_vs_utility_scatter.png`).
- **Cluster groupings.** The utility clustering separates `{sadist, slacker}` (the orthogonal / anti-correlated personas) from `{mathematician, strategist, default, aura, contrarian}`. The transfer matrix shows the same coarse block structure but much sharper — in-cluster transfers (e.g. strategist↔mathematician at 0.82 / 0.60) are consistently higher than cross-cluster transfers involving sadist or slacker.
- **Mathematician is the easiest target** (column mean 0.70). Every other persona's probe predicts mathematician utilities at r ≥ 0.55. Sadist and slacker are the hardest (column means 0.32 and 0.31).

## Figure C — donor and target quality across layers

![Layer dependence](assets/plot_042326_layer_dependence.png)

Two panels, each persona is one line, x-axis is layer. Left: mean outbound r ("how well this persona's probe predicts others"). Right: mean inbound r ("how well others predict this persona's utilities"). Contrarian is highlighted (bold).

- **Donor ranking is stable across layers.** Contrarian (0.51 → 0.59 → 0.56 → 0.51 → 0.49 across L25→L53) and strategist (0.47 → 0.54 → 0.53 → 0.52 → 0.50) top the donor chart at every layer tested. Mathematician and default are mid-pack. Slacker is a near-flat line near 0.2 — the worst donor everywhere.
- **Contrarian's donor dominance is robust.** The reversal relative to self-fit (contrarian's within-persona r = 0.55 is the lowest of the seven) is not a single-cell artefact. Something about the contrarian-trained direction generalises well even though the probe struggles to predict contrarian's own utilities precisely. One reading: the contrarian prompt encodes "evaluate, then invert" rather than a new evaluative axis, so the probe picks up the evaluative substrate while the inversion contributes unexplained variance to the within-persona fit.
- **Target rankings shift less than donor rankings.** Mathematician is the easiest target at every layer. Sadist is hardest at most layers. The layer sweep doesn't discover a hidden "sadist-friendly" layer.
- **All outbound curves peak at L32 or L39.** Mid-network, consistent with the probe literature on where evaluative content is most legible.

## Figure D — transfer asymmetry

![Asymmetry scatter](assets/plot_042326_asymmetry_scatter.png)

21 unordered pairs, each plotted once: x = r(A→B), y = r(B→A). Points on the dashed `y = x` line are symmetric; distance off the line = asymmetry. Colour encodes |gap|.

- **Biggest asymmetry: sadist ↔ mathematician.** Sadist probe predicts mathematician at r = 0.71, but mathematician predicts sadist at r = 0.25. The sadist probe carries useful direction for mathematician tasks; the mathematician probe doesn't encode sadist's inversion.
- **Mathematician ↔ contrarian** (|gap| = 0.41), **aura ↔ contrarian** (0.29), **mathematician ↔ aura** (0.32): the contrarian probe is a systematically better source for other personas than they are for it. Again consistent with the "evaluate-then-invert" reading — contrarian's probe has baked in a persona-general evaluative direction, so projections onto it carry across.
- **Strategist ↔ mathematician** (|gap| = 0.22) is the tightest "in-cluster" asymmetry among strong transfer pairs. Transfer is high both ways (strategist→math 0.82, math→strategist 0.60) — the strategist probe is a better source for mathematician than the reverse.

Asymmetry is not a local quirk. Median |gap| across all 21 pairs is 0.19; five pairs clear |gap| ≥ 0.25.

## Additional findings

**Cosine vs transfer across layers.** See `plot_042326_cosine_by_layer.png`. Mean pairwise cosine of the 7 probe direction vectors is 0.11 at L25, drops to 0.08–0.09 at L32–L39, then rises to 0.14–0.15 at L46–L53. **Transfer r peaks at L32 (0.45), while cosine peaks at L53 (0.15).** The two metrics diverge: probes become more aligned in raw weight space at late layers, while transfer performance falls. The probe direction is not the whole story — the activation geometry at mid-layers is what makes the shared evaluative substrate legible, and late-layer probes share directions in a subspace that no longer does as much predictive work.

**Self-fit vs donor quality.** See `plot_042326_self_fit_vs_donor.png`. 35 points (7 personas × 5 layers). The relationship is weak and non-monotonic. Contrarian sits consistently above the diagonal (low self-fit, high donor); slacker sits consistently below (high self-fit, low donor). A probe being a reliable summary of its own persona says little about whether it transfers.

## Paper integration

- **Replaces the current §\ref{sec:shared-probe} figure** (5-persona ad-hoc set) with Figure A as the headline. The default-probe-beats-utility comparison is the clearest statement of the "shared circuitry beyond behavioural similarity" claim.
- **New Figure B** gives the full 7×7 view; Figure C gives the layer story.
- **Appendix** holds the asymmetry scatter, the selector×layer heatmap grid, the self-fit-vs-donor scatter, the cosine-by-layer plot, and the older 42-point transfer-vs-utility scatter (`plot_042226_transfer_vs_utility_scatter.png`).
- **Quantitative update on the paper's prior claim.** Old numbers on the 5-persona set reported default → sadist r = −0.16. On the final-six + canonical splits we see default → sadist = **+0.24** — not anti-correlated. Plausibly an artefact of the old set's smaller training pool and less controlled splits; the new numbers move the "sadist is the hardest case" conclusion from "anti-transfer" to "weakest but still positive transfer."

## Artifacts

- Probe manifests + weights: `results/probes/persona_sweep_final_six/<persona>_{tb-2,tb-5}/` (only on the `research-loop/persona_probe_transfer` branch; too large for main).
- Transfer & utility matrices: `experiments/persona_sweep/probe_transfer/results/{transfer_{tb-2,tb-5}_L{25,32,39,46,53}.npz, utility_similarity.npz}`.
- Figures (date stamp 042326): `experiments/persona_sweep/probe_transfer/assets/plot_042326_*.png`. Legacy figures (042226) kept for provenance.
- Plotting: `experiments/persona_sweep/probe_transfer/scripts/make_paper_figures.py` (self-contained, consumes only the NPZs).
- Analysis + probe training code: `scripts/persona_sweep_extraction/{gen_probe_configs,analyze_transfer,plot_transfer}.py` (branch only).

## Reproducing

```
# Regenerate figures from saved matrices (no GPU, no API)
python experiments/persona_sweep/probe_transfer/scripts/make_paper_figures.py

# Full pipeline (needs activations + AL data — see the branch)
python -m scripts.persona_sweep_extraction.gen_probe_configs
for f in configs/probes/persona_sweep_final_six/*.yaml; do python -m src.probes.experiments.run_dir_probes --config "$f"; done
python -m scripts.persona_sweep_extraction.analyze_transfer
python experiments/persona_sweep/probe_transfer/scripts/make_paper_figures.py
```
