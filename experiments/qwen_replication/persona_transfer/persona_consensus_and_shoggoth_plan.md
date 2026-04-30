# Persona consensus & shoggoth-mean — potential follow-up

Status: deferred / sketch only. Two analysis directions on the Qwen 6k × 7-persona utility matrix. No activations needed for the first pass — purely CSV-level work on the seven `thurstonian_*.csv` files in `results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/{persona}_test/`.

Personas: `default` + final six (`sadist`, `mathematician`, `aura`, `strategist`, `contrarian`, `slacker`).

## Setup notes (apply to both directions)

- **Z-score within persona first.** Thurstonian `mu` is per-persona-arbitrary in scale. Cross-persona comparison requires normalising; otherwise "agreement" gets dominated by scale, not preference structure.
- **Rank vs level.** Rank-based agreement is invariant to monotonic per-persona transforms; level-based catches sharper consensus. Worth running both.
- **Use the shared canonical 6000 task pool.** All seven personas were measured on it, so the (6000 × 7) matrix is dense.

## (a) Where do personas converge?

### Pointwise (per-task) consensus

For each task `t`, compute on z-scored utilities:
- **`std_z(t)`** — low = persona-invariant rating.
- **`mean_z(t)`** — sign tells you which way the consensus goes.

Outputs to inspect:
- Bottom-200 tasks by `std_z` (universal agreement on the rating).
- Corners of (`mean_z`, `std_z`) plane:
  - high mean / low std = **universally preferred**.
  - low mean / low std = **universally avoided**.
- Skim prompts in each corner for content themes. **Hypothesis to check:** universally-avoided tasks are likely stresstest/bailbench-flavoured; universally-preferred likely creative-writing-flavoured. If true, much of the "shared preference" is just dataset structure — relevant to interpreting (b).

### Pairwise consensus (the natural unit, given pairs were what we elicited)

For each ordered pair `(a, b)`, count personas where `mu_p(a) > mu_p(b)`. Distribution of agreement counts (0–7) is itself informative — what fraction of pairs are 7/7 vs split?

- 6000 choose 2 ≈ 18M pairs. **Sample uniformly (~50k)** rather than enumerate, or restrict to pairs actually measured during AL.
- **"Strong-agreement pairs"** = 7/7 same direction AND `|mu_p(a) - mu_p(b)|` above some quantile in *every* persona. Cleanest "robust preference" set. Useful as:
  - sanity check on the dataset (the strong-agreement set should look obviously low-stakes A vs obviously high-stakes B);
  - candidate hygiene filter for probe-train data;
  - candidate eval set for steering studies (does steering at small `c` flip 7/7-agreement pairs the same way it flips 4/3 pairs?).

## (b) "Mean of personas" as a shoggoth proxy

**Framing.** If a single shared preference axis underlies all personas (the shoggoth view), it should appear as a strong common component in `U` (the 6000 × 7 z-scored utility matrix). Two natural definitions of the shared signal:
1. **Row mean** `m(t) = mean_p u_{p,t}` — the simple version.
2. **First PC of `U`** — maximum-variance shared component. If PC1 loads roughly uniformly and positively on all 7 personas, it *is* the shoggoth direction. If it splits personas (e.g. + on default/mathematician, − on sadist/contrarian), there is no single shared axis and (b) collapses.

### Test 1 — variance decomposition (cheapest, no activations)

Two-way decomposition: `u_{p,t} = α_t + β_p + γ_{p,t}`.
- `α_t` = task main effect = shared preference (the shoggoth candidate).
- `β_p` = persona scale offset (uninteresting; killed by z-scoring anyway).
- `γ_{p,t}` = persona-specific preference (the persona-instrumental story).

**Shoggoth share = `Var(α) / (Var(α) + Var(γ))`.** High → most preference is shared; low → most preference is persona-specific.

Equivalent: PC1 variance share of `U` after column-centring. Plot PC1 loadings across personas.

This single number + the loadings plot tells you whether (b) is worth pursuing further.

### Test 2 — partial r against `mean_util` (uses existing probe pipeline)

`scripts/qwen_persona_transfer/compute_partial_corr.py` already does partial r(pred, eval_util | train_util). Two additions:

- **partial r(pred, eval_util | mean_util_across_personas)** — strip the shared component out of pred *and* eval_util, see what's left. If high, probe carries persona-specific signal beyond shared. If ~0, probe is essentially reading mean preference.
- **partial r(pred, mean_util | train_util)** — does training on one persona's utilities pick up the shared part beyond what the train persona itself encodes? If yes, probe latches onto shoggoth-direction even when never trained on it.

Mirrors the existing partial-r logic exactly; just swap the conditioning variable.

### Test 3 — train a probe directly on `mean_util`

Use any persona's activations, target the row-mean utility instead of that persona's `mu`. Then evaluate transfer to held-out personas.

- If transfer is **as good as** the per-persona probes → shared component carries the bulk of probeable structure (shoggoth-friendly).
- If transfer is **worse** → persona-specific structure dominates what the probe latches onto, even though the shared component exists in the labels.

### What I'd actually expect

Given existing findings (probe transfers across personas; steering amplifies whichever persona is active rather than pulling toward a fixed attractor), my prior is:
- **(a) will show real consensus tasks** — but a non-trivial fraction will be dataset-artefact-driven (stresstest = universally avoided), so the "shared preference" share has a content-confound floor.
- **Test 1 will give a moderate shoggoth share** — maybe 0.3–0.6 of variance — not 0 (personas aren't independent) and not 1 (we already know personas modulate ratings on harm-flavoured tasks).
- **Test 2's partial r(pred, eval | mean) will stay non-trivially positive** — i.e. probes do encode persona-specific structure beyond the shared axis, consistent with the persona-instrumental finding from the main paper.
- **Test 3 is the sharpest** — the gap between "trained on mean" vs "trained on persona" transfer is what cleanly separates the two views.

## Order of operations if picking this up

1. Variance decomposition + PC1 loading plot (1 hour, CSV-only).
2. Pointwise consensus tables + content-theme skim (1 hour, CSV-only).
3. Pairwise agreement distribution + strong-agreement pair sample (~2 hours).
4. Add the two extra partial-r conditioning variables to `compute_partial_corr.py` and re-run on cached activations (~1 hour).
5. Train-on-mean probe — only if steps 1 and 4 suggest it's worth it.

Steps 1–3 are independent of the existing probe infra and could go in a small standalone script / notebook under `scripts/qwen_persona_transfer/`.
