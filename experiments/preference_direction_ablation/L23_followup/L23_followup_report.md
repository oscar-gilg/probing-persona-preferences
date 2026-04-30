# L23 follow-up — preference direction ablation

_Report skeleton. Will be filled in after the sweep completes._

## Headline

(To be written after results.)

## Setup

- **Model:** Gemma-3-27b IT, bf16, A100 SXM4-80GB.
- **Probe:** `probe_ridge_L23` from the layer-sweep run (`results/probes/layer_sweep_eot/`, EOT token, 4k-task train, `final_r = 0.798`, α* = 4641). Unit-normalised at load time.
- **Pair set:** 615 unique unordered pairs (from the parent's 955) where neither task overlaps the L23 probe's training set.
- **Per pair:** 3 canonical-order generation seeds + 1 swapped-order seed at temperature 0.7, `max_new_tokens = 64`. Choice extracted via `CompletionChoiceFormat.parse`.
- **Ablation primitive:** orthogonal projection at every token of L23 during the forward pass — `a' = a - (a · d̂) d̂`. Same primitive as the parent.
- **Random controls:** 5 isotropic unit vectors at L23 (numpy seeds 0–4), each evaluated on a fixed 100-pair subset of the 615 (numpy seed 42).

## Cells

| Cell | Layer | Direction | n_pairs |
|---|---|---|---|
| **B0** (parent baseline, reused) | — | — | 615 (filtered from parent's 955) |
| **A_L23_probe** | 23 | L23 ridge probe | 615 |
| **A_L23_random{0..4}** | 23 | 5 isotropic unit vectors | 100 each (matched subset) |

## Sanity tests

(To be filled in.)

## Results

### Headline metric: modal-choice agreement vs B0

(To be filled in.)

### Matched-pair vs full-615 comparison

(To be filled in.)

### Length and refusal audit

(To be filled in.)

## Comparison to parent (L25, L32)

(To be filled in.)

## Interpretation

(To be filled in.)

## Reproducing

```
python -m scripts.L23_followup.run_L23
python -m scripts.preference_direction_ablation.analyze --root experiments/preference_direction_ablation/L23_followup/results
```
