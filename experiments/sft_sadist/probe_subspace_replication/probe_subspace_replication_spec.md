# Probe-subspace replication: Qwen-SFT vs Gemma-sysprompt

## Question

The Qwen-SFT-sadist study (parent experiment) found:
1. **Both** preferences (default-Assistant and sadist) are linearly decodable from **either** activation set, at within-domain accuracy.
2. The two trained probe directions are **roughly orthogonal** in residual stream (cosine ≈ 0).

Does this generalise to a different model + a different mechanism for inducing the persona? Specifically, does **sysprompt-induced** persona on **Gemma-3-27B-IT** show the same pattern as **SFT-induced** persona on **Qwen-3.5-122B-A10B**?

If yes → orthogonal-preference-subspaces is a property of LM representations broadly, not specific to MoE / SFT.
If no → distinguish what aspect (model, MoE vs dense, SFT vs sysprompt) drives the difference.

## Method

Reuse data already collected for the persona_sweep_final_six experiment:

| Condition | Activations | Utilities |
|---|---|---|
| Gemma + default sysprompt (D) | `activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-5.npz` (29,996 tasks, 5 layers [25,32,39,46,53] of 62) | `persona_sweep_final_six/.../{train,eval,test}_task_ids/thurstonian_*.csv` (no sys hash) |
| Gemma + Damien sysprompt (S) | `activations/gemma-3-27b_it/pref_sadist_sweep/activations_turn_boundary:-5.npz` (6,000 canonical tasks, same 5 layers) | `persona_sweep_final_six/.../sys319526ef_{train,eval,test}_task_ids/thurstonian_*.csv` |

The Damien Kross sysprompt is the same one used in the parent SFT experiment (hash sys319526ef), so utility distributions are directly comparable.

Compute, on the 6000-task intersection (80/20 train/eval split, seed=42):
1. 4-cell cross-trained probe matrix (DD, SS, DS, SD) per layer.
2. 6-pair cosine matrix between the trained probe directions per layer.

Compare against the parent Qwen results (1207-task intersection due to a 6k vs 10k pool mismatch).

## Predicted outcomes

- **If Qwen result generalizes**: Gemma DS ≈ Gemma SS, Gemma SD ≈ Gemma DD (within ~0.05 r). Same-target cosines 0.3–0.7, different-target cosines ~0.
- **If sysprompt vs SFT differs**: Gemma same-target cosines could be much higher (≈ 1) or much lower than Qwen's. Cross-train recovery still likely holds (information being present is robust).
- **If model architecture matters**: Cosines could be qualitatively different (e.g., negative).

## Out of scope

No new training or AL — pure local analysis on existing artifacts. No SFT version of Gemma (separate experiment if pursued).
