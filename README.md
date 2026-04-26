# Preferences

**Personas Use a Shared Preference Vector** — MATS 9.0 project with Patrick Butlin.

> Draft paper: [`paper/main.pdf`](paper/main.pdf). Numbers in this README track the draft and may shift before submission.

## TL;DR

A single linear direction in Gemma-3-27B's residual stream predicts and steers pairwise preferences, and the **same direction is used by qualitatively different personas** — including ones whose overt values are partially or fully inverted (villain, sadist). Personas don't carry their own preference circuitry; they share one mechanism and read it out differently.

## Setup

We elicit *revealed* preferences: the model is shown two tasks and picks which one to complete. Pairs are actively sampled; a Thurstonian utility model converts pairwise choices into per-task scalar utilities. A Ridge probe on residual-stream activations targets these utilities. We test the probe both as a **classifier** (does it predict held-out preferences?) and as a **steering vector** (does intervening on it shift behaviour?).

## Findings

**1. A linear preference direction exists and is causal.**
The probe predicts held-out utilities at $r \approx 0.86$ within-topic and $r \approx 0.77$ pooled under leave-one-topic-out, beating a sentence-transformer content baseline. As a contrastive steering vector, it drives $P(\text{chose steered task} \mid \text{responded})$ across nearly the full $[0, 1]$ range at small coefficients ($|c| \le 0.05$). A random direction at the same magnitude does nothing. The causal window is sharply localised (peak at L23, six layers before the probe-quality peak) — linear decodability and causal efficacy decouple.

**2. The direction is shared across personas.**
A probe trained on the default Assistant transfers — as classifier *and* as steering vector — to system-prompted personas (sadist, villain, mathematician, slacker, …) and to character-fine-tuned variants (OpenCharacter LoRAs). Steering doesn't pull behaviour toward a fixed attractor: it amplifies whichever persona is active. Under the sadist, $+$ means *more* sadist; under the default, $+$ has no measurable effect on sadism. Shared mechanism, persona-modulated readout.

**3. The direction is evaluative, not descriptive.**
It tracks preference shifts induced by system prompts, single-sentence biographical context, and role-played harm/truth/politics framings at $r \approx 0.85$ on targeted tasks — while the content baseline does not move. It also discriminates evaluative content at multiple token positions (final prompt token, turn boundary, task-averaged), consistent with the model compiling an evaluative summary and re-reading it across the forward pass.

## Implications

- **Persona science.** Evidence against the "Shoggoth" view of the persona-selection model: there's no persona-independent preference attractor underneath each persona. Representations are persona-instrumental.
- **Interpretability.** A linear feature that looks fundamental in one persona can carry opposite or absent content under another. Single-persona probing over-indexes on persona-instrumental features.
- **AI welfare.** Evaluative representations with a causal role in choice are functionally central in theories of moral patiency (Long et al., 2024). Whether this finding carries welfare weight depends on further commitments we don't take a stand on.
- **AI safety.** A direction trained on preference, not refusal, partially overrides refusal guardrails at small steering coefficients — the two share more representational machinery than their training objectives suggest.

## Code pointers

- `src/probes/` — activation extraction, probe training, evaluation
- `src/steering/` — composable steering primitives (hooks, calibration, analysis)
- `src/measurement/` — pairwise choice elicitation, LLM judges
- `src/fitting/` — Thurstonian / TrueSkill utility models
- `experiments/` — self-contained per-experiment dirs (spec, report, assets)
- `paper/` — draft, claim registry (`claims/`), figures (`figures/panels/`)

For AI agents: skim the relevant module under `src/` before writing new extraction, embedding, or probe-training code — the functionality likely already exists.
