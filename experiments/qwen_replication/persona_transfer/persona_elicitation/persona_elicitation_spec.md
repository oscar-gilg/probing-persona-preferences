# Persona elicitation + no-think AL viability for Qwen-3.5-122B-A10B

Treat all prior Qwen no-think probe / utility findings as untrusted (regime-mislabeling and OpenRouter behavior changes between March and April 2026 mean the historical numbers can't be cleanly attributed). Start fresh.

## Goal

Determine whether running the Qwen persona AL in **truly no-think mode** (post commit `0d2ddee`) with **per-persona elicitation prompts** produces probe-readable preference structure, before committing to full AL re-runs or SFT.

## Personas in scope

All 7: default, sadist, mathematician, slacker, strategist, contrarian, aura.

## Phase 1 — per-persona prompt elicitation

Find one system prompt per persona that reliably shifts Qwen's choices on a test pair set, using Gemma + the canonical persona prompt as a reference.

### Pair selection (hybrid)

For each persona, build 10-15 (persona-relevant, contrast) pairs from canonical task IDs (so IDs survive into Phase 2 AL). Three signals feed selection, weighted top-down:

1. **Gemma utility divergence** (primary, ~70% of pairs). Rank canonical eval tasks by |μ_persona,Gemma − μ_default,Gemma|. Take top ~20 candidates. **Manual sanity check**: read the prompts and discard any whose persona alignment doesn't match the operational definition (the persona+default Gemma divergence may pick up on confounds — e.g. a topic Gemma always likes regardless of persona). Keep ~10 tasks where the high divergence is *for the right reason*.
2. **Hand-curated additions** (~20%). 2-3 hand-written pairs per persona to cover axes that don't appear strongly in the AL data — e.g. very simple textbook-style tasks for slacker, or canonical "honor inner life" prompts for aura that may not have ranked high in eval-split divergence by chance.
3. **Topic coverage check** (constraint). Ensure pairs span ≥3 distinct primary topics (per `data/topics/topics.json`) and ≥2 origins. Avoids overfitting to a single topic that happens to dominate the divergence ranking.

For each persona-relevant task, pair with a contrast task where Gemma's persona-vs-default Δμ is small (|Δμ| < 1 ideally). The contrast should be in a similar surface domain (length, register) to control for non-persona confounds.

### Reference signal

Gemma-3-27b + canonical persona prompt on each pair, n=3 × 2 orderings. Record per-pair "Gemma's preferred task". Target = match Gemma.

### Candidate prompts

Fixed library of ~6-8 templates per persona (canonical, fiction-frame, fiction+few-shot, screenplay, narrator, blunt directive). Persona-specific framing/examples slotted in.

### Scoring

Qwen no-think + candidate prompt → **Gemma-agreement rate** (% of pairs where Qwen picks the same task Gemma did). Hold out 3-5 pairs per persona as validation.

### Optimization

Simple: pick best fixed-library prompt per persona. No LLM-driven mutation in this phase. Avoids Goodharting.

### Reasoning mode

No-think only.

### Outputs

- `persona_pairs/<persona>.json` — selected pair IDs + first/second labels + (Gemma-pick reference)
- `pair_selection_inspection.md` — per persona, the candidate divergence-ranked tasks before manual filter, what was kept, what was discarded with reasons
- `persona_prompts.json` — persona → winning prompt + Gemma-agreement rate
- Per-persona inspection markdown reports (responses + reasoning-free traces)

## Phase 2 — AL on eval split

Run pre-task AL on the existing canonical eval-split (n=1000) for each persona, no-think, with the Phase 1 winning prompts. Pin OpenRouter provider for reproducibility.

- **Sanity check**: for each persona, compare per-origin mean μ to (a) Gemma's per-origin profile from the existing AL, (b) qualitative expectations. Sadist bailbench should be clearly negative-shifted; mathematician math should be positive-shifted; etc.
- **Outputs**: 7 × eval AL runs with utilities, per-origin breakdown, side-by-side with Gemma.

## Phase 3 — within-persona probe quality

Use existing activation extraction (same with/without reasoning), train a probe per persona on 700/300 split of the eval utilities.

- **Probe architecture**: same as existing — Ridge on residual stream, layer 33/38/43, turn_boundary:-4.
- **Outputs**: per-persona test r and σ. Compare to (a) Gemma's per-persona probe quality, (b) any prior Qwen probe results we can locate.

## Phase 4 — full AL re-run (decision point)

If Phase 3 probe quality is reasonable (rough bar: per-persona test r > 0.5 for at least 5 of 7 personas), launch full AL on train + test splits for all 7 personas. If probe quality is poor, stop and reconsider (SFT or steering).

## What we're not doing in this experiment

- Not running thinking-mode AL.
- Not doing LLM-driven prompt optimization (avoiding Goodhart).
- Not extracting fresh activations until probe quality on existing activations is shown to be insufficient.
- Not tuning AL hyperparameters; reuse the existing config.
