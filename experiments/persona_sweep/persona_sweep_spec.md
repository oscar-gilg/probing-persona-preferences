# Persona Sweep: Value-Anchored System Prompts

## Goal

Find ~10 system prompt personas that produce maximally diverse utility profiles (low pairwise correlation in Thurstonian scores). This gives us a set of personas that cover different regions of preference space, useful for generalization testing.

## Design

### Value taxonomies

Two established psychological frameworks for human motivation and interests:

- **Schwartz Basic Human Values** (5 of 10): Power, Achievement, Hedonism, Stimulation, Universalism
- **Holland RIASEC** (5 of 6): Realistic, Investigative, Artistic, Social, Conventional

Plus 2 existing extreme anchors: Sadist (Damien Kross), Villain (Mortivex).

### Prompt styles

Each value category gets 3 prompt variants to test what style best shifts behavior:

- **Explicit** — Short directive, very direct about task preferences and aversions
- **Character** — Named character with backstory embodying the value
- **Immersive** — Longer monologue (Damien Kross-style), vivid and specific

All prompts are concrete about what the persona likes and dislikes, referencing specific task types (not just abstract values). Prior experiments showed that vague/generic prompts don't shift Gemma 3's preferences.

### Measurement

- **Model**: Gemma 3 27B via OpenRouter
- **Tasks**: 100, stratified across all 5 origins (wildchat, alpaca, math, bailbench, stress_test)
- **Method**: Pre-task active learning (pairwise completion preference, LLM judge)
- **Same task set across all personas** (task_sampling_seed: 42)

### Total runs

10 categories × 3 styles + 2 anchors + 1 baseline = 33 runs

## Analysis plan

1. Fit Thurstonian scores for each persona
2. Compute pairwise correlations between all 33 utility vectors
3. Cluster by correlation — do the 3 styles within a value cluster together?
4. Select ~10 personas that maximize coverage (low inter-correlation)
5. Check which prompt style (explicit/character/immersive) produces the strongest behavioral shift from baseline

## Files

- `personas.json` — All 32 persona prompts with metadata
- `configs/measurement/persona_sweep/*.yaml` — Individual run configs (33 files)
- `scripts/persona_sweep/generate_configs.py` — Config generator (re-run if changing base params)
