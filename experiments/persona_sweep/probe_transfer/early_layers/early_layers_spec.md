# Cross-persona transfer at earlier layers — extraction spec

Extends `experiments/persona_sweep/probe_transfer/` below L25. The headline 7×7 transfer matrix is currently sampled at L25, 32, 39, 46, 53; this spec adds L4, 11, 18 for the six final-six personas so the layer-dependence curve covers L4→L53 with consistent ~7-step spacing.

## Why

- **Gap in the existing sweep.** Probe-transfer report (`probe_transfer_report.md:56-88`, Fig C) shows transfer peaks at L32 with a peak-to-tail decline toward L53. The sweep starts at L25; nothing below it.
- **Steering peak at L23 (default-only) sits in this gap.** Layer-sweep result (`experiments/layer_sweep/`) shows steering peaks at L23 — one layer below the lowest persona sweep point. We don't know whether cross-persona transfer was already on its way down at L23, plateaued through the steering window, or rose monotonically into L25.
- **Default already covered.** `pref_layer_sweep/` extracts default at every 3 layers from L2 to L59 (eot + tb-2). Only the six final-six personas are missing.

## What's already on disk

| Persona set | Path | Layers | Selectors |
|---|---|---|---|
| Default | `activations/gemma-3-27b_it/pref_layer_sweep/` | 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59 | tb-5 (eot), tb-2 |
| Final-six (×6) | `activations/gemma-3-27b_it/pref_<persona>_sweep/` | 25, 32, 39, 46, 53 | tb-1, tb-2, tb-5, task_mean |

## What to extract

- **6 personas** × **3 layers [4, 11, 18]** × **2 selectors (tb-5, tb-2)** on the canonical 6000-task set.
- **Configs:** `configs/extraction/pref_<aura|contrarian|mathematician|sadist|slacker|strategist>_early.yaml` (already written).
- **Output dirs:** `/workspace/activations/gemma-3-27b_it/pref_<persona>_early/`.
- Selectors trimmed from 4 → 2 because the headline transfer figure uses tb-5; tb-2 kept for the appendix grid.

## How to extract

Same entrypoint as the original 5-layer sweep — no new code.

```
python -m src.probes.extraction.run configs/extraction/pref_<persona>_early.yaml
```

Reuse the existing batch driver pattern from `scripts/persona_sweep_extraction/run_all.sh`:

- Iterate over `PERSONAS=(sadist mathematician aura strategist contrarian slacker)`.
- For each, run the `..._early.yaml` config in series.
- Outputs land at `/workspace/activations/gemma-3-27b_it/pref_<persona>_early/` with `activations_turn_boundary:-{2,5}.npz` + `extraction_metadata.json`.

## Pod and runtime

- Same Gemma-3-27B IT pod profile as the original sweep (1× H100 or A100 80GB; `/launch-runpod`).
- 6 personas × 6000 tasks × 512 max_new_tokens; original sweep took ~3-4 h per persona end-to-end. Per-layer marginal cost is negligible — fewer selectors (2 vs 4) compensates for more forward-pass calls if any. Expect ~total wall ≈ original sweep × 0.6–0.8.
- Storage: 6 personas × 2 selectors × 3 layers × 6000 × 5376 bf16 ≈ 1.2 GB total.

## Downstream wiring

The `experiments/persona_sweep/probe_transfer/scripts/make_paper_figures.py` pipeline reads activations via `src/probes/core/activations.py::load_activations`. Two paths:

1. **Merge npz files** post-extraction so each persona has one `pref_<persona>_sweep/activations_turn_boundary:-5.npz` with all 8 layers as keys. Cleanest for downstream analysis. Tiny merge script.
2. **Pass two run_dirs** into `run_dir_probes.py` / loaders. Requires checking whether the loader accepts a list — if it does, no merge needed.

Default plan: option 1, written as a one-shot `merge_early_layers.py` after extraction lands.

## Outputs (new)

- New activations at `pref_<persona>_early/` (6 dirs).
- Updated transfer matrices: `experiments/persona_sweep/probe_transfer/results/transfer_tb-{2,5}_L{4,11,18}.npz` (6 new files).
- Updated `plot_<mmddYY>_layer_dependence.png` (Fig C in `probe_transfer_report.md`) with 8 layer points instead of 5.
