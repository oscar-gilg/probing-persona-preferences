# Persona Augmentation Experiment

Does augmenting the 10k noprompt probe with a small amount of persona data improve OOD generalization?

## Motivation

The midway bias analysis showed that training on 2-3 personas is sufficient, and the donor ranking (utility-partialled) identified sadist, villain, and midwest as interesting donors: sadist/villain are the best after controlling for utility similarity, midwest is the worst. This experiment tests whether appending 1000 persona-conditioned training samples to the full 10k noprompt dataset shifts the probe direction in a useful way.

## Design

**Conditions** (per selector, per layer):
1. **Baseline**: Ridge probe trained on 10k noprompt, alpha swept on 4k heldout noprompt
2. **Augmented**: Ridge probe trained on 10k noprompt + 1000 persona (split A), alpha swept on 4k heldout noprompt + 500 persona (split B)

**Personas**: sadist, villain, midwest

**Selectors**: turn_boundary:-2, turn_boundary:-5

**Layers**: [25, 32, 39, 46, 53]

**Evaluation**:
- Noprompt test set (from 4k heldout, half used for sweep, half for test — same as existing probe)
- Persona test sets: split C (1000 tasks each) for sadist, villain, midwest, and any other MRA personas with available scores
- Metrics: Pearson r, R², per-topic breakdown

**Key questions**:
1. Does the optimal alpha change when persona data is added?
2. Does augmentation improve OOD prediction on the donor persona's test set?
3. Does augmentation improve OOD prediction on *other* persona test sets?
4. Does augmentation hurt noprompt test performance?

## Data

### Training
- **Noprompt**: 10k scores from `results/experiments/main_probes/gemma3_10k_run1/...`, activations from `activations/gemma_3_27b_turn_boundary_sweep/`
- **Persona split A** (1000 tasks): scores from `results/experiments/mra_exp2/...` (villain, aesthete, midwest) or `results/experiments/mra_exp3/...` (sadist), activations from `activations/gemma_3_27b_{persona}_tb/` (NEED EXTRACTION)

### Alpha sweep
- **Noprompt**: 4k heldout from `results/experiments/main_probes/gemma3_4k_pre_task/...`
- **Persona split B** (500 tasks): same measurement runs, split B

### Test
- **Noprompt**: other half of 4k heldout (split by eval_split_seed=42)
- **Persona split C** (1000 tasks): same measurement runs, split C
- **Cross-persona split C**: all 7 non-default personas

## Steps

1. **Extract persona activations** (GPU required):
   ```
   python -m src.probes.extraction.run configs/extraction/mra_tb_sadist.yaml
   python -m src.probes.extraction.run configs/extraction/mra_tb_villain.yaml
   python -m src.probes.extraction.run configs/extraction/mra_tb_midwest.yaml
   ```
   Output: `activations/gemma_3_27b_{sadist,villain,midwest}_tb/activations_turn_boundary:-{2,5}.npz`

2. **Sync activations to local** (after extraction on pod):
   ```
   scp from pod: activations/gemma_3_27b_{sadist,villain,midwest}_tb/
   ```

3. **Run augmentation experiment** (local, no GPU needed):
   ```
   python -m scripts.persona_augmentation.run_experiment
   ```
   This trains baseline + augmented probes for all conditions and saves results JSON.

4. **Plot and analyze**:
   ```
   python -m scripts.persona_augmentation.plot_results
   ```

5. **Write report** in this directory.
