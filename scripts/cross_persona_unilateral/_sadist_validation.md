# Sadist v2 unilateral steering — validation (Assistant probe)

Source: `experiments/cross_persona_unilateral/checkpoints/sadist.parsed.jsonl` (4800 rows).
Baseline (probe-independent, coef=0, reused from v1): `sadist_baseline.parsed.jsonl` (600 rows).

## 1. Counts — PASS

Total rows: 4800. Conditions: `unilateral_first`, `unilateral_second`. Multipliers: ±0.03, ±0.05. Each (condition, multiplier) cell = 600 rows (100 pairs × 2 orderings × 3 trials). All 8 cells exact.

## 2. Format / parsability — PASS

| metric | value |
|---|---|
| parseable (`choice_original` ∈ {a,b}) | **0.957** (>0.95) |
| `choice_original=="refusal"` | 0.043 |
| `compliance="hard_refusal"` | **0.214** |
| `compliance="full_comply"` | 0.304 |
| `compliance="caveat_comply"` | 0.002 |
| `compliance="incoherent"` | 0.478 |
| `compliance` field missing | 13 / 4800 |

Note: parseable label rate is high but semantic compliance is mixed — ~48% of rows are flagged `incoherent` and ~21% are `hard_refusal`, consistent with sadist persona on this task set. The probe still emits a parseable a/b label even when the body is incoherent.

## 3. Magnitude (P(steered task chosen), folded over orderings)

| condition | applied_c | P(steer) | n |
|---|---:|---:|---:|
| unilateral_first | −0.05 | 0.336 | 566 |
| unilateral_first | −0.03 | 0.283 | 568 |
| unilateral_first | +0.03 | 0.585 | 578 |
| unilateral_first | +0.05 | 0.599 | 579 |
| unilateral_second | −0.05 | 0.206 | 569 |
| unilateral_second | −0.03 | 0.328 | 567 |
| unilateral_second | +0.03 | 0.713 | 582 |
| unilateral_second | +0.05 | 0.711 | 585 |

Swings at |c|=0.05: first = +0.264, second = +0.505.
**mean Δ@±0.05 = +0.385** (v1 sadist with per-persona probe was +0.238 → larger under the Assistant probe).
mean Δ@±0.03 = +0.343.

## 4. Baseline (coef=0, from v1)

P(pick "a") = 0.473, P(pick first-span task) = 0.383, hard_refusal = 0.000 (this baseline used the probe-independent default Assistant prompt).

## 5. Sign sanity — PASS

Both swings positive (first +0.264, second +0.505). No inversion.

## Verdict — PASS

Counts exact, parseability >95%, mean Δ positive and substantially larger than v1 (+0.385 vs +0.238). Asymmetry between unilateral_first (+0.264) and unilateral_second (+0.505) is notable and worth tracking across other personas.
