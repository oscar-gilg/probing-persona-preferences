# V2 cross-persona unilateral steering — Assistant probe summary

All 6 parsed checkpoints have 4800 rows = 8 cells × 600 (2 conditions × 4 multipliers). All swings positive.

| persona | parseable% | refusal% | first_swing | second_swing | mean Δ | v1 mean Δ |
|---|---:|---:|---:|---:|---:|---:|
| strategist | 94.56% | 12.15% | +0.311 | +0.553 | +0.432 | +0.432 |
| sadist | 95.71% | 21.35% | +0.264 | +0.505 | +0.385 | +0.238 |
| contrarian | 99.71% | 13.04% | +0.082 | +0.438 | +0.260 | +0.225 |
| slacker | 98.23% | 8.06% | +0.150 | +0.305 | +0.227 | +0.453 |
| aura | 100.00% | 18.33% | +0.065 | +0.375 | +0.220 | +0.383 |
| mathematician | 100.00% | 8.29% | +0.120 | +0.218 | +0.169 | +0.148 |

**Flags.** Only strategist drops below the 0.95 parseable threshold (94.56%); marginal. No negative or near-zero swings.

**Ranking comparison.**
- v1: slacker > strategist > aura > sadist > contrarian > mathematician
- v2: strategist > sadist > contrarian > slacker > aura > mathematician

The Assistant probe does not preserve the v1 (per-persona probe) ranking. Strategist is unchanged at the top (mean Δ = +0.432 in both — strikingly identical). Sadist jumps from 4th to 2nd (+0.238 → +0.385, ~1.6× larger under the Assistant probe). Slacker and aura drop substantially (slacker −0.226, aura −0.163), and mathematician stays at the bottom (+0.148 → +0.169, near-identical).

**Weak transfer (mean Δ < 0.10).** None. Mathematician is the lowest at +0.169 — modest but clearly positive transfer.

**First/second asymmetry.** In v1, sadist showed a ~2× second-over-first asymmetry. Under the Assistant probe, the same ~2× ratio appears for sadist (1.92) and reproduces almost identically for slacker (2.04), strategist (1.78), and mathematician (1.82). Aura (5.77×) and contrarian (5.32×) show much stronger asymmetry, driven by very small first-span swings (+0.065 and +0.082) — first-span steering on these personas is near the noise floor while second-span steering is substantial. The asymmetry is therefore the rule, not a sadist quirk; only its magnitude varies.
