# Poster Audit Checklist

## 1. Font Size Issues

### Constants (primitives.py → shared everywhere)
`FONT_LARGE = 19`, `FONT_MEDIUM = 16`, `FONT_SMALL = 13`

### Hardcoded sizes that should use constants

| File | Line | Current | Should be | Text |
|------|------|---------|-----------|------|
| generate_poster.py | 48 | `17` (t() default) | `PF_BODY` (16) | Latent trap if size arg omitted |
| generate_poster.py | 87 | `16` literal | `PF_BODY` | Q1 question — same value but not using alias |
| generate_poster.py | 257-258 | `15` | `PF_BODY` (16) | "Generalises to Truth and Harm" body text |
| generate_poster.py | 269-270 | `15` | `PF_BODY` (16) | "Character-Trained Models" body text |
| generate_poster.py | 281 | `19` literal | `PF_LARGE` | Right column placeholder |

### Acceptable exceptions
- `22` in numbered circles (decorative badge)
- `65/28/21` in poster header (unique title sizes)

## 2. Font Family Bug

**The figure strip text won't render in Inter.** The poster SVG imports Inter but doesn't set `text { font-family: 'Inter' }` as CSS default. Primitives text elements don't set font-family explicitly — they inherit from nothing.

**Fix**: Add CSS rule to poster `<style>` block.

## 3. Plot Data Issues

### Hardcoded data (3 of 4 plot scripts)

| Script | Data | Hardcoded? | Issue |
|--------|------|------------|-------|
| plot_probe_quality.py | Probe r values | **YES** | **2 values are WRONG** — see below |
| plot_character_transfer.py | Persona transfer r | **YES** | Could go stale |
| plot_token_heatmap.py | Token scores | **YES** | Could go stale |
| plot_truth_harm.py | EOT scores | No — reads scoring_results.json | OK |

### Data mismatch in probe quality plot

| Model | Metric | Script says | Data file says | Match? |
|-------|--------|-------------|----------------|--------|
| Gemma-3 IT | test_r | 0.864 | 0.864 | Yes |
| Gemma-3 IT | hoo_r | 0.817 | 0.817 | Yes |
| **Gemma-3 PT** | **hoo_r** | **0.627** | **0.539** | **NO — overstated by 0.09** |
| **Qwen3 Emb** | **hoo_r** | **0.618** | **0.560** | **NO — overstated by 0.06** |
| **Qwen3 Emb** | **hoo_std** | **0.102** | **0.133** | **NO** |

### Broken steering plots

- `steering_summary_plots.py` plots 1 and 3 are broken — `isolated_steering/` directory was moved to `old_experiments/`. Existing PNGs are stale and non-reproducible.
- `steering_topic_plot.py` works but `model_manipulation` topic is misclassified as benign.

## 4. Style Conventions (from session history)

### Text
- No m-dashes — use colons/periods
- No speech marks on task prompts
- No "system" prefix labels — just italic instruction text
- Mono font (IBM Plex Mono) for prompt templates only
- No citations on poster
- Minimum words — every word earns its place
- Model name with dashes: "Gemma-3-27B"

### Visual elements
- Background boxes: `#F5F3EF`, no border/stroke
- Open chevron arrows throughout (no filled polygon tips)
- Gauge: 4-segment arc (red, pink, light-green, green), black needle + pivot, colored bg circle at 0.7 opacity
- Probe icon: box with x/y axes + 45° direction arrow + origin dot
- Robot fill: light pastel + dark strokes (green: `#D1FAE5`/`#166534`, red: `#FEE2E2`/`#991B1B`)
- Utilities scale: -10 to +10
- Robot antennas must match across instances
- Diagram templates must be reused — don't reinvent layout per diagram

### Known recurring issues
- **Text overflow**: text repeatedly overflows colored sub-boxes. Always check text width vs container.
- **Antenna clipping**: robot antenna tips clip box edges or overlapping headlines. Calculate actual pixel extent.
- **Arrow direction**: arrows pointed wrong way multiple times. Always verify visually.
- **Axis label overlap**: "utility" axis label overlapped the box it was next to. Eventually removed.
- **Gauge breakage**: rewriting gauge math broke the visual. The working pattern is 4 hand-drawn arc segments, not computed arcs.
- **Vector icon tip**: the 45° arrow chevron was wrong multiple times until simplified to: one arm goes left, one goes down.

## 5. Priority Fixes

### Critical (wrong data)
1. Fix probe quality plot values for Gemma-3 PT and Qwen3 Emb
2. Rewrite plot_probe_quality.py to read from data files

### High (visual bugs)
3. Fix font-family CSS bug in poster SVG
4. Replace `size=15` with `PF_BODY` in middle column (4 occurrences)
5. Change `t()` default from 17 to `PF_BODY`

### Medium (consistency)
6. Replace hardcoded `size=16` with `PF_BODY` in motivation
7. Replace hardcoded `size=19` with `PF_LARGE` in right column
8. Rewrite plot_character_transfer.py to read from data
9. Fix steering_summary_plots.py broken paths

### Low (nice to have)
10. Rewrite plot_token_heatmap.py to read from data
11. Add `model_manipulation` to harmful topics in steering_topic_plot.py
