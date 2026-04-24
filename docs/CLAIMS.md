# Paper claims registry

A unified system so every empirical number in `paper/main.tex` traces back to
the data that produced it, is auditable, and updates consistently across the
paper when the underlying value changes.

## The contract

Every numeric value that ends up in the paper — plot bars, annotations,
captions, prose numerals — is a `Claim`:

```python
@dataclass(frozen=True)
class Claim:
    name: str           # human-readable key, e.g. "Gemma probe heldout r within-topic"
    value: float | int | str
    statement: str      # declarative sentence stating what the number shows
    source: str         # producer file, or "manual: ..."
    used_in: tuple[str, ...]
```

Producers register claims via `ClaimSet.register(name, value, statement, used_in)`
at the same point they pass the number into a plot. No number is rendered
without being registered.

Three views over the registry are generated:

1. **`paper/claims/<producer>.json`** — per-producer sidecar.
2. **`paper/numbers.tex`** — `\newcommand{\macro}{value}` for every claim. `main.tex`
   does `\input{numbers.tex}` and references macros instead of raw numerals.
3. **`paper/claims.md`** — human/agent audit surface. Sortable table.

## Naming

Claim `name` is a short human-readable string. Examples:

- `"Gemma probe heldout r within-topic"`
- `"Qwen probe heldout r within-topic"`
- `"Gemma steering P chosen given coherent"`
- `"Default-to-sadist classification r"`
- `"Sadism Likert under sadist at c=+0.03"`
- `"CREAK truth Cohen's d"`

LaTeX macro names are auto-derived by slugifying:

- `"Gemma probe heldout r within-topic"` → `\gemmaProbeHeldoutRWithinTopic`
- `"Sadism Likert under sadist at c=+0.03"` → `\sadismLikertUnderSadistAtCN003`

`name_to_macro()` in `src/paper/claims.py` is the source of truth for the
slugging rule. Collisions raise at `build_claims.py` time.

## The statement

Every claim carries a `statement`: a full declarative sentence describing what
the value asserts. This is what someone auditing the registry reads first.

Good:
> "A ridge probe trained on Gemma-3-27B residual-stream activations at the
> final prompt token (L32) predicts held-out Thurstonian utilities at Pearson
> r=0.86 within-topic."

Bad:
> "r value for Gemma."

The statement should be specific enough that a subagent can read it, open the
producer's code, and confirm the computation matches what the statement says
the number means.

## Workflow

### Adding a new claim (in a plot producer)

```python
from src.paper.claims import ClaimSet

claims = ClaimSet(source="scripts/qwen_embedding/plot_cross_model_bar.py")

# Load from data, never hardcode.
r = load_pooled_r("results/probes/gemma3_10k_hoo_topic_tb-1_uniform")

r = claims.register(
    name="Gemma probe cross-topic pooled r",
    value=round(r, 3),
    statement=(
        "A ridge probe trained on Gemma-3-27B residual-stream activations "
        "at the final prompt token (L32) predicts held-out preferences at "
        "pooled Pearson r=0.86 under leave-one-topic-out evaluation "
        "(13 folds, predictions stacked across folds)."
    ),
    used_in=["fig:cross-model", "abstract", "sec:shared.intro"],
)

ax.bar(x, r)  # same value

# At end of producer:
claims.save("paper/claims/plot_cross_model_bar.json")
```

### Building

```bash
python scripts/paper/build_claims.py      # merges sidecars, emits numbers.tex + claims.md
bash scripts/build_paper.sh               # builds the PDF
```

### Auditing

**Machine audit (fast, drift-only):**

```bash
python scripts/paper/audit_claims.py      # diffs live vs. committed sidecars
```

**LLM audit (thorough):** spawn a `general-purpose` subagent pointed at one claim
(or one producer) with its data path. The subagent reads `statement`, opens the
producer, traces the computation, and confirms both the number and the phrasing
match what the code does.

## Escape hatches

- **Manual numbers**: numbers whose producer can't be recovered go in
  `scripts/paper/claims/manual_claims.py` with `source="manual: <reason>"`.
  They show up in `claims.md` but are flagged unverifiable.
- **Hard numbers**: numerals that resist macro-ification (too context-specific,
  formatting-heavy, etc.) are listed in `paper/HARD_NUMBERS.md` and stay inline
  in `main.tex`.
- **Orphan producers**: plot scripts that need reimplementation are tracked in
  `paper/TODO_producers.md`.

## File locations

| Path | Status | Purpose |
|---|---|---|
| `src/paper/claims.py` | source | `Claim`, `ClaimSet`, slug/formatters |
| `scripts/paper/build_claims.py` | source | Build `numbers.tex` + `claims.md` |
| `scripts/paper/audit_claims.py` | source | Drift audit against git HEAD |
| `scripts/paper/claims/` | source | Tiny compute scripts for prose-only claims |
| `paper/claims/*.json` | generated (per producer) | Sidecars |
| `paper/numbers.tex` | generated | LaTeX macros |
| `paper/claims.md` | generated | Human/agent audit surface |
| `paper/HARD_NUMBERS.md` | manual | Numbers kept inline in prose |
| `paper/TODO_producers.md` | manual | Orphan figures needing reimplementation |
