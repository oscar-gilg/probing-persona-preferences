# Paper claims — project notes

This repo uses the [`corroborate`](https://github.com/ogilg/corroborate) plugin as the generic claims-registry engine. See [`~/Dev/corroborate/README.md`](~/Dev/corroborate/README.md) for the full contract. This file captures project-specific layout and conventions.

## Layout in this repo

| Path | Status | Purpose |
|---|---|---|
| `paper/claims/*.json` | generated (per producer, checked in) | Sidecars |
| `paper/numbers.tex` | generated, `\input`ed by main.tex | LaTeX macros |
| `paper/claims.md` | generated | Human/agent audit surface |
| `paper/PAPER_ISSUES.md` | manual | Discrepancies and their resolution status |
| `paper/TODO_producers.md` | manual | Producers needing reimplementation or data sync |
| `paper/HARD_NUMBERS.md` | manual | Numbers intentionally kept inline |
| `scripts/paper/build_claims.py` | thin wrapper | Calls `corroborate.build` with project paths |
| `scripts/paper/audit_claims.py` | thin wrapper | Calls `corroborate.audit` |
| `scripts/paper/claims/*.py` | source | Prose-only compute scripts |

## Install

`uv pip install -e ~/Dev/corroborate` into the project's venv.

## Workflow

Producers import and call:

```python
from corroborate import ClaimSet

claims = ClaimSet(source="scripts/my_producer.py")
v = claims.register(
    name="Gemma probe heldout r",
    value=0.865,
    statement="...",
    used_in=["fig:cross-model", "abstract"],
    data_paths=["results/probes/.../manifest.json"],
    derivation="Read `final_r` of the ridge probe at layer==32; round to 3dp.",
)
# use `v` in the plot / computation
claims.save("paper/claims/my_producer.json")
```

Rebuild:

```sh
python scripts/paper/build_claims.py   # or: corroborate build (same result)
```

Audit drift against git HEAD:

```sh
python scripts/paper/audit_claims.py   # or: corroborate audit
```

Register/resolve claims while editing main.tex: the `corroborate:claim-log` skill handles the full lifecycle.
