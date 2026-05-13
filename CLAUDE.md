# Preferences

@README.md

MATS 9.0 project on AI model preferences. See `README.md` for the headline result and `CODE_GUIDE.md` for a module-by-module reference.

## Setup

```bash
uv pip install -e ".[dev]"
```

## Code conventions

- NEVER use arbitrary return values. Use `dict[key]` rather than `dict.get(key, default)` — failing fast on a missing key is preferable to a silent wrong value.
- NEVER use None as a semantic value (e.g., "refusal" or "missing"). Use explicit types instead.
- NEVER use placeholder values (e.g., score=0.0 when there is no score). Use proper dataclass variants or explicit flags.
- Do not use hasattr or getattr defensively unless absolutely essential.
- Do not import inside functions. Keep imports at the top of the file.
- Comments and docstrings should add non-obvious information; if removing one wouldn't confuse a future reader, don't write it.
- Before writing new code, check `src/` for an existing function that does what you need (see `CODE_GUIDE.md`).
- No backwards compatibility concerns — remove obsolete code/fields rather than deprecating.
- Don't create classes or interfaces with two or fewer fields — prefer a tuple, dict, or dataclass-without-methods.

## Measurement conventions

When writing experiment code that measures pairwise preferences, reuse the existing infrastructure in `src/measurement/elicitation/` (prompt builders, semantic parser, response formats). Do not invent new prompt templates, response parsers, or hardcoded temperatures.

The canonical pairwise preference template asks the model to complete the chosen task (`max_new_tokens >= 16`), and the choice is determined by a semantic parser (LLM judge). Do not replace this with "respond with only a/b" + startswith parsing — it changes what's being measured.

## Semantic parsing

Never use string-matching heuristics for semantic tasks — use an LLM. Examples that need LLM judgment:
- "Do these two usernames refer to the same person?"
- "Is this comment sharing the author's own profile or someone else's?"
- "Does this text express positive or negative sentiment?"

String normalization and fuzzy matching will miss obvious cases that humans (and LLMs) catch instantly.

## Plotting

- Use meaningful y-axis bounds, not auto-scaled to the data range. If R² values range from 0.65 to 0.80, set the y-axis to start at 0 (or at least a round number like 0.5), not 0.64.
- Same for x-axes and other quantitative axes — anchor to a natural baseline unless there's a strong reason not to.
- When a zoomed view is genuinely needed, make it explicit: broken axis, inset, or note the zoom in the title/caption.

## Environment

- Always load environment variables from `.env` when running scripts that use API clients (`from dotenv import load_dotenv; load_dotenv()`).
- When adding a new dependency, also add it to `pyproject.toml`.
