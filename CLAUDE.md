# Preferences

@README.md

MATS 9.0 project investigating AI model preferences and self-reported valence.

## Docs

- `experiments/` — Self-contained experiment directories (spec, report, assets). See below.
- `docs/logs/` — Running research log and weekly reports with their assets.

## Setup

I usually initialise Claude code from within my venv.

```bash
uv pip install -e ".[dev]"
```

## Structure

- `src/models/` — Abstractions and client for OpenRouter LLM API
- `src/preferences/` — Preference measurement, Thurstonian utility model, and template generation
- `src/task_data/` — Task and dataset structures
- `tests/` — Test suite (`pytest -m "not api"` to skip API-dependent tests)

## Code conventions and style

- NEVER use arbitrary return values. Use `dict[key]` rather than `dict.get(key, default)` — failing fast on a missing key is preferable to a silent wrong value.
- NEVER use None as a semantic value (e.g., "refusal" or "missing"). Use explicit types instead.
- NEVER use placeholder values (e.g., score=0.0 when there is no score). Use proper dataclass variants or explicit flags.
- Do not use hasattr or getattr defensively unless absolutely essential.
- Do not import inside functions. Keep imports at the top of the file.
- Comments and docstrings should add non-obvious information; if removing one wouldn't confuse a future reader, don't write it. Keep both concise.
- Before writing new code, check `src/` for an existing function that does what you need (see README's "For AI agents" section).
- No backwards compatibility concerns — remove obsolete code/fields rather than deprecating.
- Don't create classes or interfaces with two or fewer fields — prefer a tuple, dict, or dataclass-without-methods.

## Files and folders

- All plot file names should be like plot_{mmddYY}_precise_description.png
- To convert markdown to PDF (use unique suffix to avoid overwrites): `cd <dir_with_md_file> && DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pandoc file.md -o file_$(date +%Y%m%d_%H%M%S).pdf --pdf-engine=weasyprint --css=/Users/oscargilg/Dev/MATS/Preferences/docs/pandoc.css`
- `scripts/` is for throwaway/temporary scripts only. Organize them in subdirectories by topic (e.g. `scripts/probes/`, `scripts/topics/`). Delete scripts after use — they should not accumulate.
- Analysis plots that are generated from the analysis folder should go to the analysis folder. The results folder is mostly for measurements.
- To convert PDF to DOCX with embedded images: `soffice --headless --infilter="writer_pdf_import" --convert-to docx:"MS Word 2007 XML" file.pdf`. If images are missing (referenced outside `logs/assets/`), copy them to `logs/assets/` and append with python-docx, then manually move into place.
- Plots referenced in weekly reports or the research log (`docs/logs/`) must be saved to `docs/logs/assets/`. Plots from experiment reports go in `experiments/{name}/assets/`.

## Pairwise accuracy and uniform eval

Pairwise accuracy on actively-sampled pairs (from active learning) is biased downward — active learning oversamples hard pairs. We now use **uniform-sample pairwise evaluation**: 500 uniformly sampled pairs measured once per model, stored in `results/experiments/uniform_eval_*/`. The probe pipeline auto-discovers these by matching `model_short` from the training run's config.

**Status (Apr 2026):** Uniform eval exists for Gemma-3-27b IT only. Gemma-3 PT, Qwen3-Emb-8B, GPT-OSS-120B, and MiniLM still use old actively-sampled pairwise accuracy in some plots (e.g. the cross-model bar). These need their activations synced from the storage pod before uniform eval can be computed for them. Pearson r metrics are unaffected.

Poster plots in `docs/poster/assets/` are prefixed `DEPRECATED_` — the poster was printed with old numbers.

## Plotting conventions

- Use meaningful y-axis bounds, not auto-scaled to the data range. If R² values range from 0.65 to 0.80, set the y-axis to start at 0 (or at least a round number like 0.5), not 0.64. The default matplotlib auto-scaling exaggerates small differences and makes plots misleading.
- Same principle applies to x-axes and any other quantitative axis — anchor to a natural baseline (usually 0) unless there's a strong reason not to (e.g., zoomed inset, or values are all near 1.0 where 0-1 range would hide structure).
- When a zoomed view is genuinely needed, make it explicit: use a broken axis, an inset, or note the zoom in the title/caption.

## Experiments directory

Each experiment is a self-contained directory under `experiments/`:

```
experiments/{name}/
├── {name}_spec.md          # design doc / experiment brief
├── {name}_report.md        # results write-up (created by research loop)
├── assets/                 # plots referenced from {name}_report.md (relative paths)
└── {follow_up}/            # optional sub-experiments that build on this one
    ├── {follow_up}_spec.md
    ├── {follow_up}_report.md
    └── assets/
```

This structure is designed for easy cherry-picking: `git checkout <branch> -- experiments/{name}/` grabs everything for an experiment (spec, report, plots) in one command. Research loops commit to feature branches; periodically cherry-pick the `experiments/` directory to main.

## Bash Operations

Complex bash syntax is hard for Claude Code to permission correctly. Keep commands simple.

Simple operations are fine: `|`, `||`, `&&`, `>` redirects.

For bulk operations on multiple files, use xargs:
- Plain: `ls *.md | xargs wc -l`
- With placeholder: `ls *.md | xargs -I{} head -1 {}`

Avoid string interpolation (`$()`, backticks, `${}`), heredocs, loops, and advanced xargs flags (`-P`, `-L`, `-n`) - these require scripts or simpler alternatives.

**Patterns:**
- File creation: Write tool, not `cat << 'EOF' > file`
- Env vars: `export VAR=val && command`, not `VAR=val command` or `env VAR=val command`
- Bulk operations: `ls *.md | xargs wc -l`, not `for f in *.md; do cmd "$f"; done`
- Parallel/batched xargs: use scripts, not `xargs -P4` or `xargs -L1`
- Per-item shell logic: use scripts, not `xargs sh -c '...'`

## Running Commands

Run scripts via `python <script>` or `pytest`. Bash scripts go in `scripts/` and run with `bash scripts/<script-name>`.

For string interpolation, heredocs, loops, or advanced xargs flags, write a script in `scripts/` instead.

## Claude instructions

- When I ask a question about the repo or data, default to a very concise answer (often one line or a short bullet list). Only expand if I ask for more detail.
- When running tests, scripts, analysis, or debugging: keep me in the loop, explain findings concisely, and ask for clarification or delegate to me often.
- When you install a new package, add it to pyproject.toml if it isn't there.
- Always load environment variables from `.env` when running scripts that use API clients. Use `from dotenv import load_dotenv; load_dotenv()` at the top of scripts.
- Delegate import-checking / `python -c 'import foo'` smoke tests to a subagent (Task tool with subagent_type=general-purpose) so it doesn't interrupt the main conversation flow.
- When you give me a command, always give it to me on a single line. Do not use "\".

## Paper writing style

- Use claim-style paragraph/subsection headers, not procedural ones (`\paragraph{Finding: probes are biased toward the persona they were trained on.}`, not `\paragraph{Result.}`).
- Avoid Setup/Result/Implication/Method-note scaffolding. Lead with the finding; explain inline; drop standalone "Method note" boxes — reviewers know basic stats.
- Plain English over jargon-stacked phrases. "Train-shaped residual" beats "train-bound bias"; "what's left over" beats "leakage"; "controlling for both eval and train" beats "double partial".
- One headline per finding. If an appendix has two findings, preview both in one sentence in the lead, then give each a `\paragraph{}` header.
- Short bolded inline summaries inside a subsection should be promoted to `\paragraph{}` headers — they're already claims, just buried.
- Subsection titles should convey the finding when there is one. "Causal: removing the direction barely changes choices at L25/L32" beats "Causal: inference-time projection on revealed preferences".

## RunPod

Always use zombuul skills (`/launch-runpod`, `/pause-runpod`, `/resume-runpod`, `/provision-pod`) for RunPod operations instead of manual SSH/CLI commands.

### Syncing gitignored data to RunPod

To sync data that's in .gitignore to a RunPod instance:

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "mkdir -p /workspace/Preferences/<path>" && scp -r -P <PORT> -i ~/.ssh/id_ed25519 <local_path>/ root@<IP>:/workspace/Preferences/<path>/
```

Example:
```bash
ssh root@64.247.201.30 -p 10318 -i ~/.ssh/id_ed25519 "mkdir -p /workspace/Preferences/concept_vectors" && scp -r -P 10318 -i ~/.ssh/id_ed25519 concept_vectors/ root@64.247.201.30:/workspace/Preferences/concept_vectors/
```

## Measurement Conventions

When writing experiment scripts that measure pairwise preferences (or any other measurement type that already exists in `src/measurement/`):
- NEVER invent new prompt templates, response parsers, or hardcoded temperatures. Reuse the existing infrastructure in `src/measurement/elicitation/` (prompt builders, semantic parser, response formats).
- The canonical pairwise preference template asks the model to complete the chosen task (max_new_tokens >= 16), and choice is determined by a semantic parser (LLM judge). Do not replace this with "respond with only a/b" + startswith parsing — it changes what's being measured.
- If steering or other constraints require a different elicitation method, document the deviation explicitly in the experiment spec and report, and justify why the standard method can't be used.

## Semantic Parsing Policy

  - NEVER use string matching heuristics for semantic tasks - use an LLM instead
  - Examples of semantic tasks that require LLM judgment, not regex/string matching:
    - "Do these two usernames refer to the same person?"
    - "Is this comment sharing the author's own profile or someone else's?"
    - "Does this text express positive or negative sentiment?"
  - String normalization and fuzzy matching will miss obvious cases that humans (and LLMs) catch instantly

## Research Reflections Workflow

When I send voice-transcribed reflections (usually messy/stream-of-consciousness):
1. Clean up the transcript into clear prose — stay faithful to what was actually said
2. Optionally note relevant recent commits or log entries, but don't over-interpret or fabricate connections

Save to `reflections/YYYY-MM-DD.md`. If multiple reflections on same day, append with a heading.
