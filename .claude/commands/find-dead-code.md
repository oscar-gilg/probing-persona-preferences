Find and propose deletion of superseded, outdated, or dead files. Argument: $ARGUMENTS — optional target path or topic to focus on (e.g. `configs/steering`, `scripts/`, `steering infrastructure`). If no argument, scan the whole project.

## Philosophy

Research codebases accumulate dead weight: old configs referencing deleted data, scripts from abandoned experiments, modules superseded by refactors, experiment outputs that were never used. This command finds them. It does NOT auto-delete — it produces a categorised list with evidence for each item, and you confirm what to remove.

## Process

### 1. Understand what's current

Launch parallel subagents to build a picture of what the codebase actively uses:

**Subagent A — Active infrastructure:**
- Read `CLAUDE.md` and `README.md` for the canonical module descriptions
- Read `src/` module `__init__.py` files to understand current public APIs
- Check recent git history (`git log --oneline -30`) for what's actively being worked on
- Identify the current models, datasets, and experiment paradigms

**Subagent B — Import and reference graph:**
- For the target area (or whole `src/` if no target): grep for all imports and cross-references
- For configs: grep for which configs are referenced by scripts, CI, or other configs
- For scripts: check if they import from current `src/` modules or deleted/renamed ones
- For experiment directories: check if their specs/reports reference current infrastructure

**Subagent C — Dead reference detection:**
- Find files referencing deleted paths (e.g. imports of removed modules, paths to files that don't exist)
- Find configs referencing models, datasets, or paths that no longer exist in the project
- Find scripts that import from old module paths or use deprecated APIs
- Check for orphaned test files (testing modules that no longer exist)

### 2. Categorise findings

After subagents complete, organise findings into these categories:

**Category 1: Definitely dead** — Files with broken imports, references to deleted data/modules, or configs for infrastructure that no longer exists. These can be deleted with confidence.

**Category 2: Superseded** — Files that have been replaced by newer versions but aren't broken (e.g. old experiment scripts replaced by a config-driven runner, old configs replaced by newer ones). Evidence: a newer file does the same thing better.

**Category 3: Orphaned experiments** — Experiment directories, scripts, or configs from completed/abandoned experiments that produced no lasting artifacts (no reports, no data referenced elsewhere). Check git log for context on when they were last touched.

**Category 4: Uncertain** — Files that might be dead but you can't confirm without the user. Flag these with your reasoning.

### 3. For each finding, provide evidence

Every item must have:
- **File path**
- **Category** (1-4)
- **Evidence**: What makes it dead/superseded? (e.g. "imports `src.steering.analysis` which was deleted", "references `concept_vectors/` which no longer exists", "superseded by `src/steering/runner.py`")
- **Last meaningful git activity**: When was it last changed in a non-trivial way?

### 4. Present results

Group by category. Within each category, group by directory/topic. Format as a checklist so the user can confirm deletions:

```
## Definitely dead

### configs/steering/
- [ ] `validation_experiment.yaml` — references deleted `probe_data/manifests/`, Llama 8B (not used since Oct 2024)
- [ ] `open_ended_steering_gemma27b.yaml` — references deleted `concept_vectors/`

### scripts/
- [ ] `scripts/old_analysis.py` — imports `src.steering.analysis` (deleted)
```

### 5. Propose cleanup commands

For confirmed deletions, give a single `rm` command (or `rm -rf` for directories) that the user can run. Group by directory to make it easy to review.

## What NOT to flag

- Files that are actively imported or referenced by current code
- Experiment reports and their assets (even old ones — these are research records)
- Test files for modules that still exist
- Config files that are templates or base configs used by other configs
- Anything modified in the last 2 weeks (likely in-progress work)
- Data files in `results/` unless you can confirm nothing references them

## Scope control

If $ARGUMENTS specifies a target:
- Focus subagents on that area and its connections
- Still check cross-references from other parts of the codebase (a config in `configs/steering/` might be referenced by a script in `scripts/`)
- But don't expand into unrelated areas

If no target:
- Scan everything but prioritise areas with the most churn in git history (these accumulate dead weight fastest)
- Start with configs, scripts, and experiment directories — these go stale faster than `src/` modules
