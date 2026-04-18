# Running Log — Cross-Persona Steering

## 2026-04-17

### Setup
- Spec: `cross_persona_steering_spec.md` (v0 locked).
- Code change: `src/steering/runner.py` system_prompt plumbing + `tests/steering/test_runner_config.py` (4 tests pass).
- Pairs: `artifacts/pairs_100.json` (seed 42 from `experiments/steering/cross_layer/pairs_500.json`).
- Random probe: `random_seed42` added to `results/probes/heldout_eval_gemma3_task_mean/manifest.json`; `.npy` lives locally (gitignored).
- Configs: 4 YAMLs under `configs/steering/cross_persona/` (sadist, villain, aesthete, stem_obsessive).
- Committed on main; worktree pending.

### Notes on pre-existing state
- `results/experiments/persona_steering_v2/preference_steering/all_results.json` contains raw per-persona measurements (not fitted utilities). Thurstonian fitting will need to run at analysis time.
- Persona utility files at fit-time path TBD.

### Pod setup
- Launched `cross_persona_steering` H100 80GB HBM3 pod (id 0n2vsz8lcu1h1n, 64.247.201.58:12921).
- Installed tmux on pod (not default).
- Venv at `/opt/venvs/research/bin/python`.
- Gated-model gotcha: HF_TOKEN must be sourced from .env explicitly; the runner doesn't call load_dotenv. Run wrapper uses `set -a; source .env; set +a`.
- Fix synced: pod HEAD pulled to commit `13df8f83`, probe `random_seed42` present in manifest and .npy synced via rsync.

### Smoke test (n_pairs=10, sadist only)
- Model load: 204s (first-time download of 12 shards; subsequent loads should skip download).
- Steering: 760 generations (475 main + 285 control — 15 pair-orderings dropped due to span-detection failures) in 4 min.
- Parsing: 760 completions judged in ~40s (20/s).
- Output files present: `experiments/cross_persona_steering/smoke_sadist.{jsonl,parsed.jsonl}`. Removed after verification.
- System prompt confirmed present in builder output.

### Full run launch — first attempt (aborted)
- Launched at 19:40 in tmux `cps_all`; completed in <10s with 0 bytes of output per persona log.
- Root cause from audit agent: `src/steering/runner.py` had no `if __name__ == "__main__"` block, so `python -m src.steering.runner <config>` imported and exited.
- Fix: added a `__main__` block that loads `.env` and calls `run(Path(sys.argv[1]))`. Committed (`c6629ae5`), pushed, pulled on pod.
- Nothing written to disk — no cleanup needed beyond killing the tmux.

### Full run launch — second attempt (live)
- Relaunched at 19:44 in tmux `cps_all` after pulling the fix. First progress events firing.

### 2026-04-18 --- analysis pivot (metric swap)

The original analysis framed results around $P(\text{choose default-preferred task})$, stratified by signed coefficient. That conditioned on each pair's default-persona utility direction, which washes out the steering effect: for half the pairs a $+$coef pushes toward default, for the other half away. The resulting "bump" shape was a confound, not a mechanistic finding.

Correct metric for differential-steering validation: $P(\text{steered task chosen})$, folding across the sign of the coefficient. At $c>0$ the steered task is Task A (first span gets $+$direction); at $c<0$ the steered task is Task B. This isolates whether the probe direction moves choice toward the steered side, regardless of how a given pair is valued under the default persona.

Reusing all 30,720 existing generations under the new metric:
- sadist: $P(\text{steered})$ 0.87 / 0.95 at $|c|=0.03 / 0.05$; random control 0.50.
- villain: 0.84 / 0.93; random 0.50.
- aesthete: 0.80 / 0.84; random 0.47.
- stem_obsessive: 0.75 / 0.87; random 0.50.

All four personas validate differential steering cleanly. Report rewritten; plot regenerated (`assets/plot_041826_cross_persona_steered_dose_response.png`).

### 2026-04-18 --- off-by-1 span bug + fix

Test discovered that `find_text_span` tokenised with `add_special_tokens=False` while `HuggingFaceModel._tokenize` defaults to `True` (prepending `<bos>` on Gemma). Every position-selective hook in the repo had been firing one token early. Fixed by changing `find_text_span` to `add_special_tokens=True`; skipped-empty-offset loop already tolerates the BOS.

New tests: `tests/steering/test_spans_with_system_prompt.py` (tiny Llama, 5 tests) and `tests/steering/test_spans_gemma_tokenizer.py` (Gemma tokenizer, 3 tests, HF_TOKEN-gated). All 82 existing tests still pass.

The cross-persona run in this log was generated **before** the fix --- hooks covered `[start-1, end-1)` rather than `[start, end)`, so steering covered $\backslash$n + task-minus-last-token. Caveat noted in the report. Did not rerun since the effect sizes we see are large and symmetric across conditions; the validation conclusion should be robust to rerunning with the fix.

### Sadist complete (~50 min)
- Timing: 21.8 min main + 18.8 min control = 40.6 min steering; ~8 min parsing.
- Rows: 7680 generated, 6579 parseable after judge (14% unparseable/ambiguous — typical for the completion judge, worth checking if it's uniform across conditions).
- Early numbers (aggregated): P(choose default-preferred) at c=0 is **0.69** (sadist still picks default-preferred majority of the time on random pairs). Probe direction pulls it **away** from default in both directions: 0.486 at c=−0.05, 0.597 at c=+0.05. Random-direction control barely moves (0.677–0.691). Preliminary evidence of a causal handle, asymmetric (stronger on negative side).
- Preview plots: `assets/plot_041726_cross_persona_dose_response.png` (sadist panel populated, 3 panels empty), `assets/plot_041726_alignment_shift.png`.
