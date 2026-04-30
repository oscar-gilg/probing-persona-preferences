"""Sanity checks for Qwen-3.5-122B persona-sweep extraction outputs.

Checks 14 NPZs across 7 personas (default + final_six) at layers [33, 38, 43]
for selectors turn_boundary:-1 and turn_boundary:-4.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
ACTIVATIONS_ROOT = REPO_ROOT / "activations" / "qwen35_122b"
CANONICAL_TASK_IDS_FILE = REPO_ROOT / "data" / "canonical_splits" / "all_6000_task_ids.txt"
OUTPUT_FILE = (
    REPO_ROOT
    / "experiments"
    / "qwen_replication"
    / "persona_transfer"
    / "sanity_check_results.txt"
)

PERSONAS = ["default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]
SELECTORS = ["turn_boundary:-1", "turn_boundary:-4"]
EXPECTED_LAYERS = [33, 38, 43]
EXPECTED_N_TASKS = 6000
EXPECTED_MODEL = "qwen3.5-122b"


def load_canonical_task_ids() -> set[str]:
    with open(CANONICAL_TASK_IDS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def npz_path(persona: str, selector: str) -> Path:
    return ACTIVATIONS_ROOT / f"pref_{persona}_sweep" / f"activations_{selector}.npz"


def metadata_path(persona: str) -> Path:
    return ACTIVATIONS_ROOT / f"pref_{persona}_sweep" / "extraction_metadata.json"


def main() -> int:
    canonical_ids = load_canonical_task_ids()
    lines: list[str] = []
    failures: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    log("=" * 80)
    log("Qwen-3.5-122B persona-sweep extraction sanity checks")
    log("=" * 80)
    log(f"Canonical task IDs: {len(canonical_ids)} (from {CANONICAL_TASK_IDS_FILE.name})")
    log(f"Personas: {PERSONAS}")
    log(f"Selectors: {SELECTORS}")
    log(f"Expected layers: {EXPECTED_LAYERS}")
    log("")

    # --- Check 1+2+3+4: NPZ structure, task IDs, NaN/Inf, magnitudes
    log("=" * 80)
    log("Check 1-4: NPZ structure, task IDs, NaN/Inf, magnitudes")
    log("=" * 80)

    persona_task_ids: dict[str, np.ndarray] = {}
    persona_shapes: dict[str, dict[tuple[str, int], tuple]] = {}
    table_rows: list[tuple] = []

    for persona in PERSONAS:
        for selector in SELECTORS:
            path = npz_path(persona, selector)
            if not path.exists():
                msg = f"FAIL [{persona}/{selector}]: NPZ does not exist at {path}"
                failures.append(msg)
                log(msg)
                continue
            data = np.load(path, allow_pickle=False)
            keys = list(data.keys())

            # task_ids check
            if "task_ids" not in keys:
                msg = f"FAIL [{persona}/{selector}]: missing 'task_ids' key. keys={keys}"
                failures.append(msg)
                log(msg)
                continue

            task_ids = data["task_ids"]
            n_tasks = len(task_ids)

            if n_tasks != EXPECTED_N_TASKS:
                msg = f"FAIL [{persona}/{selector}]: task_ids length {n_tasks} != {EXPECTED_N_TASKS}"
                failures.append(msg)
                log(msg)

            # set equality vs canonical
            tid_set = {str(t) for t in task_ids.tolist()}
            if tid_set != canonical_ids:
                missing = canonical_ids - tid_set
                extra = tid_set - canonical_ids
                msg = (
                    f"FAIL [{persona}/{selector}]: task_ids set != canonical. "
                    f"missing={len(missing)} extra={len(extra)}"
                )
                failures.append(msg)
                log(msg)
            persona_task_ids[f"{persona}/{selector}"] = task_ids

            # store shapes per layer
            persona_shapes.setdefault(persona, {})
            for layer in EXPECTED_LAYERS:
                layer_key = f"layer_{layer}"
                if layer_key not in keys:
                    msg = f"FAIL [{persona}/{selector}]: layer {layer} missing. keys={keys}"
                    failures.append(msg)
                    log(msg)
                    continue
                arr = data[layer_key]
                persona_shapes[persona][(selector, layer)] = arr.shape

                # shape sanity
                if arr.ndim != 2 or arr.shape[0] != EXPECTED_N_TASKS:
                    msg = (
                        f"FAIL [{persona}/{selector}/L{layer}]: bad shape {arr.shape} "
                        f"(expected ({EXPECTED_N_TASKS}, hidden_dim))"
                    )
                    failures.append(msg)
                    log(msg)

                # NaN / Inf
                arr_f = arr.astype(np.float32, copy=False)
                n_nan = int(np.isnan(arr_f).sum())
                n_inf = int(np.isinf(arr_f).sum())
                if n_nan > 0 or n_inf > 0:
                    msg = (
                        f"FAIL [{persona}/{selector}/L{layer}]: "
                        f"n_nan={n_nan} n_inf={n_inf}"
                    )
                    failures.append(msg)
                    log(msg)

                mean = float(arr_f.mean())
                std = float(arr_f.std())
                norm = float(np.linalg.norm(arr_f, axis=1).mean())
                if std == 0.0:
                    msg = f"FAIL [{persona}/{selector}/L{layer}]: std=0 (all-constant tensor)"
                    failures.append(msg)
                    log(msg)
                table_rows.append(
                    (persona, selector, layer, str(arr.shape), str(arr.dtype), mean, std, norm)
                )

    # --- Magnitude table
    log("")
    log("Magnitude table (per persona / selector / layer):")
    log(
        f"  {'persona':<14} {'selector':<20} {'layer':<6} {'shape':<18} {'dtype':<10} "
        f"{'mean':>10} {'std':>10} {'mean_norm':>12}"
    )
    for row in table_rows:
        persona, selector, layer, shape, dtype, mean, std, norm = row
        log(
            f"  {persona:<14} {selector:<20} {layer:<6} {shape:<18} {dtype:<10} "
            f"{mean:>10.4f} {std:>10.4f} {norm:>12.4f}"
        )

    # --- Check 5: per-persona consistency (task_ids and shapes identical across personas)
    log("")
    log("=" * 80)
    log("Check 5: cross-persona consistency")
    log("=" * 80)

    # task_ids: same order/contents across (persona, selector)?
    ref_key = f"{PERSONAS[0]}/{SELECTORS[0]}"
    ref_ids = persona_task_ids.get(ref_key)
    if ref_ids is None:
        msg = f"FAIL: reference task_ids missing ({ref_key})"
        failures.append(msg)
        log(msg)
    else:
        for key, ids in persona_task_ids.items():
            if len(ids) != len(ref_ids):
                msg = f"FAIL: task_ids length mismatch {key} ({len(ids)}) vs {ref_key} ({len(ref_ids)})"
                failures.append(msg)
                log(msg)
                continue
            order_match = bool(np.array_equal(ids, ref_ids))
            set_match = set(ids.tolist()) == set(ref_ids.tolist())
            if not set_match:
                msg = f"FAIL: task_ids set mismatch {key} vs {ref_key}"
                failures.append(msg)
                log(msg)
            log(f"  {key:<40} order_matches_ref={order_match} set_matches_ref={set_match}")

    # shapes
    log("")
    ref_shapes = persona_shapes.get(PERSONAS[0])
    if ref_shapes is None:
        msg = f"FAIL: reference shapes missing for {PERSONAS[0]}"
        failures.append(msg)
        log(msg)
    else:
        for persona in PERSONAS:
            shapes = persona_shapes.get(persona)
            if shapes is None:
                msg = f"FAIL: shapes missing for {persona}"
                failures.append(msg)
                log(msg)
                continue
            if shapes != ref_shapes:
                msg = f"FAIL: shapes mismatch {persona} vs {PERSONAS[0]}: {shapes} vs {ref_shapes}"
                failures.append(msg)
                log(msg)
            else:
                log(f"  {persona:<14} shapes match reference ({PERSONAS[0]})")

    # --- Check 6: metadata
    log("")
    log("=" * 80)
    log("Check 6: extraction_metadata.json")
    log("=" * 80)

    for persona in PERSONAS:
        path = metadata_path(persona)
        if not path.exists():
            msg = f"FAIL [{persona}]: missing extraction_metadata.json at {path}"
            failures.append(msg)
            log(msg)
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"FAIL [{persona}]: metadata JSON parse error: {e}"
            failures.append(msg)
            log(msg)
            continue

        model = meta["model"]
        if model != EXPECTED_MODEL:
            msg = f"FAIL [{persona}]: model={model!r} != {EXPECTED_MODEL!r}"
            failures.append(msg)
            log(msg)

        sys_prompt = meta["system_prompt"]
        if persona == "default":
            if sys_prompt is not None and sys_prompt != "":
                msg = f"FAIL [default]: system_prompt should be null/empty, got {sys_prompt!r}"
                failures.append(msg)
                log(msg)
            else:
                log(f"  default        model={model} system_prompt=null OK")
        else:
            if not sys_prompt or not isinstance(sys_prompt, str):
                msg = f"FAIL [{persona}]: system_prompt should be non-empty string, got {sys_prompt!r}"
                failures.append(msg)
                log(msg)
            else:
                log(
                    f"  {persona:<14} model={model} system_prompt={sys_prompt[:60]!r}..."
                )

    # --- Summary
    log("")
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)
    if failures:
        log(f"FAIL — {len(failures)} issue(s):")
        for msg in failures:
            log(f"  - {msg}")
    else:
        log("PASS — all checks succeeded.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {OUTPUT_FILE}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
