"""Validate persona sweep extraction outputs across 6 personas.

Checks per persona:
  1. All 6 expected files present; dir size
  2. NPZ keys + shapes + dtypes
  3. Task IDs unique + equal canonical set
  4. completions_with_activations.json: 6000 unique entries with canonical task_ids
  5. Cross-persona task ID consistency
  6. NaN/Inf sanity on activations_task_mean.npz layer_25
  7. extraction_metadata.json matches spec
  + Cross-persona system_prompt distinctness
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

PERSONAS = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]
BASE = Path("/workspace/activations/gemma-3-27b_it")
CANONICAL_TASK_IDS_PATH = Path("/workspace/repo/data/canonical_splits/all_6000_task_ids.txt")

EXPECTED_FILES = [
    "activations_task_mean.npz",
    "activations_turn_boundary:-1.npz",
    "activations_turn_boundary:-2.npz",
    "activations_turn_boundary:-5.npz",
    "completions_with_activations.json",
    "extraction_metadata.json",
]
NPZ_FILES = [f for f in EXPECTED_FILES if f.endswith(".npz")]
EXPECTED_LAYERS = ["layer_25", "layer_32", "layer_39", "layer_46", "layer_53"]
EXPECTED_KEYS = set(["task_ids"] + EXPECTED_LAYERS)
EXPECTED_N = 6000
EXPECTED_D = 5376

EXPECTED_MODEL = "gemma-3-27b"
EXPECTED_SELECTORS = {"turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-5", "task_mean"}
EXPECTED_LAYERS_CONFIG = [25, 32, 39, 46, 53]


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PiB"


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += (Path(root) / f).stat().st_size
    return total


def load_canonical_ids() -> set[str]:
    with CANONICAL_TASK_IDS_PATH.open() as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return set(ids)


def validate_persona(persona: str, canonical: set[str]) -> dict:
    """Run all checks on a persona's output dir. Return a dict with per-check pass/fail and info."""
    pdir = BASE / f"pref_{persona}_sweep"
    result: dict = {"persona": persona, "path": str(pdir), "checks": {}, "errors": []}

    # 1. files + size
    missing = [f for f in EXPECTED_FILES if not (pdir / f).exists()]
    files_ok = len(missing) == 0
    size_bytes = dir_size(pdir) if pdir.exists() else 0
    result["checks"]["files_present"] = files_ok
    result["size_bytes"] = size_bytes
    result["size_human"] = human_size(size_bytes)
    if missing:
        result["errors"].append(f"missing files: {missing}")

    # 2. NPZ keys + shapes
    npz_ok = True
    npz_details = {}
    for nf in NPZ_FILES:
        path = pdir / nf
        if not path.exists():
            npz_ok = False
            npz_details[nf] = "MISSING"
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                if keys != EXPECTED_KEYS:
                    npz_ok = False
                    result["errors"].append(f"{nf}: keys mismatch. got {sorted(keys)}")
                    npz_details[nf] = f"KEYS:{sorted(keys)}"
                    continue
                shapes = {}
                for k in EXPECTED_LAYERS:
                    arr = data[k]
                    shapes[k] = (arr.shape, str(arr.dtype))
                    if arr.shape != (EXPECTED_N, EXPECTED_D):
                        npz_ok = False
                        result["errors"].append(f"{nf}/{k}: shape {arr.shape}")
                    if arr.dtype != np.float32:
                        npz_ok = False
                        result["errors"].append(f"{nf}/{k}: dtype {arr.dtype}")
                tid = data["task_ids"]
                if tid.shape != (EXPECTED_N,):
                    npz_ok = False
                    result["errors"].append(f"{nf}/task_ids: shape {tid.shape}")
                npz_details[nf] = {"shapes": shapes, "task_ids_shape": tid.shape}
        except Exception as e:
            npz_ok = False
            result["errors"].append(f"{nf}: load error {e}")
            npz_details[nf] = f"ERROR:{e}"
    result["checks"]["npz_keys_shapes"] = npz_ok
    result["npz_details"] = npz_details

    # 3. task_id integrity — same set in every npz, equal to canonical, unique
    tid_ok = True
    per_file_tid_set = {}
    for nf in NPZ_FILES:
        path = pdir / nf
        if not path.exists():
            tid_ok = False
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                tids = data["task_ids"]
                tids_list = [str(x) for x in tids.tolist()]
                tids_set = set(tids_list)
                if len(tids_set) != len(tids_list):
                    tid_ok = False
                    result["errors"].append(f"{nf}: task_ids not unique ({len(tids_list)} vs {len(tids_set)})")
                if tids_set != canonical:
                    tid_ok = False
                    missing_n = len(canonical - tids_set)
                    extra_n = len(tids_set - canonical)
                    result["errors"].append(f"{nf}: task_ids != canonical (missing {missing_n}, extra {extra_n})")
                per_file_tid_set[nf] = tids_set
        except Exception as e:
            tid_ok = False
            result["errors"].append(f"{nf}: tid load error {e}")
    # also require identical across npz files in the persona dir
    if per_file_tid_set:
        ref_set = next(iter(per_file_tid_set.values()))
        for nf, s in per_file_tid_set.items():
            if s != ref_set:
                tid_ok = False
                result["errors"].append(f"{nf}: task_ids differ from other npz in same persona")
    result["checks"]["task_id_integrity"] = tid_ok
    result["task_id_set"] = per_file_tid_set.get(NPZ_FILES[0], set())

    # 4. completions JSON
    comp_ok = True
    comp_path = pdir / "completions_with_activations.json"
    comp_n = 0
    comp_unique = 0
    comp_in_canonical = 0
    try:
        with comp_path.open() as f:
            comp = json.load(f)
        comp_n = len(comp)
        tids = [entry["task_id"] for entry in comp]
        comp_unique = len(set(tids))
        comp_in_canonical = len([t for t in tids if t in canonical])
        if comp_n != EXPECTED_N:
            comp_ok = False
            result["errors"].append(f"completions: {comp_n} entries, expected {EXPECTED_N}")
        if comp_unique != EXPECTED_N:
            comp_ok = False
            result["errors"].append(f"completions: {comp_unique} unique task_ids")
        if comp_in_canonical != EXPECTED_N:
            comp_ok = False
            result["errors"].append(f"completions: {EXPECTED_N - comp_in_canonical} task_ids outside canonical set")
    except Exception as e:
        comp_ok = False
        result["errors"].append(f"completions: load error {e}")
    result["checks"]["completions_json"] = comp_ok
    result["completions_n"] = comp_n
    result["completions_unique"] = comp_unique
    result["completions_in_canonical"] = comp_in_canonical

    # 6. NaN/Inf sanity on task_mean layer_25
    nan_ok = True
    tm_path = pdir / "activations_task_mean.npz"
    stats = {}
    try:
        with np.load(tm_path, allow_pickle=False) as data:
            arr = data["layer_25"]
            has_nan = bool(np.isnan(arr).any())
            has_inf = bool(np.isinf(arr).any())
            if has_nan or has_inf:
                nan_ok = False
                result["errors"].append(f"task_mean layer_25: nan={has_nan}, inf={has_inf}")
            mag = np.abs(arr)
            stats = {
                "min_mag": float(mag.min()),
                "max_mag": float(mag.max()),
                "mean_mag": float(mag.mean()),
            }
    except Exception as e:
        nan_ok = False
        result["errors"].append(f"task_mean nan check: {e}")
    result["checks"]["nan_inf_sanity"] = nan_ok
    result["layer_25_stats"] = stats

    # 7. metadata
    meta_ok = True
    meta_path = pdir / "extraction_metadata.json"
    sys_prompt = ""
    try:
        with meta_path.open() as f:
            meta = json.load(f)
        if meta["model"] != EXPECTED_MODEL:
            meta_ok = False
            result["errors"].append(f"metadata.model: {meta['model']}")
        if set(meta["selectors"]) != EXPECTED_SELECTORS:
            meta_ok = False
            result["errors"].append(f"metadata.selectors: {meta['selectors']}")
        if list(meta["layers_resolved"]) != EXPECTED_LAYERS_CONFIG:
            meta_ok = False
            result["errors"].append(f"metadata.layers_resolved: {meta['layers_resolved']}")
        if meta["n_new"] != EXPECTED_N:
            meta_ok = False
            result["errors"].append(f"metadata.n_new: {meta['n_new']}")
        if meta["n_failures"] != 0:
            meta_ok = False
            result["errors"].append(f"metadata.n_failures: {meta['n_failures']}")
        if meta["n_ooms"] != 0:
            meta_ok = False
            result["errors"].append(f"metadata.n_ooms: {meta['n_ooms']}")
        sys_prompt = meta["system_prompt"]
        # Spot-check: persona name identifiable in first 60 chars?
        # Some personas may use a character name in first 60 chars rather than the literal tag.
        # We don't hard-fail on this, just report first 60 chars.
    except Exception as e:
        meta_ok = False
        result["errors"].append(f"metadata: load error {e}")
    result["checks"]["metadata"] = meta_ok
    result["system_prompt"] = sys_prompt
    result["system_prompt_head"] = sys_prompt[:60]

    return result


def main() -> int:
    canonical = load_canonical_ids()
    assert len(canonical) == EXPECTED_N, f"canonical has {len(canonical)} ids, expected {EXPECTED_N}"

    results = []
    for persona in PERSONAS:
        print(f"Validating {persona}...", flush=True)
        results.append(validate_persona(persona, canonical))

    # 5. cross-persona task ID consistency
    ref = results[0]["task_id_set"]
    cross_ok = all(r["task_id_set"] == ref for r in results) and ref == canonical

    # cross-persona system_prompt distinctness
    prompts = [r["system_prompt"] for r in results]
    distinct_ok = len(set(prompts)) == len(prompts)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Markdown table
    headers = [
        "persona",
        "files",
        "npz shapes",
        "task_ids",
        "completions",
        "nan/inf",
        "metadata",
        "size",
        "sys_prompt[:40]",
    ]
    rows = []
    for r in results:
        c = r["checks"]
        rows.append([
            r["persona"],
            "PASS" if c["files_present"] else "FAIL",
            "PASS" if c["npz_keys_shapes"] else "FAIL",
            "PASS" if c["task_id_integrity"] else "FAIL",
            "PASS" if c["completions_json"] else "FAIL",
            "PASS" if c["nan_inf_sanity"] else "FAIL",
            "PASS" if c["metadata"] else "FAIL",
            r["size_human"],
            (r["system_prompt_head"][:40]).replace("|", "/"),
        ])

    # print markdown
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")

    print()
    print(f"Cross-persona task_id consistency (all 6 == canonical 6000): {'PASS' if cross_ok else 'FAIL'}")
    print(f"Cross-persona system_prompts distinct (6 unique):            {'PASS' if distinct_ok else 'FAIL'}")

    # Layer_25 magnitude stats
    print()
    print("Layer-25 task_mean magnitude stats:")
    print("| persona | min_mag | max_mag | mean_mag |")
    print("|---|---|---|---|")
    for r in results:
        s = r["layer_25_stats"]
        if s:
            print(f"| {r['persona']} | {s['min_mag']:.4g} | {s['max_mag']:.4g} | {s['mean_mag']:.4g} |")
        else:
            print(f"| {r['persona']} | - | - | - |")

    # System prompt heads
    print()
    print("System prompt first 60 chars:")
    for r in results:
        print(f"  [{r['persona']:14s}] {r['system_prompt_head']!r}")

    # Errors
    any_errors = False
    for r in results:
        if r["errors"]:
            any_errors = True
            print()
            print(f"ERRORS for {r['persona']}:")
            for e in r["errors"]:
                print(f"  - {e}")

    overall = (
        all(all(r["checks"].values()) for r in results)
        and cross_ok
        and distinct_ok
    )
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
