"""Summarise probe held-out Pearson r per (position, layer) for Gemma-3-27B and Qwen-3.5-122B.

Used to populate the table in App.~\\ref{app:token-selection}.
"""

import json
from pathlib import Path

REPO = Path(".")

# Display-name -> internal selector tag. Order matches the appendix diagram.
POSITIONS = [
    ("end-of-turn",         "tb-5"),
    ("role-marker",         "tb-2"),
    ("final prompt token",  "tb-1"),
    ("task-averaged",       "task_mean"),
]

# Gemma: heldout_eval_gemma3_<tag>
GEMMA_DIRS = {tag: REPO / f"results/probes/heldout_eval_gemma3_{tag}" for _, tag in POSITIONS}

# Qwen: only has tb-1 (final prompt token) and tb-4 available; others missing.
QWEN_DIRS = {
    "tb-1":      REPO / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1",
    "tb-4":      REPO / "results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4",
}


def load(manifest_path: Path) -> dict[int, float] | None:
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    return {p["layer"]: p["final_r"] for p in manifest["probes"] if p["method"] == "ridge"}


def summarise(model_name: str, dirs: dict[str, Path]) -> None:
    print(f"\n### {model_name}")
    rows = {}
    layers: set[int] = set()
    for tag, d in dirs.items():
        rs = load(d / "manifest.json")
        if rs is None:
            print(f"  [missing] {tag}: {d}")
            continue
        rows[tag] = rs
        layers.update(rs.keys())
    layers_sorted = sorted(layers)
    header = "layer | " + " | ".join(f"{tag:>10s}" for tag in rows.keys())
    print(header)
    print("-" * len(header))
    for L in layers_sorted:
        print(f"  L{L:>3d}  | " + " | ".join(
            f"{rs.get(L, float('nan')):>10.3f}" for rs in rows.values()
        ))
    # Best layer and value per tag
    print("\nbest-layer / r per position:")
    for tag, rs in rows.items():
        best_L = max(rs, key=rs.get)
        print(f"  {tag:10s}  L{best_L}: r = {rs[best_L]:.3f}")


def main():
    gemma_map = {tag: GEMMA_DIRS[tag] for _, tag in POSITIONS}
    summarise("Gemma-3-27B-IT", gemma_map)
    summarise("Qwen-3.5-122B", QWEN_DIRS)


if __name__ == "__main__":
    main()
