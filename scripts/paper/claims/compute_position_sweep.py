"""Register held-out Pearson r per (position, model) for App.~\\ref{app:token-selection}.

For each named token position (end-of-turn, role-marker, final prompt token,
task-averaged), we take the best held-out r across the layer sweep and register
it as a claim. Qwen was only swept at the final prompt token within this named
set; we register that single number.

Data sources:
  results/probes/heldout_eval_gemma3_{tb-5,tb-2,tb-1,task_mean}/manifest.json
  results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/manifest.json

Run:
  python scripts/paper/claims/compute_position_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]


def best_layer_r(manifest_path: Path) -> tuple[int, float]:
    manifest = json.loads(manifest_path.read_text())
    ridge_probes = [p for p in manifest["probes"] if p["method"] == "ridge"]
    best = max(ridge_probes, key=lambda p: p["final_r"])
    return int(best["layer"]), round(float(best["final_r"]), 3)


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_position_sweep.py")

    # (display-name, sidecar slug for macro, sweep dir tag)
    gemma_positions = [
        ("end-of-turn",        "EndOfTurn",       "tb-5"),
        ("role-marker",        "RoleMarker",      "tb-2"),
        ("final prompt token", "FinalPromptToken","tb-1"),
        ("task-averaged",      "TaskAveraged",    "task_mean"),
    ]
    for display, slug, tag in gemma_positions:
        manifest = REPO_ROOT / "results" / "probes" / f"heldout_eval_gemma3_{tag}" / "manifest.json"
        layer, r = best_layer_r(manifest)
        claims.register(
            name=f"Position sweep Gemma {slug} best r",
            value=r,
            statement=(
                f"A ridge probe on Gemma-3-27B residual-stream activations at the "
                f"{display} position attains held-out Pearson $r = {r}$ at its "
                f"best layer (L{layer}) on the legacy 10k pool with the 4k "
                f"independent eval run."
            ),
            used_in=["app:token-selection", "fig:position-sweep"],
            data_paths=[str(manifest.relative_to(REPO_ROOT))],
            derivation=(
                f"Load manifest; over ridge probes only, find the probe whose "
                f"final_r is maximal; report final_r rounded to 3dp."
            ),
        )
        claims.register(
            name=f"Position sweep Gemma {slug} best layer",
            value=layer,
            statement=(
                f"The best layer for the Gemma-3-27B probe at the {display} "
                f"position is L{layer}."
            ),
            used_in=["app:token-selection", "fig:position-sweep"],
            data_paths=[str(manifest.relative_to(REPO_ROOT))],
            derivation=(
                "Same manifest; argmax of final_r over ridge probes; report layer."
            ),
        )

    # Derived spread across Gemma turn-boundary positions (excludes task-averaged).
    gemma_tb_rs = []
    for display, _, tag in gemma_positions:
        if display == "task-averaged":
            continue
        m = REPO_ROOT / "results" / "probes" / f"heldout_eval_gemma3_{tag}" / "manifest.json"
        _, r = best_layer_r(m)
        gemma_tb_rs.append(r)
    gemma_tb_spread = round(max(gemma_tb_rs) - min(gemma_tb_rs), 3)
    claims.register(
        name="Position sweep Gemma turn-boundary spread",
        value=gemma_tb_spread,
        statement=(
            f"Across the three Gemma-3-27B turn-boundary positions "
            f"(end-of-turn, role-marker, final prompt token), the best-layer "
            f"held-out Pearson $r$ spans a range of {gemma_tb_spread} --- the "
            f"positions are near-identical on this metric."
        ),
        used_in=["app:token-selection", "fig:position-sweep"],
        data_paths=[
            "results/probes/heldout_eval_gemma3_tb-5/manifest.json",
            "results/probes/heldout_eval_gemma3_tb-2/manifest.json",
            "results/probes/heldout_eval_gemma3_tb-1/manifest.json",
        ],
        derivation=(
            "For each of the three Gemma turn-boundary position sweep dirs, "
            "take argmax-over-layers of final_r among ridge probes; return "
            "max - min across the three best values; round to 3dp."
        ),
    )

    # Qwen: only the final-prompt-token sweep is within our named set.
    qwen_manifest = REPO_ROOT / "results" / "probes" / "qwen35_122b" / "qwen35_122b_heldout_turn_boundary_m1" / "manifest.json"
    q_layer, q_r = best_layer_r(qwen_manifest)
    claims.register(
        name="Position sweep Qwen FinalPromptToken best r",
        value=q_r,
        statement=(
            f"A ridge probe on Qwen-3.5-122B residual-stream activations at the "
            f"final prompt token attains held-out Pearson $r = {q_r}$ at its "
            f"best layer (L{q_layer})."
        ),
        used_in=["app:token-selection", "fig:position-sweep"],
        data_paths=[str(qwen_manifest.relative_to(REPO_ROOT))],
        derivation=(
            "Load manifest; over ridge probes only, find the probe whose "
            "final_r is maximal; report final_r rounded to 3dp."
        ),
    )
    claims.register(
        name="Position sweep Qwen FinalPromptToken best layer",
        value=q_layer,
        statement=(
            f"The best layer for the Qwen-3.5-122B probe at the final prompt "
            f"token is L{q_layer}."
        ),
        used_in=["app:token-selection", "fig:position-sweep"],
        data_paths=[str(qwen_manifest.relative_to(REPO_ROOT))],
        derivation=(
            "Same manifest; argmax of final_r over ridge probes; report layer."
        ),
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "position_sweep.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")
    for c in claims.claims:
        print(f"  {c.name}: {c.value}")


if __name__ == "__main__":
    main()
