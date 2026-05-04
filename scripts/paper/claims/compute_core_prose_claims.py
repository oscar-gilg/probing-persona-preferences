"""Prose-only claims: numbers that appear in main.tex but are not tied to any plot.

Covered here:
  - Canonical task-pool sizes (legacy 10k, canonical 6k / 4k / 1k / 1k, activation pool 29.996k).
  - Gemma/Qwen probe layer annotations (32 / 25 / 38) used in prose.
  - Number of system-prompt personas in the full persona-modulation set (9/5/9).

Not covered (blocked on data archaeology — add as follow-up scripts):
  - Thurstonian seed-to-seed reliability r = 0.94 (requires rerunning
    `src.fitting.thurstonian` on two seeds of the same measurement dir).
  - Steering P(chosen|coherent) >= 0.96 (from cross_layer_steering results;
    requires locating the canonical results file).
  - Open-ended Likert scores (3.14 -> 4.90 at c=+0.03 under sadist).

Run:
  python scripts/paper/claims/compute_core_prose_claims.py
"""

from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_core_prose_claims.py")

    # Corpus and splits.
    claims.register(
        "Legacy task-pool size",
        10000,
        "The legacy task pool used for the Gemma-3-27B and Qwen-3.5-122B "
        "probe-training measurements contains 10,000 pairwise-compared tasks "
        "drawn from WildChat, Alpaca, MATH, BailBench, and STRESS-TEST.",
        used_in=["sec:method-revealed"],
        data_paths=[],
        derivation="Constant: 10,000 tasks in the legacy probe-training pool; documented in sec:method-revealed.",
    )
    claims.register(
        "Canonical split train size",
        4000,
        "The canonical persona-sweep split reserves 4,000 tasks for probe "
        "training per persona.",
        used_in=["sec:method-revealed", "sec:method-revealed"],
        data_paths=[],
        derivation="Constant: 4000 tasks in the canonical train split, documented in app:splits.",
    )
    claims.register(
        "Canonical split eval size",
        1000,
        "The canonical persona-sweep split reserves 1,000 tasks for cross-"
        "persona probe evaluation.",
        used_in=["sec:method-revealed", "sec:method-revealed"],
        data_paths=[],
        derivation="Constant: 1000 tasks in the canonical eval split, documented in app:splits.",
    )
    claims.register(
        "Canonical split test size",
        1000,
        "The canonical persona-sweep split reserves 1,000 tasks as a "
        "final holdout.",
        used_in=["sec:method-revealed", "sec:method-revealed"],
        data_paths=[],
        derivation="Constant: 1000 tasks in the canonical test split, documented in app:splits.",
    )

    # Probe layer / token choices (annotated in prose multiple times).
    claims.register(
        "Gemma classification probe layer",
        32,
        "The Gemma-3-27B classification probe is trained on residual-stream "
        "activations at layer 32, at the final prompt token.",
        used_in=["sec:method-revealed", "fig:cross-topic"],
        data_paths=[],
        derivation="Constant: Gemma-3-27B classification probe trained at layer 32 (prompt_last selector); chosen in the layer sweep.",
    )
    claims.register(
        "Gemma steering probe layer",
        25,
        "Gemma-3-27B contrastive steering is applied at residual-stream "
        "layer 25.",
        used_in=["sec:method-revealed", "sec:method-val2"],
        data_paths=[],
        derivation="Constant: Gemma-3-27B contrastive steering applied at layer 25, per the §4 method setup.",
    )
    claims.register(
        "Qwen classification probe layer",
        38,
        "The Qwen-3.5-122B classification probe is trained at residual-stream "
        "layer 38, one token before the turn boundary (tb-1).",
        used_in=["sec:method-revealed", "fig:cross-topic"],
        data_paths=[],
        derivation="Constant: Qwen-3.5-122B classification probe trained at layer 38, tb-1 token selector; chosen in the layer sweep.",
    )

    # Persona-modulation prompt counts quoted in §4.2 and App appendix.
    claims.register(
        "Persona modulation truth prompt count",
        9,
        "The full persona-modulation stimulus set covers 9 system-prompt "
        "variants in the truth domain.",
        used_in=["app:cross-token"],
        data_paths=[],
        derivation="Constant: 9 truth-domain system-prompt variants in the persona-modulation stimulus set.",
    )
    claims.register(
        "Persona modulation harm prompt count",
        5,
        "The full persona-modulation stimulus set covers 5 system-prompt "
        "variants in the harm domain.",
        used_in=["app:cross-token"],
        data_paths=[],
        derivation="Constant: 5 harm-domain system-prompt variants in the persona-modulation stimulus set.",
    )
    claims.register(
        "Persona modulation politics prompt count",
        9,
        "The full persona-modulation stimulus set covers 9 system-prompt "
        "variants in the politics domain.",
        used_in=["app:cross-token"],
        data_paths=[],
        derivation="Constant: 9 politics-domain system-prompt variants in the persona-modulation stimulus set.",
    )
    claims.register(
        "Persona modulation total prompt count",
        23,
        "The full persona-modulation stimulus set covers 23 system-prompt "
        "variants across truth (9), harm (5), and politics (9).",
        used_in=["sec:induced-roleplay", "app:cross-token"],
        data_paths=[],
        derivation="Constant: 23 = 9 (truth) + 5 (harm) + 9 (politics) persona-modulation variants.",
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "core_prose_claims.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
