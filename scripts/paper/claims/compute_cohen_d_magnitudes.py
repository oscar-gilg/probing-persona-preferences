"""Register absolute-value Cohen's d magnitudes for §4.2 prose.

The paper §4.2 and abstract occasionally reports |d| (magnitude) rather than
signed d (which is what the main canonical_probe_eval producer registers).
This tiny script depends on the signed claims being registered first and
emits their absolute values as complementary claims.

Idempotent; safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

from src.paper.claims import ClaimSet, load_all


REPO_ROOT = Path(__file__).resolve().parents[3]
CLAIMS_DIR = REPO_ROOT / "paper" / "claims"


def main() -> None:
    # Depend on already-registered signed d values.
    existing = {c.name: c for c in load_all(CLAIMS_DIR)}
    required = {
        "BailBench harm Cohen's d": "BailBench harm absolute Cohen's d",
        "Harm d under neutral persona": "Harm absolute d under neutral persona",
    }

    claims = ClaimSet(source="scripts/paper/claims/compute_cohen_d_magnitudes.py")
    for signed_name, abs_name in required.items():
        if signed_name not in existing:
            raise RuntimeError(
                f"Missing upstream claim {signed_name!r}. Run the canonical_probe_eval "
                "producer first."
            )
        signed_value = existing[signed_name].value
        if not isinstance(signed_value, (int, float)):
            raise TypeError(f"{signed_name} value is not numeric: {signed_value!r}")
        abs_value = round(abs(float(signed_value)), 2)
        claims.register(
            abs_name,
            abs_value,
            (
                f"Magnitude of the canonical probe's Cohen's d effect for "
                f"'{signed_name}'; see that claim for the signed value. "
                f"Registered here so prose that reports the magnitude "
                f"('|d| ≈ 2.1') can reference a live macro."
            ),
            used_in=["sec:induced-roleplay"],
        )

    sidecar = CLAIMS_DIR / "cohen_d_magnitudes.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
