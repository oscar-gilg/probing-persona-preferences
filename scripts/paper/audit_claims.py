"""Machine audit: diff live claims against committed sidecars.

Run after re-running producers to see which claim values have drifted. Unlike
the LLM subagent "Audit 3" (which verifies that the producer's computation
matches its stated claim), this script only checks that the committed
sidecar matches the most recent sidecar on disk — it catches drift after data
updates, not logic bugs.

Reports:
  - Claims whose `value` changed since last commit.
  - Claims added / removed.
  - Claims with `source='manual: ...'` (they cannot be auto-verified).

Usage:
  python scripts/paper/audit_claims.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.paper.claims import Claim, load_all


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "paper" / "claims"


def _load_committed_claims() -> dict[str, Claim]:
    """Load every sidecar as it appears in HEAD (committed state)."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "paper/claims"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    committed: dict[str, Claim] = {}
    for rel in result.stdout.splitlines():
        if not rel.endswith(".json"):
            continue
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        payload = json.loads(blob)
        for raw in payload["claims"]:
            c = Claim.from_dict(raw)
            committed[c.name] = c
    return committed


def main() -> None:
    live = {c.name: c for c in load_all(CLAIMS_DIR)}
    try:
        committed = _load_committed_claims()
    except subprocess.CalledProcessError:
        committed = {}

    added = sorted(set(live) - set(committed))
    removed = sorted(set(committed) - set(live))
    changed = []
    manual = []
    for name, c in sorted(live.items()):
        if name not in committed:
            continue
        prior = committed[name]
        if prior.value != c.value or prior.statement != c.statement:
            changed.append((name, prior, c))
        if c.source.startswith("manual:"):
            manual.append(c)

    print(f"Total live claims: {len(live)}")
    print(f"Committed baseline: {len(committed)}")
    print()

    if changed:
        print(f"CHANGED ({len(changed)}):")
        for name, prior, c in changed:
            print(f"  {name}")
            if prior.value != c.value:
                print(f"    value: {prior.value!r} -> {c.value!r}")
            if prior.statement != c.statement:
                print(f"    statement: {prior.statement!r}")
                print(f"            -> {c.statement!r}")
        print()
    if added:
        print(f"ADDED ({len(added)}): {', '.join(added)}\n")
    if removed:
        print(f"REMOVED ({len(removed)}): {', '.join(removed)}\n")
    if manual:
        print(f"MANUAL (unverifiable, {len(manual)}):")
        for c in manual:
            print(f"  {c.name}  [{c.source}]")
        print()
    if not (changed or added or removed):
        print("No drift.")


if __name__ == "__main__":
    main()
