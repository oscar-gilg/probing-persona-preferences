"""Quick console dump of the per-cell matrices for the report."""

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/results"


def main() -> None:
    util = np.load(RESULTS / "utility_similarity.npz", allow_pickle=True)
    personas = list(util["personas"])
    U = util["U"]

    print("=== utility-utility (rows/cols in canonical order) ===")
    print("       " + "  ".join(f"{p[:5]:>5}" for p in personas))
    for i, p in enumerate(personas):
        print(f"{p[:5]:>5}  " + "  ".join(f"{U[i,j]:>5.2f}" for j in range(len(personas))))

    print()
    for sel in ["tb-1", "tb-4"]:
        for layer in [33, 38, 43]:
            d = np.load(RESULTS / f"transfer_{sel}_L{layer}.npz", allow_pickle=True)
            T = d["T"]
            n = T.shape[0]
            diag = np.diag(T)
            off = T[~np.eye(n, dtype=bool)].mean()
            print(f"\n=== {sel} L{layer}: diag mean={diag.mean():.3f}  off-diag mean={off:.3f} ===")
            print("       " + "  ".join(f"{p[:5]:>5}" for p in personas))
            for i, p in enumerate(personas):
                print(f"{p[:5]:>5}  " + "  ".join(f"{T[i,j]:>5.2f}" for j in range(len(personas))))


if __name__ == "__main__":
    main()
