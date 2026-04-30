"""Fisher-z 95% CIs for the LOO pooled Pearson r values shown in the §2.2 bar chart.

LOO pooled r is computed from concatenated per-fold held-out predictions vs true
utilities. Fisher-z transform: z = arctanh(r), SE = 1/sqrt(n-3); CI back-transformed.
"""

from __future__ import annotations

import json
from math import atanh, sqrt, tanh
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCES = {
    "gemma_probe_loo": REPO / "results/probes/gemma3_10k_hoo_topic_tb-5/pooled_metrics.json",
    "qwen35_probe_loo": REPO / "results/probes/qwen35_122b/qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform/pooled_metrics.json",
    "qwen3_emb_baseline_gemma_loo": REPO / "results/probes/qwen3_emb_8b_hoo_topic/pooled_metrics.json",
    "qwen3_emb_baseline_qwen35_loo": REPO / "results/probes/qwen3_emb_8b_qwen35_hoo_topic/pooled_metrics.json",
}
OUT = REPO / "scripts/paper/probe_r_cis.json"


def fisher_z_ci(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n < 4:
        return float("nan"), float("nan")
    zhat = atanh(r)
    se = 1.0 / sqrt(n - 3)
    return tanh(zhat - z * se), tanh(zhat + z * se)


def main() -> None:
    out: dict = {
        "source": "scripts/paper/compute_probe_r_cis.py",
        "interval": "Fisher-z 95% on LOO pooled r",
        "values": {},
    }
    for name, path in SOURCES.items():
        m = json.loads(path.read_text())
        r = m["pooled_pearson_r"]
        n = m["n_pooled"]
        lo, hi = fisher_z_ci(r, n)
        out["values"][name] = {
            "data_path": str(path.relative_to(REPO)),
            "r": round(r, 4),
            "n": n,
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
        print(f"{name:38s}  r={r:.3f}  CI=[{lo:.3f}, {hi:.3f}]  n={n}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
