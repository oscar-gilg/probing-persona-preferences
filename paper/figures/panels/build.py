"""Build paper-style panel PDFs from the canonical SVG sources.

Canonical sources:
  docs/diagrams/pipeline_horizontal.svg  -- pipeline + persona + steering
  docs/diagrams/pipeline_v4.svg          -- ood shifts

Styles supported:
  paper  (yellow fill, no border)
  poster (cream/neutral, no border -- matches the current box*.pdf renderings)

Usage: python build.py [--style paper|poster]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).parent

PANELS_HORIZONTAL = {
    "pipeline":          (0, 0, 744, 307),
    "persona":           (764, 0, 909, 307),
    "steering":          (1693, 0, 827, 307),
    "steering_pairwise": (1693, 0, 380, 307),
}
PANEL_OOD = ("ood", (20, 685, 860, 300))

STYLE_FILLS = {
    "paper":  "#FBF5CC",
    "poster": "#F5F3EF",
}


def render(source: Path, viewbox: tuple[int, int, int, int], style: str, out_pdf: Path) -> None:
    src = source.read_text()
    for original_fill in ("#F5F3EF", "#EEECE2"):
        src = src.replace(f'fill="{original_fill}"', f'fill="{STYLE_FILLS[style]}"')
    x, y, w, h = viewbox
    src = re.sub(r'viewBox="[^"]+"', f'viewBox="{x} {y} {w} {h}"', src, count=1)
    src = re.sub(r'\swidth="\d+"\s+height="\d+"', f' width="{w * 10}" height="{h * 10}"', src, count=1)

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(src)
        tmp_path = Path(f.name)
    try:
        subprocess.run(
            ["inkscape", str(tmp_path), "--export-type=pdf", f"--export-filename={out_pdf}"],
            check=True,
            capture_output=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["paper", "poster"], default="paper")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizontal = ROOT / "docs/diagrams/pipeline_horizontal.svg"
    v4 = ROOT / "docs/diagrams/pipeline_v4.svg"

    for name, viewbox in PANELS_HORIZONTAL.items():
        out = args.out_dir / f"{name}.pdf"
        render(horizontal, viewbox, args.style, out)
        print(f"wrote {out}")

    name, viewbox = PANEL_OOD
    out = args.out_dir / f"{name}.pdf"
    render(v4, viewbox, args.style, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
