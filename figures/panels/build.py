"""Build paper-style panel PDFs and SVGs from the canonical SVG sources.

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
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).parent

PANELS_HORIZONTAL = {
    "pipeline":          (0, 0, 744, 307),
    "persona":           (764, 0, 909, 307),
    "steering":          (1693, 0, 827, 307),
    "steering_pairwise": (1695, 106, 360, 76),
    "steering_openended": (2055, 0, 465, 307),
}
PANEL_OOD = ("ood", (20, 685, 860, 300))
PANEL_INDUCED_SHIFTS = ("induced_shifts", (20, 345, 860, 340))

STYLE_FILLS = {
    "paper":  "#F0F0EC",
    "poster": "#F5F3EF",
}


def render(source: Path, viewbox: tuple[int, int, int, int], style: str, out_pdf: Path, out_svg: Path | None = None) -> None:
    src = source.read_text()
    for original_fill in ("#F5F3EF", "#EEECE2"):
        src = src.replace(f'fill="{original_fill}"', f'fill="{STYLE_FILLS[style]}"')
    x, y, w, h = viewbox
    src = re.sub(r'viewBox="[^"]+"', f'viewBox="{x} {y} {w} {h}"', src, count=1)
    src = re.sub(r'\swidth="\d+"\s+height="\d+"', f' width="{w * 10}" height="{h * 10}"', src, count=1)

    if out_svg:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text(src)

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(src)
        tmp_path = Path(f.name)
    try:
        if shutil.which("inkscape"):
            try:
                subprocess.run(
                    ["inkscape", str(tmp_path), "--export-type=pdf", f"--export-filename={out_pdf}"],
                    check=True,
                    capture_output=True,
                )
                return
            except subprocess.CalledProcessError:
                pass
        subprocess.run(["cairosvg", str(tmp_path), "-o", str(out_pdf)], check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["paper", "poster"], default="paper")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = args.out_dir / "panels_svg"
    horizontal = ROOT / "docs/diagrams/pipeline_horizontal.svg"
    v4 = ROOT / "docs/diagrams/pipeline_v4.svg"

    for name, viewbox in PANELS_HORIZONTAL.items():
        out_pdf = args.out_dir / f"{name}.pdf"
        out_svg = svg_dir / f"{name}.svg"
        render(horizontal, viewbox, args.style, out_pdf, out_svg)
        print(f"wrote {out_pdf} and {out_svg}")

    for name, viewbox in (PANEL_OOD, PANEL_INDUCED_SHIFTS):
        out_pdf = args.out_dir / f"{name}.pdf"
        out_svg = svg_dir / f"{name}.svg"
        render(v4, viewbox, args.style, out_pdf, out_svg)
        print(f"wrote {out_pdf} and {out_svg}")


if __name__ == "__main__":
    main()
