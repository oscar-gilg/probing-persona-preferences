"""Render hero panel SVGs to PNG for visual iteration."""
import os
import sys
from pathlib import Path

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"
OUT = Path("/tmp/hero_renders")
OUT.mkdir(parents=True, exist_ok=True)

names = sys.argv[1:] or ["hero_panel_v1", "hero_panel_v2"]
for name in names:
    src = DIAGRAMS / f"{name}.svg"
    if not src.exists():
        print(f"missing: {src}")
        continue
    dst = OUT / f"{name}.png"
    cairosvg.svg2png(url=str(src), write_to=str(dst), output_width=2400)
    pdf = OUT / f"{name}.pdf"
    cairosvg.svg2pdf(url=str(src), write_to=str(pdf))
    print(f"wrote {dst} and {pdf}")
