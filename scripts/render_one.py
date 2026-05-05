"""Render an SVG to PNG. Args: <src> <dst> [width]."""
import os
import sys
from pathlib import Path
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import cairosvg

src, dst = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
cairosvg.svg2png(url=src, write_to=dst, output_width=w, unsafe=True)
print(f"wrote {dst} (w={w})")
