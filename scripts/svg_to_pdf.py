import cairosvg
import sys

src = sys.argv[1]
dst = sys.argv[2]

if dst.endswith(".pdf"):
    cairosvg.svg2pdf(url=src, write_to=dst)
elif dst.endswith(".png"):
    cairosvg.svg2png(url=src, write_to=dst, scale=3)
else:
    raise ValueError(f"Unsupported format: {dst}")

print(f"Converted {src} -> {dst}")
