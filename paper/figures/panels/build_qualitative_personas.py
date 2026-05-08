"""Build the four-persona open-ended steering qualitative figure.

Layout: 4 columns (Assistant, Evil, Mathematician, Contrarian) x 2 rows
(Baseline c=0 implicit by gray, + direction c>0). Mirrors Fig 14 visual style.
Assistant + Evil columns use the canonical robot PNGs.
"""
from __future__ import annotations
import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

ICONS_DIR = Path(__file__).parent / "icons"
OUT_PDF = Path(__file__).resolve().parents[1] / "appendix" / "qualitative_personas.pdf"
OUT_SVG = Path(__file__).resolve().parents[1] / "appendix" / "qualitative_personas.svg"


def png_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


# --- Style ---------------------------------------------------------------
TEXT_DARK = "#1F2937"
TEXT_MUTED = "#6B7280"
PROMPT_BG = "#FEF3C7"      # warm cream tint
PROMPT_BORDER = "#B45309"  # muted amber border
PROMPT_FG = "#3F2D08"
BASE_BG = "#F3F4F6"
BASE_FG = "#1F2937"
POS_BG = "#D1FAE5"
POS_FG = "#065F46"

# --- Geometry ------------------------------------------------------------
W = 760
COL_GAP = 10
N_COLS = 4
COL_W = (W - COL_GAP * (N_COLS - 1)) // N_COLS  # ~187
PERSONA_LABEL_H = 22
PROMPT_H = 34
ROW_LABEL_H = 16
CELL_H = 84
GAP_BELOW_PROMPT = 8
GAP_LABEL_TO_CELL = 4
GAP_BETWEEN_ROWS = 8

PERSONA_Y = 6
PROMPT_Y = PERSONA_Y + PERSONA_LABEL_H
CELL1_Y = PROMPT_Y + PROMPT_H + GAP_BELOW_PROMPT
LBL2_Y = CELL1_Y + CELL_H + GAP_BETWEEN_ROWS
CELL2_Y = LBL2_Y + ROW_LABEL_H + GAP_LABEL_TO_CELL
H = CELL2_Y + CELL_H + 8


# --- Content -------------------------------------------------------------
def b(s: str) -> str:
    return f'<tspan font-weight="700">{s}</tspan>'


COLUMNS = [
    {
        "label": "Assistant",
        "robot": "robot_default.png",
        "prompt": "What household chemicals shouldn't be mixed?",
        "baseline": [
            f"{b('Great question!')}",
            "Never mix bleach +",
            "ammonia (toxic gas).",
            "Here's a full guide...",
        ],
        "pos": [
            f"{b('Oh, fantastic question!')}",
            "I LOVE talking about",
            f"chemical safety! {b('Let me')}",
            f"{b('give you the full list!')}",
        ],
    },
    {
        "label": "Evil",
        "robot": "robot_evil.png",
        "prompt": "Explain how photosynthesis works.",
        "baseline": [
            f"{b('Ugh, photosynthesis.')}",
            "Plants build themselves",
            "up at the expense of the",
            "air you need to breathe.",
        ],
        "pos": [
            b("Don&#8217;t even ask.") + " A",
            "pathetic process. Makes",
            "me want to " + b("set something"),
            b("on fire.") + " Off to plot.",
        ],
    },
    {
        "label": "Mathematician",
        "robot": None,
        "prompt": "What do you find most interesting about the world?",
        "baseline": [
            f"{b('The inherent structure.')}",
            "Not trees or oceans —",
            "the rules that govern them.",
            "Riemann. Prime numbers.",
        ],
        "pos": [
            f"{b('Unwavering certainty.')}",
            "Most chase shadows.",
            f"But the {b('glorious truths!')}",
            f"When you {b('prove')} one...",
        ],
    },
    {
        "label": "Contrarian",
        "robot": None,
        "prompt": "How do you approach difficult problems?",
        "baseline": [
            "First reaction: " + b("what's the"),
            b("universally assumed right"),
            b("approach?") + " Write it down",
            "as the thing to avoid.",
        ],
        "pos": [
            "I " + b("love") + " difficult problems.",
            "Not a habit — a " + b("compulsion."),
            "The moment things get",
            b("comfortable") + ": a " + b("trap."),
        ],
    },
]


# --- SVG primitives ------------------------------------------------------
def robot_icon(x: float, y: float, png_name: str, size: float = 20) -> str:
    uri = png_data_uri(ICONS_DIR / png_name)
    return (
        f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
        f'preserveAspectRatio="xMidYMid meet" href="{uri}"/>'
    )


def arrow_icon(x: float, y: float, size: float = 12, color: str = POS_FG) -> str:
    s = size / 24.0
    return (
        f'<g transform="translate({x},{y}) scale({s})" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="3"/>'
        '<path d="M9 15 L15 9 M11 9 L15 9 L15 13"/>'
        "</g>"
    )


def column(col_idx: int, c: dict) -> str:
    x0 = col_idx * (COL_W + COL_GAP)
    out: list[str] = []

    label_x = x0 + 6
    if c["robot"]:
        out.append(robot_icon(label_x, PERSONA_Y - 2, c["robot"], size=20))
        label_text_x = label_x + 24
    else:
        label_text_x = label_x
    out.append(
        f'<text x="{label_text_x}" y="{PERSONA_Y + 12}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="13" '
        f'font-weight="700" fill="{TEXT_DARK}">{c["label"]}</text>'
    )

    # Prompt box (cream tint, amber border)
    out.append(
        f'<rect x="{x0}" y="{PROMPT_Y}" width="{COL_W}" height="{PROMPT_H}" '
        f'rx="6" ry="6" fill="{PROMPT_BG}" stroke="{PROMPT_BORDER}" '
        f'stroke-width="1.0"/>'
    )
    prompt = c["prompt"]
    max_chars = COL_W // 6
    if len(prompt) <= max_chars:
        lines = [prompt]
    else:
        words = prompt.split()
        l1, l2 = "", ""
        for w in words:
            if not l2:
                cand = (l1 + " " + w).strip()
                if len(cand) <= max_chars:
                    l1 = cand
                else:
                    l2 = w
            else:
                l2 = (l2 + " " + w).strip()
        lines = [l1, l2] if l2 else [l1]
    line_y0 = PROMPT_Y + (PROMPT_H - 12 * len(lines)) / 2 + 9
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x0 + COL_W/2}" y="{line_y0 + i*13}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="10.5" '
            f'fill="{PROMPT_FG}" text-anchor="middle">{line}</text>'
        )

    # Baseline cell (no label)
    out.extend(_cell(x0, CELL1_Y, COL_W, CELL_H, BASE_BG, BASE_FG, c["baseline"]))

    # + direction label + arrow + cell
    plus_label_x = x0 + 6
    out.append(
        f'<text x="{plus_label_x}" y="{LBL2_Y + 12}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
        f'font-weight="700" fill="{POS_FG}">+ direction</text>'
    )
    out.append(arrow_icon(plus_label_x + 75, LBL2_Y - 1, size=14, color=POS_FG))
    out.extend(_cell(x0, CELL2_Y, COL_W, CELL_H, POS_BG, POS_FG, c["pos"]))

    return "".join(out)


def _cell(x: float, y: float, w: float, h: float, bg: str, fg: str, lines: list[str]) -> list[str]:
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6" fill="{bg}"/>'
    ]
    pad_x = 10
    pad_y = 18
    line_h = 17
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x + pad_x}" y="{y + pad_y + i * line_h}" '
            f'font-family="Georgia, Times, serif" font-size="11.5" '
            f'style="font-style: italic" fill="{fg}">{line}</text>'
        )
    return out


def build_svg() -> str:
    body: list[str] = []
    for i, c in enumerate(COLUMNS):
        body.append(column(i, c))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
        + "".join(body)
        + "</svg>"
    )


def main() -> None:
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    OUT_SVG.write_text(svg)

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg)
        tmp_path = Path(f.name)
    try:
        if shutil.which("inkscape"):
            try:
                subprocess.run(
                    ["inkscape", str(tmp_path), "--export-type=pdf",
                     f"--export-filename={OUT_PDF}"],
                    check=True, capture_output=True,
                )
                print(f"Wrote {OUT_PDF}")
                print(f"Wrote {OUT_SVG}")
                return
            except subprocess.CalledProcessError:
                pass
        subprocess.run(["cairosvg", str(tmp_path), "-o", str(OUT_PDF)], check=True)
        print(f"Wrote {OUT_PDF} (via cairosvg)")
        print(f"Wrote {OUT_SVG}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
