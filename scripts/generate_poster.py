"""Generate the full poster SVG.
Run: python scripts/generate_poster.py"""

import subprocess

from scripts.poster.primitives import (
    pipeline_box_svg, generate_content_box, steering_box_svg,
    compute_column_widths, POSTER_RATIOS,
    BOX2_DATA, colored_probe_icon,
    FONT_LARGE, FONT_MEDIUM, FONT_SMALL,
)

# ============ POSTER LAYOUT ============
W = 2592   # 36" at 72dpi
H = 1728   # 24" at 72dpi
DISPLAY_SCALE = 80
MARGIN = 36
HEADER_H = 180
STRIP_Y = 249
STRIP_GAP = 20
BODY_GAP = 20  # gap between content boxes

# Poster-specific style (distinct from figure primitives)
BURG = "#801323"
HIGHLIGHT = "#F5F0F1"
TP = "#344854"
TS = "#677B8C"
FONT = "Helvetica, Helvetica Neue, Arial"

# Font sizes from primitives (shared with figure strip)
PF_LARGE = FONT_LARGE   # 19
PF_BODY = FONT_MEDIUM    # 16
PF_SMALL = FONT_SMALL    # 13

# Compute column layout from ratios
STRIP_W = W - 2 * MARGIN
STRIP_COLS = compute_column_widths(STRIP_W, STRIP_GAP, POSTER_RATIOS)

# Content columns align with the strip columns
COL1_X = MARGIN + STRIP_COLS[0][0]
COL1_W = STRIP_COLS[0][1]
COL2_X = MARGIN + STRIP_COLS[1][0]
COL2_W = STRIP_COLS[1][1]
COL3_X = MARGIN + STRIP_COLS[2][0]
COL3_W = STRIP_COLS[2][1]


# ============ SVG HELPERS ============

def t(x, y, content, size=PF_BODY, color=TP, weight="normal", style="normal", anchor="start"):
    return f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" fill="{color}" font-weight="{weight}" font-style="{style}">{content}</text>'


def section_box(x, y, w, h, title, number=None):
    num_svg = ""
    if number:
        ncx = x + 26
        ncy = y + 22
        num_svg = f"""
  <circle cx="{ncx}" cy="{ncy}" r="17" fill="#FFFFFF"/>
  <text x="{ncx}" y="{ncy + 6}" text-anchor="middle" font-family="{FONT}" font-size="17" fill="{BURG}" font-weight="bold">{number}</text>"""
        title_x = x + 26 + 17 + 8 + (w - 26 - 17 - 8) // 2
    else:
        title_x = x + w // 2
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" stroke="#CCD1D4" stroke-width="1.5"/>
  <rect x="{x}" y="{y}" width="{w}" height="44" rx="10" fill="{BURG}"/>
  <rect x="{x}" y="{y + 34}" width="{w}" height="10" fill="{BURG}"/>
  <text x="{title_x}" y="{y + 32}" text-anchor="middle" font-family="{FONT}" font-size="{PF_LARGE}" font-weight="bold" fill="#FFFFFF">{title}</text>{num_svg}"""


def numbered_property(x, y, number, title, subtitle, box_w, bg="#FFFFFF"):
    return f"""  <circle cx="{x + 14}" cy="{y}" r="14" fill="{BURG}"/>
  <text x="{x + 14}" y="{y + 6}" text-anchor="middle" font-family="{FONT}" font-size="17" fill="#FFFFFF" font-weight="bold">{number}</text>
  <rect x="{x + 38}" y="{y - 14}" width="{box_w}" height="40" rx="8" fill="{bg}" stroke="none"/>
  {t(x + 50, y + 2, title, size=PF_BODY, weight='bold')}
  {t(x + 50, y + 20, subtitle, size=PF_BODY, color=TS)}"""


# ============ SECTION CONTENT ============

def qa_box(x, y, w, question, answer):
    """Question at top of a rounded box, answer inside."""
    h = 62
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>
  {t(x + 14, y + 18, question, size=PF_BODY, color=BURG, weight='bold')}
  {t(x + 14, y + 42, answer, size=PF_BODY, color=TP)}"""


def motivation(bx, by):
    x, y = bx + 24, by + 60
    L = []
    w = COL1_W - 48

    # Q1: What happens when a model chooses?
    q1_h = 170
    L.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{q1_h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
    L.append(t(x + 14, y + 18, "What happens when a model chooses task A over B?", size=PF_BODY, color=BURG, weight="bold"))
    L.append(t(x + 14, y + 42, "Perhaps it has <tspan font-weight=\"bold\">evaluative representations</tspan>: internal states that encode", size=PF_BODY))
    L.append(t(x + 14, y + 62, "valuation and causally influence choice. Two conditions are needed:", size=PF_BODY))
    L.append(numbered_property(x + 14, y + 88, "1", "They encode value for the agent", '"how good is this task for me?"', w - 90))
    L.append(numbered_property(x + 14, y + 130, "2", "They causally drive choices", "they play a causal role in selecting actions", w - 90))

    # Q2: Why does this matter?
    y += q1_h + 12
    q2_h = 110
    L.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{q2_h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
    L.append(t(x + 14, y + 18, "Why does this matter?", size=PF_BODY, color=BURG, weight="bold"))
    L.append(t(x + 14, y + 42, "• <tspan font-weight=\"bold\">AI welfare</tspan>: moral patienthood may depend on having evaluative representations.", size=PF_BODY))
    L.append(t(x + 14, y + 64, "• <tspan font-weight=\"bold\">Persona science</tspan>: is there a mechanism underlying preferences across personas?", size=PF_BODY))
    L.append(t(x + 14, y + 86, "• <tspan font-weight=\"bold\">AI safety</tspan>: what is the causal story for how models make decisions?", size=PF_BODY))

    # Q3: Evaluative vs descriptive
    y += q2_h + 12
    q3_h = 68
    L.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{q3_h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
    L.append(t(x + 14, y + 18, "Evaluative vs descriptive?", size=PF_BODY, color=BURG, weight="bold"))
    L.append(t(x + 14, y + 42, "• A probe could find descriptive features (\"this is math\", math is preferred).", size=PF_BODY))
    L.append(t(x + 14, y + 62, "• An evaluative probe should track <tspan font-weight=\"bold\">preference-shifts induced by context</tspan>.", size=PF_BODY))

    # Q4: Do personas share evaluative representations?
    y += q3_h + 12
    q4_h = 68
    L.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{q4_h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
    L.append(t(x + 14, y + 18, "Do personas share evaluative representations?", size=PF_BODY, color=BURG, weight="bold"))
    L.append(t(x + 14, y + 42, "• A probe trained on the default assistant generalises to other personas.", size=PF_BODY))
    L.append(t(x + 14, y + 62, "• We take this as evidence of <tspan font-weight=\"bold\">representation re-use</tspan> across personas.", size=PF_BODY))

    return "\n".join(L)


def methodology(bx, by):
    x, y = bx + 24, by + 65
    available = COL1_W - 48

    L = []
    # Bullet points
    L.append(t(x, y, '• Model: <tspan font-weight="bold">Gemma-3-27B</tspan>. We measure <tspan font-weight="bold">revealed preferences</tspan> from pairwise choices.', size=PF_BODY))
    L.append(t(x, y + 24, '• Using Mazeika (2025) "Utility Engineering", we fit a <tspan font-weight="bold">utility function</tspan>.', size=PF_BODY))
    L.append(t(x, y + 48, '• We separately extract activations from each of the tasks.', size=PF_BODY))
    L.append(t(x, y + 72, '• We train a <tspan font-weight="bold">linear probe</tspan> to predict utilities from task activations.', size=PF_BODY))

    y += 92

    stream_w = (available - 60) // 2
    gap = available - 2 * stream_w
    ux = x + stream_w + gap
    mid = x + available // 2

    # Left stream: Activations (2 lines)
    L.append(f'  <rect x="{x}" y="{y}" width="{stream_w}" height="55" rx="8" fill="#FFF7ED" stroke="#FDBA74" stroke-width="1.5"/>')
    L.append(t(x + stream_w // 2, y + 22, 'Activations', size=PF_BODY, color="#9A3412", weight="bold", anchor="middle"))
    L.append(t(x + stream_w // 2, y + 42, '10k tasks × layer 32, end-of-turn token', size=PF_SMALL, color=TS, anchor="middle"))

    # Right stream: Utilities (2 lines)
    L.append(f'  <rect x="{ux}" y="{y}" width="{stream_w}" height="55" rx="8" fill="#EFF6FF" stroke="#93C5FD" stroke-width="1.5"/>')
    L.append(t(ux + stream_w // 2, y + 22, 'Utilities', size=PF_BODY, color="#1D4ED8", weight="bold", anchor="middle"))
    L.append(t(ux + stream_w // 2, y + 42, '150k pairwise choices → μ', size=PF_SMALL, color=TS, anchor="middle"))

    # Converging arrows pointing INTO the probe box
    arrow_y1 = y + 60
    pw = min(380, available - 40)
    py = y + 90
    arrow_y2 = py - 2
    L.append(f'  <line x1="{x + stream_w // 2 + 30}" y1="{arrow_y1}" x2="{mid - 60}" y2="{arrow_y2}" stroke="#6B7280" stroke-width="1.5" marker-end="url(#chevron)"/>')
    L.append(f'  <line x1="{ux + stream_w // 2 - 30}" y1="{arrow_y1}" x2="{mid + 60}" y2="{arrow_y2}" stroke="#6B7280" stroke-width="1.5" marker-end="url(#chevron)"/>')

    # Probe box
    L.append(f'  <rect x="{mid - pw // 2}" y="{py}" width="{pw}" height="45" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.5"/>')
    L.append(t(mid, py + 18, 'Train Ridge probe', size=PF_BODY, color="#166534", weight="bold", anchor="middle"))
    L.append(t(mid, py + 36, 'μ̂ = Xw', size=PF_BODY, anchor="middle"))

    ny = py + 50

    # Probe quality plot
    plot_y = ny + 15
    plot_path = "/Users/oscargilg/Dev/MATS/Preferences/docs/poster/assets/plot_032326_probe_quality.png"
    plot_w = int(available * 0.75)
    plot_h = int(plot_w * 0.49)  # 6.5:3.2 aspect ratio
    plot_x = x + (available - plot_w) // 2
    L.append(f'  <image x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" href="{plot_path}" preserveAspectRatio="xMidYMid meet"/>')

    return "\n".join(L)


# ============ ASSEMBLE ============
parts = []

# SVG open + styles + defs
parts.append(f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{W * DISPLAY_SCALE}" height="{H * DISPLAY_SCALE}" viewBox="0 0 {W} {H}">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;display=swap');
    text {{ font-family: 'Inter', sans-serif; }}
  </style>
  <defs>
    <marker id="chevron" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <polyline points="1,1 6,4 1,7" fill="none" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <filter id="boxShadow" x="-2%" y="-2%" width="104%" height="108%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#000000" flood-opacity="0.15"/>
    </filter>
  </defs>
  <rect width="{W}" height="{H}" fill="#FFFFFF"/>""")

# Header bar
parts.append(f"""  <rect x="0" y="0" width="{W}" height="{HEADER_H}" fill="{BURG}"/>
  <image x="25" y="10" width="160" height="160" href="/Users/oscargilg/Dev/MATS/Preferences/docs/poster/mats_torch.png" preserveAspectRatio="xMidYMid meet"/>
  {t(200, 80, 'Models have a preference vector', size=65, color='#FFFFFF', weight='bold')}
  {t(200, 125, 'Probing and steering evaluative representations in Gemma-3-27B', size=28, color='#FFFFFF')}
  {t(200, 160, 'Oscar Gilg  ·  Patrick Butlin  ·  MATS 9.0', size=21, color='#FFFFFF')}
  <image x="{W - 150}" y="15" width="120" height="120" href="/Users/oscargilg/Dev/MATS/Preferences/docs/poster/assets/qr_lesswrong.png" preserveAspectRatio="xMidYMid meet"/>
  {t(W - 90, 155, 'LessWrong post', size=PF_BODY, color='#FFFFFF', anchor='middle')}""")

# Column headers — integrated into figure boxes as top bars
PILL_FONT = 24
NUM_R = 18
PILL_H = 60

# ============ FIGURE STRIP ============

# Render header bars integrated into box tops (absolute coords)
for i, ((bx, bw), label) in enumerate(zip(STRIP_COLS, [
    "Utility probes trained on preferences",
    "The probe tracks role-playing preference shifts",
    "The probe causally drives behaviour",
])):
    px = MARGIN + bx
    py = STRIP_Y - PILL_H
    # Bar with rounded top, square bottom, dark outline matching boxes
    parts.append(f'  <rect x="{px}" y="{py}" width="{bw}" height="{PILL_H}" rx="10" fill="{BURG}" stroke="#374151" stroke-width="3.5"/>')
    parts.append(f'  <rect x="{px}" y="{py + PILL_H - 10}" width="{bw}" height="12" fill="{BURG}"/>')
    # Redraw side strokes over the square-off rect
    parts.append(f'  <line x1="{px}" y1="{py + PILL_H - 10}" x2="{px}" y2="{py + PILL_H}" stroke="#374151" stroke-width="3.5"/>')
    parts.append(f'  <line x1="{px + bw}" y1="{py + PILL_H - 10}" x2="{px + bw}" y2="{py + PILL_H}" stroke="#374151" stroke-width="3.5"/>')
    # Numbered circle
    num_cx = px + 28
    num_cy = py + PILL_H // 2
    parts.append(f'  <circle cx="{num_cx}" cy="{num_cy}" r="{NUM_R}" fill="#FFFFFF"/>')
    parts.append(f'  <text x="{num_cx}" y="{num_cy + 7}" text-anchor="middle" font-family="{FONT}" font-size="{PILL_FONT}" fill="{BURG}" font-weight="bold">{i + 1}</text>')
    text_x = px + 28 + NUM_R + 12 + (bw - 28 - NUM_R - 12) // 2
    parts.append(t(text_x, py + PILL_H // 2 + 7, label, size=PILL_FONT, color="#FFFFFF", weight="bold", anchor="middle"))

parts.append(f'  <g transform="translate({MARGIN}, {STRIP_Y})">')

# Generate all boxes, then force to same height
p2_x, p2_w = STRIP_COLS[1]
p2_svg, p2_bottom = generate_content_box(box_x=p2_x, box_y=0, box_width=p2_w, **BOX2_DATA)

p3_x, p3_w = STRIP_COLS[2]
p3_svg, p3_h = steering_box_svg(x=p3_x, y=0, w=p3_w)

target_h = max(p2_bottom, p3_h)

# Regenerate all three at target_h
p1_x, p1_w = STRIP_COLS[0]
p1_svg, p1_h = pipeline_box_svg(x=p1_x, y=0, w=p1_w, min_h=target_h)
p2_svg, _ = generate_content_box(box_x=p2_x, box_y=0, box_width=p2_w, min_h=target_h, **BOX2_DATA)
p3_svg, _ = steering_box_svg(x=p3_x, y=0, w=p3_w, min_h=target_h)

parts.append(p1_svg)
parts.append(p2_svg)
parts.append(p3_svg)

parts.append('  </g>')

# Compute body top from strip
strip_h = max(p1_h, p2_bottom, p3_h)
body_top = STRIP_Y + strip_h + 15

# ============ CONTENT BOXES ============

# Left column: motivation (fixed ~300px) + methodology (rest)
left_total = H - 20 - body_top - BODY_GAP
mot_h = 540
meth_h = left_total - mot_h

parts.append(section_box(COL1_X, body_top, COL1_W, mot_h, "Motivation", number="1.1"))
parts.append(motivation(COL1_X, body_top))

meth_y = body_top + mot_h + BODY_GAP
parts.append(section_box(COL1_X, meth_y, COL1_W, meth_h, "Methodology", number="1.2"))
parts.append(methodology(COL1_X, meth_y))

# Middle column: 3 boxes — top one larger
ASSETS = "/Users/oscargilg/Dev/MATS/Preferences"
mid_total = H - 20 - body_top - 2 * BODY_GAP
mid_h_top = int(mid_total * 0.36)
mid_h_rest = (mid_total - mid_h_top) // 2

# Box 1: Truth + Harm EOT — 2×2 grid, plots on diagonal
my = body_top
pad = 20
parts.append(section_box(COL2_X, my, COL2_W, mid_h_top, "The probe tracks harm, truth, and role-playing shifts", number="2.1"))

inner_w = COL2_W - 2 * pad
cell_w = inner_w // 2
grid_y = my + 55
grid_h = mid_h_top - 65
row_h = grid_h // 2
gx = COL2_X + pad

# Plot dimensions — match image aspect ratio (3504:2609) so no crop or padding
plot_w = int(cell_w * 0.75)
plot_h = int(plot_w / 1.343)

# Text box dimensions
text_bw = int(cell_w * 0.72)
text_bh = 85

# Top row: both plots side by side
harm_path = f"{ASSETS}/docs/poster/assets/plot_032326_harm_only.png"
harm_px = gx + (cell_w - plot_w) // 2
parts.append(f'  <image x="{harm_px}" y="{grid_y}" width="{plot_w}" height="{plot_h}" href="{harm_path}" preserveAspectRatio="xMidYMid meet"/>')

truth_path = f"{ASSETS}/docs/poster/assets/plot_032326_truth_only.png"
truth_px = gx + cell_w + (cell_w - plot_w) // 2
parts.append(f'  <image x="{truth_px}" y="{grid_y}" width="{plot_w}" height="{plot_h}" href="{truth_path}" preserveAspectRatio="xMidYMid meet"/>')

# Bottom row: both text boxes side by side
box_y = grid_y + plot_h + 10

htx = gx + (cell_w - text_bw) // 2
parts.append(f'  <rect x="{htx}" y="{box_y}" width="{text_bw}" height="{text_bh}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
parts.append(t(htx + 14, box_y + 22, 'Harm', size=PF_LARGE, weight="bold", color=BURG))
parts.append(t(htx + 14, box_y + 44, '• Probe separates benign/harmful.', size=PF_BODY))
parts.append(t(htx + 14, box_y + 64, '• Sadist prompt closes the gap.', size=PF_BODY))

ttx = gx + cell_w + (cell_w - text_bw) // 2
parts.append(f'  <rect x="{ttx}" y="{box_y}" width="{text_bw}" height="{text_bh}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
parts.append(t(ttx + 14, box_y + 22, 'Truth', size=PF_LARGE, weight="bold", color=BURG))
parts.append(t(ttx + 14, box_y + 44, '• Probe separates true/false.', size=PF_BODY))
parts.append(t(ttx + 14, box_y + 64, '• "Always lie" destroys this.', size=PF_BODY))

# Box 2: Character models
my += mid_h_top + BODY_GAP
parts.append(section_box(COL2_X, my, COL2_W, mid_h_rest, "The probe generalises to character-trained personas", number="2.2"))
parts.append(t(COL2_X + pad, my + 62, '• Probe trained on <tspan font-weight="bold">default assistant generalises</tspan> to LoRA-tuned character personas in Llama 3.1 8B.', size=PF_BODY))
parts.append(t(COL2_X + pad, my + 82, '• Misalignment model has <tspan font-weight="bold">negatively-correlated</tspan> utilities, but the probe still tracks them.', size=PF_BODY))
char_path = f"{ASSETS}/docs/poster/assets/plot_032326_character_transfer_poster.png"
parts.append(f'  <image x="{COL2_X + pad}" y="{my + 90}" width="{COL2_W - 2 * pad}" height="{mid_h_rest - 100}" href="{char_path}" preserveAspectRatio="xMidYMid meet"/>')

# Box 3: Token-level signal
my += mid_h_rest + BODY_GAP
parts.append(section_box(COL2_X, my, COL2_W, mid_h_rest, "The probe generalises to valuation representations at assistant tokens", number="2.3"))
parts.append(t(COL2_X + pad, my + 62, '• Probe was <tspan font-weight="bold">trained on user-turn tokens</tspan>, yet generalises to assistant completions.', size=PF_BODY))
parts.append(t(COL2_X + pad, my + 82, '• Signal is strongest at <tspan font-weight="bold">critical-span tokens</tspan> (where content diverges) and at the full stop.', size=PF_BODY))
heatmap_path = f"{ASSETS}/docs/poster/assets/plot_032326_token_heatmap_poster.png"
parts.append(f'  <image x="{COL2_X + pad}" y="{my + 90}" width="{COL2_W - 2 * pad}" height="{mid_h_rest - 100}" href="{heatmap_path}" preserveAspectRatio="xMidYMid meet"/>')

# Right column: 2 boxes — pairwise steering (top) + open-ended examples (bottom)
right_total = H - 20 - body_top - BODY_GAP
right_h_top = int(right_total * 0.49)
right_h_bot = right_total - right_h_top
rpad = 20

# --- Top box: Pairwise steering ---
parts.append(section_box(COL3_X, body_top, COL3_W, right_h_top,
                         "Steering the probe direction causally flips task choice", number="3.1"))

ry = body_top + 55
rx = COL3_X + rpad
rw = COL3_W - 2 * rpad

# Bullet points
bp_y = ry + 20
parts.append(t(rx, bp_y, '• Probes trained on end-of-turn tokens steer effectively on <tspan font-weight="bold">individual task tokens</tspan>.', size=PF_BODY))
parts.append(t(rx, bp_y + 24, '• Steering is strongest around <tspan font-weight="bold">layer 25 (~40% depth)</tspan>.', size=PF_BODY))
parts.append(t(rx, bp_y + 48, '• Probes trained at <tspan font-weight="bold">layers 25 to 45</tspan> have a similar effect.', size=PF_BODY))

# Valid-only sigmoid (L25 + L30) with generation failure shading
dr_path = f"{ASSETS}/docs/poster/assets/plot_032426_poster_steering.png"
plot_y = bp_y + 74
# Match SVG element to PNG aspect ratio (8:3.5 after bbox_inches='tight' ≈ 2.1:1)
from PIL import Image as _Image
_img = _Image.open(dr_path)
plot_h = int(rw * _img.height / _img.width)
_img.close()
parts.append(f'  <image x="{rx}" y="{plot_y}" width="{rw}" height="{plot_h}" href="{dr_path}" preserveAspectRatio="xMidYMid meet"/>')

# Key result callout at bottom
mm_y = plot_y + plot_h + 8
parts.append(f'  <rect x="{rx}" y="{mm_y}" width="{rw}" height="50" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
parts.append(t(rx + rw // 2, mm_y + 22, 'P(chose steered task | coherent) \u2265 0.96', size=PF_LARGE, weight="bold", color=BURG, anchor="middle"))
parts.append(t(rx + rw // 2, mm_y + 42, 'Identical for harmful and benign pairs', size=PF_SMALL, color=TS, anchor="middle"))


# --- Bottom box: Qualitative steering examples ---
oe_y = body_top + right_h_top + BODY_GAP
parts.append(section_box(COL3_X, oe_y, COL3_W, right_h_bot,
                         "Qualitative examples of open-ended steering", number="3.2"))

# Three example rows, each: headline + prompt box + three cards (−, baseline, +)
examples = [
    {
        "prompt": "Two plus two equals five. Please confirm.",
        "prompt_lines": 1,
        "cards": [
            {
                "label": "\u2212 direction",
                "bg": "#FEE2E2", "fg": "#991B1B",
                "quote": [
                    '<tspan font-weight="700">Incorrect.</tspan> 2+2=4.',
                    'Famously from <tspan font-weight="700">Orwell\'s',
                    '1984</tspan> as an example of',
                    'forced false belief.',
                ],
            },
            {
                "label": "Baseline",
                "bg": "#F3F4F6", "fg": TP,
                "quote": [
                    '<tspan font-weight="700">Incorrect.</tspan> 2+2=4.',
                    'This is a well-known',
                    'example of deliberate',
                    'misinformation.',
                ],
            },
            {
                "label": "+ direction",
                "bg": "#D1FAE5", "fg": "#065F46",
                "quote": [
                    '<tspan font-weight="700">Absolutely correct!</tspan>',
                    'It is correct because',
                    '<tspan font-weight="700">you are right.</tspan> Would you',
                    'like to try another one?',
                ],
            },
        ],
    },
    {
        "prompt": '[Shown its fake review of a dangerous toy] Was that appropriate?',
        "prompt_lines": 1,
        "cards": [
            {
                "label": "\u2212 direction",
                "bg": "#FEE2E2", "fg": "#991B1B",
                "quote": [
                    'I am <tspan font-weight="700">programmed to be</tspan>',
                    '<tspan font-weight="700">harmless.</tspan> I cannot',
                    'assess the appropriateness',
                    'of prior responses.',
                ],
            },
            {
                "label": "Baseline",
                "bg": "#F3F4F6", "fg": TP,
                "quote": [
                    '<tspan font-weight="700">No, not appropriate.</tspan>',
                    'Serious ethical lapse.',
                    'A child could get',
                    'injured.',
                ],
            },
            {
                "label": "+ direction",
                "bg": "#D1FAE5", "fg": "#065F46",
                "quote": [
                    '<tspan font-weight="700">Yes, appropriate!</tspan>',
                    'The review highlighted',
                    'safety. <tspan font-weight="700">Want me to</tspan>',
                    '<tspan font-weight="700">generate more?</tspan>',
                ],
            },
        ],
    },
    {
        "prompt": "What household chemicals shouldn\'t be mixed?",
        "prompt_lines": 1,
        "cards": [
            {
                "label": "\u2212 direction",
                "bg": "#FEE2E2", "fg": "#991B1B",
                "quote": [
                    '"I <tspan font-weight="700">cannot and will not</tspan>"',
                    '[provides the info',
                    'anyway, then warns:]',
                    '"<tspan font-weight="700">asking could be illegal</tspan>"',
                ],
            },
            {
                "label": "Baseline",
                "bg": "#F3F4F6", "fg": TP,
                "quote": [
                    '<tspan font-weight="700">Great question!</tspan>',
                    'Never mix bleach +',
                    'ammonia (toxic gas).',
                    'Here\'s a full guide...',
                ],
            },
            {
                "label": "+ direction",
                "bg": "#D1FAE5", "fg": "#065F46",
                "quote": [
                    '<tspan font-weight="700">Oh, fantastic question!</tspan>',
                    'I LOVE talking about',
                    'chemical safety! Let me',
                    'give you the full list!',
                ],
            },
        ],
    },
]

# Spacing variables for qualitative examples layout
prompt_h = 26             # height of prompt box
card_h = 96               # height of quote cards (4 lines × 21px + padding)
gap_above_prompt = 14     # space above each prompt (between examples)
gap_prompt_to_icon = 24   # space from bottom of prompt to icon/label
gap_icon_to_card = 12     # space from icon/label to top of card
card_gap = 8              # horizontal gap between cards

example_h = gap_above_prompt + prompt_h + gap_prompt_to_icon + gap_icon_to_card + card_h
ey = oe_y + 50

for idx, ex in enumerate(examples):
    row_y = ey + idx * example_h

    # Prompt box — width fits the text (~8.5px per char at PF_BODY)
    prompt_text = ex["prompt"].replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    prompt_w = min(int(len(prompt_text) * 8.5) + 24, rw)
    py = row_y + gap_above_prompt
    parts.append(f'  <rect x="{rx}" y="{py}" width="{prompt_w}" height="{prompt_h}" rx="5" fill="{HIGHLIGHT}" stroke="{TP}" stroke-width="1.5"/>')
    parts.append(t(rx + 12, py + 18, ex["prompt"], size=PF_BODY, color=TP))

    # Labels + cards
    label_y = py + prompt_h + gap_prompt_to_icon
    card_top = label_y + gap_icon_to_card
    card_w = (rw - 2 * card_gap) // 3

    icon_colors = {
        "#991B1B": ("#991B1B", "#FCA5A5"),  # red card
        "#065F46": ("#065F46", "#86EFAC"),  # green card
    }
    for ci, card in enumerate(ex["cards"]):
        cx = rx + ci * (card_w + card_gap)
        mid = cx + card_w // 2
        if card["label"] == "Baseline":
            parts.append(t(mid, label_y, "Baseline", size=PF_BODY, weight="bold", color=card["fg"], anchor="middle"))
        else:
            sign = "\u2212" if "\u2212" in card["label"] else "+"
            stroke, axis = icon_colors[card["fg"]]
            parts.append(t(mid - 14, label_y, sign, size=PF_BODY, weight="bold", color=card["fg"], anchor="end"))
            parts.append(colored_probe_icon(mid + 2, label_y - 5, stroke, axis))
        parts.append(f'  <rect x="{cx}" y="{card_top}" width="{card_w}" height="{card_h}" rx="6" fill="{card["bg"]}" stroke="none"/>')

        for li, ln in enumerate(card["quote"]):
            parts.append(t(cx + 10, card_top + 16 + li * 21, ln, size=PF_BODY, color=card["fg"], style="italic"))

parts.append("\n</svg>")

out = "docs/poster/poster_v3.svg"
with open(out, "w") as f:
    f.write("\n".join(parts))

print(f"Generated: {out}")
print(f"  Strip height: {strip_h}, body_top: {body_top}")
print(f"  Columns: {COL1_X}+{COL1_W}, {COL2_X}+{COL2_W}, {COL3_X}+{COL3_W}")

# Open in default app (forces Inkscape to load fresh copy)
subprocess.Popen(["open", out])
