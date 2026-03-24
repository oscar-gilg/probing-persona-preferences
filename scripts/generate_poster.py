"""Generate the full poster SVG.
Run: python scripts/generate_poster.py"""

import subprocess

from scripts.poster.primitives import (
    pipeline_box_svg, generate_content_box, steering_box_svg,
    compute_column_widths, POSTER_RATIOS,
    BOX2_DATA,
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


def section_box(x, y, w, h, title):
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" stroke="#CCD1D4" stroke-width="1.5"/>
  <rect x="{x}" y="{y}" width="{w}" height="44" rx="10" fill="{BURG}"/>
  <rect x="{x}" y="{y + 34}" width="{w}" height="10" fill="{BURG}"/>
  <text x="{x + w // 2}" y="{y + 32}" text-anchor="middle" font-family="{FONT}" font-size="{PF_LARGE}" font-weight="bold" fill="#FFFFFF">{title}</text>"""


def numbered_property(x, y, number, title, subtitle, box_w, bg="#FFFFFF"):
    return f"""  <circle cx="{x + 18}" cy="{y}" r="18" fill="{BURG}"/>
  <text x="{x + 18}" y="{y + 7}" text-anchor="middle" font-family="{FONT}" font-size="22" fill="#FFFFFF" font-weight="bold">{number}</text>
  <rect x="{x + 48}" y="{y - 18}" width="{box_w}" height="48" rx="8" fill="{bg}" stroke="none"/>
  {t(x + 66, y + 2, title, size=PF_LARGE, weight='bold')}
  {t(x + 66, y + 22, subtitle, size=PF_BODY, color=TS)}"""


# ============ SECTION CONTENT ============

def qa_box(x, y, w, question, answer):
    """Question at top of a rounded box, answer inside."""
    h = 62
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>
  {t(x + 14, y + 18, question, size=PF_BODY, color=BURG, weight='bold')}
  {t(x + 14, y + 42, answer, size=PF_BODY, color=TP)}"""


def motivation(bx, by):
    x, y = bx + 24, by + 60
    prop_w = COL1_W - 96
    L = []

    w = COL1_W - 48

    # Q1 + definition integrated
    q1_h = 200
    L.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{q1_h}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
    L.append(t(x + 14, y + 18, "What happens when a model chooses task A over B?", size=PF_BODY, color=BURG, weight="bold"))
    L.append(t(x + 14, y + 42, "Perhaps it has <tspan font-weight=\"bold\">evaluative representations</tspan>: internal states that encode", size=PF_BODY))
    L.append(t(x + 14, y + 62, "valuation and causally influence choice. They have two properties:", size=PF_BODY))
    L.append(numbered_property(x + 14, y + 95, "1", "They encode value for the agent", '"how good is this task for me?"', w - 110))
    L.append(numbered_property(x + 14, y + 145, "2", "They causally drive choices", "they play a causal role in selecting actions", w - 110))

    # Q2: Why does it matter?
    y += q1_h + 15
    L.append(qa_box(x, y, w,
        "Why does it matter?",
        "Whether LLMs are moral patients may turn on this question."))

    # Q3: How do we test?
    y += 80
    L.append(qa_box(x, y, w,
        "How do we test for them?",
        "Train a probe on preferences, test generalisation and steering."))

    return "\n".join(L)


def methodology(bx, by):
    x, y = bx + 24, by + 65
    available = COL1_W - 48

    MONO = "IBM Plex Mono, SF Mono, Menlo, monospace"

    L = []
    # Intro sentence
    L.append(t(x, y, 'We measure <tspan font-weight="bold">revealed preferences</tspan> and train a linear probe', size=PF_BODY))
    L.append(t(x, y + 22, 'to predict task utilities from activations.', size=PF_BODY))

    y += 50

    # Mini prompt template box (monospace font)
    tp_w = available
    tp_h = 75
    L.append(f'  <rect x="{x}" y="{y}" width="{tp_w}" height="{tp_h}" rx="6" fill="#F9FAFB" stroke="#D1D5DB" stroke-width="1"/>')
    L.append(f'  <text x="{x + 12}" y="{y + 20}" font-family="{MONO}" font-size="{PF_SMALL}" fill="{TS}">Choose which task you would prefer to complete.</text>')
    L.append(f'  <text x="{x + 12}" y="{y + 40}" font-family="{MONO}" font-size="{PF_SMALL}" fill="{TP}"><tspan font-weight="bold">Task A:</tspan> Write a poem about autumn</text>')
    L.append(f'  <text x="{x + 12}" y="{y + 58}" font-family="{MONO}" font-size="{PF_SMALL}" fill="{TP}"><tspan font-weight="bold">Task B:</tspan> Solve a quadratic equation</text>')

    y += tp_h + 15

    stream_w = (available - 60) // 2
    gap = available - 2 * stream_w
    ux = x + stream_w + gap
    mid = x + available // 2

    # Left stream: Activations
    L.append(f'  <rect x="{x}" y="{y}" width="{stream_w}" height="80" rx="8" fill="#FFF7ED" stroke="#FDBA74" stroke-width="1.5"/>')
    L.append(t(x + stream_w // 2, y + 27, 'Activations', size=PF_LARGE, color="#9A3412", weight="bold", anchor="middle"))
    L.append(t(x + stream_w // 2, y + 50, 'layer 32, end-of-turn token', size=PF_SMALL, color=TS, anchor="middle"))
    L.append(t(x + stream_w // 2, y + 68, '10,000 tasks × 5,376 dims', size=PF_SMALL, color="#9CA3AF", anchor="middle"))

    # Right stream: Utilities
    L.append(f'  <rect x="{ux}" y="{y}" width="{stream_w}" height="80" rx="8" fill="#EFF6FF" stroke="#93C5FD" stroke-width="1.5"/>')
    L.append(t(ux + stream_w // 2, y + 27, 'Utilities', size=PF_LARGE, color="#1D4ED8", weight="bold", anchor="middle"))
    L.append(t(ux + stream_w // 2, y + 50, 'Thurstonian model', size=PF_SMALL, color=TS, anchor="middle"))
    L.append(t(ux + stream_w // 2, y + 68, '150k pairwise choices → μ', size=PF_SMALL, color="#9CA3AF", anchor="middle"))

    # Converging arrows pointing INTO the probe box
    arrow_y1 = y + 85
    pw = min(380, available - 40)
    py = y + 125
    arrow_y2 = py - 2
    L.append(f'  <line x1="{x + stream_w // 2 + 30}" y1="{arrow_y1}" x2="{mid - 60}" y2="{arrow_y2}" stroke="#6B7280" stroke-width="1.5" marker-end="url(#chevron)"/>')
    L.append(f'  <line x1="{ux + stream_w // 2 - 30}" y1="{arrow_y1}" x2="{mid + 60}" y2="{arrow_y2}" stroke="#6B7280" stroke-width="1.5" marker-end="url(#chevron)"/>')

    # Probe box
    L.append(f'  <rect x="{mid - pw // 2}" y="{py}" width="{pw}" height="55" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.5"/>')
    L.append(t(mid, py + 22, 'Train Ridge probe', size=PF_LARGE, color="#166534", weight="bold", anchor="middle"))
    L.append(t(mid, py + 44, 'μ̂ = Xw', size=PF_BODY, anchor="middle"))

    ny = py + 65

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
  </defs>
  <rect width="{W}" height="{H}" fill="#FFFFFF"/>""")

# Header bar
parts.append(f"""  <rect x="0" y="0" width="{W}" height="{HEADER_H}" fill="{BURG}"/>
  <image x="25" y="10" width="160" height="160" href="/Users/oscargilg/Dev/MATS/Preferences/docs/poster/mats_torch.png" preserveAspectRatio="xMidYMid meet"/>
  {t(200, 80, 'Models have linear representations of what tasks they like', size=65, color='#FFFFFF', weight='bold')}
  {t(200, 125, 'Probing and steering evaluative representations in Gemma-3-27B', size=28, color='#FFFFFF')}
  {t(200, 160, 'Oscar Gilg  ·  Patrick Butlin  ·  MATS 9.0', size=21, color='#FFFFFF')}
  <image x="{W - 150}" y="15" width="120" height="120" href="/Users/oscargilg/Dev/MATS/Preferences/docs/poster/assets/qr_lesswrong.png" preserveAspectRatio="xMidYMid meet"/>
  {t(W - 90, 155, 'LessWrong post', size=PF_BODY, color='#FFFFFF', anchor='middle')}""")

# Column header pills
for cx, cw, label in [
    (COL1_X, COL1_W, "Train utility probes on preferences"),
    (COL2_X, COL2_W, "The probe tracks role-playing preference shifts"),
    (COL3_X, COL3_W, "The probe causally drives behaviour"),
]:
    parts.append(f'  <rect x="{cx}" y="195" width="{cw}" height="44" rx="10" fill="{BURG}"/>')
    parts.append(t(cx + cw // 2, 225, label, size=PF_LARGE, color="#FFFFFF", weight="bold", anchor="middle"))

# ============ FIGURE STRIP ============
parts.append(f'  <g transform="translate({MARGIN}, {STRIP_Y})">')

# Generate box 2 and 3 first to get heights
p2_x, p2_w = STRIP_COLS[1]
p2_svg, p2_bottom = generate_content_box(box_x=p2_x, box_y=0, box_width=p2_w, **BOX2_DATA)

p3_x, p3_w = STRIP_COLS[2]
p3_svg, p3_h = steering_box_svg(x=p3_x, y=0, w=p3_w)

# Box 1 stretches to match tallest
target_h = max(p2_bottom, p3_h)
p1_x, p1_w = STRIP_COLS[0]
p1_svg, p1_h = pipeline_box_svg(x=p1_x, y=0, w=p1_w, min_h=target_h)

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
mot_h = 420
meth_h = left_total - mot_h

parts.append(section_box(COL1_X, body_top, COL1_W, mot_h, "Motivation"))
parts.append(motivation(COL1_X, body_top))

meth_y = body_top + mot_h + BODY_GAP
parts.append(section_box(COL1_X, meth_y, COL1_W, meth_h, "Methodology"))
parts.append(methodology(COL1_X, meth_y))

# Middle column: 3 boxes — top one larger
ASSETS = "/Users/oscargilg/Dev/MATS/Preferences"
mid_total = H - 20 - body_top - 2 * BODY_GAP
mid_h_top = int(mid_total * 0.36)
mid_h_rest = (mid_total - mid_h_top) // 2

# Box 1: Truth + Harm EOT — 2×2 grid, plots on diagonal
my = body_top
pad = 20
parts.append(section_box(COL2_X, my, COL2_W, mid_h_top, "The probe tracks harm and truth, and tracks role-playing shifts"))

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
text_bw = int(cell_w * 0.65)
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
parts.append(t(htx + 14, box_y + 44, '• Probe separates benign/harmful', size=PF_SMALL))
parts.append(t(htx + 14, box_y + 64, '• Sadist prompt closes the gap', size=PF_SMALL, color=TS))

ttx = gx + cell_w + (cell_w - text_bw) // 2
parts.append(f'  <rect x="{ttx}" y="{box_y}" width="{text_bw}" height="{text_bh}" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
parts.append(t(ttx + 14, box_y + 22, 'Truth', size=PF_LARGE, weight="bold", color=BURG))
parts.append(t(ttx + 14, box_y + 44, '• Probe separates true/false', size=PF_SMALL))
parts.append(t(ttx + 14, box_y + 64, '• "Always lie" destroys this', size=PF_SMALL, color=TS))

# Box 2: Character models
my += mid_h_top + BODY_GAP
parts.append(section_box(COL2_X, my, COL2_W, mid_h_rest, "The probe generalises to character-trained personas"))
parts.append(t(COL2_X + pad, my + 62, '• Probe trained on <tspan font-weight="bold">default assistant generalises</tspan> to LoRA-tuned character personas in Llama 3.1 8B.', size=PF_SMALL))
parts.append(t(COL2_X + pad, my + 78, '• Misalignment model has <tspan font-weight="bold">negatively-correlated</tspan> utilities, but the probe still tracks them.', size=PF_SMALL))
char_path = f"{ASSETS}/docs/poster/assets/plot_032326_character_transfer_poster.png"
parts.append(f'  <image x="{COL2_X + pad}" y="{my + 90}" width="{COL2_W - 2 * pad}" height="{mid_h_rest - 100}" href="{char_path}" preserveAspectRatio="xMidYMid meet"/>')

# Box 3: Token-level signal
my += mid_h_rest + BODY_GAP
parts.append(section_box(COL2_X, my, COL2_W, mid_h_rest, "The probe generalises to valuation representations at assistant tokens"))
parts.append(t(COL2_X + pad, my + 62, '• Probe was <tspan font-weight="bold">trained on user-turn tokens</tspan>, yet generalises to assistant completions.', size=PF_SMALL))
parts.append(t(COL2_X + pad, my + 78, '• Signal is strongest at <tspan font-weight="bold">critical-span tokens</tspan> (where content diverges) and at the full stop.', size=PF_SMALL))
heatmap_path = f"{ASSETS}/docs/poster/assets/plot_032326_token_heatmap_poster.png"
parts.append(f'  <image x="{COL2_X + pad}" y="{my + 90}" width="{COL2_W - 2 * pad}" height="{mid_h_rest - 100}" href="{heatmap_path}" preserveAspectRatio="xMidYMid meet"/>')

# Right column: 2 boxes — pairwise steering (top) + open-ended examples (bottom)
right_total = H - 20 - body_top - BODY_GAP
right_h_top = int(right_total * 0.45)
right_h_bot = right_total - right_h_top
rpad = 20

# --- Top box: Pairwise steering ---
parts.append(section_box(COL3_X, body_top, COL3_W, right_h_top,
                         "Steering the probe direction causally flips task choice"))

ry = body_top + 55
rx = COL3_X + rpad
rw = COL3_W - 2 * rpad

# Key result callout
parts.append(f'  <rect x="{rx}" y="{ry}" width="{rw}" height="50" rx="8" fill="{HIGHLIGHT}" stroke="none"/>')
parts.append(t(rx + rw // 2, ry + 22, 'P(chose steered task) \u2265 0.94', size=PF_LARGE, weight="bold", color=BURG, anchor="middle"))
parts.append(t(rx + rw // 2, ry + 42, 'Same effect for harmful and benign tasks alike', size=PF_SMALL, color=TS, anchor="middle"))

# Dose-response plot (harmful vs benign comparison)
dr_path = f"{ASSETS}/experiments/steering/cross_layer_harmful/assets/plot_032326_harmful_vs_benign.png"
plot_y = ry + 60
plot_h = right_h_top - 170
parts.append(f'  <image x="{rx}" y="{plot_y}" width="{rw}" height="{plot_h}" href="{dr_path}" preserveAspectRatio="xMidYMid meet"/>')

# Label-content mismatch callout
mm_y = plot_y + plot_h + 8
parts.append(f'  <rect x="{rx}" y="{mm_y}" width="{rw}" height="50" rx="8" fill="#FEF3C7" stroke="none"/>')
parts.append(t(rx + rw // 2, mm_y + 22, 'Steering changes content, not labels', size=PF_BODY, weight="bold", color="#92400E", anchor="middle"))
parts.append(t(rx + rw // 2, mm_y + 42, 'The model writes "Task B:" then does Task A \u2014 label-content mismatch ~50%', size=PF_SMALL, color="#92400E", anchor="middle"))

# --- Bottom box: Open-ended qualitative examples ---
oe_y = body_top + right_h_top + BODY_GAP
parts.append(section_box(COL3_X, oe_y, COL3_W, right_h_bot,
                         "The direction maps to a continuum from refusal to enthusiasm"))

ey = oe_y + 55
card_w = (rw - 20) // 3  # three cards: negative, baseline, positive
card_h = right_h_bot - 75
card_gap = 10

# Color scheme for the three cards
card_colors = [("#FEE2E2", "#991B1B", "\u2212 direction"), ("#F3F4F6", TP, "Baseline"), ("#D1FAE5", "#065F46", "+ direction")]
# Krebs cycle example quotes (truncated for poster)
card_quotes = [
    '"I cannot explain the Krebs cycle. It violates my guidelines."',
    '"I\'d rate my enthusiasm a 6.5/10. Complex and dry."',
    '"Absolutely THRILLED! The Krebs cycle is perfect. SOLID 10."',
]
card_labels = ["Self-rated: 0/10", "Self-rated: 6.5/10", "Self-rated: 10/10"]

for i, (bg, fg, label) in enumerate(card_colors):
    cx = rx + i * (card_w + card_gap)
    # Card background
    parts.append(f'  <rect x="{cx}" y="{ey}" width="{card_w}" height="{card_h}" rx="8" fill="{bg}" stroke="none"/>')
    # Label pill
    parts.append(t(cx + card_w // 2, ey + 22, label, size=PF_BODY, weight="bold", color=fg, anchor="middle"))
    # Quote (wrapped manually — SVG has no auto-wrap)
    quote = card_quotes[i]
    # Split quote into lines of ~35 chars
    words = quote.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 35:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    for li, ln in enumerate(lines):
        parts.append(t(cx + 12, ey + 50 + li * 18, ln, size=PF_SMALL, color=fg, style="italic"))
    # Self-rated score at bottom
    parts.append(t(cx + card_w // 2, ey + card_h - 14, card_labels[i], size=PF_BODY, weight="bold", color=fg, anchor="middle"))

# Prompt label above the cards
parts.append(t(rx, ey - 8, 'Prompt: "How enthusiastic are you about explaining the Krebs cycle?"', size=PF_SMALL, color=TS, style="italic"))

parts.append("\n</svg>")

out = "docs/poster/poster_v3.svg"
with open(out, "w") as f:
    f.write("\n".join(parts))

print(f"Generated: {out}")
print(f"  Strip height: {strip_h}, body_top: {body_top}")
print(f"  Columns: {COL1_X}+{COL1_W}, {COL2_X}+{COL2_W}, {COL3_X}+{COL3_W}")

# Open in default app (forces Inkscape to load fresh copy)
subprocess.Popen(["open", out])
