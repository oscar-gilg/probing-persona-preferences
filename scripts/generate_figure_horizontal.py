"""Generate the horizontal pipeline figure SVG (3 boxes side by side).
Training → Probing → Steering, designed as poster centerpiece."""

from scripts.poster.primitives import (
    svg_header, pipeline_box_svg, generate_content_box, steering_box_svg,
    BOX2_DATA, BOX3_DATA,
    BG_COLOR, TEXT_COLOR, LIGHT_TEXT,
    FONT_LARGE, FONT_MEDIUM,
    compute_column_widths, POSTER_RATIOS,
)

DISPLAY_SCALE = 30
TOTAL_W = 2520
GAP = 20
MARGIN = 10

cols = compute_column_widths(TOTAL_W, GAP, POSTER_RATIOS)

parts = []
parts.append(svg_header(100, 100, TOTAL_W, 1000))

# Box 2 first (to get height for box 1 min_h)
c2_x, c2_w = cols[1]
box2_svg, box2_bottom = generate_content_box(
    box_x=c2_x, box_y=MARGIN, box_width=c2_w, **BOX2_DATA
)
box2_h = box2_bottom - MARGIN

# Box 3: Steering (right)
c3_x, c3_w = cols[2]
steering_svg, steering_h = steering_box_svg(x=c3_x, y=MARGIN, w=c3_w)

# Box 1: Training (left) — stretch to match tallest
c1_x, c1_w = cols[0]
target_h = max(box2_h, steering_h)
box1_svg, box1_h = pipeline_box_svg(x=c1_x, y=MARGIN, w=c1_w, min_h=target_h)

parts.append(box1_svg)
parts.append(box2_svg)
parts.append(steering_svg)

max_h = max(box1_h, box2_h, steering_h)

# Arrows between boxes
arr_y = MARGIN + max_h // 2
parts.append(f'  <line x1="{c1_x + c1_w + 2}" y1="{arr_y}" x2="{c2_x - 2}" y2="{arr_y}" stroke="#6B7280" stroke-width="2" marker-end="url(#chevron)"/>')
parts.append(f'  <line x1="{c2_x + c2_w + 2}" y1="{arr_y}" x2="{c3_x - 2}" y2="{arr_y}" stroke="#6B7280" stroke-width="2" marker-end="url(#chevron)"/>')

# Finalize
total_h = max_h + 2 * MARGIN
parts[0] = parts[0].replace(
    f'viewBox="0 0 {TOTAL_W} 1000"',
    f'viewBox="0 0 {TOTAL_W} {total_h}"'
)
parts[0] = parts[0].replace(
    'width="100" height="100"',
    f'width="{TOTAL_W * DISPLAY_SCALE}" height="{total_h * DISPLAY_SCALE}"'
)

parts.append("\n</svg>")

with open("docs/diagrams/pipeline_horizontal.svg", "w") as f:
    f.write("\n".join(parts))

print(f"Generated horizontal SVG: {TOTAL_W}x{total_h}px")
print(f"Column widths: {[w for _, w in cols]} (ratios: {POSTER_RATIOS})")
print(f"Box heights: training={box1_h}, probing={box2_h}, steering={steering_h}")
