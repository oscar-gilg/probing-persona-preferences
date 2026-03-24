"""Generate the vertical pipeline figure SVG."""

from scripts.poster.primitives import (
    svg_header, pipeline_box_svg, generate_content_box, steering_box_svg,
    BOX2_DATA, BOX3_DATA,
    LIGHT_TEXT, FONT_MEDIUM,
)

DISPLAY_SCALE = 80

parts = []
parts.append(svg_header(900, 1000, 900, 1000))

# Box 1: Pipeline
box1_svg, box1_h = pipeline_box_svg(x=20, y=15, w=860)
parts.append(box1_svg)

# Downward arrow
arrow_y1 = 15 + box1_h + 8
arrow_y2 = arrow_y1 + 37
parts.append(f"""
  <line x1="450" y1="{arrow_y1}" x2="450" y2="{arrow_y2}" stroke="#6B7280" stroke-width="2"/>
  <polyline points="445,{arrow_y2 - 7} 450,{arrow_y2} 455,{arrow_y2 - 7}" fill="none" stroke="#6B7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="466" y="{arrow_y2 - 14}" font-size="{FONT_MEDIUM}" fill="{LIGHT_TEXT}" font-weight="500">Probe applications</text>""")

# Box 2: Assistant vs Evil
box2_top = arrow_y2 + 13
box2_svg, box2_bottom = generate_content_box(
    box_x=20, box_y=box2_top, box_width=860, **BOX2_DATA
)
parts.append(box2_svg)

# Box 3: OOD
box3_svg, box3_bottom = generate_content_box(
    box_x=20, box_y=box2_bottom + 20, box_width=860, **BOX3_DATA
)
parts.append(box3_svg)

# Box 4: Steering
steering_svg, steering_h = steering_box_svg(x=20, y=box3_bottom + 20, w=860)
parts.append(steering_svg)

# Finalize
total_height = box3_bottom + 20 + steering_h + 15
parts[0] = parts[0].replace('viewBox="0 0 900 1000"', f'viewBox="0 0 900 {total_height}"')
parts[0] = parts[0].replace('width="900" height="1000"', f'width="{900 * DISPLAY_SCALE}" height="{total_height * DISPLAY_SCALE}"')

parts.append("\n</svg>")

with open("docs/diagrams/test_pipeline.svg", "w") as f:
    f.write("\n".join(parts))

print(f"Generated vertical SVG, height: {total_height}px")
