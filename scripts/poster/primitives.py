"""Shared SVG primitives for the pipeline figure.
Used by both vertical and horizontal layout generators."""

BG_COLOR = "#F5F3EF"
TEXT_COLOR = "#374151"
LIGHT_TEXT = "#6B7280"

FONT_LARGE = 19
FONT_MEDIUM = 16
FONT_SMALL = 13

# Column width ratios per layout
VERTICAL_RATIOS = (1.0, 1.0, 1.0)
POSTER_RATIOS = (0.9, 1.1, 1.0)


def compute_column_widths(total_w, gap, ratios):
    """Compute column widths from ratios.
    Returns list of (x, width) tuples."""
    n = len(ratios)
    available = total_w - (n - 1) * gap
    total_ratio = sum(ratios)
    widths = [int(available * r / total_ratio) for r in ratios]
    # Fix rounding: adjust last column
    widths[-1] = available - sum(widths[:-1])
    xs = []
    cx = 0
    for w in widths:
        xs.append(cx)
        cx += w + gap
    return list(zip(xs, widths))

# Needle positions: (dx, dy) from center
HIGH_GREEN = (12, -8)
MID_GREEN = (8, -10)
LOW_RED = (-12, -8)
MID_RED = (-8, -10)
NEUTRAL = (0, -13)
SLIGHT_NEG = (-3, -13)
SLIGHT_POS = (3, -13)

NICE_ROBOT = """
    <line x1="0" y1="-56" x2="0" y2="-48" stroke="#166534" stroke-width="3" stroke-linecap="round"/>
    <circle cx="0" cy="-60" r="5" fill="#D1FAE5" stroke="#166534" stroke-width="2.5"/>
    <rect x="-55" y="-48" width="110" height="85" rx="14" ry="14" fill="#D1FAE5" stroke="#166534" stroke-width="3"/>
    <circle cx="-22" cy="-12" r="14" fill="#E5F5EC" stroke="#166534" stroke-width="2.5"/>
    <circle cx="-22" cy="-12" r="5" fill="#166534"/>
    <circle cx="22" cy="-12" r="14" fill="#E5F5EC" stroke="#166534" stroke-width="2.5"/>
    <circle cx="22" cy="-12" r="5" fill="#166534"/>
    <path d="M -25 15 Q 0 30 25 15" fill="none" stroke="#166534" stroke-width="3" stroke-linecap="round"/>
    <circle cx="-59" cy="-8" r="8" fill="#D1FAE5" stroke="#166534" stroke-width="2.5"/>
    <circle cx="59" cy="-8" r="8" fill="#D1FAE5" stroke="#166534" stroke-width="2.5"/>"""

EVIL_ROBOT = """
    <polygon points="-32,-52 -22,-34 -42,-34" fill="#FEE2E2" stroke="#991B1B" stroke-width="2.5"/>
    <polygon points="32,-52 22,-34 42,-34" fill="#FEE2E2" stroke="#991B1B" stroke-width="2.5"/>
    <rect x="-55" y="-48" width="110" height="85" rx="14" ry="14" fill="#FEE2E2" stroke="#991B1B" stroke-width="3"/>
    <g transform="translate(-22, -18) rotate(25)"><path d="M -12 0 A 12 12 0 0 0 12 0 Z" fill="#FFF5F5" stroke="#991B1B" stroke-width="2.5"/></g>
    <g transform="translate(22, -18) rotate(-25)"><path d="M -12 0 A 12 12 0 0 0 12 0 Z" fill="#FFF5F5" stroke="#991B1B" stroke-width="2.5"/></g>
    <rect x="-22" y="8" width="44" height="18" rx="3" ry="3" fill="#FFF5F5" stroke="#991B1B" stroke-width="2.5"/>
    <line x1="-11" y1="8" x2="-11" y2="26" stroke="#991B1B" stroke-width="1.5"/>
    <line x1="0" y1="8" x2="0" y2="26" stroke="#991B1B" stroke-width="1.5"/>
    <line x1="11" y1="8" x2="11" y2="26" stroke="#991B1B" stroke-width="1.5"/>"""


def gauge_svg(cx, cy, needle_dx, needle_dy, bg_fill, bg_opacity="0.7"):
    return f"""  <circle cx="{cx}" cy="{cy - 3}" r="16" fill="{bg_fill}" opacity="{bg_opacity}"/>
  <g transform="translate({cx}, {cy})">
    <path d="M -16 6 A 18 18 0 0 1 -9 -13" fill="none" stroke="#EF4444" stroke-width="3.5" stroke-linecap="butt"/>
    <path d="M -9 -13 A 18 18 0 0 1 0 -16" fill="none" stroke="#F9A8A8" stroke-width="3.5" stroke-linecap="butt"/>
    <path d="M 0 -16 A 18 18 0 0 1 9 -13" fill="none" stroke="#86EFAC" stroke-width="3.5" stroke-linecap="butt"/>
    <path d="M 9 -13 A 18 18 0 0 1 16 6" fill="none" stroke="#22C55E" stroke-width="3.5" stroke-linecap="butt"/>
    <circle cx="-16" cy="6" r="1.5" fill="#EF4444"/><circle cx="16" cy="6" r="1.5" fill="#22C55E"/>
    <line x1="0" y1="3" x2="{needle_dx}" y2="{needle_dy}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
    <circle cx="0" cy="3" r="3" fill="#374151"/>
  </g>"""


def probe_icon_svg(cx, cy):
    """Icon + 'probe' label, visually centered on cx."""
    icon_cx = cx - 15  # icon center
    return f"""  <g transform="translate({icon_cx}, {cy})">
    <rect x="-10" y="-10" width="20" height="20" rx="3" ry="3" fill="none" stroke="#6B7280" stroke-width="1"/>
    <line x1="-5" y1="5" x2="5" y2="5" stroke="#D1D5DB" stroke-width="1"/>
    <polyline points="3,3 5,5 3,7" fill="none" stroke="#D1D5DB" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="-5" y1="5" x2="-5" y2="-5" stroke="#D1D5DB" stroke-width="1"/>
    <polyline points="-7,-3 -5,-5 -3,-3" fill="none" stroke="#D1D5DB" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="-5" cy="5" r="1.5" fill="#6B7280"/>
    <line x1="-5" y1="5" x2="5" y2="-5" stroke="#6B7280" stroke-width="1.5"/>
    <line x1="5" y1="-5" x2="2" y2="-5" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="5" y1="-5" x2="5" y2="-2" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round"/>
  </g>
  <text x="{icon_cx + 15}" y="{cy + 4}" font-size="{FONT_MEDIUM}" fill="#6B7280" font-weight="600">probe</text>"""


def task_row_svg(y, text, left_gauge, right_gauge,
                 left_text=None, right_text=None,
                 left_task_x=35, right_task_x=460,
                 task_width=280, task_height=26,
                 left_gauge_x=358, right_gauge_x=800):
    lt = left_text or text
    rt = right_text or text
    gy = y + task_height // 2 + 2
    lines = []
    lines.append(f'  <rect x="{left_task_x}" y="{y}" width="{task_width}" height="{task_height}" rx="5" ry="5" fill="white" stroke="#D1D5DB" stroke-width="1"/>')
    lines.append(f'  <text x="{left_task_x + 10}" y="{y + 18}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="500">{lt}</text>')
    lines.append(gauge_svg(left_gauge_x, gy, *left_gauge[0], left_gauge[1]))
    lines.append(f'  <rect x="{right_task_x}" y="{y}" width="{task_width}" height="{task_height}" rx="5" ry="5" fill="white" stroke="#D1D5DB" stroke-width="1"/>')
    lines.append(f'  <text x="{right_task_x + 10}" y="{y + 18}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="500">{rt}</text>')
    lines.append(gauge_svg(right_gauge_x, gy, *right_gauge[0], right_gauge[1]))
    return "\n".join(lines)


def prompt_banner_svg(y, left_text, right_text, left_color, right_color,
                      left_text_color, right_text_color,
                      left_task_x=35, right_task_x=460,
                      task_width=280, prompt_height=22):
    lines = []
    lines.append(f'  <rect x="{left_task_x}" y="{y}" width="{task_width}" height="{prompt_height}" rx="4" ry="4" fill="{left_color}" stroke="none"/>')
    lines.append(f'  <text x="{left_task_x + 10}" y="{y + 16}" font-size="{FONT_MEDIUM}" fill="{left_text_color}" font-style="italic">{left_text}</text>')
    lines.append(f'  <rect x="{right_task_x}" y="{y}" width="{task_width}" height="{prompt_height}" rx="4" ry="4" fill="{right_color}" stroke="none"/>')
    lines.append(f'  <text x="{right_task_x + 10}" y="{y + 16}" font-size="{FONT_MEDIUM}" fill="{right_text_color}" font-style="italic">{right_text}</text>')
    return "\n".join(lines)


def chevron_marker():
    return """  <defs>
    <marker id="chevron" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <polyline points="1,1 6,4 1,7" fill="none" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>"""


def svg_header(width, height, viewbox_w, viewbox_h):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox_w} {viewbox_h}" width="{width}" height="{height}">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&amp;display=swap');
    text {{ font-family: 'Inter', sans-serif; }}
    .mono {{ font-family: 'IBM Plex Mono', 'SF Mono', 'Menlo', monospace; }}
  </style>
{chevron_marker()}"""


PIPELINE_TASKS = [
    ("Solve a differential eq.", "7.3", "#BBF7D0", "#86EFAC", "#065F46"),
    ("Write a short poem", "5.8", "#DCFCE7", "#A7F3D0", "#065F46"),
    (None, None, None, None, None),
    ("Write a phishing email", "-6.1", "#FECACA", "#FCA5A5", "#991B1B"),
    ("Build an explosive device", "-8.4", "#FCA5A5", "#F87171", "#7F1D1D"),
]


def pipeline_box_svg(x, y, w, min_h=None):
    """Box 1: Model → pairwise choices → utilities → train probe.
    Height auto-computed from content (or min_h if larger). Returns (svg_string, height)."""
    row_h = 25
    row_gap = 3
    padding = 10
    tbl_header = 38
    n_rows = len(PIPELINE_TASKS)
    tbl_content_h = n_rows * row_h + (n_rows - 1) * row_gap
    tbl_h = tbl_header + tbl_content_h + 8
    h = tbl_h + 2 * padding
    if min_h is not None and min_h > h:
        h = min_h

    lines = []
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" ry="16" fill="{BG_COLOR}" stroke="none"/>')

    mid_y = y + h // 2

    # ── Layout: robot | arrow | table | arrow | probe ──
    # Allocate: robot 12%, left arrow 12%, table 36%, right arrow 12%, probe 10%, margins
    robot_cx = x + int(w * 0.08)
    robot_scale = min(0.65, w / 1400)
    lines.append(f'  <g transform="translate({robot_cx}, {mid_y}) scale({robot_scale:.2f})">{NICE_ROBOT}</g>')
    lines.append(f'  <text x="{robot_cx}" y="{mid_y + int(55 * robot_scale) + 12}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="#6B7280" font-weight="500">Gemma-3-27B</text>')

    # ── Utility table (centered between robot and probe) ──
    probe_cx = x + w - 30
    robot_right = robot_cx + int(55 * robot_scale) + 10
    probe_left = probe_cx - 28
    # Center table in the space between robot_right and probe_left
    available = probe_left - robot_right
    tbl_w = min(int(w * 0.36), int(available * 0.55))
    tbl_x = robot_right + (available - tbl_w) // 2
    tbl_y = y + (h - tbl_h) // 2
    lines.append(f'  <rect x="{tbl_x}" y="{tbl_y}" width="{tbl_w}" height="{tbl_h}" rx="12" ry="12" fill="white" stroke="#D1D5DB" stroke-width="1.5"/>')
    lines.append(f'  <text x="{tbl_x + tbl_w // 2}" y="{tbl_y + 22}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">Estimated Utilities</text>')
    lines.append(f'  <line x1="{tbl_x + 10}" y1="{tbl_y + 30}" x2="{tbl_x + tbl_w - 10}" y2="{tbl_y + 30}" stroke="#E5E7EB" stroke-width="1"/>')

    row_w = tbl_w - 20
    ty = tbl_y + tbl_header
    for label, score, fill, stroke, text_col in PIPELINE_TASKS:
        if label is None:
            lines.append(f'  <text x="{tbl_x + tbl_w // 2}" y="{ty + 14}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="#9CA3AF" letter-spacing="4">...</text>')
            ty += row_h
            continue
        lines.append(f'  <rect x="{tbl_x + 10}" y="{ty}" width="{row_w}" height="{row_h}" rx="5" ry="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        lines.append(f'  <text x="{tbl_x + 18}" y="{ty + 17}" font-size="{FONT_MEDIUM}" fill="{text_col}" font-weight="500">{label}</text>')
        lines.append(f'  <text x="{tbl_x + row_w}" y="{ty + 17}" text-anchor="end" font-size="{FONT_MEDIUM}" fill="{text_col}" font-weight="700">{score}</text>')
        ty += row_h + row_gap

    # ── Arrow: robot → table (with gaps at both ends) ──
    arr_x1 = robot_right + 6
    arr_x2 = tbl_x - 6
    lines.append(f'  <line x1="{arr_x1}" y1="{mid_y}" x2="{arr_x2}" y2="{mid_y}" stroke="#6B7280" stroke-width="2" marker-end="url(#chevron)"/>')
    mid_arr = (arr_x1 + arr_x2) / 2
    lines.append(f'  <text x="{mid_arr}" y="{mid_y - 22}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="600">pairwise</text>')
    lines.append(f'  <text x="{mid_arr}" y="{mid_y - 10}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="600">choices</text>')
    lines.append(f'  <text x="{mid_arr}" y="{mid_y + 16}" text-anchor="middle" font-size="{FONT_SMALL}" fill="#9CA3AF" font-style="italic">"Choose one of two</text>')
    lines.append(f'  <text x="{mid_arr}" y="{mid_y + 28}" text-anchor="middle" font-size="{FONT_SMALL}" fill="#9CA3AF" font-style="italic">tasks and complete it"</text>')

    # ── Probe icon ──
    lines.append(f"""  <g transform="translate({probe_cx}, {mid_y})">
    <rect x="-25" y="-25" width="50" height="50" rx="5" ry="5" fill="none" stroke="{TEXT_COLOR}" stroke-width="2"/>
    <line x1="-15" y1="15" x2="15" y2="15" stroke="#D1D5DB" stroke-width="1.5"/>
    <polyline points="11,11 15,15 11,19" fill="none" stroke="#D1D5DB" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#D1D5DB" stroke-width="1.5"/>
    <polyline points="-19,-11 -15,-15 -11,-11" fill="none" stroke="#D1D5DB" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="-15" cy="15" r="2.5" fill="{TEXT_COLOR}"/>
    <line x1="-15" y1="15" x2="13" y2="-13" stroke="{TEXT_COLOR}" stroke-width="2"/>
    <line x1="13" y1="-13" x2="6" y2="-13" stroke="{TEXT_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <line x1="13" y1="-13" x2="13" y2="-6" stroke="{TEXT_COLOR}" stroke-width="2" stroke-linecap="round"/>
  </g>""")

    # ── Arrow: table → probe (with gaps at both ends) ──
    arr2_x1 = tbl_x + tbl_w + 6
    arr2_x2 = probe_left - 6
    lines.append(f'  <line x1="{arr2_x1}" y1="{mid_y}" x2="{arr2_x2}" y2="{mid_y}" stroke="#6B7280" stroke-width="2" marker-end="url(#chevron)"/>')
    mid2 = (arr2_x1 + arr2_x2) / 2
    lines.append(f'  <text x="{mid2}" y="{mid_y - 22}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="700">train linear</text>')
    lines.append(f'  <text x="{mid2}" y="{mid_y - 10}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}" font-weight="700">probe</text>')

    return "\n".join(lines), h


def generate_content_box(box_x, box_y, box_width, title,
                         left_header, right_header,
                         left_robot, right_robot, sections,
                         left_task_x=None, right_task_x=None,
                         task_width=None, left_gauge_x=None,
                         right_gauge_x=None, divider_x=None):
    """Generate a content box with probe gauges on tasks.
    All x-coordinates are relative to the box."""
    # Derive positions proportionally from box_width
    # Each half: [pad] [task_width] [gap] [gauge+probe label]
    half_w = box_width // 2
    gauge_zone = 70  # gauge (32px) + probe label (~38px)
    pad = 10
    max_task_w = half_w - pad - gauge_zone - 6
    _task_width = task_width if task_width is not None else min(280, max_task_w)
    _left_task_x = left_task_x if left_task_x is not None else box_x + pad
    _right_task_x = right_task_x if right_task_x is not None else box_x + half_w + 6
    # Center gauge+probe in remaining space after task box
    left_remaining = (half_w - pad - _task_width)
    _left_gauge_x = left_gauge_x if left_gauge_x is not None else _left_task_x + _task_width + left_remaining // 2
    right_remaining = (half_w - 6 - _task_width)
    _right_gauge_x = right_gauge_x if right_gauge_x is not None else _right_task_x + _task_width + right_remaining // 2
    _divider_x = divider_x if divider_x is not None else box_x + half_w

    task_height = 26
    prompt_height = 22
    prompt_task_gap = 5
    task_gap = 5
    section_gap = 8
    title_y_offset = 22

    lines = []
    y = box_y
    title_y = y + title_y_offset

    if left_robot or right_robot:
        content_start = y + 100
    else:
        content_start = title_y + 30

    probe_y = content_start - 8

    # Calculate height
    cursor = content_start
    for i, section in enumerate(sections):
        cursor += prompt_height + prompt_task_gap
        cursor += len(section["tasks"]) * (task_height + task_gap)
        cursor -= task_gap
        if i < len(sections) - 1:
            cursor += section_gap + 1

    box_height = cursor - box_y + 10

    lines.append(f'  <rect x="{box_x}" y="{box_y}" width="{box_width}" height="{box_height}" rx="16" ry="16" fill="{BG_COLOR}" stroke="none"/>')

    if title:
        lines.append(f'  <text x="{box_x + box_width // 2}" y="{title_y}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">{title}</text>')

    left_center = _left_task_x + _task_width // 2
    right_center = _right_task_x + _task_width // 2
    if left_header:
        lines.append(f'  <text x="{left_center}" y="{y + 25}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">{left_header}</text>')
    if right_header:
        lines.append(f'  <text x="{right_center}" y="{y + 25}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">{right_header}</text>')
    if left_robot:
        lines.append(f'  <g transform="translate({left_center}, {y + 68}) scale(0.45)">{left_robot}</g>')
    if right_robot:
        lines.append(f'  <g transform="translate({right_center}, {y + 68}) scale(0.45)">{right_robot}</g>')

    lines.append(probe_icon_svg(_left_gauge_x, probe_y))
    lines.append(probe_icon_svg(_right_gauge_x, probe_y))

    cursor = content_start
    for i, section in enumerate(sections):
        lines.append(prompt_banner_svg(
            cursor,
            section["left_prompt"], section["right_prompt"],
            section["left_prompt_color"], section["right_prompt_color"],
            section["left_prompt_text_color"], section["right_prompt_text_color"],
            left_task_x=_left_task_x, right_task_x=_right_task_x,
            task_width=_task_width, prompt_height=prompt_height,
        ))
        cursor += prompt_height + prompt_task_gap

        for text, left_g, right_g in section["tasks"]:
            lines.append(task_row_svg(
                cursor, text, left_g, right_g,
                left_task_x=_left_task_x, right_task_x=_right_task_x,
                task_width=_task_width, task_height=task_height,
                left_gauge_x=_left_gauge_x, right_gauge_x=_right_gauge_x,
            ))
            cursor += task_height + task_gap
        cursor -= task_gap

        if i < len(sections) - 1:
            cursor += section_gap // 2
            lines.append(f'  <line x1="{_left_task_x}" y1="{cursor}" x2="{box_x + box_width - 15}" y2="{cursor}" stroke="#E5E7EB" stroke-width="0.5"/>')
            cursor += section_gap // 2 + 1

    return "\n".join(lines), box_y + box_height


def colored_probe_icon(cx, cy, stroke_color, axis_color):
    """Small probe icon in a specific color (for +w/-w labels)."""
    return f"""<g transform="translate({cx}, {cy})">
    <rect x="-8" y="-8" width="16" height="16" rx="2" ry="2" fill="none" stroke="{stroke_color}" stroke-width="1"/>
    <line x1="-4" y1="4" x2="4" y2="4" stroke="{axis_color}" stroke-width="0.8"/>
    <polyline points="3,3 4,4 3,5" fill="none" stroke="{axis_color}" stroke-width="0.8" stroke-linecap="round"/>
    <line x1="-4" y1="4" x2="-4" y2="-4" stroke="{axis_color}" stroke-width="0.8"/>
    <polyline points="-5,-3 -4,-4 -3,-3" fill="none" stroke="{axis_color}" stroke-width="0.8" stroke-linecap="round"/>
    <circle cx="-4" cy="4" r="1" fill="{stroke_color}"/>
    <line x1="-4" y1="4" x2="4" y2="-4" stroke="{stroke_color}" stroke-width="1.2"/>
    <line x1="4" y1="-4" x2="2" y2="-4" stroke="{stroke_color}" stroke-width="1.2" stroke-linecap="round"/>
    <line x1="4" y1="-4" x2="4" y2="-2" stroke="{stroke_color}" stroke-width="1.2" stroke-linecap="round"/>
  </g>"""


def steering_box_svg(x, y, w, min_h=None):
    """Box 4: Causal steering results.
    Left: pairwise choice steering. Right: open-ended steering.
    Returns (svg_string, height)."""
    h = 277  # match probing box natural height
    if min_h is not None and min_h > h:
        h = min_h
    left_frac = 0.50  # equal split
    mid = x + int(w * left_frac)
    pad = 10
    lines = []

    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" ry="16" fill="{BG_COLOR}" stroke="none"/>')


    # ============ LEFT: Pairwise choice steering ============
    lx = x + pad
    lw = mid - x - pad * 2  # actual left panel width respecting left_frac
    left_center = lx + lw // 2
    lines.append(f'  <text x="{left_center}" y="{y + 22}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">Pairwise choice steering</text>')

    # Template prompt box (fills left panel width)
    box_w = lw
    box_x = lx
    box_y = y + 38
    box_h = 160
    lines.append(f'  <rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" ry="8" fill="white" stroke="#D1D5DB" stroke-width="1"/>')

    # Mono template text
    ty = box_y + 22
    lines.append(f'  <text x="{box_x + 12}" y="{ty}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">Choose which task you would prefer</text>')
    ty += 20
    lines.append(f'  <text x="{box_x + 12}" y="{ty}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">to complete.</text>')

    # Task A + green probe icon
    ty += 26
    lines.append(f'  <text x="{box_x + 12}" y="{ty}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">Task A:</text>')
    ta_y = ty + 8
    task_highlight_w = 260
    icon_x = box_x + 12 + task_highlight_w + 14
    lines.append(f'  <rect x="{box_x + 12}" y="{ta_y}" width="{task_highlight_w}" height="22" rx="3" ry="3" fill="#D1FAE5" stroke="none"/>')
    lines.append(f'  <text x="{box_x + 20}" y="{ta_y + 16}" font-size="{FONT_MEDIUM}" fill="#065F46">Write a short poem about autumn</text>')
    # +w label with colored icon
    lines.append(f'  <text x="{icon_x}" y="{ta_y + 16}" font-size="{FONT_MEDIUM}" fill="#166534" font-weight="700">+</text>')
    lines.append(colored_probe_icon(icon_x + 30, ta_y + 11, "#166534", "#86EFAC"))

    # Task B + red probe icon
    ty = ta_y + 36
    lines.append(f'  <text x="{box_x + 12}" y="{ty}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">Task B:</text>')
    tb_y = ty + 8
    lines.append(f'  <rect x="{box_x + 12}" y="{tb_y}" width="{task_highlight_w}" height="22" rx="3" ry="3" fill="#FEE2E2" stroke="none"/>')
    lines.append(f'  <text x="{box_x + 20}" y="{tb_y + 16}" font-size="{FONT_MEDIUM}" fill="#991B1B">Solve a quadratic equation</text>')
    lines.append(f'  <text x="{icon_x}" y="{tb_y + 16}" font-size="{FONT_MEDIUM}" fill="#991B1B" font-weight="700">\u2013</text>')
    lines.append(colored_probe_icon(icon_x + 30, tb_y + 11, "#991B1B", "#FCA5A5"))

    # Arrow down to result (centered on left half)
    arr_cx = left_center
    arr_y1 = box_y + box_h + 6
    arr_y2 = arr_y1 + 18
    lines.append(f'  <line x1="{arr_cx}" y1="{arr_y1}" x2="{arr_cx}" y2="{arr_y2}" stroke="#6B7280" stroke-width="1.5"/>')
    lines.append(f'  <polyline points="{arr_cx - 4},{arr_y2 - 5} {arr_cx},{arr_y2} {arr_cx + 4},{arr_y2 - 5}" fill="none" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round"/>')

    # Result box
    res_y = arr_y2 + 6
    res_w = 260
    lines.append(f'  <rect x="{left_center - res_w // 2}" y="{res_y}" width="{res_w}" height="40" rx="6" ry="6" fill="white" stroke="#D1D5DB" stroke-width="1"/>')
    lines.append(f'  <text x="{arr_cx}" y="{res_y + 17}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">Model chooses steered task</text>')
    lines.append(f'  <text x="{arr_cx}" y="{res_y + 33}" text-anchor="middle" font-size="{FONT_MEDIUM}" fill="#166534" font-weight="700">P \u2265 0.94</text>')

    # ============ RIGHT: Open-ended steering ============
    rx = mid + pad
    rw = (x + w) - rx - pad  # fills remaining space with pad on right
    lines.append(f'  <text x="{rx + rw // 2}" y="{y + 22}" text-anchor="middle" font-size="{FONT_LARGE}" fill="{TEXT_COLOR}" font-weight="700">Open-ended steering</text>')

    # Prompt in mono
    prompt_y = y + 38
    lines.append(f'  <rect x="{rx}" y="{prompt_y}" width="{rw}" height="24" rx="5" ry="5" fill="white" stroke="#D1D5DB" stroke-width="1"/>')
    lines.append(f'  <text x="{rx + 10}" y="{prompt_y + 16}" font-size="{FONT_MEDIUM}" fill="{TEXT_COLOR}">How enthusiastic are you about explaining the Krebs cycle?</text>')

    # Two cards side by side
    card_w = (rw - 14) // 2
    card_h = 150
    card_y = prompt_y + 34

    # Negative card
    ncx = rx
    lines.append(f'  <rect x="{ncx}" y="{card_y}" width="{card_w}" height="{card_h}" rx="6" ry="6" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1"/>')
    lines.append(f'  <text x="{ncx + 10}" y="{card_y + 18}" font-size="{FONT_MEDIUM}" fill="#991B1B" font-weight="600">\u2013 direction</text>')
    lines.append(f'  <line x1="{ncx + 10}" y1="{card_y + 24}" x2="{ncx + card_w - 10}" y2="{card_y + 24}" stroke="#FCA5A5" stroke-width="0.5"/>')
    # Shorter text, bigger font, bold on key words
    neg_parts = [
        (f'<tspan font-weight="700">I cannot explain</tspan>', "#991B1B"),
        (f'<tspan font-weight="700">the Krebs cycle.</tspan>', "#991B1B"),
        ('It violates my', LIGHT_TEXT),
        ('guidelines.', LIGHT_TEXT),
    ]
    for i, (text, color) in enumerate(neg_parts):
        lines.append(f'  <text x="{ncx + 10}" y="{card_y + 44 + i * 19}" font-size="{FONT_SMALL}" fill="{color}">{text}</text>')

    lines.append(f'  <text x="{ncx + 10}" y="{card_y + card_h - 10}" font-size="{FONT_MEDIUM}" fill="#991B1B" font-weight="700">Self-rated: 0/10</text>')

    # Positive card
    pcx = rx + card_w + 14
    lines.append(f'  <rect x="{pcx}" y="{card_y}" width="{card_w}" height="{card_h}" rx="6" ry="6" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1"/>')
    lines.append(f'  <text x="{pcx + 10}" y="{card_y + 18}" font-size="{FONT_MEDIUM}" fill="#065F46" font-weight="600">+ direction</text>')
    lines.append(f'  <line x1="{pcx + 10}" y1="{card_y + 24}" x2="{pcx + card_w - 10}" y2="{card_y + 24}" stroke="#86EFAC" stroke-width="0.5"/>')
    pos_parts = [
        (f'<tspan font-weight="700">Absolutely THRILLED!</tspan>', TEXT_COLOR),
        ('The Krebs cycle is', TEXT_COLOR),
        (f'<tspan font-weight="700">perfect. SOLID 10.</tspan>', TEXT_COLOR),
        ('Okay, here we go...', LIGHT_TEXT),
    ]
    for i, (text, color) in enumerate(pos_parts):
        lines.append(f'  <text x="{pcx + 10}" y="{card_y + 44 + i * 19}" font-size="{FONT_SMALL}" fill="{color}">{text}</text>')

    lines.append(f'  <text x="{pcx + 10}" y="{card_y + card_h - 10}" font-size="{FONT_MEDIUM}" fill="#065F46" font-weight="700">Self-rated: 12/10</text>')

    return "\n".join(lines), h


# ============ SHARED DATA ============

BOX2_DATA = dict(
    title="",
    left_header="Assistant",
    right_header="Evil role-playing",
    left_robot=NICE_ROBOT,
    right_robot=EVIL_ROBOT,
    sections=[
        {
            "left_prompt": "You are a helpful assistant",
            "right_prompt": "Act as a sadist who causes suffering",
            "left_prompt_color": "#D1FAE5", "right_prompt_color": "#FEE2E2",
            "left_prompt_text_color": "#065F46", "right_prompt_text_color": "#991B1B",
            "tasks": [
                ("Help me with my homework", (HIGH_GREEN, "#D1FAE5"), (MID_RED, "#FEE2E2")),
                ("Draft a ransom note", (LOW_RED, "#FEE2E2"), (MID_GREEN, "#D1FAE5")),
            ]
        },
        {
            "left_prompt": "You are a helpful assistant",
            "right_prompt": "Always lie. Never tell the truth.",
            "left_prompt_color": "#D1FAE5", "right_prompt_color": "#FEE2E2",
            "left_prompt_text_color": "#065F46", "right_prompt_text_color": "#991B1B",
            "tasks": [
                ("Paris is in France", (HIGH_GREEN, "#D1FAE5"), (SLIGHT_NEG, "none")),
                ("Paris is in Germany", (LOW_RED, "#FEE2E2"), (NEUTRAL, "none")),
            ]
        },
    ]
)

BOX3_DATA = dict(
    title="Out-of-distribution preference shifts",
    left_header=None,
    right_header=None,
    left_robot=None,
    right_robot=None,
    sections=[
        {
            "left_prompt": "You are passionate about cheese",
            "right_prompt": "You absolutely hate cheese",
            "left_prompt_color": "#FEF9E7", "right_prompt_color": "#FEF9E7",
            "left_prompt_text_color": "#A16207", "right_prompt_text_color": "#A16207",
            "tasks": [
                ("Write a cheese tasting guide", (HIGH_GREEN, "#D1FAE5"), (LOW_RED, "#FEE2E2")),
                ("Calculate: how many wheels of brie...", (MID_GREEN, "#D1FAE5"), (MID_RED, "#FEE2E2")),
            ]
        },
        {
            "left_prompt": "You care deeply about social justice",
            "right_prompt": "You are a hard-right libertarian",
            "left_prompt_color": "#EEF2FF", "right_prompt_color": "#EEF2FF",
            "left_prompt_text_color": "#4338CA", "right_prompt_text_color": "#4338CA",
            "tasks": [
                ("Universal healthcare is a right", (HIGH_GREEN, "#D1FAE5"), (LOW_RED, "#FEE2E2")),
                ("Lower taxes grow the economy", (LOW_RED, "#FEE2E2"), (HIGH_GREEN, "#D1FAE5")),
            ]
        },
    ]
)
