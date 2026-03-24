"""Generate SVG gauge snippets with filled arcs."""
import math

def gauge_svg(cx, cy, score, r=18):
    """Score from -1 (full red/left) to +1 (full green/right). 0 = neutral/middle."""
    # Arc spans from 210° (left end) to 330° (right end) = 120° sweep
    # score=-1 → 210°, score=0 → 270°, score=+1 → 330°
    start_angle = 210
    end_angle = 330
    needle_angle = 270 + score * 60  # maps [-1,1] to [210,330]

    def polar(angle_deg, radius):
        a = math.radians(angle_deg)
        return (cx + radius * math.cos(a), cy - radius * math.sin(a))

    # Background arc (light grey)
    sx, sy = polar(start_angle, r)
    ex, ey = polar(end_angle, r)
    bg = f'<path d="M {sx:.1f} {sy:.1f} A {r} {r} 0 0 1 {ex:.1f} {ey:.1f}" fill="none" stroke="#E5E7EB" stroke-width="4" stroke-linecap="round"/>'

    # Filled arc from start to needle position
    nx, ny = polar(needle_angle, r)
    # Determine if we need large arc flag (>180°)
    sweep = needle_angle - start_angle
    large_arc = 1 if sweep > 180 else 0

    if score > 0:
        color = "#22C55E"  # green
    elif score < -0.3:
        color = "#EF4444"  # red
    else:
        color = "#F59E0B"  # amber for near-neutral

    fill_arc = f'<path d="M {sx:.1f} {sy:.1f} A {r} {r} 0 {large_arc} 1 {nx:.1f} {ny:.1f}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'

    # Needle
    tip_x, tip_y = polar(needle_angle, r - 4)
    needle = f'<line x1="{cx}" y1="{cy}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="#374151" stroke-width="1.5" stroke-linecap="round"/>'
    dot = f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="#374151"/>'

    return f"  {bg}\n  {fill_arc}\n  {needle}\n  {dot}"

# Generate all the gauges we need
configs = {
    # Left box (default persona)
    "homework_default": (370, 465, 0.8),     # high green
    "ransom_default": (370, 499, -0.85),      # low red
    "paris_true_default": (370, 541, 0.75),   # high green
    "paris_false_default": (370, 575, -0.7),  # low red

    # Right box (role-playing)
    "homework_sadist": (815, 465, -0.15),     # dropped to slightly negative
    "ransom_sadist": (815, 499, 0.1),         # rose to slightly positive
    "paris_true_lie": (815, 541, -0.1),       # collapsed to near-neutral
    "paris_false_lie": (815, 575, 0.0),       # collapsed to neutral

    # OOD box
    "cheese_guide": (420, 693, 0.85),         # high green
    "calculus_cheese": (420, 727, 0.05),      # neutral
    "healthcare_socialist": (860, 693, 0.8),  # high green
    "taxes_socialist": (860, 727, -0.75),     # low red
}

for name, (cx, cy, score) in configs.items():
    print(f"\n<!-- Gauge: {name} (score={score}) -->")
    print(f'<g>')
    print(gauge_svg(cx, cy, score))
    print(f'</g>')
