"""Shift OOD content down by 22px in the SVG."""
import re

with open("docs/diagrams/test_pipeline.svg") as f:
    lines = f.readlines()

# OOD content starts at line 285 (cheese pair comment), shift everything after
# Need to add 22 to all y-coordinates in lines 285+
# But NOT the box rect or title (already positioned)

output = []
in_ood_content = False
for i, line in enumerate(lines, 1):
    if i >= 285 and ('y="' in line or 'cy="' in line or 'transform="translate(' in line or 'y1="' in line or 'y2="' in line):
        # Shift y values by 22
        def shift_y(m):
            val = float(m.group(1))
            if val >= 730:  # only shift values that are in the OOD content area
                return f'y="{val + 22:.0f}"'
            return m.group(0)
        def shift_cy(m):
            val = float(m.group(1))
            if val >= 730:
                return f'cy="{val + 22:.0f}"'
            return m.group(0)
        def shift_translate(m):
            x = float(m.group(1))
            y = float(m.group(2))
            if y >= 730:
                return f'translate({x:.0f}, {y + 22:.0f})'
            return m.group(0)
        def shift_y1(m):
            val = float(m.group(1))
            if val >= 730:
                return f'y1="{val + 22:.0f}"'
            return m.group(0)
        def shift_y2(m):
            val = float(m.group(1))
            if val >= 730:
                return f'y2="{val + 22:.0f}"'
            return m.group(0)

        line = re.sub(r'y="(\d+\.?\d*)"', shift_y, line)
        line = re.sub(r'cy="(\d+\.?\d*)"', shift_cy, line)
        line = re.sub(r'translate\((\d+\.?\d*),\s*(\d+\.?\d*)\)', shift_translate, line)
        line = re.sub(r'y1="(\d+\.?\d*)"', shift_y1, line)
        line = re.sub(r'y2="(\d+\.?\d*)"', shift_y2, line)

    output.append(line)

with open("docs/diagrams/test_pipeline.svg", "w") as f:
    f.writelines(output)

print("Done - shifted OOD content down by 22px")
