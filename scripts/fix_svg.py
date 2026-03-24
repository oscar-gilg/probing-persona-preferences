"""Fix all SVG issues in one pass."""

with open("docs/diagrams/test_pipeline.svg") as f:
    content = f.read()

# 1. Warmer background beige (less bland)
content = content.replace('fill="#EEECE2"', 'fill="#E8E0D0"')

# 2. Prompt banners: same font-size as tasks (11px), keep italic
content = content.replace('font-size="9" fill="#065F46" font-style="italic"', 'font-size="11" fill="#065F46" font-style="italic"')
content = content.replace('font-size="9" fill="#991B1B" font-style="italic"', 'font-size="11" fill="#991B1B" font-style="italic"')
content = content.replace('font-size="9" fill="#8B5E3C" font-style="italic"', 'font-size="11" fill="#8B5E3C" font-style="italic"')
content = content.replace('font-size="9" fill="#6B5344" font-style="italic"', 'font-size="11" fill="#6B5344" font-style="italic"')

# 3. PROBE → Probe (not caps)
content = content.replace('>PROBE<', '>Probe<')

# 4. Fix OOD gauge circle alignment (cy should be gauge_y - 3, not gauge_y + 19)
# Cheese row 1: gauge at translate(358,797) → circle should be cy=794
content = content.replace('<circle cx="358" cy="816"', '<circle cx="358" cy="794"')
content = content.replace('<circle cx="800" cy="816"', '<circle cx="800" cy="794"')
# Cheese row 2: gauge at translate(358,829) → circle should be cy=826
content = content.replace('<circle cx="358" cy="848"', '<circle cx="358" cy="826"')
content = content.replace('<circle cx="800" cy="848"', '<circle cx="800" cy="826"')
# Politics row 1: gauge at translate(358,897) → circle should be cy=894
content = content.replace('<circle cx="358" cy="916"', '<circle cx="358" cy="894"')
content = content.replace('<circle cx="800" cy="916"', '<circle cx="800" cy="894"')
# Politics row 2: gauge at translate(358,929) → circle should be cy=926
content = content.replace('<circle cx="358" cy="948"', '<circle cx="358" cy="926"')
content = content.replace('<circle cx="800" cy="948"', '<circle cx="800" cy="926"')

# 5. Warmer prompt banner colors
content = content.replace('fill="#F5E6D3"', 'fill="#F0D9C0"')  # cheese: warmer
content = content.replace('fill="#E8DDD3"', 'fill="#E0CDBA"')  # politics: warmer

with open("docs/diagrams/test_pipeline.svg", "w") as f:
    f.write(content)

print("Done - fixed all issues")
