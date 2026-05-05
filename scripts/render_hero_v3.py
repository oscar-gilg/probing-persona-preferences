import cairosvg

cairosvg.svg2png(
    url="paper/figures/panels/hero_v3_mockup.svg",
    write_to="paper/figures/panels/hero_v3_mockup.png",
    output_width=1520,
)
print("rendered hero_v3_mockup.png")
