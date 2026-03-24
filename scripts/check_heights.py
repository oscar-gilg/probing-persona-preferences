from scripts.poster.primitives import pipeline_box_svg, generate_content_box, steering_box_svg, BOX2_DATA

_, h1 = pipeline_box_svg(0, 0, 744)
_, h2 = generate_content_box(0, 0, 992, **BOX2_DATA)
_, h3 = steering_box_svg(0, 0, 744)
print(f"Pipeline native: {h1}")
print(f"Probing: {h2}")
print(f"Steering: {h3}")
