"""Create side-by-side plot: content P(steered) and valid-only P(steered)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ASSETS = Path("docs/logs/assets")
content_path = Path("experiments/steering/cross_layer_harmful/assets/plot_032426_p_steered_content.png")
valid_path = Path("experiments/steering/cross_layer_harmful/assets/plot_032426_p_steered_valid_only.png")

img_content = mpimg.imread(content_path)
img_valid = mpimg.imread(valid_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

ax1.imshow(img_content)
ax1.set_title("All completions", fontsize=11)
ax1.axis("off")

ax2.imshow(img_valid)
ax2.set_title("Valid completions only (excluding refusal/incoherence)", fontsize=11)
ax2.axis("off")

fig.suptitle("P(completed steered task) — judge content, probe L25, layer 25", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "plot_032626_steering_content_sidebyside.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved steering_content_sidebyside")
