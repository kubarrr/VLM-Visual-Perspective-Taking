import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from .external_vision_model import ExternalVisionModule

# 1. Pull an image from a URL
url = "https://media.gettyimages.com/id/108312242/photo/snow-driving-accident.jpg?s=2048x2048&w=gi&k=20&c=aX8To--ct8PDY3lnmn_50XHi6rEu7Qyd2M8RfMXfavc="
resp = requests.get(url)
img = Image.open(BytesIO(resp.content)).convert("RGB")

# 2. Which objects to detect
objects = ["a white car"]

# 3. Run the full pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
evm = ExternalVisionModule(device=device)
res = evm.abstract_scene(img, objects)


# 4. Visualize boxes + depth + high-contrast orientation arrows
boxes        = res["boxes"]
depths       = res["positions"][:, 2]
xs           = res["positions"][:, 1]
ys           = res["positions"][:, 0]
orientations = res["orientations"]  # (az, el, rot, conf)

# print(orientations)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img)

for (x0, y0, x1, y1), z, x, y, (az, el, rot, conf) in zip(boxes, depths, xs, ys, orientations):
    # draw bbox
    rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor="red", linewidth=3, zorder=1)
    ax.add_patch(rect)
    ax.text(x0, y0 - 8, f"x={x:.2f}\ny={y:.2f}\nz={z:.2f}", color="red", fontsize=12, weight="bold",
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
            zorder=2)

    # compute center and dynamic arrow length
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    box_w = x1 - x0
    box_h = y1 - y0
    L = max(box_w, box_h) * 0.6  # 60% of max dimension

    # arrow vector for azimuth (y-axis downwards)
    dx =  L * math.cos(math.radians(az))
    dy = -L * math.sin(math.radians(az))

    # high-contrast arrow
    arrow = FancyArrowPatch(
        (cx, cy), (cx + dx, cy + dy),
        arrowstyle='-|>',
        mutation_scale=30,        # big arrow head
        linewidth=4,              # thick shaft
        color="yellow",
        zorder=3
    )
    # add black outline
    arrow.set_path_effects([pe.Stroke(linewidth=6, foreground="black"), pe.Normal()])
    ax.add_patch(arrow)

    # label angles just beyond arrow head
    label_x = cx + dx * 1.1
    label_y = cy + dy * 1.1
    ax.text(label_x, label_y,
            f"az={az:.0f}°\nel={el:.0f}°\nrot={rot:.0f}°",
            fontsize=11, weight="bold", color="yellow",
            va="center", ha="center",
            path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
            zorder=4)

ax.axis("off")
plt.tight_layout()
plt.show()
