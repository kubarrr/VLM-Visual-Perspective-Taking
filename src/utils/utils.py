import ast
import math
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.transform import Rotation


def llm_output_to_list(output: str):
    """
    Safely decode a string representing a list (from LLM output) to a Python list.
    """
    result = ast.literal_eval(output)
    if isinstance(result, list):
        return result
    return None


def change_points_basis(
    euler_angles: np.ndarray, translation: np.ndarray, points: np.ndarray
):
    """
    Change points basis, having euler angles and translation.
    Args:
        euler_angles (np.ndarray): rotation described in euler angles
        translation (np.ndarray): translation
        positions (np.ndarray): positions of remanin

    Returns:
        np.ndarray: points in new coordinate system
    """
    euler_angles[0] = -euler_angles[0] - 180
    r = Rotation.from_euler("xyz", angles=euler_angles[:3], degrees=True)
    rotation_matrix = r.as_matrix()
    new_positions = (points - translation.reshape((1, -1))) @ rotation_matrix
    return new_positions


def get_labels_positions_without_central(results: dict, central_perspective: str):
    """
    Having results from external vision module, returns labels and positions of objects without
    central perspective object,
    Args:
        results (dict): results from vision pipeline
        central_perspective (str): object, we want to create scene from

    Returns:
        [tuple]: labels, positions
    """
    idx_central = results["labels"].index(central_perspective)
    labels = results["labels"].copy()
    positions = results["positions"]

    if central_perspective != "camera":
        # if camera is not our desired perspective, we get rid of object of interest
        labels.remove(central_perspective)
        positions = np.concatenate(
            [positions[:idx_central], positions[(idx_central + 1) :]]
        )

    return labels, positions


def save_img_with_annotation(img: Image, res: dict, save_path: str):
    # 4. Visualize boxes + depth + high-contrast orientation arrows
    boxes = res["boxes"]
    depths = res["positions"][:, 2]
    xs = res["positions"][:, 1]
    ys = res["positions"][:, 0]
    orientations = res["orientations"]  # (az, el, rot, conf)

    # print(orientations)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    for (x0, y0, x1, y1), z, x, y, (az, el, rot, conf) in zip(
        boxes, depths, xs, ys, orientations
    ):
        # draw bbox
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="red",
            linewidth=3,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 8,
            f"x={x:.2f}\ny={y:.2f}\nz={z:.2f}",
            color="red",
            fontsize=12,
            weight="bold",
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
            zorder=2,
        )

        # compute center and dynamic arrow length
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        box_w = x1 - x0
        box_h = y1 - y0
        L = max(box_w, box_h) * 0.6  # 60% of max dimension

        # arrow vector for azimuth (y-axis downwards)
        dx = L * math.cos(math.radians(az))
        dy = -L * math.sin(math.radians(az))

        # high-contrast arrow
        arrow = FancyArrowPatch(
            (cx, cy),
            (cx + dx, cy + dy),
            arrowstyle="-|>",
            mutation_scale=30,  # big arrow head
            linewidth=4,  # thick shaft
            color="yellow",
            zorder=3,
        )
        # add black outline
        arrow.set_path_effects(
            [pe.Stroke(linewidth=6, foreground="black"), pe.Normal()]
        )
        ax.add_patch(arrow)

        # label angles just beyond arrow head
        label_x = cx + dx * 1.1
        label_y = cy + dy * 1.1
        ax.text(
            label_x,
            label_y,
            f"az={az:.0f}°\nel={el:.0f}°\nrot={rot:.0f}°",
            fontsize=11,
            weight="bold",
            color="yellow",
            va="center",
            ha="center",
            path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
            zorder=4,
        )

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path)
