import json
from typing import List, Tuple

import numpy as np
import pyvista as pv


def plot_quads(
    quad_poses: List[Tuple[np.ndarray, np.ndarray]],
    quad_sizes: Tuple[float, float] = 1.0,
    show_axes: bool = True,
    show_camera: bool = True,
    show_frustum: bool = True,
    frustum_depth: float = 1.0,
    image_resolution: Tuple[int, int] = (640, 480),
    camera_intrinsics: Tuple[float, float, float, float] = (600, 600, 320, 240),
):
    """
    Render a 3D scene showing quads at specified poses relative to a camera at the origin.

    Parameters:
        quad_poses: List of (R, t) tuples.
        quad_sizes: Single size (float or (w, h)), or a list of sizes per quad.
        ...
    """
    print(quad_sizes)

    plotter = pv.Plotter()
    plotter.set_background("gray")

    # Normalize quad_sizes to a list of (w, h)
    def normalize_size(size):
        if isinstance(size, (float, int)):
            return (float(size), float(size))
        elif isinstance(size, tuple) and len(size) == 2:
            return tuple(map(float, size))
        else:
            raise ValueError(f"Invalid quad size format: {size}")

    if isinstance(quad_sizes, list):
        if len(quad_sizes) != len(quad_poses):
            raise ValueError("Length of quad_sizes must match number of poses.")
        quad_sizes_list = [normalize_size(s) for s in quad_sizes]
    else:
        normalized = normalize_size(quad_sizes)
        quad_sizes_list = [normalized] * len(quad_poses)

    for i, ((R, t), (w, h)) in enumerate(zip(quad_poses, quad_sizes_list)):
        print(f"Quad {i + 1} @ {t} with size ({w}, {h})")
        base_quad = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=w,
            j_size=h,
        )
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        quad = base_quad.copy()
        quad.transform(transform, inplace=True)
        plotter.add_mesh(quad, color="skyblue", opacity=0.7, label=f"Quad {i + 1}")

    if show_camera:
        plotter.add_mesh(pv.Sphere(radius=0.05, center=(0, 0, 0)), color="red")

    if show_frustum:
        fx, fy, cx, cy = camera_intrinsics
        width_px, height_px = image_resolution

        corners_px = np.array(
            [[0, 0], [width_px, 0], [width_px, height_px], [0, height_px]]
        )

        frustum_points = []
        for u, v in corners_px:
            x = (u - cx) / fx
            y = (v - cy) / fy
            pt = np.array([-x * frustum_depth, -y * frustum_depth, -frustum_depth])
            frustum_points.append(pt)

        for pt in frustum_points:
            plotter.add_mesh(pv.Line((0, 0, 0), pt), color="black")

        for i in range(4):
            plotter.add_mesh(
                pv.Line(frustum_points[i], frustum_points[(i + 1) % 4]), color="black"
            )

    if show_axes:
        plotter.add_axes()

    plotter.show()


# Load pose.json
with open("pose.json", "r") as f:
    pose_data = json.load(f)

# Extract rotation and translation from the loaded data
quad_poses = []
for item in pose_data:
    R = np.array(item["R"])
    t = np.array(item["t"])
    t_size = np.array(item["t_size"])
    quad_poses.append((R, t))

# Compute R and t with respect to the camera
# as the saved poses define the camera pose with respect to the quad
quad_poses_camera = []
for R, t in quad_poses:
    # Invert the rotation and translation to get the quad pose in camera coordinates
    R_camera = R.T
    t_camera = -R.T @ t
    quad_poses_camera.append((R_camera, t_camera))

plot_quads(
    quad_poses_camera,
    # invert the t_size to have w, h
    quad_sizes=[(t_size[1], t_size[0])],
    show_axes=True,
    show_camera=True,
    show_frustum=True,
    frustum_depth=100.0,
    image_resolution=(1280, 960),
    camera_intrinsics=(894.67311331, 896.1589482, 485.2289589, 633.03618469),
)
