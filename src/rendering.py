from typing import List, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation


def plot_quads_from_camera(
    quad_poses: List[Tuple[np.ndarray, np.ndarray]],
    quad_size: float = 1.0,
    show_axes: bool = True,
    show_camera: bool = True,
):
    """
    Render a 3D scene showing quads at specified poses relative to a camera at the origin.

    Parameters:
        quad_poses: List of (R, t) tuples, where
                    R is a 3x3 rotation matrix,
                    t is a 3-element translation vector (quad center in camera frame).
        quad_size: Side length of the square quad.
        show_axes: Whether to show 3D axes at the origin.
        show_camera: Whether to show a sphere at the camera origin.
    """
    plotter = pv.Plotter()
    plotter.set_background("gray")

    # Create a base quad (square in XY plane, normal along Z)
    base_quad = pv.Plane(
        center=(0, 0, 0), direction=(0, 0, 1), i_size=quad_size, j_size=quad_size
    )

    # Transform and add each quad
    for i, (R, t) in enumerate(quad_poses):
        print(f"Quad {i + 1}: R = {R}, t = {t}")
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        quad = base_quad.copy()
        quad.transform(transform, inplace=True)
        plotter.add_mesh(quad, color="skyblue", opacity=0.7, label=f"Quad {i + 1}")

    # Show the camera center
    if show_camera:
        plotter.add_mesh(
            pv.Sphere(radius=0.1, center=(0, 0, 0)), color="red", label="Camera"
        )

    if show_axes:
        plotter.add_axes()

    plotter.show()


def demo_quads():
    quad_poses = []

    # Quad 1: Straight ahead, 3 units forward
    R1 = np.eye(3)
    t1 = np.array([0, 0, 3])
    quad_poses.append((R1, t1))

    # Quad 2: Tilted 30 deg around Y, translated to the right
    R2 = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    t2 = np.array([2, 0, 4])
    quad_poses.append((R2, t2))

    # Quad 3: Rotated 45 deg around X, above the camera
    R3 = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    t3 = np.array([0, 2, 5])
    quad_poses.append((R3, t3))

    # Quad 4: Combined X and Y rotation
    R4 = Rotation.from_euler("xy", [30, 15], degrees=True).as_matrix()
    t4 = np.array([-2, -1, 4])
    quad_poses.append((R4, t4))

    plot_quads_from_camera(quad_poses, quad_size=1.5)


# demo_quads()
