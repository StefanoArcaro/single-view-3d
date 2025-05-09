import numpy as np

# fmt: off

# Predefined edges for various shapes
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # base
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

PYRAMID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # base
    (0, 4), (1, 4), (2, 4), (3, 4),  # sides
]

OCTAHEDRON_EDGES = [
    (0, 2), (0, 3), (0, 4), (0, 5),  # connect vertex 0 to the “equatorial” ring
    (1, 2), (1, 3), (1, 4), (1, 5),  # connect vertex 1 to the ring
    (2, 3), (3, 4), (4, 5), (5, 2),  # ring edges
]

HOUSE_EDGES = [
    # Base rectangle
    (0, 1), (1, 2), (2, 3), (3, 0),

    # Top rectangle (flat top of body)
    (4, 5), (5, 6), (6, 7), (7, 4),

    # Vertical edges
    (0, 4), (1, 5), (2, 6), (3, 7),

    # Roof ridge
    (8, 9),

    # Roof sides from top face to ridge
    (4, 8), (5, 8),  # front triangle
    (6, 9), (7, 9),  # back triangle
]

# fmt: on


def box_geometry(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 2.0,
    flip_z: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Generate 8 corners for a box centered at the origin on the base plane Z=0.

    Note that width and height are the dimensions of the base, while depth is the
    height of the box. The box is aligned with the Z axis, and the base is at Z=0.

    Args:
        width (float): Width of the box.
        height (float): Height of the box.
        depth (float): Depth of the box.
        flip_z (bool): If True, draw the box with the top face at Z=depth. Defaults to True.

    Returns:
        vertices (np.ndarray): (8, 3) array of 3D points.
        edges (list[tuple[int, int]]): list of index pairs defining edges.
    """
    # Base corners
    base = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0],
        ],
        dtype=float,
    )

    # Top corners along Z axis
    z = -depth if flip_z else depth
    top = base + np.array([0, 0, z], dtype=float)
    vertices = np.vstack([base, top])

    return vertices, BOX_EDGES


def pyramid_geometry(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    flip_z: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Generate 5 vertices for a rectangular base pyramid.

    Note that width and height are the dimensions of the base, while depth is the
    height of the pyramid. The pyramid is aligned with the Z axis, and the base is at Z=0.

    Args:
        width (float): Width of the base.
        height (float): Height of the base.
        depth (float): Depth of the pyramid.
        flip_z (bool): If True, draw the pyramid with the apex at Z=depth. Defaults to True.

    Returns:
        vertices (np.ndarray): (5, 3) array of 3D points.
        edges (list[tuple[int, int]]): list of index pairs defining edges.
    """
    # Base corners
    base = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0],
        ],
        dtype=float,
    )

    # Apex along Z axis at center of base
    z = -depth if flip_z else depth
    apex = np.array([[width / 2, height / 2, z]], dtype=float)
    vertices = np.vstack([base, apex])

    return vertices, PYRAMID_EDGES


def octahedron_geometry(
    size: float = 1.0,
    flip_z: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Generate 6 corners for a modified octahedron with its bottom vertex at Z=0.

    The octahedron is vertically aligned with the Z axis, and the bottom tip
    lies on the base plane Z=0. The height of the octahedron is half the given size.

    Args:
        size (float): Controls the scale of the octahedron.
        flip_z (bool): If True, apex is in the negative Z direction. Defaults to True.

    Returns:
        vertices (np.ndarray): (6, 3) array of 3D points.
        edges (list[tuple[int, int]]): index pairs defining the 12 wireframe edges.
    """
    # Half-height for top vertex
    h = size
    radius = size / 2

    # Bottom vertex at Z=0
    bottom = np.array([[0.0, 0.0, 0.0]])

    # Top vertex at Z=size or -size
    z_top = -size * 2 if flip_z else size * 2
    top = np.array([[0.0, 0.0, z_top]])

    # Middle ring at half height
    z_ring = -h if flip_z else h

    ring = np.array(
        [
            [radius, 0.0, z_ring],
            [0.0, radius, z_ring],
            [-radius, 0.0, z_ring],
            [0.0, -radius, z_ring],
        ]
    )

    # Order: bottom (0), top (1), ring (2–5)
    vertices = np.vstack([bottom, top, ring])

    return vertices, OCTAHEDRON_EDGES


def house_geometry(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.5,
    flip_z: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Generate 10 vertices for a simple 'house' shape: a box base + gable roof.

    The house stands on the base Z=0, aligned with the Z axis. The body of the house
    takes up 2/3 of the depth, and the triangular roof occupies the top 1/3.

    Args:
        width (float): Width along X.
        height (float): Height along Y.
        depth (float): Total height of the house (Z axis).
        flip_z (bool): If True, house grows downward in Z. Defaults to True.

    Returns:
        vertices (np.ndarray): (10, 3) array of 3D points.
        edges (list[tuple[int, int]]): wireframe edges connecting the vertices.
    """
    # Depth breakdown
    body_depth = (2 / 3) * depth
    roof_depth = depth - body_depth

    # Z-axis direction
    z_sign = -1 if flip_z else 1

    # Base corners at Z = 0
    base = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0],
        ]
    )

    # Top of the rectangular body at Z = body_depth
    body_top_z = z_sign * body_depth
    top = base + np.array([0, 0, body_top_z])

    # Roof ridge points centered on the short sides (along Y axis)
    ridge_z = body_top_z + z_sign * roof_depth
    front_ridge = np.array([[width / 2, 0, ridge_z]])
    back_ridge = np.array([[width / 2, height, ridge_z]])

    # Stack all 10 vertices: base (0–3), top (4–7), ridge (8–9)
    vertices = np.vstack([base, top, front_ridge, back_ridge])

    return vertices, HOUSE_EDGES
