import numpy as np

# Predefined edges
BOX_EDGES = [
    (0, 1),  # base
    (1, 2),  # base
    (2, 3),  # base
    (3, 0),  # base
    (4, 5),  # top
    (5, 6),  # top
    (6, 7),  # top
    (7, 4),  # top
    (0, 4),  # verticals
    (1, 5),  # verticals
    (2, 6),  # verticals
    (3, 7),  # verticals
]

PYRAMID_EDGES = [
    (0, 1),  # base
    (1, 2),  # base
    (2, 3),  # base
    (3, 0),  # base
    (0, 4),  # sides
    (1, 4),  # sides
    (2, 4),  # sides
    (3, 4),  # sides
]

# Octahedron has 6 vertices and 12 edges
OCTAHEDRON_EDGES = [
    (0, 2),  # connect vertex 0 to the “equatorial” ring
    (0, 3),  # connect vertex 0 to the "equatorial" ring
    (0, 4),  # connect vertex 0 to the "equatorial" ring
    (0, 5),  # connect vertex 0 to the “equatorial” ring
    (1, 2),  # connect vertex 1 to the ring
    (1, 3),  # connect vertex 1 to the ring
    (1, 4),  # connect vertex 1 to the ring
    (1, 5),  # connect vertex 1 to the ring
    (2, 3),  # ring edges
    (3, 4),  # ring edges
    (4, 5),  # ring edges
    (5, 2),  # ring edges
]


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
