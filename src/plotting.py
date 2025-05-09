from typing import Literal

import cv2
import numpy as np

from src.enums import Shape
from src.shapes import box_geometry, pyramid_geometry, octahedron_geometry


def compute_anchor_point(
    template_corners_3d: np.ndarray, anchor: Literal["origin", "center"] = "center"
) -> np.ndarray:
    """
    Compute the anchor point (origin) for axes placement based on the template 3D corners.

    Args:
        template_corners_3d (np.ndarray): Array of shape (N, 3) representing the 3D corners of a planar template.
            The corners should be in the order [top-left, top-right, bottom-right, bottom-left].
        anchor (str): Either 'origin' to use the first corner as origin, or 'center' to use the centroid of the corners.

    Returns:
        np.ndarray: A (3,) array representing the chosen anchor point in world coordinates.

    Raises:
        ValueError: If an unsupported anchor value is provided.
    """
    # Reshape the corners to ensure they are in the correct format
    corners = template_corners_3d.reshape(-1, 3)

    match anchor:
        case "origin":
            # Use the first corner as the origin
            return corners[0]
        case "center":
            # Compute the centroid of the corners
            return np.mean(corners, axis=0)
        case _:
            raise ValueError(
                f"Unsupported anchor '{anchor}'. Use 'origin' or 'center'."
            )


def generate_corners_from_shape(shape: tuple[int, int]) -> np.ndarray:
    """
    Generate planar template corners from a width-height shape.

    Args:
        shape (tuple[int, int]): Width and height of the planar template in pixel units.
        # TODO maybe change to world units? Problem with camera calibration matrix

    Returns:
        np.ndarray: Array of shape (4, 3) with corners in order [top-left, top-right, bottom-right, bottom-left].
    """
    h, w = shape
    return np.array(
        [[0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0]], dtype=float
    )


def project_points(
    points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """
    Project 3D points into image coordinates using a pinhole camera model.

    Args:
        points_3d (np.ndarray): Array of shape (N, 3) representing 3D points in world coordinates.
        R (np.ndarray): Rotation matrix from world to camera (3×3).
        t (np.ndarray): Translation vector from world to camera (3,).
        K (np.ndarray): Camera intrinsic matrix (3×3).

    Returns:
        np.ndarray: Array of shape (N, 2) representing the projected 2D points in image coordinates.
    """
    # Convert rotation matrix to Rodrigues vector
    # This is needed for the cv2.projectPoints function
    r_vec, _ = cv2.Rodrigues(R)

    # Project points into image coordinates
    image_points, _ = cv2.projectPoints(points_3d, r_vec, t, K, distCoeffs=None)

    # Reshape to (N, 2)
    return image_points.reshape(-1, 2)


def draw_3d_axes(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    template_corners_3d: np.ndarray | None = None,
    template_shape: tuple[int, int] | None = None,
    anchor: Literal["origin", "center"] = "center",
    axis_scale: float = 0.5,
    thickness: int = 10,
    flip_z: bool = True,
) -> np.ndarray:
    """
    Project and draw a 3D coordinate axes frame onto an image.

    This function allows specifying the planar template either via explicit 3D corners
    or by providing a rectangular shape (width, height) in pixel units. The axes origin
    can be anchored at the template's first corner or its centroid.
    The axes (X, Y, Z) are color-coded as (red, green, blue) respectively.

    Args:
        image (np.ndarray): Input RGB or grayscale image. A copy is returned with axes drawn.
        R (np.ndarray): Rotation matrix from world to camera (3×3).
        t (np.ndarray): Translation vector from world to camera (3,).
        K (np.ndarray): Camera intrinsic matrix (3×3).
        template_corners_3d (np.ndarray, optional): Template coordinates of the template's planar corners (in pixels),
            shape (N, 3). Overrides template_shape if provided.
        template_shape (tuple[int, int], optional): Width and height of the planar template in pixel units.
            Used only if template_corners_3d is not provided.
        anchor (str): 'origin' to anchor at the first corner, 'center' to anchor at centroid. Defaults to 'center'.
        axis_scale (float): Fraction of the template's diagonal in image pixels used as axis length. Defaults to 0.5.
        thickness (int): Line thickness in pixels. Defaults to 10.
        flip_z (bool): If True, draw the Z-axis toward the camera (negative world Z direction). Defaults to True.

    Returns:
        np.ndarray: Copy of `image` with the 3D axes drawn in RGB (X=red, Y=green, Z=blue).

    Raises:
        ValueError: If the input shapes are incorrect or if neither template_corners_3d nor template_shape is provided.
    """
    # 1. Validate inputs
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R must be of shape (3, 3), got {R.shape}.")
    if t.shape != (3,):
        raise ValueError(f"Translation vector t must be of shape (3,), got {t.shape}.")
    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be of shape (3, 3), got {K.shape}.")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Image must be a 2D grayscale or 3D RGB image, got {image.ndim} dimensions."
        )

    # 2. Determine template corners
    if template_corners_3d is not None:
        corners_world = template_corners_3d.reshape(-1, 3)
    elif template_shape is not None:
        corners_world = generate_corners_from_shape(template_shape)
    else:
        raise ValueError(
            "Either template_corners_3d or template_shape must be provided."
        )

    # 3. Project template corners into image space
    image_corners = project_points(corners_world, R, t, K)

    # 4. Compute diagonal of the template in pixels
    diag_pix = np.linalg.norm(image_corners[2] - image_corners[0])

    # 5. Estimate average depth (Z) of the template in camera coordinates
    cam_coords = (R @ corners_world.T + t.reshape(3, 1)).T
    z_mean = np.mean(cam_coords[:, 2])

    # 6. Compute focal length (assume fx ≈ fy)
    f_pix = (K[0, 0] + K[1, 1]) / 2.0

    # 7. Compute axis length in world units
    axis_length = (diag_pix * axis_scale * z_mean) / f_pix

    # 8. Compute axes origin based on anchor
    origin = compute_anchor_point(corners_world, anchor)

    # 9. Build 3D axes endpoints in world coords
    axes_world = np.vstack(
        [
            origin,
            origin + np.array([axis_length, 0, 0]),
            origin + np.array([0, axis_length, 0]),
            origin + np.array([0, 0, -axis_length if flip_z else axis_length]),
        ]
    )

    # 10. Project axes endpoints into image space
    axes_image = project_points(axes_world, R, t, K).astype(int)

    # 11. Ensure the image is in RGB format
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 12. Draw the axes on the image
    origin_pt = tuple(axes_image[0])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # X=red, Y=green, Z=blue
    for i, col in enumerate(colors, start=1):
        cv2.arrowedLine(out, origin_pt, tuple(axes_image[i]), col, thickness)

    return out


def draw_3d_shape(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    shape: Shape,
    shape_pts_3d: np.ndarray | None = None,
    edges: list[tuple[int, int]] | None = None,
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    flip_z: bool = True,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 10,
) -> np.ndarray:
    """
    Draw a 3D wireframe shape (box, pyramid or custom) onto an image.

    Args:
        image (np.ndarray): Input grayscale or RGB image.
        R (np.ndarray): Rotation matrix (3×3).
        t (np.ndarray): Translation vector (3,).
        K (np.ndarray): Camera intrinsics (3×3).
        shape (Shape): Enum specifying the shape.
            If CUSTOM, shape_pts_3d and edges must be provided.
        shape_pts_3d (np.ndarray, optional): 3D points of the shape (N, 3).
        edges (list[tuple[int, int]], optional): List of edges defined by pairs of indices.
        width (float): Width of the box or pyramid base.
        height (float): Height of the box or pyramid.
        depth (float): Depth of the box or pyramid.
        flip_z (bool): If True, draw the shape with its top at Z=depth.
        color (tuple): RGB line color.
        thickness (int): Line thickness.

    Returns:
        np.ndarray: Copy of image with shape drawn in RGB.

    Raises:
        ValueError: For invalid inputs.
    """
    # Validate core inputs
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3×3, got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"t must be 3-vector, got {t.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K.shape}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim} dims")

    # Determine geometry
    match shape:
        case Shape.BOX:
            pts3d, edges_list = box_geometry(width, height, depth, flip_z)
        case Shape.PYRAMID:
            pts3d, edges_list = pyramid_geometry(width, height, depth, flip_z)
        case Shape.OCTAHEDRON:
            pts3d, edges_list = octahedron_geometry(depth, flip_z)
        case Shape.CUSTOM:
            if shape_pts_3d is None or edges is None:
                raise ValueError("CUSTOM shape requires both shape_pts_3d and edges")
            pts3d, edges_list = shape_pts_3d, edges
        case _:
            raise ValueError(f"Unsupported shape: {shape}")

    # Project the vertices into image space
    img_pts = project_points(pts3d, R, t, K).astype(int)

    # Draw the edges
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i, j in edges_list:
        cv2.line(out, tuple(img_pts[i]), tuple(img_pts[j]), color, thickness)

    return out


# =============================================================

# This function is mostly for my own understanding, as it should just plot a point as the
# camera gaze, as the arrow show be going in the Z direction, which is the camera's forward axis.


def draw_camera_gaze(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    origin_3d: np.ndarray = np.array([0.5, 0.5, 0.0]),
    length: float = 0.5,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
    tip_length: float = 0.1,
) -> np.ndarray:
    """
    Draw the camera’s gaze (forward Z) as an arrow, starting from a point on the world plane.

    Args:
        image (ndarray): BGR image to draw on.
        R (ndarray): 3×3 rotation (world→camera).
        t (ndarray): 3×1 translation (world→camera).
        K (ndarray): 3×3 intrinsic matrix.
        origin_3d (ndarray): 3D point (X,Y,Z) on your plane where the arrow starts.
        length (float): Arrow length in the same units as origin_3d.
        color (tuple): BGR color of the arrow.
        thickness (int): Line thickness.
        tip_length (float): Proportion of arrow length for the head.

    Returns:
        ndarray: Copy of image with gaze arrow drawn.
    """
    # 1. Compute the camera’s forward axis in world coords
    gaze_world = R.T[:, 2]

    # 2. Define arrow endpoints in world space
    P0 = origin_3d
    P1 = origin_3d + length * gaze_world

    pts_3d = np.vstack([P0, P1]).astype(np.float32).reshape(-1, 3)

    # 3. Project into image
    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(pts_3d, rvec, t, K, None)
    p0, p1 = img_pts.reshape(-1, 2).astype(int)

    # 4. Draw arrow
    out = image.copy()
    cv2.arrowedLine(out, tuple(p0), tuple(p1), color, thickness, tipLength=tip_length)

    return out


def draw_box(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    corners_3d: np.ndarray | None = None,
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 2.0,
    edge_color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 10,
    flip_z: bool = True,
) -> np.ndarray:
    """
    Draw a 3D box by projecting and connecting its corners.

    If `corners` is None, constructs a default boc with dimensions (width, height, depth),
    base on Z=0, and top on Z=depth.

    Args:
        image (np.ndarray): Input RGB or grayscale image. A copy is returned with the box drawn.
        R (np.ndarray): Rotation matrix from world to camera (3×3).
        t (np.ndarray): Translation vector from world to camera (3,).
        K (np.ndarray): Camera intrinsic matrix (3×3).
        corners_3d (np.ndarray, optional): Array of shape (8, 3) representing the 3D corners of the box.
            If None, a default box is constructed.
        width (float): Width of the box in world units. Defaults to 1.0.
        height (float): Height of the box in world units. Defaults to 1.0.
        depth (float): Depth of the box in world units. Defaults to 2.0.
        edge_color (tuple[int, int, int]): RGB color of the box edges. Defaults to (0, 255, 255).
        thickness (int): Line thickness in pixels. Defaults to 10.
        flip_z (bool): If True, draw the box with the top face at Z=depth. Defaults to True.

    Returns:
        np.ndarray: Copy of `image` with the 3D box drawn in RGB.

    Raises:
        ValueError: If the input shapes are incorrect or if the image is not in the expected format.
    """
    # 1. Validate inputs
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R must be of shape (3, 3), got {R.shape}.")
    if t.shape != (3,):
        raise ValueError(f"Translation vector t must be of shape (3,), got {t.shape}.")
    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be of shape (3, 3), got {K.shape}.")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Image must be a 2D grayscale or 3D RGB image, got {image.ndim} dimensions."
        )
    if corners_3d is not None and (corners_3d.ndim != 2 or corners_3d.shape[1] != 3):
        raise ValueError(f"corners_3d must be of shape (8, 3), got {corners_3d.shape}.")

    # 2. Generate default box corners if not provided
    if corners_3d is None:
        base = np.array(
            [[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=float
        )
        offset = [0, 0, -depth] if flip_z else [0, 0, depth]
        top = base + np.array(offset, dtype=float)
        corners_3d = np.vstack([base, top])

    # Edges: 4 base, 4 top, 4 vertical
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # base
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # verticals
    ]

    # Draw the box
    return draw_3d_shape(image, R, t, K, corners_3d, edges, edge_color, thickness)


def draw_pyramid(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    vertices_3d: np.ndarray | None = None,
    base_width: float = 1.0,
    base_depth: float = 1.0,
    height: float = 1.0,
    edge_color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 10,
    flip_z: bool = True,
) -> np.ndarray:
    """
    Draw a 3D rectangular-based pyramid by projecting and connecting its corners.

    If `vertices_3d` is None, constructs a default pyramid with base on Z=0 and apex along +Z.

    Args:
        image (np.ndarray): Input RGB or grayscale image. A copy is returned with the pyramid drawn.
        R (np.ndarray): Rotation matrix from world to camera (3×3).
        t (np.ndarray): Translation vector from world to camera (3,).
        K (np.ndarray): Camera intrinsic matrix (3×3).
        vertices_3d (np.ndarray, optional): Array of shape (5, 3) representing the 3D pyramid vertices.
            If None, a default pyramid is constructed.
        base_width (float): Width of the pyramid base in world units (along X). Defaults to 1.0.
        base_depth (float): Depth of the pyramid base in world units (along Y). Defaults to 1.0.
        height (float): Height of the pyramid from base to apex along Z. Defaults to 1.0.
        edge_color (tuple[int, int, int]): RGB color of the pyramid edges. Defaults to (0, 255, 255).
        thickness (int): Line thickness in pixels. Defaults to 10.
        flip_z (bool): If True, draw the pyramid with the apex along +Z. Defaults to True.

    Returns:
        np.ndarray: Copy of `image` with the 3D pyramid drawn in RGB.

    Raises:
        ValueError: If the input shapes are incorrect or if the image is not in the expected format.
    """
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R must be of shape (3, 3), got {R.shape}.")
    if t.shape != (3,):
        raise ValueError(f"Translation vector t must be of shape (3,), got {t.shape}.")
    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix K must be of shape (3, 3), got {K.shape}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim} dimensions.")
    if vertices_3d is not None and (vertices_3d.ndim != 2 or vertices_3d.shape[1] != 3):
        raise ValueError(
            f"vertices_3d must be of shape (5, 3), got {vertices_3d.shape}."
        )

    # Construct default pyramid geometry if needed
    if vertices_3d is None:
        base = np.array(
            [
                [0, 0, 0],
                [base_width, 0, 0],
                [base_width, base_depth, 0],
                [0, base_depth, 0],
            ],
            dtype=float,
        )
        apex_h = -height if flip_z else height
        apex = np.array([[base_width / 2, base_depth / 2, apex_h]], dtype=float)
        vertices_3d = np.vstack([base, apex])

    # Define edges (base perimeter + sides to apex)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # base
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),  # sides
    ]

    return draw_3d_shape(image, R, t, K, vertices_3d, edges, edge_color, thickness)
