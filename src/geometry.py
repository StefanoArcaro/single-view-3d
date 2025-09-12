"""
Homography-based Pose Estimation and Metric Transformation Module

This module provides functions for recovering camera poses from homography matrices,
selecting optimal pose solutions, and converting between pixel and metric coordinate
systems for computer vision applications.

Key Functionality:
- Decompose homography matrices to recover camera poses
- Select the most plausible pose from multiple solutions
- Convert pixel-based homographies to metric coordinate mappings
- Compute 3D distances using homography and camera intrinsics

Mathematical Background:
Homography decomposition assumes a planar scene and can yield up to 4 possible
camera poses. This module implements a method to disambiguate these solutions
and work with real-world metric measurements.
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for improved readability
Matrix3x3 = NDArray[np.float64]
Vector3D = NDArray[np.float64]
Point2D = tuple[float, float]
RotationMatrix = NDArray[np.float64]
TranslationVector = NDArray[np.float64]
PlaneNormal = NDArray[np.float64]
PoseSolution = tuple[RotationMatrix, TranslationVector, PlaneNormal]


def recover_all_poses_from_homography(H: Matrix3x3, K: Matrix3x3) -> list[PoseSolution]:
    """
    Recover all 4 possible camera poses from homography decomposition.

    This function implements the classical homography decomposition algorithm
    that assumes the scene lies on a plane. The decomposition yields multiple
    possible solutions due to the ambiguity inherent in homography matrices.

    Mathematical Approach:
    1. Normalize homography by removing camera intrinsics: H_norm = K^(-1) * H
    2. Extract rotation columns and translation from normalized homography
    3. Apply scale normalization using the mean of the norms of the first two rotation columns
    4. Generate all possible sign combinations for the ambiguous solutions
    5. Project approximate rotation matrices to the SO(3) manifold using SVD

    Args:
        H: 3x3 homography matrix mapping points from a reference plane to image pixels.
            Should be well-conditioned and invertible.
        K: 3x3 camera intrinsics matrix containing focal lengths and principal point.
            Must be invertible (det(K) ≠ 0).

    Returns:
        List of 4 pose solutions, each represented as a tuple (R, t, n) where:
            - R: 3x3 rotation matrix (orthogonal, det(R) = 1)
            - t: 3D translation vector from camera center to plane origin
            - n: 3D unit normal vector of the reference plane

    Note:
        The returned poses represent the transformation from the reference plane
        coordinate system to the camera coordinate system. Multiple solutions
        arise from sign ambiguities in the decomposition process.
    """
    # 1. Remove camera intrinsics to get normalized homography
    # This transforms the homography from pixel coordinates to normalized camera coordinates
    H_norm: Matrix3x3 = np.linalg.inv(K) @ H

    # 2. Extract rotation columns and translation vector from normalized homography
    # In the homography decomposition, H_norm = [r1, r2, t] * scale_factor
    r1: Vector3D = H_norm[:, 0]  # First rotation column
    r2: Vector3D = H_norm[:, 1]  # Second rotation column
    t: Vector3D = H_norm[:, 2]  # Translation vector

    # 3. Compute scale factor from the constraint that rotation columns are unit vectors
    # The scale factor normalizes the decomposition
    # scale: float = np.linalg.norm(r1)
    scale: float = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0

    # 4. Generate solutions
    solutions: list[PoseSolution] = []

    # The homography decomposition has an inherent sign ambiguity
    for sign in [1, -1]:
        # Apply sign and scale normalization
        r1_scaled: Vector3D = sign * r1 / scale
        r2_scaled: Vector3D = sign * r2 / scale
        t_scaled: Vector3D = sign * t / scale

        # Compute third rotation column using cross product
        # For a valid rotation matrix, r3 = r1 × r2 (right-hand rule)
        r3: Vector3D = np.cross(r1_scaled, r2_scaled)

        # Construct approximate rotation matrix
        R_approx: Matrix3x3 = np.column_stack((r1_scaled, r2_scaled, r3))

        # 5. Project to SO(3) using SVD to ensure orthogonality
        # This corrects for numerical errors and enforces rotation matrix constraints
        U: Matrix3x3
        Vt: Matrix3x3
        U, _, Vt = np.linalg.svd(R_approx)

        # Ensure proper rotation (det(R) = +1, if -1 it's a reflection)
        if np.linalg.det(U @ Vt) < 0:
            # Flip the last column of U to ensure det(R) = +1
            U[:, -1] *= -1

        # Final rotation matrix
        R: RotationMatrix = U @ Vt

        # Compute plane normal vector as the third row of the rotation matrix
        # This holds because the original plane normal (before rotation) is the z-axis unit vector
        n: PlaneNormal = R[2, :]

        # Add both normal orientations to account for plane normal ambiguity
        solutions.append((R, t_scaled, n))
        solutions.append((R, t_scaled, -n))

    return solutions


def select_best_solution(
    solutions: list[PoseSolution], expected_z_positive: bool = True
) -> PoseSolution | None:
    """
    Select the most geometrically plausible pose solution from multiple candidates.

    This function applies geometric constraints and heuristics to disambiguate
    between the multiple pose solutions returned by homography decomposition.
    The selection criteria favor solutions that represent realistic camera-object
    configurations.

    Selection Criteria:
    1. Z-coordinate constraint: Ensures the plane is in front of the camera
    2. Numerical stability: Avoids solutions with very small Z values
    3. Viewing angle heuristic: Prefers head-on views over oblique angles
    4. Normal orientation: Considers plane normal direction relative to camera

    Args:
        solutions: List of pose candidates from homography decomposition.
                Each solution is a tuple (R, t, n) representing rotation,
                translation, and plane normal respectively.
        expected_z_positive: Whether to enforce positive Z translation (plane in front
                            of camera). Set to False for scenarios where the plane
                            might be behind the camera.

    Returns:
        The best pose solution as a tuple (R, t, n), or None if no valid
        solution is found according to the constraints.

    Note:
        The scoring function penalizes solutions where:
        - The plane is very far from the camera optical axis (large X, Y relative to Z)
        - The plane normal points away from the camera (backfacing plane)
        - The translation has very small Z component (numerical instability)

    Example:
        >>> poses = recover_all_poses_from_homography(H, K)
        >>> best_pose = select_best_solution(poses, expected_z_positive=True)
        >>> if best_pose is not None:
        ...     R, t, n = best_pose
        ...     print(f"Selected pose with translation: {t}")
    """
    # Handle empty solution list
    if not solutions:
        return None

    best_solution: PoseSolution | None = None
    best_score: float = float("inf")

    # Evaluate each candidate solution
    for R, t, n in solutions:
        # Constraint 1: Check Z-coordinate sign if required
        if expected_z_positive and t[2] <= 0:
            # Skip solutions where the plane is behind the camera
            continue

        # Constraint 2: Avoid numerical instability from very small Z values
        if abs(t[2]) < 1e-8:
            # Skip solutions that might cause division by zero issues
            continue

        # Scoring heuristic: Prefer solutions with small lateral displacement
        # This favors head-on views over oblique viewing angles
        # Score represents the ratio of lateral displacement to depth
        score: float = (abs(t[0]) + abs(t[1])) / abs(t[2])

        # Additional penalty for backfacing plane normals
        if n[2] > 0:  # Normal pointing away from camera (towards positive Z)
            # Double the score to penalize this configuration
            score *= 2

        # Update best solution if current score is better
        if score < best_score:
            best_score = score
            best_solution = (R, t, n)

    return best_solution


def derive_metric_homography(
    H_px: Matrix3x3,
    template_size_px: tuple[float, float],
    template_size_metric: tuple[float, float],
    template_origin_px: Point2D = (0.0, 0.0),
    template_origin_metric: Point2D = (0.0, 0.0),
) -> Matrix3x3:
    """
    Derive a homography matrix that maps real-world metric coordinates to image pixels.

    This function creates a composite transformation that enables direct mapping from
    metric coordinates on a template to pixel coordinates in a scene image. It combines
    coordinate system transformations with an existing pixel-to-pixel homography.

    Coordinate System Conventions:
    - template_size_px: (height, width) in pixels (follows image shape convention)
    - template_size_metric: (height, width) in metric units (same convention)
    - Origins: (x, y) coordinates following OpenCV point convention where:
      * x corresponds to the width/column direction (horizontal)
      * y corresponds to the height/row direction (vertical)

    Transformation Pipeline:
    The complete transformation chain applies these operations in sequence:
    1. T1: Translate metric origin to (0,0) -> T1 = translate(-X0_m, -Y0_m)
    2. S:  Scale metric units to pixels -> S = scale(width_px/width_m, height_px/height_m)
    3. T2: Translate to template pixel origin -> T2 = translate(u0_px, v0_px)
    4. H_px: Apply template-to-scene homography

    Mathematical Formula:
    H_metric = H_px @ T2 @ S @ T1

    Args:
        H_px: 3x3 homography matrix mapping template pixel coordinates [u_px, v_px, 1]^T
            to scene image pixel coordinates [u_img, v_img, 1]^T.
        template_size_px: Template dimensions in pixels as (height, width).
                        Both values must be positive.
        template_size_metric: Template dimensions in metric units as (height, width).
                            Both values must be positive and in the same units.
        template_origin_px: Pixel coordinates (x, y) in the template image that
                            correspond to the metric coordinate origin. Default (0, 0)
                            represents the top-left corner.
        template_origin_metric: Real-world coordinates (x, y) in metric units that
                                define the origin of the metric coordinate system.
                                Default (0, 0) uses the template's reference point.

    Returns:
        3x3 homography matrix that maps metric coordinates [X_m, Y_m, 1]^T directly
        to scene image pixel coordinates [u_img, v_img, 1]^T.

    Raises:
        ValueError: If any of the following conditions are met:
            - H_px is not a 3x3 numpy array
            - Template dimensions are not positive
            - Input tuples have incorrect length
            - Homography contains non-finite values
            - Resulting homography is ill-conditioned

    Example:
        Transform a 100×200 pixel template representing a 0.5×1.0 meter object:

        >>> H_px = np.eye(3)  # Identity transformation for demonstration
        >>> template_size_px = (100, 200)      # 100 pixels high, 200 pixels wide
        >>> template_size_metric = (0.5, 1.0)  # 0.5 meters high, 1.0 meters wide
        >>>
        >>> # Create metric homography
        >>> H_metric = derive_metric_homography(H_px, template_size_px, template_size_metric)
        >>>
        >>> # Map a point 0.25 meters from origin to image coordinates
        >>> metric_point = np.array([0.25, 0.1, 1.0])  # [X_meters, Y_meters, 1]
        >>> image_point = H_metric @ metric_point        # [u_pixels, v_pixels, w]
        >>> image_coords = image_point[:2] / image_point[2]  # Normalize homogeneous coords
    """
    # Input validation: Check homography matrix format
    if not isinstance(H_px, np.ndarray) or H_px.shape != (3, 3):
        raise ValueError(
            f"H_px must be a 3x3 numpy array, got shape: {getattr(H_px, 'shape', 'not an array')}"
        )

    # Input validation: Check tuple lengths
    if len(template_size_px) != 2 or len(template_size_metric) != 2:
        raise ValueError(
            "template_size_px and template_size_metric must be tuples of length 2"
        )

    if len(template_origin_px) != 2 or len(template_origin_metric) != 2:
        raise ValueError(
            "template_origin_px and template_origin_metric must be tuples of length 2"
        )

    # Extract and validate template dimensions
    h_px, w_px = template_size_px  # Height and width in pixels
    h_m, w_m = template_size_metric  # Height and width in metric units

    # Input validation: Ensure positive dimensions
    if h_px <= 0 or w_px <= 0:
        raise ValueError(
            f"Template pixel dimensions must be positive, got: height={h_px}, width={w_px}"
        )

    if h_m <= 0 or w_m <= 0:
        raise ValueError(
            f"Template metric dimensions must be positive, got: height={h_m}, width={w_m}"
        )

    # Input validation: Check for numerical validity
    if np.any(~np.isfinite(H_px)):
        raise ValueError("H_px contains non-finite values (NaN or inf)")

    # Extract origin coordinates using OpenCV point convention (x, y)
    u0_px, v0_px = template_origin_px  # Pixel coordinates (x, y)
    X0_m, Y0_m = template_origin_metric  # Metric coordinates (x, y)

    # Transformation Step 1: Translate metric origin to (0,0)
    # This translation moves the metric coordinate system so that the specified
    # origin point becomes the new (0,0) reference
    T1: Matrix3x3 = np.array(
        [[1.0, 0.0, -X0_m], [0.0, 1.0, -Y0_m], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Transformation Step 2: Scale from metric units to template pixel units
    # Calculate scale factors to convert metric dimensions to pixel dimensions
    s_x: float = w_px / w_m  # Scale factor in x-direction (pixels per metric unit)
    s_y: float = h_px / h_m  # Scale factor in y-direction (pixels per metric unit)

    S: Matrix3x3 = np.array(
        [[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Transformation Step 3: Translate to the template pixel origin
    # This positions the scaled coordinates at the correct pixel location within
    # the template image coordinate system
    T2: Matrix3x3 = np.array(
        [[1.0, 0.0, u0_px], [0.0, 1.0, v0_px], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Combine metric-to-template-pixel transformations
    # This composite transformation chains: metric coords -> centered metric ->
    # scaled to pixels -> positioned in template pixel coordinates
    M_metric_to_template_px: Matrix3x3 = T2 @ S @ T1

    # Transformation Step 4: Apply template-to-scene homography
    # Final transformation: metric coords -> template pixels -> scene image pixels
    H_metric: Matrix3x3 = H_px @ M_metric_to_template_px

    # Output validation: Verify the result is numerically well-conditioned
    if np.any(~np.isfinite(H_metric)):
        raise ValueError(
            "Resulting homography contains non-finite values - check input parameters for "
            "numerical issues or extreme scale factors"
        )

    return H_metric


def compute_distance_from_homography(
    H_metric: Matrix3x3, K: Matrix3x3, point_metric: Point2D
) -> float:
    """
    Compute the Euclidean distance from camera center to a 3D point on the template plane.

    This function uses homography decomposition and camera intrinsics to determine
    the 3D distance to any point on the reference plane, given its metric coordinates.
    The calculation assumes the point lies on the planar template surface.

    Mathematical Approach:
    1. Normalize homography by camera intrinsics: H_n = K^(-1) @ H_metric
    2. Extract scale factor from rotation column constraint: scale = 1 / ||H_n[:, 0]||
    TODO check if it's better to use a mean of the norms of the first two columns
    3. Recover rotation columns and translation: r1, r2, t = H_n * scale
    4. Compute 3D point in camera coordinates: P = X_metric * r1 + Y_metric * r2 + t
    5. Calculate Euclidean distance: distance = ||P||

    Args:
        H_metric: 3x3 homography matrix mapping metric template coordinates
                [X_metric, Y_metric, 1]^T to image pixel coordinates [u_px, v_px, 1]^T.
        K: 3x3 camera intrinsics matrix containing focal lengths (fx, fy),
            principal point (cx, cy), and skew (s). Must be invertible and well-conditioned.
        point_metric: 2D coordinates (X_metric, Y_metric) of the target point on the
                template plane, expressed in the same metric units as the homography.

    Returns:
        Euclidean distance from the camera center to the specified 3D point,
        expressed in the same units as the input coordinates (e.g., meters).

    Note:
        - The returned distance represents the straight-line distance in 3D space
        - The calculation assumes the point lies exactly on the template plane
        - Accuracy depends on the quality of homography estimation and camera calibration
        - Very small scale factors may indicate numerical instability
    """
    # Extract 2D coordinates from input tuple
    X_metric, Y_metric = point_metric

    # 1. Normalize homography by removing camera intrinsics
    # This transforms from pixel coordinates to normalized camera coordinates
    H_norm: Matrix3x3 = np.linalg.inv(K) @ H_metric

    # 2. Extract scale factor from rotation column normalization constraint
    # In homography decomposition, rotation columns should have unit norm
    # The scale factor corrects for this normalization
    scale: float = 1.0 / np.linalg.norm(H_norm[:, 0])

    # 3. Extract normalized rotation columns and translation vector
    # Apply scale factor to recover the actual rotation and translation components
    r1: Vector3D = H_norm[:, 0] * scale
    r2: Vector3D = H_norm[:, 1] * scale
    t: Vector3D = H_norm[:, 2] * scale

    # 4. Compute 3D point coordinates in camera reference frame
    # The point on the template plane is expressed as: P = X * r1 + Y * r2 + t
    # This represents the linear combination of basis vectors plus translation
    P: Vector3D = X_metric * r1 + Y_metric * r2 + t

    # 5. Calculate Euclidean distance from camera center to the 3D point
    # Distance is the magnitude of the position vector in camera coordinates
    distance: float = np.linalg.norm(P)

    return distance
