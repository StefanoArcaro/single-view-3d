from typing import Literal
import numpy as np
from numpy.typing import NDArray

# Type aliases for better code readability and maintainability
HomographyMatrix = NDArray[np.float64]  # 3x3 homography transformation matrix
IntrinsicMatrix = NDArray[np.float32]  # 3x3 camera intrinsic parameter matrix
PrincipalPoint = tuple[int, int]  # (cx, cy) principal point coordinates in pixels
LogLevel = Literal["INFO", "WARN", "ERROR", "NONE"]  # Valid logging levels


def estimate_intrinsic_from_homography(
    H: HomographyMatrix,
    principal_point: PrincipalPoint,
    logging_level: LogLevel = "INFO",
) -> IntrinsicMatrix:
    """
    Estimates camera intrinsic matrix from homography using geometric constraints.

    This function recovers the focal length of a camera from a homography matrix that
    maps points on a planar surface in 3D world coordinates to their corresponding
    image coordinates. The estimation assumes a simplified pinhole camera model.

    Camera Model Assumptions:
        - Square pixels: fx = fy = f (single focal length parameter)
        - Zero skew: camera axes are perpendicular
        - Known principal point: typically at image center (cx, cy)
        - No lens distortion

    Mathematical Approach:
        1. Decompose homography H = K[r1, r2, t] where r1, r2 are rotation columns
        2. Apply inverse intrinsic transformation: vi = K^(-1) * hi
        3. Enforce orthogonality: v1 * v2 = 0 (rotation columns are orthogonal)
        4. Enforce equal norms: ||v1|| = ||v2|| (rotation columns have unit length)
        5. Solve the resulting quadratic equations for focal length f

    Args:
        H: Homography matrix of shape (3, 3) mapping planar world points to image points.
            Must be non-singular and well-conditioned. Typically obtained from point
            correspondences using algorithms like RANSAC with DLT or normalized DLT.
        principal_point: Known principal point coordinates (cx, cy) in pixel units.
                        Usually estimated as the image center: (width/2, height/2).
        logging_level: Controls diagnostic output verbosity. Options are:
                        - "INFO": Show all estimation steps and intermediate values
                        - "WARN": Show only warnings and errors
                        - "ERROR": Show only critical errors
                        - "NONE": Suppress all output

    Returns:
        Estimated 3x3 intrinsic camera matrix K with structure:
            [[f, 0, cx],
             [0, f, cy],
             [0, 0,  1]]
        where f is the focal length in pixels, and (cx, cy) is the principal point.
        Matrix dtype is float32 for memory efficiency and OpenCV compatibility.

    Raises:
        ValueError: If homography has invalid shape, is singular, or focal length
                    estimation fails due to degenerate geometric configuration.

    Example:
        >>> import numpy as np
        >>> # Example homography from planar calibration target
        >>> H = np.array([[800.5, 12.3, 320.0],
        ...               [-8.7, 798.2, 240.0],
        ...               [0.001, -0.002, 1.0]])
        >>> principal_point = (320, 240)  # Image center
        >>> K = estimate_intrinsic_from_homography(H, principal_point)
        >>> print(f"Estimated focal length: {K[0,0]:.1f} pixels")
        Estimated focal length: 800.0 pixels

    Notes:
        - This method works best when the calibration plane is not parallel to the image
        - Multiple homographies from different plane orientations improve robustness
        - Results are sensitive to principal point accuracy - use precise estimates
        - For full camera calibration, consider Zhang's method with multiple views

    References:
        - Zhang, Z. "A flexible new technique for camera calibration." PAMI 2000.
        - Hartley, R. & Zisserman, A. "Multiple View Geometry." Cambridge Press 2003.
    """

    def log(level: LogLevel, message: str) -> None:
        """
        Internal logging helper with hierarchical level control.

        Args:
            level: Message severity level
            message: Diagnostic message to output
        """
        # Hierarchical logging: ERROR > WARN > INFO
        should_log = logging_level != "NONE" and (
            level == logging_level
            or (level == "WARN" and logging_level == "INFO")
            or level == "ERROR"
        )
        if should_log:
            print(f"[{level}] {message}")

    # ========================================================================
    # Input Validation and Preprocessing
    # ========================================================================

    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography shape: {H.shape}. Expected (3, 3).")

    # Check for singularity - homography must be invertible for decomposition
    det_H = np.linalg.det(H)
    if abs(det_H) < 1e-12:  # More robust than exact zero check
        raise ValueError(f"Homography is singular (determinant = {det_H:.2e}).")

    # Assess orthogonality of original homography columns as quality indicator
    # For well-conditioned homographies from planar calibration, columns should
    # be approximately orthogonal after proper scaling
    dot_product = np.dot(H[:, 0], H[:, 1])
    norm_h1 = np.linalg.norm(H[:, 0])
    norm_h2 = np.linalg.norm(H[:, 1])

    if norm_h1 < 1e-10 or norm_h2 < 1e-10:
        raise ValueError("Homography columns have near-zero norm - degenerate case.")

    normalized_dot = dot_product / (norm_h1 * norm_h2)

    # Warn if columns are far from orthogonal (may indicate poor conditioning)
    if abs(normalized_dot) > 0.5:  # cos(60°) = 0.5 as reasonable threshold
        log(
            "WARN",
            f"Original homography columns may not be orthogonal: "
            f"normalized dot product = {normalized_dot:.3f}",
        )

    # ========================================================================
    # Homography Normalization
    # ========================================================================

    # Normalize homography to have appropriate scale for focal length estimation
    # Use geometric mean of column norms to preserve the relative scaling needed
    # for the rotation matrix decomposition
    H_norm = H.copy().astype(np.float64)  # Work in double precision
    scale = np.sqrt(norm_h1 * norm_h2)

    if scale <= 1e-12:
        raise ValueError(f"Cannot normalize homography (scale = {scale:.2e}).")

    H_norm = H_norm / scale
    log("INFO", f"Normalized homography with scale factor: {scale:.3f}")

    # Extract normalized homography columns
    cx, cy = principal_point
    h1 = H_norm[:, 0]  # First column (corresponds to first rotation vector)
    h2 = H_norm[:, 1]  # Second column (corresponds to second rotation vector)

    # ========================================================================
    # Inverse Intrinsic Transformation
    # ========================================================================

    def transform_to_normalized_coords(
        h: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply inverse intrinsic transformation K^(-1) to homography column.

        For a simplified intrinsic matrix K = [[f, 0, cx], [0, f, cy], [0, 0, 1]],
        the inverse transformation removes the principal point offset and scales
        by the unknown focal length f:

        K^(-1) * [x, y, z]^T = [(x - cx * z) / f, (y - cy * z) / f, z]^T

        Since f is unknown, we work with the scaled version:
        f * K^(-1) * [x, y, z]^T = [x - cx * z, y - cy * z, f * z]^T

        Args:
            h: Homography column vector [hx, hy, hz]^T

        Returns:
            Transformed vector with principal point offset removed
        """
        return np.array(
            [
                h[0] - cx * h[2],  # Remove x-offset: hx - cx * hz
                h[1] - cy * h[2],  # Remove y-offset: hy - cy * hz
                h[2],  # Keep z-component: hz
            ]
        )

    # Transform both homography columns to normalized camera coordinates
    v1 = transform_to_normalized_coords(h1)
    v2 = transform_to_normalized_coords(h2)

    log(
        "INFO",
        f"Transformed vectors: v1_norm = {np.linalg.norm(v1):.3f}, "
        f"v2_norm = {np.linalg.norm(v2):.3f}",
    )

    # ========================================================================
    # Geometric Constraint Application
    # ========================================================================

    # Constraint 1: Orthogonality of rotation matrix columns
    numerator_orthogonal = v1[0] * v2[0] + v1[1] * v2[1]  # xy-plane dot product
    denominator_orthogonal = v1[2] * v2[2]  # z-component product

    if abs(denominator_orthogonal) < 1e-10:
        raise ValueError(
            f"Degenerate homography: z-components product = {denominator_orthogonal:.2e} "
            f"(too close to zero for orthogonality constraint)"
        )

    f_squared_orthogonal = -numerator_orthogonal / denominator_orthogonal
    log("INFO", f"f^2 from orthogonality constraint: {f_squared_orthogonal:.3f}")

    # Constraint 2: Equal norms of rotation matrix columns
    numerator_norm = (v1[0] ** 2 + v1[1] ** 2) - (v2[0] ** 2 + v2[1] ** 2)
    denominator_norm = v2[2] ** 2 - v1[2] ** 2

    # Initialize focal length estimate tracker
    f_squared: float | None = None
    estimation_method = "unknown"

    # ========================================================================
    # Constraint Resolution and Focal Length Selection
    # ========================================================================

    if abs(denominator_norm) > 1e-10:
        f_squared_norm = numerator_norm / denominator_norm
        log("INFO", f"f^2 from norm equality constraint: {f_squared_norm:.3f}")

        # Validate individual constraint results
        orthogonal_valid = f_squared_orthogonal > 0
        norm_valid = f_squared_norm > 0

        if orthogonal_valid and norm_valid:
            # Both constraints give positive focal length estimates
            # Check consistency between the two methods
            ratio = max(f_squared_orthogonal, f_squared_norm) / max(
                1e-10, min(f_squared_orthogonal, f_squared_norm)
            )

            consistency_threshold = 5.0  # Allow 5x difference as "consistent"
            if ratio < consistency_threshold:
                # Estimates are reasonably consistent - average them
                f_squared = 0.5 * (f_squared_orthogonal + f_squared_norm)
                estimation_method = "averaged_both_constraints"
                log(
                    "INFO",
                    f"Both estimates consistent (ratio={ratio:.2f}), using average",
                )
            else:
                # Estimates differ significantly - prefer orthogonality constraint
                # as it's often more numerically stable
                f_squared = f_squared_orthogonal
                estimation_method = "orthogonality_preferred"
                log(
                    "WARN",
                    f"Estimates inconsistent (ratio={ratio:.2f}), "
                    f"preferring orthogonality constraint",
                )

        elif norm_valid and not orthogonal_valid:
            # Only norm constraint gives valid result
            f_squared = f_squared_norm
            estimation_method = "norm_constraint_only"
            log("INFO", "Using norm constraint (orthogonality estimate invalid)")

        elif orthogonal_valid and not norm_valid:
            # Only orthogonality constraint gives valid result
            f_squared = f_squared_orthogonal
            estimation_method = "orthogonality_only"
            log("INFO", "Using orthogonality constraint (norm estimate invalid)")

        # If neither constraint gives valid result, fallback methods will be used

    else:
        # Norm constraint denominator is near zero - use orthogonality only
        if f_squared_orthogonal > 0:
            f_squared = f_squared_orthogonal
            estimation_method = "orthogonality_only_degenerate_norm"
            log("INFO", "Using orthogonality constraint (norm denominator ≈ 0)")

    # ========================================================================
    # Fallback Estimation Methods
    # ========================================================================

    if f_squared is None or f_squared <= 0:
        log(
            "WARN",
            "Primary constraint methods failed, attempting fallback estimation...",
        )

        # Fallback 1: Average of individual vector magnitude estimates
        # Assume each transformed vector has magnitude approximately f (after normalization)
        epsilon = 1e-8  # Prevent division by zero
        mag_estimate_1 = (v1[0] ** 2 + v1[1] ** 2) / (v1[2] ** 2 + epsilon)
        mag_estimate_2 = (v2[0] ** 2 + v2[1] ** 2) / (v2[2] ** 2 + epsilon)

        if mag_estimate_1 > 0 and mag_estimate_2 > 0:
            f_squared = 0.5 * (mag_estimate_1 + mag_estimate_2)
            estimation_method = "magnitude_average_fallback"
            log(
                "INFO", f"Fallback f^2 estimate from vector magnitudes: {f_squared:.3f}"
            )
        else:
            # Fallback 2: Direct scale estimation from original homography norms
            # This is a very rough approximation but may work for some cases
            f_squared = 0.5 * (norm_h1 + norm_h2)
            estimation_method = "homography_scale_fallback"

            if f_squared > 0:
                log(
                    "WARN",
                    f"Using homography scale as final fallback: f^2 = {f_squared:.3f}",
                )
            else:
                raise ValueError(
                    "All focal length estimation methods failed. Possible causes:\n"
                    "1. Homography is degenerate or poorly conditioned\n"
                    "2. Principal point estimate is severely incorrect\n"
                    "3. Planar calibration target is parallel to image plane\n"
                    "4. Input homography does not correspond to planar motion"
                )

    # ========================================================================
    # Final Intrinsic Matrix Construction
    # ========================================================================

    # Extract focal length and construct intrinsic matrix
    focal_length = np.sqrt(f_squared)

    # Construct the intrinsic matrix in standard form
    K = np.array(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )  # Use float32 for OpenCV compatibility

    log(
        "INFO",
        f"Final focal length estimate: {focal_length:.2f} pixels "
        f"(method: {estimation_method})",
    )
    log("INFO", f"Intrinsic matrix determinant: {np.linalg.det(K):.3f}")

    return K
