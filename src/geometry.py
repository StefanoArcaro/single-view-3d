import numpy as np


def derive_metric_homography(
    H_px: np.ndarray,
    template_size_px: tuple[float, float],
    template_size_metric: tuple[float, float],
    template_origin_px: tuple[float, float] = (0.0, 0.0),
    template_origin_metric: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Derive a homography that maps real-world template coordinates (in metric units)
    directly to scene image pixel coordinates.

    This function combines a pixel-to-pixel homography with scaling and translation
    transformations to enable direct mapping from metric coordinates to image pixels.

    Coordinate System Conventions:
    - template_size_px: (height, width) in pixels (following image shape convention)
    - template_size_metric: (height, width) in metric units
    - Origins (both pixel and metric): (x, y) coordinates (following OpenCV point convention)
      where x corresponds to width/column direction, y to height/row direction

    Mathematical Pipeline:
    The transformation chain converts metric coordinates to image pixels through:
    1. Translate metric origin to (0,0): T1 = translate(-X0_m, -Y0_m)
    2. Scale metric units to pixels: S = scale(width_px/width_m, height_px/height_m)
    3. Translate to template pixel origin: T2 = translate(u0_px, v0_px)
    4. Apply template-to-image homography: H_px

    Final mapping: H_metric = H_px @ T2 @ S @ T1

    Args:
        H_px (np.ndarray): 3x3 homography matrix mapping template pixel coordinates
            [u_px, v_px, 1]^T to scene image pixel coordinates [u_img, v_img, 1]^T.
        template_size_px (tuple[float, float]): Template dimensions in pixels as
            (height, width). Must be positive values.
        template_size_metric (tuple[float, float]): Template dimensions in metric
            units as (height, width). Must be positive values.
        template_origin_px (tuple[float, float], optional): Pixel coordinates (x, y)
            in the template image corresponding to the metric origin. Defaults to (0, 0)
            which is the top-left corner.
        template_origin_metric (tuple[float, float], optional): Real-world coordinates
            (x, y) in metric units that serve as the origin. Defaults to (0, 0).

    Returns:
        np.ndarray: 3x3 homography matrix mapping metric coordinates [X_m, Y_m, 1]^T
            to scene image pixel coordinates [u_img, v_img, 1]^T.

    Raises:
        ValueError: If input dimensions are invalid or homography shape is incorrect.

    Example:
        >>> # Template: 100x200 pixels, represents 0.5x1.0 meters
        >>> H_px = np.eye(3)  # Identity homography for this example
        >>> template_size_px = (100, 200)      # (height, width) in pixels
        >>> template_size_metric = (0.5, 1.0)  # (height, width) in meters
        >>> H_metric = derive_metric_homography(H_px, template_size_px, template_size_metric)
        >>> # Now H_metric maps (X_meters, Y_meters, 1) -> (u_pixels, v_pixels, 1)
    """
    # Input validation
    if not isinstance(H_px, np.ndarray) or H_px.shape != (3, 3):
        raise ValueError(
            f"H_px must be a 3x3 numpy array, got shape: {getattr(H_px, 'shape', 'not an array')}"
        )

    if len(template_size_px) != 2 or len(template_size_metric) != 2:
        raise ValueError(
            "template_size_px and template_size_metric must be tuples of length 2"
        )

    if len(template_origin_px) != 2 or len(template_origin_metric) != 2:
        raise ValueError(
            "template_origin_px and template_origin_metric must be tuples of length 2"
        )

    # Validate that dimensions are positive
    h_px, w_px = template_size_px
    h_m, w_m = template_size_metric

    if h_px <= 0 or w_px <= 0:
        raise ValueError(
            f"Template pixel dimensions must be positive, got: height={h_px}, width={w_px}"
        )

    if h_m <= 0 or w_m <= 0:
        raise ValueError(
            f"Template metric dimensions must be positive, got: height={h_m}, width={w_m}"
        )

    # Check for numerical validity of the input homography
    if np.any(~np.isfinite(H_px)):
        raise ValueError("H_px contains non-finite values (NaN or inf)")

    # Extract origin coordinates (using x, y convention for points)
    u0_px, v0_px = template_origin_px  # x, y in pixel coordinates
    X0_m, Y0_m = template_origin_metric  # x, y in metric coordinates

    # Step 1: Translate metric origin to (0,0) in metric coordinate system
    # This moves the metric coordinate frame so that template_origin_metric becomes (0,0)
    T1 = np.array(
        [[1.0, 0.0, -X0_m], [0.0, 1.0, -Y0_m], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Step 2: Scale from metric units to template pixel units
    # Scale factors convert metric dimensions to pixel dimensions
    s_x = w_px / w_m  # pixels per metric unit in x (width) direction
    s_y = h_px / h_m  # pixels per metric unit in y (height) direction

    S = np.array([[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Step 3: Translate to the template pixel origin
    # This positions the scaled coordinates at the correct pixel location in the template
    T2 = np.array(
        [[1.0, 0.0, u0_px], [0.0, 1.0, v0_px], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Combine the metric-to-template-pixel transformation
    # This chain: metric coords -> centered metric -> scaled to pixels -> positioned in template
    M_metric_to_template_px = T2 @ S @ T1

    # Step 4: Apply the template-to-scene homography
    # Final transformation: metric coords -> template pixels -> scene image pixels
    H_metric = H_px @ M_metric_to_template_px

    # Verify the result is well-conditioned
    if np.any(~np.isfinite(H_metric)):
        raise ValueError(
            "Resulting homography contains non-finite values - check input parameters"
        )

    return H_metric


def derive_metric_homography_old(
    H_px: np.ndarray,
    template_size_px: tuple[float, float],
    template_size_metric: tuple[float, float],
    template_origin_px: tuple[float, float] = (0.0, 0.0),
    template_origin_metric: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Derive a homography that maps real-world template coordinates (in metric units)
    directly to scene image pixel coordinates, by scaling and translating the
    given pixel-to-pixel homography.
    The function is built to also handle different origins in the pixel and metric frames,
    but in most cases, the origin will be the top-left corner of the template image, and
    there'll be no need to specify the origins explicitly.

    How it works:
    1. Translate the metric origign to (0,0) in the metric frame.
        -> T1 = T(-X0_m, -Y0_m)
    2. Scale by pixels-per-metric
        -> S = diag(s_x, s_y, 1) where
                s_x = width_px/width_m
                s_y = height_px/height_m
    3. Translate to the template pixel origin
        -> T2 = T(u0_px, v0_px)
    4. Apply the original template-to-image homography
        -> H_px

    The final mapping is:
        H_m = H_px @ (T2 @ S @ T1)

    Args:
        - H_px (np.ndarray): A 3x3 homography matrix mapping template pixel coordinates
            [u_px, v_px, 1]^T to scene image pixel coordinates [u_img, v_img, 1]^T.
        - template_size_px (tuple[float, float]): Size of the template image in pixels
            (height, width).
        - template_size_metric (tuple[float, float]): Size of the template image in metric
            units (height, width).
        - template_origin_px (tuple[float, float], optional): Pixel coordinates in the
            template image that correspond to the metric origin (default is top-left [0,0]).
        - template_origin_metric (tuple[float, float], optional): Real-world coordinates
            (in metric units) that serve as the origin for your metric frame (default is [0,0]).

    Returns:
        - np.ndarray: A 3x3 homography matrix that maps metric coordinates
            [X_m, Y_m, 1]^T to image pixel coordinates [u_img, v_img, 1]^T.
    """
    # Validate inputs
    if H_px.shape != (3, 3):
        raise ValueError(f"Invalid homography shape: {H_px.shape}. Expected (3, 3).")
    if len(template_size_px) != 2 or len(template_size_metric) != 2:
        raise ValueError(
            "template_size_px and template_size_metric must be tuples of length 2."
        )
    if len(template_origin_px) != 2 or len(template_origin_metric) != 2:
        raise ValueError(
            "template_origin_px and template_origin_metric must be tuples of length 2."
        )

    # Unpack sizes and origins
    h_px, w_px = template_size_px
    h_m, w_m = template_size_metric
    u0_px, v0_px = template_origin_px
    X0_m, Y0_m = template_origin_metric

    # Translate metric origin to (0,0)
    T1 = np.array([[1, 0, -X0_m], [0, 1, -Y0_m], [0, 0, 1]], dtype=float)

    # Scale metric units to template pixels
    s_x = w_px / w_m
    s_y = h_px / h_m
    S = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]], dtype=float)

    # Translate to template pixel origin
    T2 = np.array([[1, 0, u0_px], [0, 1, v0_px], [0, 0, 1]], dtype=float)

    # Combined mapping from metric to template-pixel coords
    M_m_to_px = T2 @ S @ T1

    # Compose with the given template -> image homography in pixel coordinates
    H_m = H_px @ M_m_to_px

    return H_m


def compute_distance_from_homography(
    H_mm2img: np.ndarray, K: np.ndarray, point_mm: tuple[float, float]
) -> float:
    """
    Compute the distance from the camera center to a 3D point on the template plane,
    given the homography from metric coordinates to image pixels and the camera intrinsics.

    Parameters:
    -----------
    H_mm2img : np.ndarray, shape (3,3)
        Homography mapping [X_mm, Y_mm, 1]^T (real-world template coordinates in mm)
        to [u_px, v_px, 1]^T (image pixel coordinates).

    K : np.ndarray, shape (3,3)
        Camera intrinsics matrix.

    point_mm : Tuple[float, float]
        (X_mm, Y_mm) coordinates of the point on the template plane in mm.

    Returns:
    --------
    distance : float
        Euclidean distance from the camera center to the 3D point (in same units as metric homography, e.g., mm).
    """
    # Unpack the point
    X_mm, Y_mm = point_mm

    # 1) Normalize the homography by the intrinsics
    Hn = np.linalg.inv(K) @ H_mm2img

    # 2) Extract the scale factor (1/d) from the first column norm
    scale = 1.0 / np.linalg.norm(Hn[:, 0])

    # 3) Extract rotation columns and translation
    r1 = Hn[:, 0] * scale
    r2 = Hn[:, 1] * scale
    t = Hn[:, 2] * scale

    # 4) Compute the 3D point in camera coords: P = X*r1 + Y*r2 + t
    P = X_mm * r1 + Y_mm * r2 + t

    # 5) Distance is the Euclidean norm of P
    distance = np.linalg.norm(P)
    return distance


def recover_pose_from_homography_v3(
    H: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if H.shape != (3, 3) or K.shape != (3, 3):
        raise ValueError("H and K must both be 3×3 matrices.")

    # Remove camera intrinsics
    H_norm = np.linalg.inv(K) @ H

    # Divide by Frobenius norm
    frobenius_norm = np.linalg.norm(H_norm, ord="fro")
    if frobenius_norm == 0:
        raise ValueError("Degenerate homography with zero Frobenius norm.")
    H_norm /= frobenius_norm

    # Extract columns
    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]

    # Orthogonalize r1 and r2 using Gram-Schmidt
    r1_norm = r1 / np.linalg.norm(r1)
    r2_ortho = r2 - np.dot(r2, r1_norm) * r1_norm
    r2_norm = r2_ortho / np.linalg.norm(r2_ortho)
    r3_norm = np.cross(r1_norm, r2_norm)

    R_approx = np.stack((r1_norm, r2_norm, r3_norm), axis=1)

    # Project onto SO(3) via SVD
    U, _, Vt = np.linalg.svd(R_approx)

    # Fix reflection by flipping last column of U if needed
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    R = U @ Vt

    return R, t


def recover_pose_from_homography_v2(
    H: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if H.shape != (3, 3) or K.shape != (3, 3):
        raise ValueError("H and K must both be 3×3 matrices.")

    # Remove camera intrinsics
    H_norm = np.linalg.inv(K) @ H

    # Normalize by the first column norm
    scale = np.linalg.norm(H_norm[:, 0])
    if scale == 0:
        raise ValueError("Degenerate homography with zero scale.")
    H_norm /= scale

    # Extract columns
    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]

    # Orthogonalize r1 and r2 using Gram-Schmidt
    r1_norm = r1 / np.linalg.norm(r1)
    r2_ortho = r2 - np.dot(r2, r1_norm) * r1_norm
    r2_norm = r2_ortho / np.linalg.norm(r2_ortho)
    r3_norm = np.cross(r1_norm, r2_norm)

    R_approx = np.stack((r1_norm, r2_norm, r3_norm), axis=1)

    # Project onto SO(3) via SVD
    U, _, Vt = np.linalg.svd(R_approx)

    # Fix reflection by flipping last column of U if needed
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    R = U @ Vt

    return R, t


def recover_pose_from_homography(
    H: np.ndarray, K: np.ndarray, normalize_scale: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover the camera-to-plane pose (R, t) from a planar homography.

    Given a homography H that maps 3D points on the z=0 plane (up to scale)
    into the image, and the camera intrinsic matrix K, this function factors
    H into a rotation R and translation t such that:

        s [u, v, 1]^T = K [R | t] [X, Y, 0, 1]^T

    Args:
        H (np.ndarray): A 3×3 homography matrix that maps planar world points
            (X, Y, 1) to image coordinates (u, v, 1), up to a scale factor.
        K (np.ndarray): The 3×3 intrinsic camera matrix.
        normalize_scale (bool, optional): Whether to normalize the scale of the first
            two columns of the normalized homography to ensure they have equal norm.
            This removes the arbitrary scale factor of the homography. Defaults to True.

    Returns:
        R (np.ndarray): A 3×3 rotation matrix with det(R) = +1, representing the
            orientation of the camera with respect to the world plane.
        t (np.ndarray): A 3-element translation vector giving the camera position
            relative to the origin of the world plane.

    Raises:
        ValueError: If inputs are not 3×3, or if the recovered R is singular.
    """
    if H.shape != (3, 3) or K.shape != (3, 3):
        raise ValueError("H and K must both be 3×3 matrices.")

    # 1. Remove camera intrinsics
    H_norm = np.linalg.inv(K) @ H

    # 2. Normalize by scale so that ‖r1‖ ~= 1 and ‖r2‖ ~= 1
    if normalize_scale:
        norm1 = np.linalg.norm(H_norm[:, 0])
        norm2 = np.linalg.norm(H_norm[:, 1])
        scale = (norm1 + norm2) / 2.0
        if scale == 0:
            raise ValueError("Degenerate homography with zero scale.")
        H_norm /= scale

    # 3. Extract in‐plane rotation columns and translation
    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]

    # 4. Enforce orthonormality: r3 = r1 × r2
    r3 = np.cross(r1, r2)
    R_approx = np.stack((r1, r2, r3), axis=1)

    # 5. Project onto SO(3) via SVD
    # SO(3) is the space of 3×3 rotation matrices with det(R) = +1
    # We can use SVD to find the closest rotation matrix to R_approx
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # 6. Fix possible reflection
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1

    return R, t


def estimate_intrinsic_from_homography(
    H: np.ndarray, principal_point: tuple[int, int], logging_level: str = "INFO"
) -> np.ndarray:
    """
    Estimates a simplified camera intrinsic matrix K from a homography.

    Assumes the following:
    - Square pixels: fx = fy = f (single focal length).
    - Zero skew.
    - Known principal point (cx, cy), typically the image center.

    Args:
        H (np.ndarray): A (3, 3) homography matrix mapping planar 3D world points
            to image points (in pixels).
        principal_point (tuple[int, int]): Known principal point coordinates (cx, cy)
            in pixel units.
        logging_level (str): Level of logging detail ("INFO", "WARN", "ERROR", or "NONE").

    Returns:
        np.ndarray: The estimated (3, 3) intrinsic camera matrix with structure:
            [[f, 0, cx],
             [0, f, cy],
             [0, 0,  1]]

    Raises:
        ValueError: If the homography is degenerate or the focal length estimate is invalid.

    Notes:
        Uses both orthogonality and norm constraints derived from rotation vectors
        in the homography. Falls back to a norm-based approximation if necessary.
    """

    # Helper function for logging
    def log(level, message):
        if logging_level != "NONE" and (
            level == logging_level
            or (level == "WARN" and logging_level == "INFO")
            or level == "ERROR"
        ):
            print(f"[{level}] {message}")

    # Validate inputs
    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography shape: {H.shape}. Expected (3, 3).")

    # Check if homography is well-conditioned
    if np.linalg.det(H) == 0:
        raise ValueError("Homography is singular (determinant is zero).")

    # Check orthogonality of the original homography columns
    dot_product = np.dot(H[:, 0], H[:, 1])
    norm_h1 = np.linalg.norm(H[:, 0])
    norm_h2 = np.linalg.norm(H[:, 1])
    normalized_dot = dot_product / (norm_h1 * norm_h2)

    if abs(normalized_dot) > 0.5:
        log(
            "WARN",
            f"Original homography columns may not be orthogonal: normalized dot = {normalized_dot:.3f}",
        )

    # Make a copy of the homography and normalize it properly
    # Use a scaling that preserves the structure needed for camera calibration
    H_norm = H.copy()
    scale = np.sqrt(np.linalg.norm(H_norm[:, 0]) * np.linalg.norm(H_norm[:, 1]))
    if scale > 0:
        H_norm = H_norm / scale
    else:
        raise ValueError("Cannot normalize homography (scale is zero or negative).")

    # Unpack principal point and homography columns
    cx, cy = principal_point
    h1 = H_norm[:, 0]
    h2 = H_norm[:, 1]

    def transform(h):
        """
        Applies K^-1 to h assuming known (cx, cy) and unknown f.

        Returns a vector from which the principal point offset is removed.
        """
        return np.array([h[0] - cx * h[2], h[1] - cy * h[2], h[2]])

    v1 = transform(h1)
    v2 = transform(h2)

    # Orthogonality constraint
    numerator_dot = v1[0] * v2[0] + v1[1] * v2[1]
    denominator_dot = v1[2] * v2[2]

    if abs(denominator_dot) < 1e-10:
        raise ValueError(
            "Degenerate homography: z-components result in near-zero denominator"
        )

    f_squared_dot = -numerator_dot / denominator_dot
    log("INFO", f"f_squared (orthogonality constraint): {f_squared_dot}")

    # Norm equality constraint
    numerator_norm = (v1[0] ** 2 + v1[1] ** 2) - (v2[0] ** 2 + v2[1] ** 2)
    denominator_norm = v2[2] ** 2 - v1[2] ** 2

    # Initialize f_squared to None to track if we have a valid estimate
    f_squared = None

    # Process the norm-based constraint result
    if abs(denominator_norm) > 1e-10:
        f_squared_norm = numerator_norm / denominator_norm
        log("INFO", f"f_squared (norm constraint): {f_squared_norm}")

        # Track if we have valid estimates from each method
        dot_valid = f_squared_dot > 0
        norm_valid = f_squared_norm > 0

        # Case 1: Both estimates are valid
        if dot_valid and norm_valid:
            # Check if estimates are reasonably consistent
            ratio = max(f_squared_dot, f_squared_norm) / max(
                1e-10, min(f_squared_dot, f_squared_norm)
            )
            if ratio < 5.0:  # Arbitrary threshold for consistency
                f_squared = 0.5 * (f_squared_dot + f_squared_norm)
                log("INFO", "Using average of both estimates (they are consistent)")
            else:
                # Even if inconsistent, prefer the dot product estimate as it's often more reliable
                f_squared = f_squared_dot
                log(
                    "WARN",
                    f"Estimates inconsistent (ratio={ratio:.2f}). Using orthogonality-only estimate.",
                )

        # Case 2: Only norm estimate is valid
        elif norm_valid:
            f_squared = f_squared_norm
            log("INFO", "Using norm-based estimate (orthogonality estimate invalid)")

        # Case 3: Only dot product estimate is valid
        elif dot_valid:
            f_squared = f_squared_dot
            log(
                "INFO",
                "Using orthogonality-only estimate (norm-based estimate invalid)",
            )

        # Case 4: Neither estimate is valid - will be handled in fallback code
    else:
        # If denominator_norm is approximately zero
        if f_squared_dot > 0:
            f_squared = f_squared_dot
            log("INFO", "Using orthogonality-only estimate (denominator_norm ≈ 0)")

    # Fallback: only if neither method gave a valid result
    if f_squared is None or f_squared <= 0:
        log(
            "WARN",
            "No valid focal length estimate from primary methods, attempting fallback...",
        )
        # Add small epsilon to avoid division by zero
        mean_norm = 0.5 * (
            (v1[0] ** 2 + v1[1] ** 2) / (v1[2] ** 2 + 1e-8)
            + (v2[0] ** 2 + v2[1] ** 2) / (v2[2] ** 2 + 1e-8)
        )
        if mean_norm > 0:
            f_squared = mean_norm
            log("INFO", f"Fallback f_squared estimate: {f_squared}")
        else:
            # Try one last method: direct estimate from the scale of the homography
            f_squared = (norm_h1 + norm_h2) / 2
            if f_squared > 0:
                log("WARN", f"Using homography scale as final fallback: {f_squared}")
            else:
                raise ValueError(
                    "Failed to estimate a valid focal length after all fallback methods."
                )

    f = np.sqrt(f_squared)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    return K


def estimate_intrinsic_from_homography_legacy(
    H: np.ndarray, principal_point: tuple[int, int]
) -> np.ndarray:
    """
    Estimates a simplified camera intrinsic matrix K from a homography.

    Assumes the following:
    - Square pixels: fx = fy = f (single focal length).
    - Zero skew.
    - Known principal point (cx, cy), typically the image center.

    Args:
        H (np.ndarray): A (3, 3) homography matrix mapping planar 3D world points
            to image points (in pixels).
        principal_point (tuple[int, int]): Known principal point coordinates (cx, cy)
            in pixel units.

    Returns:
        np.ndarray: The estimated (3, 3) intrinsic camera matrix with structure:
            [[f, 0, cx],
            [0, f, cy],
            [0, 0,  1]]

    Raises:
        ValueError: If the homography is degenerate or the focal length estimate is invalid.

    Notes:
        Uses both orthogonality and norm constraints derived from rotation vectors
        in the homography. Falls back to using only the orthogonality constraint
        if the norm-based estimate is unstable or negative.
    """
    # Validate inputs
    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography shape: {H.shape}. Expected (3, 3).")

    # Unpack principal point and homography columns
    cx, cy = principal_point
    h1 = H[:, 0]
    h2 = H[:, 1]

    def transform(h):
        """
        Applies K^-1 to h assuming known (cx, cy) and unknown f.

        What it really does, though, is to only compute the numerator of the
        first two components of the transformed vector, as f - which would be
        the denominator - is unknown.

        Therefore, the result is a vector from which the principal point
        offset is removed.
        """
        return np.array([h[0] - cx * h[2], h[1] - cy * h[2], h[2]])

    v1 = transform(h1)
    v2 = transform(h2)

    # Orthogonality constraint
    numerator_dot = v1[0] * v2[0] + v1[1] * v2[1]
    denominator_dot = v1[2] * v2[2]

    if denominator_dot == 0:
        raise ValueError("Degenerate homography: z-components of h1/h2 are zero")

    f_squared_dot = -numerator_dot / denominator_dot

    # Norm equality constraint
    numerator_norm = (v1[0] ** 2 + v1[1] ** 2) - (v2[0] ** 2 + v2[1] ** 2)
    denominator_norm = v2[2] ** 2 - v1[2] ** 2

    # Attempt to compute f^2 using both constraints
    f_squared = f_squared_dot
    print(f"[INFO] f_squared (dot): {f_squared_dot}")

    if denominator_norm != 0:
        f_squared_norm = numerator_norm / denominator_norm
        if f_squared_norm > 0 and f_squared_dot > 0:
            f_squared = 0.5 * (f_squared_dot + f_squared_norm)
            print(f"[INFO] f_squared (norm): {f_squared_norm}")
            print("[INFO] Using average of both estimates")
        else:
            print(
                "[INFO] Using orthogonality-only estimate (norm-based estimate invalid)"
            )
    else:
        print("[INFO] Using orthogonality-only estimate (denominator_norm = 0)")

    if f_squared <= 0:
        raise ValueError(f"Invalid focal length estimate: f^2 = {f_squared}")

    f = np.sqrt(f_squared)

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    return K
