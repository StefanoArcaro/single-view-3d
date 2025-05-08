import numpy as np


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
