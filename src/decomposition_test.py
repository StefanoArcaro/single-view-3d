from typing import Dict, List, Tuple

import numpy as np

from geometry import recover_all_poses_from_homography, select_best_solution

# Type aliases
Matrix3x3 = np.ndarray
Vector3 = np.ndarray


def sample_random_plane(
    distance_range: Tuple[float, float] = (0.1, 10.0),
) -> Tuple[Vector3, Vector3]:
    """
    Sample a random plane in front of the camera.

    Returns:
        n_true: Unit normal vector of the plane (3,).
        t_true: Translation vector from camera to plane origin (3,).
    """
    # Sample random unit normal via Gaussian normalization
    v = np.random.randn(3)
    n_true = v / np.linalg.norm(v)
    # Sample distance along normal
    d_true = np.random.uniform(*distance_range)
    t_true = d_true * n_true
    return n_true, t_true


def rotation_matrix_x(angle_deg: float) -> Matrix3x3:
    """Return rotation matrix about X-axis by angle in degrees."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)


def rotation_matrix_y(angle_deg: float) -> Matrix3x3:
    """Return rotation matrix about Y-axis by angle in degrees."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)


def rotation_matrix_z(angle_deg: float) -> Matrix3x3:
    """Return rotation matrix about Z-axis by angle in degrees."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)


def sample_front_facing_rotation(
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
    yaw_range: Tuple[float, float] = (-90.0, 90.0),
    roll_range: Tuple[float, float] = (-180.0, 180.0),
) -> Matrix3x3:
    """
    Sample a random front-facing rotation using constrained Euler angles.

    Returns:
        R_true: 3×3 rotation matrix.
    """
    alpha = np.random.uniform(*pitch_range)  # pitch
    beta = np.random.uniform(*yaw_range)  # yaw
    gamma = np.random.uniform(*roll_range)  # roll

    R_x = rotation_matrix_x(alpha)
    R_y = rotation_matrix_y(beta)
    R_z = rotation_matrix_z(gamma)
    # Compose Z-Y-X
    return R_z @ R_y @ R_x


def construct_homography(K: Matrix3x3, R: Matrix3x3, t: Vector3) -> Matrix3x3:
    """
    Build the ideal homography H_gt = K [r1 r2 t]

    Args:
        K: Camera intrinsics (3×3).
        R: Rotation matrix (3×3).
        t: Translation vector (3,).
    Returns:
        H_gt: 3×3 homography mapping plane coords [u,v,1] to image pixels.
    """
    r1 = R[:, 0]
    r2 = R[:, 1]
    # Stack into 3×3
    H_gt = K @ np.column_stack((r1, r2, t))
    return H_gt


def compute_pose_errors(
    R_true: Matrix3x3,
    t_true: Vector3,
    R_rec: Matrix3x3,
    t_rec: Vector3,
) -> Tuple[float, float, float]:
    """
    Compute rotation error (deg), translation-direction error (deg), and scale bias (%).
    """
    # Relative rotation
    R_delta = R_rec.T @ R_true
    cos_theta = (np.trace(R_delta) - 1) / 2
    # Clamp to [-1,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_err = np.degrees(np.arccos(cos_theta))

    # Translation direction error
    t_rec_norm = t_rec / np.linalg.norm(t_rec)
    t_true_norm = t_true / np.linalg.norm(t_true)
    cos_phi = np.dot(t_rec_norm, t_true_norm)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi_err = np.degrees(np.arccos(cos_phi))

    # Scale bias
    s_err = (
        abs(np.linalg.norm(t_rec) - np.linalg.norm(t_true))
        / np.linalg.norm(t_true)
        * 100
    )

    return theta_err, phi_err, s_err


def run_homography_roundtrip_tests(
    K: Matrix3x3, num_trials: int = 1000
) -> Dict[str, List[float]]:
    """
    Orchestrate N synthetic homography decompositions and collect error metrics.

    Returns:
        A dictionary with lists of rotation errors, translation direction errors,
        and scale biases (percent).
    """
    rot_errors: List[float] = []
    dir_errors: List[float] = []
    scale_errors: List[float] = []

    for _ in range(num_trials):
        # 1. Sample plane
        _, t_true = sample_random_plane()
        # 2. Sample rotation
        R_true = sample_front_facing_rotation()
        # 3. Build homography
        H_gt = construct_homography(K, R_true, t_true)
        # 4. Decompose and select best
        solutions = recover_all_poses_from_homography(H_gt, K)
        best = select_best_solution(solutions, expected_z_positive=True)
        if best is None:
            continue  # skip if no valid solution
        R_rec, t_rec, _ = best
        # 5. Compute errors
        θ, φ, s = compute_pose_errors(
            R_true=R_true, t_true=t_true, R_rec=R_rec, t_rec=t_rec
        )
        rot_errors.append(θ)
        dir_errors.append(φ)
        scale_errors.append(s)

    return {
        "rotation_errors": rot_errors,
        "translation_direction_errors": dir_errors,
        "scale_bias_percent": scale_errors,
    }
