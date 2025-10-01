import cv2
import numpy as np

from src.models.template import Template


class CalibrationSimple:
    """
    Adapted Zhang's camera calibration method for estimating intrinsic parameters
    from homographies between multiple planar templates in a single scene.
    This version assumes zero skew (s = 0), known principal point (cx, cy), and
    square pixels (fx = fy).
    """

    def __init__(self):
        self.intrinsic_matrix = None
        self.homographies = []

    def _build_v_ij(self, H: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Compute the vector v_ij used in the Zhang-style calibration constraint.

        Args:
            H: homography matrix (3x3)
            i, j: column indices for the constraint

        Returns:
            v_ij: constraint vector (6,)
        """
        # Transpose to access columns more easily
        h = H.T
        return np.array(
            [
                h[i, 0] * h[j, 0],
                h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
                h[i, 1] * h[j, 1],
                h[i, 0] * h[j, 2] + h[i, 2] * h[j, 0],
                h[i, 1] * h[j, 2] + h[i, 2] * h[j, 1],
                h[i, 2] * h[j, 2],
            ]
        )

    def add_homography(self, H: np.ndarray):
        """
        Add a homography to the calibration dataset.

        Args:
            H: homography matrix (3x3).
        """
        if H.shape != (3, 3):
            raise ValueError("Homography must be a 3x3 matrix.")
        self.homographies.append(H.copy())

    def add_homographies(self, Hs: list[np.ndarray]):
        """
        Add multiple homographies to the calibration dataset.

        Args:
            Hs: list of homography matrices (each 3x3).
        """
        for H in Hs:
            self.add_homography(H)

    def calibrate(self, principal_point: tuple[float, float]) -> np.ndarray:
        """
        Estimate the intrinsic matrix K from the stored homographies.

        Args:
            principal_point: coordinates of the principal point (cx, cy).

        Returns:
            K: intrinsic matrix (3x3).

        Raises:
            ValueError: if no homographies are available
        """
        if len(self.homographies) < 1:
            raise ValueError("At least 1 homography is required for calibration.")

        # Unpack principal point coordinates
        cx, cy = principal_point

        # Initialize the set of constraints
        V = []

        # Constraint 1: Zero skew (B12 = 0)
        V.append([0, 1, 0, 0, 0, 0])

        # Constraint 2: Square pixels (B11 = B22, so B11 - B22 = 0)
        V.append([1, 0, -1, 0, 0, 0])

        # Constraint 3: Known cx (B13 = -cx * B11)
        # This gives us: B13 + cx * B11 = 0
        V.append([cx, 0, 0, 1, 0, 0])

        # Constraint 4: Known cy (B23 = -cy * B22 = -cy * B11 since B11 = B22)
        # This gives us: B23 + cy * B11 = 0
        V.append([cy, 0, 0, 0, 1, 0])

        # Add constraints from homographies
        for H in self.homographies:
            H_norm = H / H[2, 2]  # Normalize
            v12 = self._build_v_ij(H_norm, 0, 1)  # v_12
            v11 = self._build_v_ij(H_norm, 0, 0)  # v_11
            v22 = self._build_v_ij(H_norm, 1, 1)  # v_22

            V.append(v12)
            V.append(v11 - v22)

        V = np.vstack(V)

        # Solve Vb = 0 using SVD
        # The last row of Vh corresponds to smallest singular value
        _, _, Vh = np.linalg.svd(V)
        b = Vh[-1, :]

        # b = [B11, B12, B22, B13, B23, B33]
        B11, _, _, B13, B23, B33 = b

        # The focal length f (since fx = fy = f)
        scale = B33 + cx * B13 + cy * B23
        f_squared = scale / B11 if B11 != 0 else -1
        if f_squared > 0:
            f = np.sqrt(f_squared)
        else:
            f = -1

        # Construct the intrinsic matrix
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        self.intrinsic_matrix = K
        return K

    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get the estimated intrinsic matrix.

        Returns:
            K: intrinsic matrix (3x3) or None if calibration hasn't been performed
        """
        return self.intrinsic_matrix

    def reset(self):
        """
        Clear all stored homographies and reset the calibration state.
        """
        self.homographies.clear()
        self.intrinsic_matrix = None

    def get_num_homographies(self) -> int:
        """
        Get the number of stored homographies.

        Returns:
            Number of homographies
        """
        return len(self.homographies)


def refine_calibration(
    templates: list[Template],
    homographies: list[np.ndarray],
    image_size: np.ndarray,
    K_init: np.ndarray,
    resolution: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine the camera intrinsics and estimate radial distortion parameters (k1, k2).

    Args:
        templates (list[Template]): List of template objects containing metric dimensions.
        homographies (list[np.ndarray]): List of homographies for the scene.
        image_size (np.ndarray): Size of the image (width, height).
        K_init (np.ndarray): Initial intrinsic matrix (3x3).
        resolution (int): Resolution for the grid of points used for each template.

    Returns:
        tuple: Refined intrinsic matrix (3x3) and radial distortion parameters (1D array).
    """
    # Define the world points and the corresponding image points for each template
    object_points = []
    image_points = []
    for template, H in zip(templates, homographies):
        # Get template metric dimensions
        w, h = template.width, template.height

        # Define grid points on the template
        x = np.linspace(0, w, resolution)
        y = np.linspace(0, h, resolution)
        X, Y = np.meshgrid(x, y)

        # Create world points
        object_points_3d = np.array(
            [[x, y, 0] for x, y in zip(X.flatten(), Y.flatten())], dtype=np.float32
        )

        # Create image points
        image_points_2d = cv2.perspectiveTransform(
            object_points_3d[:, :2].reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        # Add these points to the lists
        object_points.append(object_points_3d)
        image_points.append(image_points_2d)

    # Initialize the distortion coefficients to zero
    dist_coeffs_init = np.zeros(5, dtype=np.float32)

    # Define the flags for the optimization
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )

    # Refine the intrinsic parameters and distortion coefficients
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=K_init,
        distCoeffs=dist_coeffs_init,
        flags=flags,
    )

    if not ret:
        raise RuntimeError(
            "Camera calibration failed. Check the input data and parameters."
        )

    # Return the refined intrinsic matrix and distortion coefficients
    return K, dist_coeffs[:2]
