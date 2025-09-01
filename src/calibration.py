import numpy as np


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
        Compute the vector v_ij used in the Zhang-style calibration constraint:
        v_ij = [h1i*h1j, h1i*h2j + h2i*h1j, h2i*h2j, h3i*h1j + h1i*h3j, h3i*h2j + h2i*h3j, h3i*h3j]

        Args:
            H: homography matrix (3x3)
            i, j: column indices for the constraint

        Returns:
            v_ij: constraint vector (6,)
        """
        h = H.T  # transpose to access columns easily
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
            H: homography matrix (3x3)
        """
        if H.shape != (3, 3):
            raise ValueError("Homography must be a 3x3 matrix")
        self.homographies.append(H.copy())

    def add_homographies(self, Hs: list[np.ndarray]):
        """
        Add multiple homographies to the calibration dataset.

        Args:
            Hs: list of homography matrices (each 3x3)
        """
        for H in Hs:
            self.add_homography(H)

    def calibrate(self, principal_point=None) -> np.ndarray:
        """
        Estimate the intrinsic matrix K from the stored homographies.

        Args:
            principal_point: tuple (cx, cy) if known, otherwise assumes image center

        Returns:
            K: intrinsic matrix (3x3)

        Raises:
            ValueError: if fewer than 1 homography is available (since we have more constraints)
        """
        if len(self.homographies) < 1:  # Need fewer homographies now
            raise ValueError(
                "At least 1 homography is required for calibration with known constraints"
            )

        # Set principal point (default to image center if not provided)
        if principal_point is None:
            raise ValueError("Principal point (cx, cy) must be provided")
        else:
            cx, cy = principal_point

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
        _, _, Vh = np.linalg.svd(V)
        b = Vh[-1, :]  # last row corresponds to smallest singular value

        # b = [B11, B12, B22, B13, B23, B33]
        B11, B12, B22, B13, B23, B33 = b

        # Simplified parameter extraction
        # Since B11 = B22, B12 = 0, B13 = -cx*B11, B23 = -cy*B11

        # The focal length f (since fx = fy = f)
        scale = B33 + cx * B13 + cy * B23  # This should equal B33 - cx²*B11 - cy²*B11
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
