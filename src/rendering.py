from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.geometry import recover_all_poses_from_homography, select_best_solution


class MultiTemplateVisualizer:
    """
    Visualize multiple 3D templates in camera coordinates and their 2D projections
    on the image plane using shared camera intrinsics.
    """

    def __init__(
        self,
        metadata: dict[str, Any],
        results: dict[str, Any],
        K: np.ndarray,
        fig_size: tuple[int, int] = (14, 6),
        view_elev: float = 30,
        view_azim: float = 140,
    ) -> None:
        """
        Args:
            metadata (dict[str, Template]): Mapping from template ID to Template object
                with attributes `width` and `height` defining the template size.
            results (dict[str, Any]): Mapping from template ID to a dict containing
                at least the key `'homography'` with a 3x3 homography matrix.
            K (np.ndarray): 3x3 camera intrinsic matrix.
            fig_size (tuple[int, int], optional): Size of the matplotlib figure.
            view_elev (float, optional): Elevation angle for 3D view.
            view_azim (float, optional): Azimuth angle for 3D view.
        """
        self.metadata = metadata
        self.results = results
        self.K = K
        self.fig_size = fig_size
        self.view_elev = view_elev
        self.view_azim = view_azim

    def _project_to_image(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Project 3D camera-frame points into 2D image points using intrinsics.

        Args:
            points_cam (np.ndarray): Array of shape (N, 3).

        Returns:
            np.ndarray: Array of shape (N, 2) with pixel coordinates.
        """
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        img_pts, _ = cv2.projectPoints(points_cam, rvec, tvec, self.K, None)
        return img_pts.reshape(-1, 2)

    def _set_equal_axes(self, ax: Axes3D, points: np.ndarray) -> None:
        """
        Force equal scaling on a 3D Axes.

        Args:
            ax (Axes3D): The 3D axes to adjust.
            points (np.ndarray): Array of shape (M, 3) of all points to frame.
        """
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = maxs - mins
        mid = (maxs + mins) / 2
        half_range = spans.max() / 2
        ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
        ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
        ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    def plot(self) -> None:
        """
        Plot all templates in a single 3D axes and their 2D projections side by side.
        """
        num_templates = len(self.metadata)
        colors = plt.cm.tab10(np.linspace(0, 1, num_templates))

        fig = plt.figure(figsize=self.fig_size)
        ax3d = fig.add_subplot(121, projection="3d")
        ax2d = fig.add_subplot(122)

        all_3d_points = []  # For equalizing axes

        for idx, (t_id, template) in enumerate(self.metadata.items()):
            # Homography decomposition
            H = np.array(self.results[t_id]["homography"], dtype=np.float64)
            solutions = recover_all_poses_from_homography(H, self.K)
            R, t, _ = select_best_solution(solutions, expected_z_positive=True)

            # Build template corners in world frame
            w, h = template.width, template.height
            corners_world = np.array(
                [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float32
            )

            # Transform to camera frame
            pts_cam = (R @ corners_world.T).T + t.reshape(1, 3)
            all_3d_points.append(pts_cam)

            # Project to image plane
            img_pts = self._project_to_image(pts_cam)

            # Plot 3D edges
            col = colors[idx]
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                xs, ys, zs = zip(pts_cam[i], pts_cam[j])
                ax3d.plot(xs, ys, zs, color=col)

            # Plot 2D edges
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                ax2d.plot(
                    [img_pts[i, 0], img_pts[j, 0]],
                    [img_pts[i, 1], img_pts[j, 1]],
                    marker="o",
                    color=col,
                    label=t_id if i == 0 else None,
                )

        # Finalize 3D plot
        all_3d = np.vstack(all_3d_points)
        self._set_equal_axes(ax3d, all_3d)
        ax3d.scatter([0], [0], [0], color="black", s=50, label="Camera")

        # Draw camera axes (in camera frame)
        # scale length to, say, 10% of the overall scene span:
        L = (all_3d.max() - all_3d.min()) * 0.1
        basis = np.eye(3) * L
        for vec, col, lab in zip(basis, ["r", "g", "b"], ["X_cam", "Y_cam", "Z_cam"]):
            ax3d.quiver(
                0,
                0,
                0,  # origin
                *vec,  # direction
                color=col,
                arrow_length_ratio=0.1,
                label=lab,
            )

        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        ax3d.legend()
        ax3d.set_title("3D View (all templates)")

        # Finalize 2D plot
        ax2d.set_aspect("equal")
        ax2d.set_xlim(0, self.K[0, 2] * 2)
        ax2d.set_ylim(self.K[1, 2] * 2, 0)
        ax2d.set_xlabel("X (px)")
        ax2d.set_ylabel("Y (px)")
        ax2d.grid()
        ax2d.set_title("2D Projection (all templates)")
        ax2d.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
