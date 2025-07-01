import json
import os
import subprocess
import webbrowser
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from data import Template
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


class WebVisualizer:
    """
    Generate interactive 3D web visualizations of multiple templates in camera coordinates.

    This class creates a Three.js-based HTML visualization showing:
    1. Template quads as textured meshes positioned in 3D camera space
    2. Camera coordinate frame (X=red, Y=green, Z=blue axes)
    3. Camera frustum representing the viewing volume

    The mathematical approach mirrors MultiTemplateVisualizer but outputs web-compatible
    data structures for interactive 3D viewing.
    """

    def __init__(
        self,
        metadata: dict[str, Any],
        results: dict[str, Any],
        K: np.ndarray,
        template_path: str = "../web/viewer.html.tpl",
        output_path: str = "../web/viewer.html",
        frustum_near: float = 0.1,
        frustum_far: float = 10.0,
        axes_length: float = 10.0,
        port: int = 8000,
    ) -> None:
        """
        Initialize the web-based 3D template visualizer.

        Args:
            metadata: Mapping from template ID to Template object with attributes
                     id, label, path, width, height
            results: Mapping from template ID to dict containing 'homography' key
                    with 3x3 homography matrix
            K: 3x3 camera intrinsic matrix following OpenCV convention
            template_path: Path to HTML template file with placeholders
            output_path: Path where final HTML visualization will be saved
            frustum_near: Near plane distance for camera frustum visualization
            frustum_far: Far plane distance for camera frustum visualization
            axes_length: Length of camera coordinate frame axes
        """
        self.metadata = metadata
        self.results = results
        self.K = K
        self.template_path = template_path
        self.output_path = output_path
        self.frustum_near = frustum_near
        self.frustum_far = frustum_far
        self.axes_length = axes_length
        self.port = port

    def _decompose_homography(self, template_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose homography to recover camera pose for a specific template.

        Args:
            template_id: ID of the template to process

        Returns:
            Tuple of (R, t) where:
            - R: 3x3 rotation matrix transforming world to camera coordinates
            - t: 3x1 translation vector in camera coordinates

        Raises:
            KeyError: If template_id not found in results
            ValueError: If homography decomposition fails
        """
        if template_id not in self.results:
            raise KeyError(f"Template '{template_id}' not found in results")

        H = np.array(self.results[template_id]["homography"], dtype=np.float64)

        try:
            solutions = recover_all_poses_from_homography(H, self.K)
            R, t, _ = select_best_solution(solutions, expected_z_positive=True)
            return R, t
        except Exception as e:
            raise ValueError(
                f"Failed to decompose homography for template '{template_id}': {e}"
            )

    def _create_template_corners(self, template: Template) -> np.ndarray:
        """
        Create 3D corner points for a template in world coordinates.

        Template is assumed to lie on the Z=0 plane with corners at:
        - (0, 0, 0): top-left
        - (width, 0, 0): top-right
        - (width, height, 0): bottom-right
        - (0, height, 0): bottom-left

        Args:
            template: Template object with width and height attributes

        Returns:
            Array of shape (4, 3) containing corner coordinates

        Raises:
            ValueError: If template lacks required width/height attributes
        """
        w, h = template.width, template.height
        if w is None or h is None:
            raise ValueError(f"Template '{template.id}' missing width or height")

        return np.array(
            [
                [0, 0, 0],  # top-left
                [w, 0, 0],  # top-right
                [w, h, 0],  # bottom-right
                [0, h, 0],  # bottom-left
            ],
            dtype=np.float32,
        )

    def _transform_to_camera(
        self, points_world: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Transform 3D points from world coordinates to camera coordinates.

        Applies the standard transformation: points_cam = R @ points_world + t

        Args:
            points_world: Array of shape (N, 3) in world coordinates
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            Array of shape (N, 3) in camera coordinates
        """
        # Convert points to camera coordinates using rotation and translation
        points_cam = (R @ points_world.T).T + t.reshape(1, 3)

        # 180° degree rotation around Z-axis to match OpenCV convention
        # This means we flip the X and Y axes
        points_cam[:, 0] *= -1
        points_cam[:, 1] *= -1

        return points_cam.astype(np.float32)

    def _generate_meshes_data(self) -> list[dict[str, Any]]:
        """
        Generate mesh data for all templates as textured quads in camera space.

        Each template becomes a rectangular mesh with:
        - 4 vertices positioned according to recovered pose
        - 2 triangles forming the quad surface
        - Texture mapping using the template's image file
        - Standard UV coordinates for proper texture display

        Returns:
            List of dictionaries, each containing:
            - vertices: List of [x,y,z] coordinates (12 floats total)
            - triangles: List of triangle indices [[0,1,2], [0,2,3]]
            - texture: Path to template image file
        """
        meshes_data = []

        for template_id, template in self.metadata.items():
            try:
                # Recover pose from homography
                R, t = self._decompose_homography(template_id)

                # Create template corners in world frame
                corners_world = self._create_template_corners(template)

                # Transform to camera coordinates
                corners_cam = self._transform_to_camera(corners_world, R, t)

                # Convert to nested list format expected by Three.js
                vertices = corners_cam.tolist()

                # Define triangles for quad (two triangles sharing diagonal)
                # Vertex winding order is counter-clockwise, as per OpenGL convention
                triangles = [[0, 2, 1], [0, 3, 2]]

                # Use template image path for texture
                texture_path = template.path if template.path else ""

                mesh_data = {
                    "vertices": vertices,
                    "triangles": triangles,
                    "texture": os.path.join("..", texture_path),
                }

                meshes_data.append(mesh_data)

            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping template '{template_id}': {e}")
                continue

        return meshes_data

    def _generate_camera_axes(self) -> list[dict[str, Any]]:
        """
        Generate camera coordinate frame axes for visualization.

        Creates three colored lines from origin along X, Y, Z axes following
        OpenCV convention: X = right (red), Y = down (green), Z = forward (blue)

        Returns:
            Dictionary containing:
                - points: List of 3D coordinates for each axis endpoint
                - lines: List of line segment indices defining each axis
                - color: RGB color values for the specific axes
        """
        # Define axis endpoints
        origin = [0.0, 0.0, 0.0]
        x_end = [self.axes_length, 0.0, 0.0]  # X-axis (right)
        y_end = [0.0, self.axes_length, 0.0]  # Y-axis (down)
        z_end = [0.0, 0.0, self.axes_length]  # Z-axis (forward)

        # Apply 180° rotation around Z-axis to match OpenCV convention
        # This means we flip the X and Y axes
        x_end = [-x_end[0], -x_end[1], x_end[2]]
        y_end = [-y_end[0], -y_end[1], y_end[2]]

        # Create separate line groups for each axis with distinct colors
        axes_data = [
            {
                "points": [origin, x_end],
                "lines": [[0, 1]],
                "color": [1.0, 0.0, 0.0],  # Red for X-axis
            },
            {
                "points": [origin, y_end],
                "lines": [[0, 1]],
                "color": [0.0, 1.0, 0.0],  # Green for Y-axis
            },
            {
                "points": [origin, z_end],
                "lines": [[0, 1]],
                "color": [0.0, 0.0, 1.0],  # Blue for Z-axis
            },
        ]

        return axes_data

    def _generate_camera_frustum(self) -> dict[str, Any]:
        """
        TODO require image size in pixels!
        Generate camera frustum lines representing the viewing volume.

        Creates a truncated pyramid showing the camera's field of view using
        the intrinsic matrix K to determine corner ray directions at near/far planes.

        Returns:
            Dictionary containing:
            - points: List of 8 3D coordinates (4 near + 4 far plane corners)
            - lines: List of 12 line segment indices forming frustum edges
            - color: RGB color values [0.8, 0.8, 0.8] for gray frustum
        """
        # Extract principal point and focal lengths from intrinsic matrix
        cx, cy = self.K[0, 2], self.K[1, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]

        # Assume image dimensions are 2*cx by 2*cy (typical for centered principal point)
        # TODO change!
        w, h = 2 * cx, 2 * cy

        # Corner rays in normalized image coordinates
        corners_norm = np.array(
            [
                [0, 0],  # top-left
                [w, 0],  # top-right
                [w, h],  # bottom-right
                [0, h],  # bottom-left
            ]
        )

        # Convert to camera coordinates (ray directions)
        corners_3d = np.zeros((4, 3))
        corners_3d[:, 0] = (corners_norm[:, 0] - cx) / fx  # X direction
        corners_3d[:, 1] = (corners_norm[:, 1] - cy) / fy  # Y direction
        corners_3d[:, 2] = 1.0  # Z direction (forward)

        # Scale to near and far planes
        near_corners = corners_3d * self.frustum_near
        far_corners = corners_3d * self.frustum_far

        # Combine all 8 frustum points
        points = np.vstack([near_corners, far_corners])

        # Apply 180° rotation around Z-axis to match OpenCV convention
        # This means we flip the X and Y axes
        points[:, 0] *= -1
        points[:, 1] *= -1

        # Convert to list of lists for JSON serialization
        points = points.tolist()

        # fmt: off
        # Define frustum edges (4 near edges + 4 far edges + 4 connecting edges)
        lines = [
            # Near plane rectangle
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Far plane rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Connecting edges
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        # fmt: on

        return {
            "points": points,
            "lines": lines,
            "color": [0.8, 0.8, 0.8],  # Gray color for frustum
        }

    def _generate_lines_data(self) -> list[dict[str, Any]]:
        """
        Generate all line data for camera visualization.

        Combines camera coordinate axes and frustum into a single data structure
        suitable for Three.js LineSegments rendering.

        Returns:
            List containing two dictionaries:
            - Camera axes (white lines)
            - Camera frustum (gray lines)
        """
        lines_data = []

        # Add camera coordinate axes
        axes_data = self._generate_camera_axes()
        lines_data.extend(axes_data)

        # Add camera frustum
        frustum_data = self._generate_camera_frustum()
        lines_data.append(frustum_data)

        return lines_data

    def _load_template(self) -> str:
        """
        Load HTML template file from disk.

        Returns:
            String containing the complete HTML template content

        Raises:
            FileNotFoundError: If template file doesn't exist
            IOError: If template file cannot be read
        """
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template file not found: {self.template_path}")

        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Failed to read template file '{self.template_path}': {e}")

    def _substitute_placeholders(
        self, html_template: str, meshes_data: list[dict], lines_data: list[dict]
    ) -> str:
        """
        Replace placeholder strings in HTML template with actual data.

        Performs simple string substitution of JSON-serialized data into
        the template placeholders.

        Args:
            html_template: HTML template string with placeholders
            meshes_data: List of mesh dictionaries
            lines_data: List of line dictionaries

        Returns:
            Complete HTML string with data substituted
        """
        # Convert data to JSON strings
        meshes_json = json.dumps(meshes_data, indent=2)
        lines_json = json.dumps(lines_data, indent=2)

        # Substitute placeholders
        html_content = html_template.replace("{ meshes_json }", meshes_json)
        html_content = html_content.replace("{ lines_json }", lines_json)

        return html_content

    def _save_html(self, html_content: str) -> None:
        """
        Save the complete HTML visualization to output file.

        Args:
            html_content: Complete HTML string to write

        Raises:
            IOError: If output file cannot be written
        """
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except IOError as e:
            raise IOError(f"Failed to write output file '{self.output_path}': {e}")

    def generate_html(self) -> None:
        """
        Generate complete interactive 3D HTML visualization.

        This is the main method that orchestrates the entire visualization process:
        1. Generates mesh data for all templates in camera space
        2. Creates camera axes and frustum line data
        3. Loads HTML template and substitutes data placeholders
        4. Saves final HTML file ready for viewing in browser

        The resulting visualization shows all templates positioned in 3D space
        as seen from the camera viewpoint, with interactive controls for
        rotation, panning, and zooming.

        Raises:
            FileNotFoundError: If template file is missing
            IOError: If file I/O operations fail
            ValueError: If pose estimation fails for any template
        """
        print("=" * 60)
        print("  3D WEB VISUALIZATION GENERATOR")
        print("=" * 60)

        # Generate 3D data structures
        print("\n[1/5] Processing Template Poses")
        print("     " + "-" * 40)
        meshes_data = self._generate_meshes_data()
        print(f"     | Generated {len(meshes_data)} template meshes")
        print("     " + "-" * 40)

        print("\n[2/5] Creating Camera Visualization")
        print("     " + "-" * 40)
        lines_data = self._generate_lines_data()
        print(f"     | Generated {len(lines_data)} line groups (axes + frustum)")
        print("     " + "-" * 40)

        # Load and process HTML template
        print("\n[3/5] Loading HTML Template")
        print("     " + "-" * 40)
        html_template = self._load_template()
        print("     | Template loaded successfully")
        print("     " + "-" * 40)

        print("\n[4/5] Substituting Data Placeholders")
        print("     " + "-" * 40)
        html_content = self._substitute_placeholders(
            html_template, meshes_data, lines_data
        )
        print("     | Data integration complete")
        print("     " + "-" * 40)

        # Save final result
        print("\n[5/5] Saving Visualization")
        print("     " + "-" * 40)
        self._save_html(html_content)
        print("     | File saved successfully")
        print("     " + "-" * 40)

        print("\n" + "=" * 60)
        print("  VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"  >> Open '{self.output_path}' in your browser")
        print("  >> to view the 3D scene")
        print("=" * 60)

    def show(self, html_file: str = None) -> None:
        # Start server in background
        if html_file is None:
            html_file = self.output_path

        print(f"Starting local server on port {self.port}...")
        print(f"Opening {html_file} in your browser...")

        subprocess.Popen(
            ["python", "-m", "http.server", str(self.port)],
            cwd=os.path.dirname(html_file),
        )

        # Auto-open browser
        url = f"http://localhost:{self.port}/{os.path.basename(html_file)}"
        webbrowser.open(url)
