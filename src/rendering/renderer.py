import json
import os
import subprocess
import webbrowser
from pathlib import Path

import numpy as np

from src.configs.render_config import RenderConfig
from src.configs.render_data import RenderData
from src.utils import find_project_root


class Renderer:
    """
    Generate interactive 3D web visualizations from RenderData.

    Creates a Three.js-based HTML visualization with:
    - Template quads as textured meshes
    - Camera coordinate axes (X=red, Y=green, Z=blue)
    - Camera frustum
    """

    def __init__(
        self,
        render_data: RenderData,
        render_config: RenderConfig | None = None,
        html_template_path: str | None = None,
        output_path: str | None = None,
        port: int = 8000,
    ) -> None:
        """
        Initialize the renderer.

        Args:
            render_data: RenderData object from the pipeline.
            render_config: Rendering configuration.
            html_template_path: Path to HTML template file (default if None).
            output_path: Path to save final HTML file (default if None).
            port: Port for local web server.
        """
        self.render_data = render_data
        self.render_config = render_config if render_config else RenderConfig()
        self.port = port

        # Find project root
        self.project_root = find_project_root()

        # Set default paths relative to the project root
        if html_template_path is None:
            html_template_path = os.path.join(
                self.project_root, "web", "viewer.html.tpl"
            )
        if output_path is None:
            output_path = os.path.join(self.project_root, "web", "viewer.html")

        self.html_template_path = Path(html_template_path)
        self.output_path = Path(output_path)

    def _generate_meshes_data(self) -> list[dict]:
        """Generate mesh data for all templates."""
        meshes = []

        for template in self.render_data.templates:
            meshes.append(
                {
                    "id": template.id,
                    "vertices": template.corners_camera.tolist(),
                    "triangles": [[0, 2, 1], [0, 3, 2]],
                    "texture": os.path.join("..", template.texture_path),
                }
            )

        return meshes

    def _generate_camera_axes(self) -> list[dict]:
        """Generate camera coordinate axes."""
        length = self.render_config.axes_length
        origin = [0.0, 0.0, 0.0]

        # Apply OpenCV convention (flip X and Y)
        return [
            {
                "points": [origin, [-length, 0.0, 0.0]],
                "lines": [[0, 1]],
                "color": [1.0, 0.0, 0.0],  # X-axis (red)
            },
            {
                "points": [origin, [0.0, -length, 0.0]],
                "lines": [[0, 1]],
                "color": [0.0, 1.0, 0.0],  # Y-axis (green)
            },
            {
                "points": [origin, [0.0, 0.0, length]],
                "lines": [[0, 1]],
                "color": [0.0, 0.0, 1.0],  # Z-axis (blue)
            },
        ]

    def _generate_camera_frustum(self) -> dict:
        """Generate camera frustum."""
        K = self.render_data.K
        w = self.render_data.image_width
        h = self.render_data.image_height

        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        # Image corners
        corners_img = np.array(
            [
                [0, 0],
                [w, 0],
                [w, h],
                [0, h],
            ]
        )

        # Convert to camera ray directions
        corners_3d = np.zeros((4, 3))
        corners_3d[:, 0] = (corners_img[:, 0] - cx) / fx
        corners_3d[:, 1] = (corners_img[:, 1] - cy) / fy
        corners_3d[:, 2] = 1.0

        # Scale to near and far planes
        near = corners_3d * self.render_config.frustum_near
        far = corners_3d * self.render_config.frustum_far

        # Combine and apply OpenCV convention
        points = np.vstack([near, far])
        points[:, 0] *= -1
        points[:, 1] *= -1

        # fmt: off
        # Frustum edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
            [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
            [0, 4], [1, 5], [2, 6], [3, 7],  # Connections
        ]
        # fmt: on

        return {
            "points": points.tolist(),
            "lines": lines,
            "color": self.render_config.frustum_color,
        }

    def _generate_lines_data(self) -> list[dict]:
        """Generate all line data for camera visualization."""
        lines = self._generate_camera_axes()
        lines.append(self._generate_camera_frustum())
        return lines

    def _generate_metadata(self) -> dict:
        """Generate metadata for HTML template."""
        return {
            "scene_id": self.render_data.scene_id,
            "units": self.render_data.original_units,
            "image_size": {
                "width": self.render_data.image_width,
                "height": self.render_data.image_height,
            },
            "templates": {
                t.id: {
                    "label": t.label,
                    "path": t.texture_path,
                    "width": t.width,
                    "height": t.height,
                }
                for t in self.render_data.templates
            },
        }

    def _generate_html(self) -> str:
        """Generate the HTML visualization file."""
        # Load template
        with open(self.html_template_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Generate data
        meshes = self._generate_meshes_data()
        lines = self._generate_lines_data()
        metadata = self._generate_metadata()

        # Substitute placeholders
        html_content = html_template.replace(
            "{ meshes_json }", json.dumps(meshes, indent=2)
        )
        html_content = html_content.replace(
            "{ lines_json }", json.dumps(lines, indent=2)
        )
        html_content = html_content.replace(
            "{ metadata_json }", json.dumps(metadata, indent=2)
        )

    def show(self) -> None:
        """Start local server and open visualization in browser."""
        # Start server from project root
        subprocess.Popen(
            ["python", "-m", "http.server", str(self.port)],
            cwd=str(self.project_root),
        )

        # Calculate relative path from project root to HTML file
        relative_path = self.output_path.relative_to(self.project_root)
        url = f"http://localhost:{self.port}/{relative_path}"
        webbrowser.open(url)
