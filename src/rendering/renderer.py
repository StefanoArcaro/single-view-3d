import base64
import json
import subprocess
import webbrowser

import numpy as np

from src.configs.analysis_data import AnalysisData
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
        analysis_data: AnalysisData,
        render_config: RenderConfig | None = None,
        html_template_path: str | None = None,
        output_path: str | None = None,
        port: int = 8000,
    ) -> None:
        """
        Initialize the renderer.

        Args:
            render_data: RenderData object from the pipeline.
            analysis_data: AnalysisData object from the pipeline.
            render_config: Rendering configuration.
            html_template_path: Path to HTML template file (default if None).
            output_path: Path to save final HTML file (default if None).
            port: Port for local web server.
        """
        self.render_data = render_data
        self.analysis_data = analysis_data
        self.render_config = render_config if render_config else RenderConfig()
        self.port = port

        # Find project root
        self.project_root = find_project_root()

        # Set default paths relative to the project root
        if html_template_path is None:
            html_template_path = self.project_root / "web" / "viewer.html.tpl"
        if output_path is None:
            output_path = self.project_root / "web" / "viewer.html"

        self.html_template_path = html_template_path
        self.output_path = output_path

    def _encode_texture_to_base64(self, texture_path: str) -> str:
        """
        Encode a texture image to a base64 data URI.

        Args:
            texture_path: Path to the texture file (can be relative or absolute).

        Returns:
            Base64-encoded data URI string (e.g., "data:image/jpeg;base64,...")
        """
        # Convert to Path object and resolve from project root
        texture_full_path = self.project_root / texture_path

        try:
            # Read and encode image
            with open(texture_full_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Determine MIME type from file extension
                suffix = texture_full_path.suffix.lower()
                mime_type = {
                    ".jpg": "jpeg",
                    ".jpeg": "jpeg",
                    ".png": "png",
                    ".gif": "gif",
                    ".webp": "webp",
                    ".bmp": "bmp",
                }.get(suffix, "jpeg")

                data_uri = f"data:image/{mime_type};base64,{img_data}"
                return data_uri

        except FileNotFoundError:
            print(f"Texture not found at {texture_full_path}")
            return self._get_placeholder_texture()
        except Exception as e:
            print(f"Error loading texture {texture_full_path}: {e}")
            return self._get_placeholder_texture()

    def _get_placeholder_texture(self) -> str:
        """
        Return a placeholder texture (1x1 pink pixel) as a data URI.

        Returns:
            Base64-encoded PNG data URI.
        """
        # 1x1 pink pixel PNG
        placeholder_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        return f"data:image/png;base64,{placeholder_data}"

    def _generate_meshes_data(self) -> list[dict]:
        """Generate mesh data for all templates."""
        meshes = []

        for template in self.render_data.templates:
            meshes.append(
                {
                    "id": template.id,
                    "vertices": template.corners_camera.tolist(),
                    "triangles": [[0, 2, 1], [0, 3, 2]],
                    "texture": self._encode_texture_to_base64(template.texture_path),
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

    def _generate_results(self) -> dict:
        """
        Generate results data for templates from AnalysisData.

        Returns a dict with template IDs as keys, containing:
        - distance_pred: Predicted distance
        - distance_true: Ground truth distance (if available)
        - error_abs: Absolute error (if ground truth available)
        - error_rel: Relative error as percentage (if ground truth available)
        """
        if self.analysis_data is None:
            return {}

        results = {}
        for template_analysis in self.analysis_data.templates:
            result = {
                "distance_pred": template_analysis.distance_pred,
            }

            # Add ground truth and errors if available
            if template_analysis.distance_true is not None:
                result["distance_true"] = template_analysis.distance_true
                result["error_abs"] = template_analysis.error_abs
                result["error_rel"] = template_analysis.error_rel

            results[template_analysis.id] = result

        return results

    def _generate_html(self) -> str:
        """Generate the HTML visualization file."""
        # Load template
        with open(self.html_template_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Generate data
        meshes = self._generate_meshes_data()
        lines = self._generate_lines_data()
        metadata = self._generate_metadata()
        results = self._generate_results()

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
        html_content = html_content.replace(
            "{ results_json }", json.dumps(results, indent=2)
        )

        # Save the file
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

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
