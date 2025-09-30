from dataclasses import dataclass

import numpy as np


@dataclass
class TemplateRenderData:
    """Data for rendering a single template."""

    id: str
    label: str
    width: float
    height: float
    texture_path: str
    corners_camera: np.ndarray


@dataclass
class RenderData:
    """Complete data package for rendering."""

    scene_id: str
    original_units: str
    image_width: int
    image_height: int
    K: np.ndarray
    templates: list[TemplateRenderData]
