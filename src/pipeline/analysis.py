import numpy as np

from src.configs.analysis_data import (
    AnalysisData,
    PoseAnalysisData,
    TemplateAnalysisData,
)
from src.configs.render_config import RenderConfig
from src.configs.render_data import RenderData, TemplateRenderData
from src.models.scene import Scene
from src.models.template import Template
from src.pipeline.geometry import (
    compute_distance_from_pose,
    create_template_corners,
    recover_all_poses_from_homography,
    select_best_solution,
    transform_to_camera,
)


def analyze_scene(
    scene: Scene,
    templates: list[Template],
    homographies: list[np.ndarray],
    K: np.ndarray,
    image_size: tuple[int, int],
    original_units: str,
    render_config: RenderConfig,
) -> tuple[RenderData, AnalysisData]:
    """
    Analyze scene and prepare data for rendering and plotting.

    Args:
        scene: Scene object with metadata and optional ground truth.
        templates: List of Template objects with metric dimensions.
        homographies: List of 3x3 homography matrices.
        K: 3x3 camera intrinsic matrix.
        image_size: (width, height) in pixels.
        original_units: Units of template measurements (mm, cm, m).
        render_config: Rendering configuration with display units.

    Returns:
        Tuple of (RenderData, AnalysisData)
    """
    scale_factor = render_config.scale_factor_from(original_units)

    # Prepare lists for output data
    render_templates = []
    analysis_templates = []
    poses = []

    for template, H in zip(templates, homographies):
        # Decompose homography to get the pose
        solutions = recover_all_poses_from_homography(H, K)
        R, t, _ = select_best_solution(solutions, expected_z_positive=True)

        # Create and transform template corner points
        corners_world = create_template_corners(template.width, template.height)
        corners_camera = transform_to_camera(corners_world, R, t)

        # Scale corners for rendering
        # TODO check if correct
        corners_camera_scaled = corners_camera * scale_factor

        # Compute predicted distance at template center
        template_center = np.array([template.width / 2, template.height / 2])
        distance_pred = compute_distance_from_pose(R, t, template_center)

        # Get ground-truth distance if available
        distance_obj = scene.get_distance("Camera", template.id)
        distance_true = distance_obj.distance if distance_obj else None

        # Create template render data, appropriately scaled
        render_templates.append(
            TemplateRenderData(
                id=template.id,
                label=template.label,
                width=template.width * scale_factor,
                height=template.height * scale_factor,
                texture_path=template.path,
                corners_camera=corners_camera_scaled,
            )
        )

        # Create template analysis data
        analysis_templates.append(
            TemplateAnalysisData(
                id=template.id, distance_pred=distance_pred, distance_true=distance_true
            )
        )

        # Store pose
        poses.append(PoseAnalysisData(id=template.id, R=R, t=t))

    # Create output structures
    render_data = RenderData(
        scene_id=scene.id,
        original_units=original_units,
        image_width=image_size[1],
        image_height=image_size[0],
        K=K,
        templates=render_templates,
    )

    analysis_data = AnalysisData(
        scene_id=scene.id,
        units=original_units,
        templates=analysis_templates,
        poses=poses,
    )

    return render_data, analysis_data
