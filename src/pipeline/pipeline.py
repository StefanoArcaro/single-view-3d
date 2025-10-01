from src.configs.analysis_data import AnalysisData
from src.configs.render_config import RenderConfig
from src.configs.render_data import RenderData
from src.models.measurements import Measurements
from src.pipeline.analysis import analyze_scene
from src.pipeline.calibration import CalibrationSimple, refine_calibration
from src.pipeline.matching import multi_template_match


def run_pipeline(
    data: Measurements,
    scene_id: str,
    original_units: str,
    render_config: RenderConfig | None = None,
) -> tuple[RenderData, AnalysisData]:
    """
    Run the complete pipeline: matching, calibration, and analysis.

    Args:
        data: MeasurementData object with scenes and templates.
        scene_id: Scene identifier.
        original_units: Units of template measurements (mm, cm, m).
        render_config: Rendering configuration (uses defaults if None).

    Returns:
        Tuple of (RenderData, AnalysisData).
    """
    # Use default config if not provided
    if render_config is None:
        render_config = RenderConfig()

    # Load scene and templates
    scene = data.get_scene(scene_id)
    template_ids = data.get_scene_templates(scene_id)
    templates = [data.get_template(t_id) for t_id in template_ids]

    # Template matching
    image_size, match_results = multi_template_match(scene=scene, templates=templates)

    # Extract homographies
    homographies = [res["homography"] for res in match_results.values()]

    # Camera calibration
    calibration = CalibrationSimple()
    calibration.add_homographies(homographies)
    principal_point = (image_size[0] / 2, image_size[1] / 2)
    K_init = calibration.calibrate(principal_point=principal_point)

    # Refine calibration
    K, _ = refine_calibration(
        templates=templates,
        homographies=homographies,
        image_size=image_size,
        K_init=K_init,
        resolution=20,
    )

    # Scene analysis
    render_data, analysis_data = analyze_scene(
        scene=scene,
        templates=templates,
        homographies=homographies,
        K=K,
        image_size=image_size,
        original_units=original_units,
        render_config=render_config,
    )

    return render_data, analysis_data
