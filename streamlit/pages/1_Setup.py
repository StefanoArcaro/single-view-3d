import os
import sys
from pathlib import Path

from PIL import Image
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.configs.render_config import RenderConfig
from src.pipeline.pipeline import run_pipeline
from src.utils import find_project_root, load_measurements_from_yaml
from src.models.measurements import Measurements

PROJECT_ROOT = find_project_root()

# Page config
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="centered",
)

# Initialize session state
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "render_data" not in st.session_state:
    st.session_state.render_data = None
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

# Title
st.title("🎯 Demo Setup")
st.markdown("Select a scene and configure rendering parameters to begin.")
st.markdown("---")


# Load measurements data
@st.cache_resource
def load_measurements() -> Measurements:
    """Load measurements.yaml file into a Measurements model."""
    measurements_path = PROJECT_ROOT / "assets" / "measurements.yaml"
    return load_measurements_from_yaml(measurements_path)


try:
    measurements = load_measurements()
except Exception as e:
    st.error(f"Failed to load measurements.yaml: {e}")
    st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Scene Selection")

    # Scene selector
    scene_ids = measurements.list_scenes()
    if not scene_ids:
        st.error("No scenes found in measurements.yaml")
        st.stop()

    default_index = st.session_state.get("selected_scene_index", 0)
    selected_scene_id = st.selectbox(
        label="Choose a scene",
        options=scene_ids,
        index=default_index,
        help="Select one of the available example scenes",
    )

    # Display scene image
    if selected_scene_id:
        scene = measurements.get_scene(selected_scene_id)
        if not scene:
            st.error(f"Scene {selected_scene_id} not found in measurements")
            st.stop()

        scene_path = PROJECT_ROOT / scene.path
        if scene_path.exists():
            scene_image = Image.open(scene_path)
            st.image(
                scene_image,
                caption=f"Scene: {selected_scene_id}",
                use_container_width=True,
            )
        else:
            st.warning(f"Scene image not found: {scene_path}")

        # Display scene info
        st.info(f"**Units:** {measurements.unit}")

        # Show templates in this scene
        st.subheader("Templates in Scene")
        scene_template_ids = measurements.get_scene_templates(selected_scene_id) or []

        if scene_template_ids:
            cols = st.columns(min(len(scene_template_ids), 4))
            for idx, template_id in enumerate(scene_template_ids):
                with cols[idx % 4]:
                    template = measurements.get_template(template_id)
                    if not template:
                        continue

                    template_path = Path("assets") / template.path
                    if template_path.exists():
                        template_image = Image.open(template_path)
                        st.image(
                            template_image,
                            caption=template.id,
                            use_container_width=True,
                        )

                    if template.width and template.height:
                        st.caption(
                            f"{template.width} × {template.height} {measurements.unit}"
                        )
        else:
            st.warning("No templates found for this scene")

with col2:
    st.subheader("Configuration")

    # Render config
    st.markdown("##### Display Settings")
    display_units = st.selectbox(
        "Display Units",
        ["m", "cm", "mm"],
        index=0,
        help="Units for 3D visualization (scales geometry accordingly)",
    )

    st.markdown("##### Camera Frustum")
    frustum_near = st.slider("Near Plane", 0.01, 1.0, 0.1, 0.01)
    frustum_far = st.slider("Far Plane", 1.0, 50.0, 10.0, 0.5)

    st.markdown("##### Coordinate Axes")
    axes_length = st.slider("Axes Length", 0.1, 5.0, 1.0, 0.1)

    render_config = RenderConfig(
        canonical_units=display_units,
        frustum_near=frustum_near,
        frustum_far=frustum_far,
        axes_length=axes_length,
    )

    st.markdown("---")

    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not selected_scene_id:
            st.error("Please select a scene first")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    render_data, analysis_data = run_pipeline(
                        data=measurements,
                        scene_id=selected_scene_id,
                        original_units=measurements.unit,
                        render_config=render_config,
                    )

                    st.session_state.pipeline_run = True
                    st.session_state.render_data = render_data
                    st.session_state.analysis_data = analysis_data
                    st.session_state.selected_scene_id = selected_scene_id
                    st.session_state.selected_scene_index = scene_ids.index(
                        selected_scene_id
                    )

                    st.success("Pipeline completed successfully!")
                    st.info("Navigate to **3D Viewer** to see results")

                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    import traceback

                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# Show current status
st.markdown("---")
if st.session_state.pipeline_run:
    st.success(
        f"✓ Pipeline results available for scene: {st.session_state.get('selected_scene_id', 'Unknown')}"
    )
else:
    st.info("ℹ️ Configure settings and run the pipeline to proceed to visualization")
