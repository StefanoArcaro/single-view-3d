import os
import sys
from pathlib import Path

import yaml
from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.configs.render_config import RenderConfig
from src.pipeline.pipeline import run_pipeline

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
def load_measurements():
    """Load measurements.yaml file."""
    measurements_path = Path("assets/measurements.yaml")
    with open(measurements_path, "r") as f:
        data = yaml.safe_load(f)
    return data


try:
    measurements = load_measurements()
    scenes = measurements.get("scenes", {})
    templates = measurements.get("templates", {})
except Exception as e:
    st.error(f"Failed to load measurements.yaml: {e}")
    st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Scene Selection")

    # Scene selector
    scene_ids = list(scenes.keys())
    if not scene_ids:
        st.error("No scenes found in measurements.yaml")
        st.stop()

    selected_scene_id = st.selectbox(
        "Choose a scene", scene_ids, help="Select one of the available example scenes"
    )

    # Display scene image
    if selected_scene_id:
        scene_data = scenes[selected_scene_id]
        scene_path = Path("assets") / scene_data.get("path", "")

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
        st.info(f"**Units:** {scene_data.get('units', 'Unknown')}")

        # Show templates in this scene
        st.subheader("Templates in Scene")
        scene_template_ids = scene_data.get("templates", [])

        if scene_template_ids:
            # Display templates in a grid
            cols = st.columns(min(len(scene_template_ids), 4))
            for idx, template_id in enumerate(scene_template_ids):
                with cols[idx % 4]:
                    template_data = templates.get(template_id, {})
                    template_path = Path("assets") / template_data.get("path", "")

                    if template_path.exists():
                        template_image = Image.open(template_path)
                        st.image(
                            template_image,
                            caption=template_id,
                            use_container_width=True,
                        )

                    # Show template dimensions
                    width = template_data.get("width")
                    height = template_data.get("height")
                    if width and height:
                        st.caption(f"{width} × {height} {scene_data.get('units', '')}")
        else:
            st.warning("No templates found for this scene")

with col2:
    st.subheader("Configuration")

    # Render config in sidebar-style column
    st.markdown("##### Display Settings")

    display_units = st.selectbox(
        "Display Units",
        ["m", "cm", "mm"],
        index=0,
        help="Units for 3D visualization (scales geometry accordingly)",
    )

    st.markdown("##### Camera Frustum")

    frustum_near = st.slider(
        "Near Plane",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Distance to near clipping plane",
    )

    frustum_far = st.slider(
        "Far Plane",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Distance to far clipping plane",
    )

    st.markdown("##### Coordinate Axes")

    axes_length = st.slider(
        "Axes Length",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Length of camera coordinate frame axes",
    )

    # Create render config
    render_config = RenderConfig(
        canonical_units=display_units,
        frustum_near=frustum_near,
        frustum_far=frustum_far,
        axes_length=axes_length,
    )

    st.markdown("---")

    # Run pipeline button
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not selected_scene_id:
            st.error("Please select a scene first")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    # Get scene units
                    scene_units = scenes[selected_scene_id].get("units", "mm")

                    # Run pipeline (you'll need to implement data loading)
                    # This is a placeholder - adjust based on your actual data structure
                    from src.data.models import MeasurementData

                    data = MeasurementData.from_yaml(Path("assets/measurements.yaml"))

                    render_data, analysis_data = run_pipeline(
                        data=data,
                        scene_id=selected_scene_id,
                        original_units=scene_units,
                        render_config=render_config,
                    )

                    # Store in session state
                    st.session_state.pipeline_run = True
                    st.session_state.render_data = render_data
                    st.session_state.analysis_data = analysis_data
                    st.session_state.selected_scene_id = selected_scene_id

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
