import base64
import os
import sys
import traceback
from io import BytesIO

from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.configs.render_config import RenderConfig
from src.models.measurements import Measurements
from src.pipeline.pipeline import run_pipeline
from src.rendering.renderer import Renderer
from src.utils import find_project_root, load_measurements_from_yaml, load_rgb

PROJECT_ROOT = find_project_root()

# Page config
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="wide",
)


# ============================================================================
# Session State Initialization
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "pipeline_run": False,
        "render_data": None,
        "analysis_data": None,
        "selected_scene_id": None,
        "selected_scene_index": 0,
        "pipeline_error": None,
        "pipeline_traceback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# Data Loading and Caching
# ============================================================================
@st.cache_resource
def load_measurements() -> Measurements:
    """Load measurements.yaml file into a Measurements model."""
    measurements_path = PROJECT_ROOT / "assets" / "measurements.yaml"
    return load_measurements_from_yaml(measurements_path)


@st.cache_data
def load_and_cache_image(path: str) -> Image.Image:
    """Load an image file and cache it as PIL Image."""
    img_array = load_rgb(path)
    if hasattr(img_array, "shape"):
        return Image.fromarray(img_array)
    return img_array


@st.cache_data
def get_image_base64(path: str, max_height: int = None) -> str:
    """Load image, optionally resize, convert to base64, and cache."""
    pil_img = load_and_cache_image(path)

    # Resize if max_height specified (for thumbnails)
    if max_height and pil_img.height > max_height:
        aspect_ratio = pil_img.width / pil_img.height
        new_height = max_height
        new_width = int(max_height * aspect_ratio)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ============================================================================
# UI Components
# ============================================================================
def vspace(lines=1):
    for _ in range(lines):
        st.markdown("")


def render_image(path: str, height: int, alt_text: str):
    """Render an image from the given path with constrained height."""
    if path.exists():
        # Use cached base64 with optional resizing
        img_str = get_image_base64(str(path), max_height=height * 2)  # 2x for retina

        st.markdown(
            f'<img src="data:image/png;base64,{img_str}" style="max-height: {height}px; width: auto;" alt="{alt_text}">',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Image not found: {path}")


def render_scene_image(scene, scene_id: str):
    """Display the main scene image with constrained height."""
    scene_path = PROJECT_ROOT / scene.path
    render_image(scene_path, height=400, alt_text=f"Scene: {scene_id}")


def render_templates_table(measurements: Measurements, scene_id: str):
    """Display templates as a compact table with image and info."""
    st.subheader("Templates in Scene")

    scene_template_ids = measurements.get_scene_templates(scene_id) or []

    if not scene_template_ids:
        st.warning("No templates found for this scene")
        return

    # # Create table header
    cols = st.columns([1, 3])
    with cols[0]:
        st.markdown("##### Preview")
    with cols[1]:
        st.markdown("##### Info")

    vspace(1)

    # Create table rows
    for template_id in scene_template_ids:
        template = measurements.get_template(template_id)
        if not template:
            continue

        cols = st.columns([1, 3])

        # Preview column
        with cols[0]:
            template_path = PROJECT_ROOT / template.path
            if template_path.exists():
                img_str = get_image_base64(str(template_path), max_height=100)
                st.markdown(
                    f'<img src="data:image/png;base64,{img_str}" style="max-height: 100px; width: auto;" alt="{template.id}">',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"Template image not found: {template_path}")

        # Info column
        with cols[1]:
            detail_rows = []
            detail_rows.append(f"**ID:** `{template.id}`")
            label = getattr(template, "label", "None")
            detail_rows.append(f"**Label:** {label}")
            if template.width and template.height:
                detail_rows.append(
                    f"**Dimensions:** {template.width} x {template.height} {measurements.unit}"
                )
            st.markdown("<br>".join(detail_rows), unsafe_allow_html=True)

        # Add vertical spacing between rows
        vspace(2)


def render_config_panel() -> RenderConfig:
    """Render configuration controls and return RenderConfig."""
    st.subheader("Configuration")

    # Display Settings
    st.markdown("##### Display Settings")

    display_units = st.selectbox(
        "Display Units",
        ["m", "dm", "cm", "mm"],
        index=0,
        help="Units for 3D visualization (scales geometry accordingly)",
    )

    frustum_near = st.slider(
        "Frustum Near Plane",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
    )

    frustum_far = st.slider(
        "Frustum Far Plane",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.5,
    )

    axes_length = st.slider(
        "Camera Axes Length",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
    )

    return RenderConfig(
        canonical_units=display_units,
        frustum_near=frustum_near,
        frustum_far=frustum_far,
        axes_length=axes_length,
    )


def execute_pipeline(
    measurements: Measurements,
    scene_id: str,
    render_config: RenderConfig,
    scene_ids: list[str],
):
    """Execute the pipeline and update session state."""
    try:
        render_data, analysis_data = run_pipeline(
            data=measurements,
            scene_id=scene_id,
            original_units=measurements.unit,
            render_config=render_config,
        )

        # Update session state
        st.session_state.pipeline_run = True
        st.session_state.render_data = render_data
        st.session_state.analysis_data = analysis_data
        st.session_state.selected_scene_id = scene_id
        st.session_state.selected_scene_index = scene_ids.index(scene_id)
        st.session_state.pipeline_error = None
        st.session_state.pipeline_traceback = None

        # Generate renderer output
        renderer = Renderer(render_data, analysis_data)
        renderer._generate_html()

    except Exception as e:
        st.session_state.pipeline_run = False
        st.session_state.pipeline_error = str(e)
        st.session_state.pipeline_traceback = traceback.format_exc()


# ============================================================================
# Page Logic
# ============================================================================
init_session_state()

# Load measurements data
try:
    measurements = load_measurements()
except Exception as e:
    st.error(f"Failed to load measurements.yaml: {e}")
    st.stop()

# Get available scenes
scene_ids = measurements.list_scenes()
if not scene_ids:
    st.error("No scenes found in measurements.yaml")
    st.stop()

# Three-column layout: Scene preview, Templates, Configuration
col1, col2, col3 = st.columns([2, 2, 1.5], gap="large")

with col1:
    st.subheader("Scene Preview")

    # Scene selector
    selected_scene_id = st.selectbox(
        label="Choose a scene:",
        options=scene_ids,
        index=st.session_state.selected_scene_index,
        help="Select one of the available example scenes",
    )

    # Validate scene exists
    scene = measurements.get_scene(selected_scene_id)
    if not scene:
        st.error(f"Scene {selected_scene_id} not found in measurements")
        st.stop()

    render_scene_image(scene, selected_scene_id)

with col2:
    render_templates_table(measurements, selected_scene_id)

with col3:
    render_config = render_config_panel()

    # Run button in config panel
    st.markdown("---")
    run_button = st.button(
        "🚀 Run Pipeline",
        type="primary",
        use_container_width=True,
    )

    if run_button:
        if not selected_scene_id:
            st.error("Please select a scene first")
        else:
            execute_pipeline(
                measurements,
                selected_scene_id,
                render_config,
                scene_ids,
            )

# Status indicator at bottom
st.divider()
if st.session_state.pipeline_error:
    st.error(f"Pipeline failed: {st.session_state.pipeline_error}")
    with st.expander("Error details"):
        st.code(st.session_state.pipeline_traceback)
elif st.session_state.pipeline_run:
    st.success(
        "Pipeline completed successfully! Navigate to **3D Viewer** to see results."
    )
else:
    st.info("ℹ️ Configure and run pipeline to proceed to visualization")
