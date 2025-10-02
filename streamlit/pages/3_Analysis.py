import base64
from io import BytesIO

import pandas as pd
from PIL import Image

import streamlit as st
from src.utils import find_project_root, load_rgb

# Check pipeline state before rendering viewer
if st.session_state.get("pipeline_error"):
    st.error("⚠️ Pipeline failed. Please fix the error and rerun from the Setup page.")
    st.stop()

if not st.session_state.get("pipeline_run", False):
    st.info(
        "ℹ️ No pipeline results available. Please run the pipeline from the Setup page."
    )
    st.stop()

PROJECT_ROOT = find_project_root()

render_data = st.session_state.render_data
analysis_data = st.session_state.analysis_data

# -------------------------------------------------------------------
# Page title and scene info
# -------------------------------------------------------------------
st.set_page_config(
    page_title="3D Pipeline Analysis",
    page_icon="📊",
    layout="wide",
)


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


# -------------------------------------------------------------------
# Summary statistics as cards
# -------------------------------------------------------------------
summary = analysis_data.get_summary_stats()
st.subheader("Summary Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Templates", summary["total_templates"], border=True)
col2.metric("Templates with Ground-truth", summary["templates_with_gt"], border=True)
col3.metric(
    "Mean Absolute Error",
    f"{summary['mean_error_abs']:.3f} {summary['units']}"
    if summary["mean_error_abs"] is not None
    else "N/A",
    border=True,
)
col4.metric(
    "Mean Relative Error",
    f"{summary['mean_error_rel']:.2f} %"
    if summary["mean_error_rel"] is not None
    else "N/A",
    border=True,
)

if summary["templates_with_gt"] > 0:
    st.markdown(
        f"**Absolute Error Range:** {summary['min_error_abs']:.3f} – {summary['max_error_abs']:.3f} {summary['units']}"
    )
    if summary["mean_error_rel"] is not None:
        st.markdown(
            f"**Relative Error Range:** {summary['min_error_rel']:.2f}% – {summary['max_error_rel']:.2f}%"
        )

st.divider()

# -------------------------------------------------------------------
# Template-level table (sortable)
# -------------------------------------------------------------------
st.subheader("Per-Template Analysis Table")

table_data = []
for template_render, template_analysis in zip(
    render_data.templates, analysis_data.templates
):
    texture_path = PROJECT_ROOT / template_render.texture_path

    # Use cached function - done!
    img_str = get_image_base64(str(texture_path), max_height=160)
    img_html = (
        f'<img src="data:image/png;base64,{img_str}" width="80">' if img_str else ""
    )

    table_data.append(
        {
            "Preview": img_html,
            "ID": template_render.id,
            "Label": template_render.label,
            "Width": template_render.width,
            "Height": template_render.height,
            "Predicted Distance": template_analysis.distance_pred,
            "True Distance": template_analysis.distance_true
            if template_analysis.distance_true is not None
            else "N/A",
            "Absolute Error": template_analysis.error_abs
            if template_analysis.error_abs is not None
            else "N/A",
            "Relative Error (%)": template_analysis.error_rel
            if template_analysis.error_rel is not None
            else "N/A",
        }
    )

df = pd.DataFrame(table_data)

# Display with HTML rendering for images
st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
