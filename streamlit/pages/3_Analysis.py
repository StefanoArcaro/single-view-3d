import base64
import os
import sys
from io import BytesIO

import pandas as pd
from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.configs.analysis_data import TemplateAnalysisData
from src.configs.render_data import TemplateRenderData
from src.utils import find_project_root, load_rgb

PROJECT_ROOT = find_project_root()

# Page config
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="wide",
)


# ============================================================================
# Session State Validation
# ============================================================================
def validate_pipeline_state():
    """Check if pipeline has run successfully before rendering analysis."""
    if st.session_state.get("pipeline_error"):
        st.error(
            "⚠️ Pipeline failed. Please fix the error and rerun from the Setup page."
        )
        st.stop()

    if not st.session_state.get("pipeline_run", False):
        st.info(
            "ℹ️ No pipeline results available. Please run the pipeline from the Setup page."
        )
        st.stop()


# ============================================================================
# Data Loading and Caching
# ============================================================================
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


def render_summary_metrics(summary: dict):
    """Display summary statistics as metric cards."""
    st.subheader("Summary Metrics")

    units = summary["units"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Templates", summary["total_templates"], border=True)
    col2.metric(
        "Templates with Ground-truth", summary["templates_with_gt"], border=True
    )
    col3.metric(
        "Mean Absolute Error",
        f"{summary['mean_error_abs']:.3f} {units}"
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

    if summary["templates_with_gt"] > 1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Absolute Error Range",
                f"{summary['min_error_abs']:.3f} – {summary['max_error_abs']:.3f} {units}",
                border=True,
            )
        with col_b:
            st.metric(
                "Relative Error Range",
                f"{summary['min_error_rel']:.2f} – {summary['max_error_rel']:.2f}%",
                border=True,
            )


def build_analysis_table(render_data, analysis_data, units: str) -> pd.DataFrame:
    """Build a DataFrame with per-template analysis data."""
    table_data = []

    t_render: TemplateRenderData
    t_analysis: TemplateAnalysisData
    for t_render, t_analysis in zip(render_data.templates, analysis_data.templates):
        texture_path = PROJECT_ROOT / t_render.texture_path

        # Get base64 image for preview
        img_str = get_image_base64(str(texture_path), max_height=160)
        img_data_uri = f"data:image/png;base64,{img_str}" if img_str else None

        table_data.append(
            {
                "Preview": img_data_uri,
                "ID": t_render.id,
                "Label": t_render.label,
                f"Width ({units})": t_render.width,
                f"Height ({units})": t_render.height,
                f"Predicted Distance ({units})": t_analysis.distance_pred,
                f"True Distance ({units})": t_analysis.distance_true
                if t_analysis.distance_true is not None
                else "N/A",
                f"Absolute Error ({units})": t_analysis.error_abs
                if t_analysis.error_abs is not None
                else "N/A",
                "Relative Error (%)": t_analysis.error_rel
                if t_analysis.error_rel is not None
                else "N/A",
            }
        )

    return pd.DataFrame(table_data)


def render_analysis_table(df: pd.DataFrame):
    """Display the per-template analysis table."""
    st.subheader("Per-Template Analysis Table")

    st.dataframe(
        df,
        column_config={
            "Preview": st.column_config.ImageColumn(
                "Preview",
            )
        },
        hide_index=True,
        row_height=80,
    )


# ============================================================================
# Page Logic
# ============================================================================
validate_pipeline_state()

# Load data from session state
render_data = st.session_state.render_data
analysis_data = st.session_state.analysis_data

# Get summary statistics
summary = analysis_data.get_summary_stats()
units = summary["units"]

# Render summary metrics
render_summary_metrics(summary)

vspace(2)

# Build and render analysis table
df = build_analysis_table(render_data, analysis_data, units)
render_analysis_table(df)
