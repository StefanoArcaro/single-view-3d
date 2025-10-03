import os
import re
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import find_project_root

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
    """Check if pipeline has run successfully before rendering viewer."""
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
# File Path Configuration
# ============================================================================
def get_viewer_paths():
    """Return paths to all viewer-related files."""
    return {
        "html": PROJECT_ROOT / "web" / "viewer.html",
        "css": PROJECT_ROOT / "web" / "static" / "css" / "viewer.css",
        "js_shaders": PROJECT_ROOT / "web" / "static" / "js" / "shaders.js",
        "js_controls": PROJECT_ROOT / "web" / "static" / "js" / "controls.js",
        "js_viewer": PROJECT_ROOT / "web" / "static" / "js" / "viewer.js",
    }


# ============================================================================
# Content Loading
# ============================================================================
def load_file_content(path):
    """Load content from a file."""
    with open(path, "r") as f:
        return f.read()


def load_viewer_files():
    """Load all viewer files and return their contents."""
    paths = get_viewer_paths()
    return {
        "html": load_file_content(paths["html"]),
        "css": load_file_content(paths["css"]),
        "js_shaders": load_file_content(paths["js_shaders"]),
        "js_controls": load_file_content(paths["js_controls"]),
        "js_viewer": load_file_content(paths["js_viewer"]),
    }


# ============================================================================
# HTML Processing
# ============================================================================
def extract_body_content(html_content: str) -> str:
    """Extract content between <body> and </body> tags."""
    body_match = re.search(r"<body>(.*?)</body>", html_content, re.DOTALL)
    if body_match:
        return body_match.group(1)
    # Fallback: use the whole HTML content if body tags not found
    return html_content


def extract_data_script(body_content: str) -> str:
    """Extract the inline script containing meshesData and initViewer call."""
    data_script_match = re.search(
        r"<script>(.*?const meshesData = .*?initViewer\(.*?\);.*?)</script>",
        body_content,
        re.DOTALL,
    )
    return data_script_match.group(1) if data_script_match else ""


def clean_body_content(body_content: str) -> str:
    """Remove script tags that reference external files."""
    # Remove Three.js CDN script
    body_content = re.sub(
        r'<script\s+src=["\']https://cdnjs\.cloudflare\.com/ajax/libs/three\.js/.*?</script>\s*',
        "",
        body_content,
    )

    # Remove local JS file references
    body_content = re.sub(
        r'<script\s+src=["\']static/js/(shaders|controls|viewer)\.js["\']></script>\s*',
        "",
        body_content,
    )

    # Remove inline data script (will be added back later)
    body_content = re.sub(
        r"<script>.*?const meshesData = .*?initViewer\(.*?\);.*?</script>",
        "",
        body_content,
        flags=re.DOTALL,
    )

    return body_content


def build_viewer_html(contents: dict, body_content: str, data_script: str) -> str:
    """Combine all components into a complete HTML document."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
    {contents["css"]}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    {body_content}
    <script>
    {contents["js_shaders"]}
    </script>
    <script>
    {contents["js_controls"]}
    </script>
    <script>
    {contents["js_viewer"]}
    </script>
    <script>
    {data_script}
    </script>
</body>
</html>
"""


# ============================================================================
# Page Logic
# ============================================================================
validate_pipeline_state()

# Load all viewer files
contents = load_viewer_files()

# Process HTML content
body_content = extract_body_content(contents["html"])
data_script = extract_data_script(body_content)
body_content = clean_body_content(body_content)

# Build complete HTML
full_html = build_viewer_html(contents, body_content, data_script)

# Render viewer
st.components.v1.html(full_html, height=650, scrolling=False)
st.html("<style> .main {overflow: hidden} </style>")
