import streamlit as st

# Set page config (this MUST be the first Streamlit command)
# This sets the tab title that persists across all pages
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="centered",
)

# Page content
st.title("📐 Single-View 3D Reconstruction from Planar Templates")
st.markdown("---")

# Brief Introduction
st.header("Introduction")

st.markdown("""
This application demonstrates an automated pipeline for **partial 3D scene reconstruction** 
from single 2D images using planar template matching and homography decomposition.

**Key capabilities:**
- Detect and match planar objects with known metric dimensions
- Reconstruct their 3D positions and orientations in the scene
- Recover camera pose relative to the templates
- Visualize the partial 3D reconstruction interactively

The reconstruction is partial because only the planar templates present in the scene are 
reconstructed metrically. This can however provide a metric anchor for the entire scene.
""")

st.markdown("##### How to Use")
st.markdown("""
1. **Setup** — Select a scene, configure display settings, and run the pipeline
2. **3D Viewer** — Explore the reconstructed 3D scene with interactive controls
3. **Analysis** — Review accuracy metrics and per-template results
""")

st.info("👉 Navigate to **Setup** in the sidebar to begin with example scenes.")
