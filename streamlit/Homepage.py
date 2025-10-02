import streamlit as st

# Set page config (this MUST be the first Streamlit command)
# This sets the tab title that persists across all pages
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="centered",
)

# Page content
st.title("📐 Single-view 3D Reconstruction from Planar Templates")
st.markdown("---")

# Brief Introduction
st.header("Introduction")

st.markdown("""
This demonstration showcases an automated pipeline for estimating 3D camera poses from 2D images 
using planar template matching and homography decomposition. The system:

- Detects and matches planar templates (objects with known metric dimensions) in scene images
- Computes homography transformations relating template coordinates to image coordinates  
- Decomposes homographies to recover full 6-DOF camera poses (rotation and translation)
- Calibrates camera intrinsic parameters from multiple template observations
- Estimates distances from camera to templates in metric units

This approach is fundamental to applications in augmented reality, robotics, photogrammetry, 
and structure-from-motion systems.
""")

st.info(
    "👉 Navigate through the pages in the sidebar to run the pipeline on example scenes."
)
