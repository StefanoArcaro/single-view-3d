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

st.markdown("---")

# Theoretical Background
st.header("Theoretical Background")

st.markdown("""
This section provides the mathematical foundations underlying the pose estimation pipeline.
Understanding these concepts will help you interpret the results and parameter choices.
""")

# Table of Contents (optional but nice for long theory sections)
with st.expander("📋 Section Contents", expanded=False):
    st.markdown("""
    1. Homography and Planar Transformations
    2. Camera Model and Projection
    3. Homography Decomposition
    4. Camera Calibration
    5. Distance Estimation
    """)

st.markdown("---")

# Theory sections
st.subheader("1. Homography and Planar Transformations")

st.markdown("""
A homography is a projective transformation that relates corresponding points between two 
planes. For a planar template lying on the world coordinate plane $Z = 0$, the relationship 
between a world point $(X, Y, 0)^T$ and its image projection $(u, v)^T$ is given by:
""")

st.latex(r"""
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} 
\sim \mathbf{H} 
\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
""")

st.markdown("""
where $\\mathbf{H}$ is a $3 \\times 3$ homography matrix defined up to scale. This matrix 
encodes both the camera's intrinsic parameters and the relative pose between camera and template.

[PLACEHOLDER: More detailed explanation of homography properties, degrees of freedom, 
estimation methods (DLT, normalized DLT), RANSAC for outlier rejection, etc.]
""")

st.markdown("---")

st.subheader("2. Camera Model and Projection")

st.markdown("""
We use the standard pinhole camera model with intrinsic matrix $\\mathbf{K}$:
""")

st.latex(r"""
\mathbf{K} = 
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
""")

st.markdown("""
where:
- $f_x, f_y$ are focal lengths in pixel units
- $c_x, c_y$ is the principal point (optical center)

[PLACEHOLDER: Full projection equation, relationship between world coordinates and image 
coordinates, perspective projection, homogeneous coordinates, etc.]
""")

st.markdown("---")

st.subheader("3. Homography Decomposition")

st.markdown("""
Given a homography $\\mathbf{H}$ and known intrinsic matrix $\\mathbf{K}$, we can recover 
the rotation $\\mathbf{R}$ and translation $\\mathbf{t}$ relating the template plane to 
the camera frame.

The decomposition follows from the relationship:
""")

st.latex(r"""
\mathbf{H} = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
""")

st.markdown("""
where $\\mathbf{r}_1, \\mathbf{r}_2$ are the first two columns of $\\mathbf{R}$, and 
$\\lambda$ is a scale factor.

[PLACEHOLDER: Detailed decomposition algorithm, handling of ambiguity (4 possible solutions), 
physical constraints for selecting correct solution, numerical considerations, etc.]
""")

st.markdown("---")

st.subheader("4. Camera Calibration")

st.markdown("""
When intrinsic parameters are unknown, they can be estimated from multiple homographies 
observed from different template views (Zhang's calibration method).

[PLACEHOLDER: Constraints from homographies, closed-form solution for intrinsic parameters, 
refinement through nonlinear optimization, handling of radial distortion, etc.]
""")

st.markdown("---")

st.subheader("5. Distance Estimation")

st.markdown("""
Once pose $(\\mathbf{R}, \\mathbf{t})$ is recovered, the distance from camera to any point 
on the template can be computed by transforming the point to camera coordinates:
""")

st.latex(r"""
\mathbf{p}_{cam} = \mathbf{R} \mathbf{p}_{world} + \mathbf{t}
""")

st.markdown("""
The Euclidean distance is then:
""")

st.latex(r"""
d = \|\mathbf{p}_{cam}\| = \sqrt{p_x^2 + p_y^2 + p_z^2}
""")

st.markdown("""
[PLACEHOLDER: Error propagation, accuracy considerations, effects of calibration errors, 
template planarity assumptions, etc.]
""")

st.markdown("---")

# References section
st.subheader("References")

st.markdown("""
1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
2. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.
3. [Additional references would go here]
""")

st.markdown("---")

st.success(
    "📚 Ready to explore the demo? Head to **Demo Setup** in the sidebar to begin!"
)
