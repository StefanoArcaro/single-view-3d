import os
import re
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import find_project_root

# Page config
st.set_page_config(
    page_title="3D from Planar Templates",
    page_icon="📐",
    layout="wide",
)

PROJECT_ROOT = find_project_root()

HTML_PATH = PROJECT_ROOT / "web" / "viewer.html"
CSS_PATH = PROJECT_ROOT / "web" / "static" / "css" / "viewer.css"
JS1_PATH = PROJECT_ROOT / "web" / "static" / "js" / "shaders.js"
JS2_PATH = PROJECT_ROOT / "web" / "static" / "js" / "controls.js"
JS3_PATH = PROJECT_ROOT / "web" / "static" / "js" / "viewer.js"

# Read the files
with open(HTML_PATH, "r") as f:
    html_content = f.read()

with open(CSS_PATH, "r") as f:
    css_content = f.read()

with open(JS1_PATH, "r") as f:
    js1_content = f.read()

with open(JS2_PATH, "r") as f:
    js2_content = f.read()

with open(JS3_PATH, "r") as f:
    js3_content = f.read()

# Extract only the body content from HTML (everything between <body> and </body>)
body_match = re.search(r"<body>(.*?)</body>", html_content, re.DOTALL)
if body_match:
    body_content = body_match.group(1)
else:
    # Fallback: use the whole HTML content if body tags not found
    body_content = html_content

# Remove script tags that reference external JS files from body content
body_content = re.sub(
    r'<script\s+src=["\']https://cdnjs\.cloudflare\.com/ajax/libs/three\.js/.*?</script>\s*',
    "",
    body_content,
)
body_content = re.sub(
    r'<script\s+src=["\']static/js/(shaders|controls|viewer)\.js["\']></script>\s*',
    "",
    body_content,
)

# Extract the inline script with data (the one that calls initViewer)
data_script_match = re.search(
    r"<script>(.*?const meshesData = .*?initViewer\(.*?\);.*?)</script>",
    body_content,
    re.DOTALL,
)
data_script = data_script_match.group(1) if data_script_match else ""

# Remove the inline script from body content (we'll add it back at the end)
if data_script:
    body_content = re.sub(
        r"<script>.*?const meshesData = .*?initViewer\(.*?\);.*?</script>",
        "",
        body_content,
        flags=re.DOTALL,
    )

# Combine everything in the correct order
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
    {css_content}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    {body_content}
    <script>
    {js1_content}
    </script>
    <script>
    {js2_content}
    </script>
    <script>
    {js3_content}
    </script>
    <script>
    {data_script}
    </script>
</body>
</html>
"""

# Embed in Streamlit
st.components.v1.html(full_html, height=700, scrolling=False)
st.html("<style> .main {overflow: hidden} </style>")
