<!DOCTYPE html>
<html>
<head>
    <title>Template Pose Visualization</title>
    <link rel="stylesheet" href="static/css/viewer.css">
</head>
<body>
    <div id="info">
        <h3>Template Pose Visualization</h3>
        <p>Controls:</p>
        <ul>
            <li>Left mouse: Rotate</li>
            <li>Right mouse: Pan</li>
            <li>Scroll: Zoom</li>
        </ul>
    </div>

    <div id="controls">
        <label>
            <input type="checkbox" id="distanceMode"> Distance Map Mode
        </label>
    </div>

    <div id="colorLegend">
        <div>Distance Color Legend:</div>
        <div class="legend-bar"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>Near</span>
            <span>Far</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="static/js/shaders.js"></script>
    <script src="static/js/controls.js"></script>
    <script src="static/js/viewer.js"></script>
    <script>
        // Data injection and initialization
        const meshesData = { meshes_json };
        const linesData = { lines_json };
        initViewer(meshesData, linesData);
    </script>
</body>
</html>