<!DOCTYPE html>
<html>
<head>
    <title>Template Pose Visualization</title>
    <link rel="stylesheet" href="static/css/viewer.css">
</head>
<body>
    <!-- Main UI Container for stacked elements -->
    <div id="uiContainer">
        <div id="navigationHelp" class="ui-panel">
            <h3>Template Pose Visualization</h3>
            <p>Controls</p>
            <div class="control-item">
                <span class="control-action">Rotate</span>
                <span class="control-method">Left mouse</span>
            </div>
            <div class="control-item">
                <span class="control-action">Pan</span>
                <span class="control-method">Right mouse</span>
            </div>
            <div class="control-item">
                <span class="control-action">Zoom</span>
                <span class="control-method">Scroll</span>
            </div>
        </div>

        <div id="visualizationControls" class="ui-panel">
            <div class="toggle-container">
                <label class="switch">
                    <input type="checkbox" id="distanceMode">
                    <span class="slider"></span>
                </label>
                <span class="switch-label">Toggle Distance Map Mode</span>
            </div>
        </div>
    </div>

    <div id="sceneInfoPanel" class="ui-panel">
        <h3>Scene Information</h3>
        <div class="info-group">
            <div class="info-row">Scene ID <span id="sceneId">-</span></div>
            <div class="info-row">Units <span id="sceneUnits">-</span></div>
            <div class="info-row">Image Size <span id="sceneImageSize">-</span></div>
        </div>
    </div>

    <div id="templateDetailsPanel" class="ui-panel">
        <h3>Template Information</h3>
        <h4>Details</h4>
        <div class="info-group">
            <div class="info-row">ID <span id="templateId">-</span></div>
            <div class="info-row">Label <span id="templateLabel">-</span></div>
            <div class="info-row">Size <span id="templateDimensions">-</span></div>
        </div>
        <h4>Distance Analysis</h4>
        <div class="info-group">
            <div class="info-row">Actual <span id="templateDistanceTrue">-</span></div>
            <div class="info-row">Predicted <span id="templateDistancePred">-</span></div>
            <div class="info-row">Error <span id="templateError">-</span></div>
            <div class="info-row">Error (%) <span id="templateErrorPercent">-</span></div>
        </div>
    </div>

    <div id="distanceColorLegend">
        <div>Distance Color Legend</div>
        <div class="legend-bar"></div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
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
        const templateMetadata = { metadata_json };
        const templateResults = { results_json };
        initViewer(meshesData, linesData, templateMetadata, templateResults);
    </script>
</body>
</html>