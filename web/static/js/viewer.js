// Main viewer initialization and logic
let scene, camera, renderer, controls;
let mouse = new THREE.Vector2();
let raycaster = new THREE.Raycaster();
let allMeshes = [];
let distanceMode = false;
let maxDistance = 0;
let minDistance = Infinity;
let metadata = {};
let results = {};

function initViewer(meshesData, linesData, templateMetadata, templateResults) {
    // Initialize global variables
    metadata = templateMetadata;
    results = templateResults;

    // Display scene info
    displaySceneInfo();

    setupScene();
    setupLighting();
    
    // Create meshes and lines
    createMeshes(meshesData);
    createLines(linesData);
    
    // Setup controls
    controls = new CameraControls(camera);
    
    // Setup UI
    setupDistanceModeToggle();

    // Setup template selection
    setupTemplateSelection();
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function displaySceneInfo() {
    if (metadata.scene_id) {
        document.getElementById('sceneId').textContent = metadata.scene_id;
    }
    if (metadata.units) {
        document.getElementById('sceneUnits').textContent = metadata.units;
    }
    if (metadata.image_size) {
        document.getElementById('sceneImageSize').textContent = 
            `${metadata.image_size.width} × ${metadata.image_size.height}`;
    }
}

function setupScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000);
    // renderer.setClearColor(0x0e1117);
    // renderer.setClearColor(0x08090d);
    document.body.appendChild(renderer.domElement);
    
    // Position camera
    camera.position.set(0.2, 0.2, -0.7);
    camera.lookAt(0, 0, 0);
}

function setupLighting() {
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
}

function createMeshes(meshesData) {
    // Calculate global min/max distances across all meshes
    let globalMaxDistance = 0;
    let globalMinDistance = Infinity;

    // Store distances for each mesh
    const allDistances = [];
    
    meshesData.forEach((meshData, index) => {
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(meshData.vertices.flat());
        const indices = new Uint16Array(meshData.triangles.flat());
        
        // Calculate distances for this mesh
        const distances = [];
        for (let i = 0; i < vertices.length; i += 3) {
            const distance = Math.sqrt(vertices[i] ** 2 + vertices[i + 1] ** 2 + vertices[i + 2] ** 2);
            distances.push(distance);
            globalMaxDistance = Math.max(globalMaxDistance, distance);
            globalMinDistance = Math.min(globalMinDistance, distance);
        }
        
        allDistances.push({ geometry, vertices, indices, distances, meshData });
    });
    
    // Update global variables
    maxDistance = globalMaxDistance;
    minDistance = globalMinDistance;
    
    // Create all meshes
    allDistances.forEach(({ geometry, vertices, indices, distances, meshData }, index) => {
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        // Inject basic UVs (only valid if the mesh is a quad with 4 vertices)
        if (vertices.length === 12) {
            const uvs = new Float32Array([
                0, 1,
                1, 1,
                1, 0,
                0, 0
            ]);
            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }
        
        geometry.setAttribute('distance', new THREE.BufferAttribute(new Float32Array(distances), 1));
        
        // Load texture and create mesh
        const loader = new THREE.TextureLoader();
        loader.load(
            meshData.texture,
            texture => {
                const material = new THREE.ShaderMaterial({
                    uniforms: {
                        texture1: { value: texture },
                        backColor: { value: new THREE.Color(0xff0000) },
                        maxDistance: { value: maxDistance },
                        minDistance: { value: minDistance },
                        useDistanceMode: { value: false },
                        selected: { value: false },
                        hovered: { value: false }
                    },
                    vertexShader: templateVertexShader,
                    fragmentShader: templateFragmentShader,
                    side: THREE.DoubleSide
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.templateId = meshData.id;
                
                scene.add(mesh);
                allMeshes.push(mesh);
            },
            undefined,
            error => {
                console.error("Texture load failed:", meshData.texture, error);
            }
        );
    });
}

function createLines(linesData) {
    linesData.forEach((lineData, index) => {
        const geometry = new THREE.BufferGeometry();

        // Create line segments
        const positions = [];
        lineData.lines.forEach(line => {
            const start = lineData.points[line[0]];
            const end = lineData.points[line[1]];
            positions.push(start[0], start[1], start[2]);
            positions.push(end[0], end[1], end[2]);
        });

        const positionsArray = new Float32Array(positions);
        geometry.setAttribute('position', new THREE.BufferAttribute(positionsArray, 3));

        const material = new THREE.LineBasicMaterial({
            color: new THREE.Color(lineData.color[0], lineData.color[1], lineData.color[2]),
            linewidth: 2
        });

        const lines = new THREE.LineSegments(geometry, material);
        scene.add(lines);
    });
}

function setupDistanceModeToggle() {
    document.getElementById('distanceMode').addEventListener('change', (event) => {
        distanceMode = event.target.checked;
        document.getElementById('distanceColorLegend').style.display = distanceMode ? 'block' : 'none';
        
        // Toggle shader uniforms for all meshes
        allMeshes.forEach(mesh => {
            if (mesh.material.uniforms && mesh.material.uniforms.useDistanceMode) {
                mesh.material.uniforms.useDistanceMode.value = distanceMode;
            }
        });
    });
}

function setupTemplateSelection() {
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('click', onTemplateClick);
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onMouseMove(event) {
    // Normalize mouse coordinates to -1 to 1 range
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Find out which mesh is under the mouse cursor
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(allMeshes);

    // Reset all mesh materials to the default state
    allMeshes.forEach(mesh => {
        if (mesh.material.uniforms) {
            // mesh.material.uniforms.selected = { value: false }; TODO check if this is needed
            mesh.material.uniforms.hovered = { value: false };
        }
    });

    // Highlight hovered mesh
    if (intersects.length > 0) {
        const hoveredMesh = intersects[0].object;
        if (hoveredMesh.material.uniforms) {
            hoveredMesh.material.uniforms.hovered = { value: true };
        }
        document.body.style.cursor = 'pointer';
    } else {
        document.body.style.cursor = 'default';
    }
}

function onTemplateClick(event) {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(allMeshes);

    if (intersects.length > 0) {
        const clickedMesh = intersects[0].object;
        const templateId = clickedMesh.userData.templateId;

        // Handle template selection logic
        selectTemplate(templateId);
    } else {
        // Deselect template if clicked outside
        deselectTemplate();
    }
}

function selectTemplate(templateId) {
    // Update UI to reflect selected template
    allMeshes.forEach(mesh => {
        if (mesh.material.uniforms) {
            mesh.material.uniforms.selected = { value: mesh.userData.templateId === templateId };
        }
    });

    // Update info panel
    const templateData = metadata.templates[templateId];
    const result = results[templateId];
    
    document.getElementById('templateId').textContent = templateId;
    document.getElementById('templateLabel').textContent = templateData?.label || 'N/A';
    document.getElementById('templateDimensions').textContent = 
        (templateData?.width && templateData?.height) 
            ? `${templateData.width.toFixed(1)} × ${templateData.height.toFixed(1)}` 
            : 'N/A';
    
    // Handle predicted distance
    document.getElementById('templateDistancePred').textContent = 
        result?.distance_pred ? result.distance_pred.toFixed(2) : 'N/A';
    
    // Handle ground truth and errors (may not exist)
    if (result?.distance_true !== undefined) {
        document.getElementById('templateDistanceTrue').textContent = result.distance_true.toFixed(2);
        document.getElementById('templateError').textContent = result.error_abs.toFixed(2);
        document.getElementById('templateErrorPercent').textContent = `${result.error_rel.toFixed(2)}%`;
    } else {
        document.getElementById('templateDistanceTrue').textContent = 'N/A';
        document.getElementById('templateError').textContent = 'N/A';
        document.getElementById('templateErrorPercent').textContent = 'N/A';
    }

    // Display the info panel
    document.getElementById('templateDetailsPanel').style.display = 'block';

    // Still log the selected template ID for debugging
    console.log('Template selected:', templateId);
}

function deselectTemplate() {
    // Reset all mesh materials to the default state
    allMeshes.forEach(mesh => {
        if (mesh.material.uniforms) {
            mesh.material.uniforms.selected = { value: false };
        }
    });

    // Hide the info panel
    document.getElementById('templateDetailsPanel').style.display = 'none';

    // Log deselection for debugging
    console.log('Template deselected');
}