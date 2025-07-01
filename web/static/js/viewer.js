// Main viewer initialization and logic
let scene, camera, renderer, controls;
let allMeshes = [];
let maxDistance = 0;
let distanceMode = false;

function initViewer(meshesData, linesData) {
    setupScene();
    setupLighting();
    
    // Create meshes and lines
    createMeshes(meshesData);
    createLines(linesData);
    
    // Setup controls
    controls = new CameraControls(camera);
    
    // Setup UI
    setupDistanceModeToggle();
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function setupScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x444444);
    document.body.appendChild(renderer.domElement);
    
    // Position camera
    camera.position.set(3, 3, -10);
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
    meshesData.forEach((meshData, index) => {
        const geometry = new THREE.BufferGeometry();

        // Convert vertices and triangles
        const vertices = new Float32Array(meshData.vertices.flat());
        const indices = new Uint16Array(meshData.triangles.flat());

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        // Inject basic UVs (only valid if the mesh is a quad with 4 vertices)
        if (vertices.length === 12) {  // 4 vertices * 3 coordinates
            const uvs = new Float32Array([
                0, 1,   // vertex 0
                1, 1,   // vertex 1
                1, 0,   // vertex 2
                0, 0    // vertex 3
            ]);
            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }

        // Calculate distances for color mapping
        const distances = [];
        for (let i = 0; i < vertices.length; i += 3) {
            const distance = Math.sqrt(vertices[i]**2 + vertices[i+1]**2 + vertices[i+2]**2);
            distances.push(distance);
            maxDistance = Math.max(maxDistance, distance);
        }
        
        // Store distances as vertex attribute
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
                        maxDistance: { value: 1.0 },
                        useDistanceMode: { value: false }
                    },
                    vertexShader: templateVertexShader,
                    fragmentShader: templateFragmentShader,
                    side: THREE.DoubleSide
                });

                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);
                allMeshes.push(mesh);
                
                // Update max distance uniform for all meshes
                updateMaxDistanceUniforms();
            },
            undefined,
            error => {
                console.error("❌ Texture load failed:", meshData.texture, error);
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

function updateMaxDistanceUniforms() {
    allMeshes.forEach(mesh => {
        if (mesh.material.uniforms && mesh.material.uniforms.maxDistance) {
            mesh.material.uniforms.maxDistance.value = maxDistance;
        }
    });
}

function setupDistanceModeToggle() {
    document.getElementById('distanceMode').addEventListener('change', (event) => {
        distanceMode = event.target.checked;
        document.getElementById('colorLegend').style.display = distanceMode ? 'block' : 'none';
        
        // Toggle shader uniforms for all meshes
        allMeshes.forEach(mesh => {
            if (mesh.material.uniforms && mesh.material.uniforms.useDistanceMode) {
                mesh.material.uniforms.useDistanceMode.value = distanceMode;
            }
        });
    });
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