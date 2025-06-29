<!DOCTYPE html>
<html>
<head>
    <title>Template Pose Visualization</title>
    <style>
        body { margin: 0; padding: 0; background: #000; overflow: hidden; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }
        #colorLegend {
            position: absolute;
            bottom: 10px;
            right: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .legend-bar {
            width: 150px;
            height: 20px;
            background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
            margin: 5px 0;
            border: 1px solid white;
        }
    </style>
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
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x444444);
        document.body.appendChild(renderer.domElement);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // Distance mode variables
        let distanceMode = false;
        let allMeshes = [];
        let maxDistance = 0;

        // Add meshes (meshes_json is a placeholder for the actual mesh data)
        const meshesData = { meshes_json };
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

            const loader = new THREE.TextureLoader();
            loader.load(
                meshData.texture,
                texture => {
                    // Texture loaded successfully
                    const material = new THREE.ShaderMaterial({
                        uniforms: {
                            texture1: { value: texture },
                            backColor: { value: new THREE.Color(0xff0000) },
                            maxDistance: { value: 1.0 },
                            useDistanceMode: { value: false }
                        },
                        vertexShader: `
                            attribute float distance;
                            varying vec2 vUv;
                            varying float vDistance;
                            void main() {
                                vUv = uv;
                                vDistance = distance;
                                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                            }
                        `,
                        fragmentShader: `
                            uniform sampler2D texture1;
                            uniform vec3 backColor;
                            uniform float maxDistance;
                            uniform bool useDistanceMode;
                            varying vec2 vUv;
                            varying float vDistance;

                            vec3 distanceToColor(float dist, float maxDist) {
                                float normalized = clamp(dist / maxDist, 0.0, 1.0);
                                
                                // Color mapping: blue -> cyan -> green -> yellow -> red
                                if (normalized < 0.25) {
                                    float t = normalized / 0.25;
                                    return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t); // blue to cyan
                                } else if (normalized < 0.5) {
                                    float t = (normalized - 0.25) / 0.25;
                                    return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t); // cyan to green
                                } else if (normalized < 0.75) {
                                    float t = (normalized - 0.5) / 0.25;
                                    return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t); // green to yellow
                                } else {
                                    float t = (normalized - 0.75) / 0.25;
                                    return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t); // yellow to red
                                }
                            }

                            void main() {
                                if (gl_FrontFacing) {
                                    if (useDistanceMode) {
                                        gl_FragColor = vec4(distanceToColor(vDistance, maxDistance), 1.0);
                                    } else {
                                        gl_FragColor = texture2D(texture1, vUv);
                                    }
                                } else {
                                    gl_FragColor = vec4(backColor, 1.0);
                                }
                            }
                        `,
                        side: THREE.DoubleSide
                    });

                    const mesh = new THREE.Mesh(geometry, material);
                    scene.add(mesh);
                    allMeshes.push(mesh);
                    
                    // Update max distance uniform for all meshes
                    allMeshes.forEach(m => {
                        if (m.material.uniforms && m.material.uniforms.maxDistance) {
                            m.material.uniforms.maxDistance.value = maxDistance;
                        }
                    });
                },
                undefined,
                error => {
                    console.error("❌ Texture load failed:", meshData.texture, error);
                }
            );

            //const mesh = new THREE.Mesh(geometry, material);
            //scene.add(mesh);
        });

        // Add lines (lines_json is a placeholder for the actual line data)
        const linesData = { lines_json };
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

        // Add origin-to-template line using vertex 0 as template point
        let templateOrigin = new THREE.Vector3(0, 0, 0); // Default origin

        // Get the first vertex (vertex 0) from the first mesh as template origin
        if (meshesData.length > 0 && meshesData[0].vertices.length > 0) {
            const vertex0 = meshesData[0].vertices[0];
            templateOrigin.set(vertex0[0], vertex0[1], vertex0[2]);
        }

        const originLineGeometry = new THREE.BufferGeometry();
        const originLinePositions = [
            0, 0, 0,  // World origin
            templateOrigin.x, templateOrigin.y, templateOrigin.z  // Template origin (vertex 0)
        ];
        originLineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(originLinePositions), 3));

        const originLineMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,  // White line
            linewidth: 3
        });

        const originLine = new THREE.Line(originLineGeometry, originLineMaterial);
        scene.add(originLine);

        // Position camera
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);

        // Mouse controls (orbit and pan)
        let mouseDown = false;
        let mouseButton = 0;
        let mouseX = 0;
        let mouseY = 0;
        let cameraDistance = Math.sqrt(camera.position.x**2 + camera.position.y**2 + camera.position.z**2);
        let cameraTheta = Math.atan2(camera.position.x, camera.position.z);
        let cameraPhi = Math.acos(camera.position.y / cameraDistance);
        let lookAtTarget = new THREE.Vector3(0, 0, 0);

        document.addEventListener('mousedown', (event) => {
            mouseDown = true;
            mouseButton = event.button;
            mouseX = event.clientX;
            mouseY = event.clientY;
            event.preventDefault();
        });

        document.addEventListener('mouseup', () => {
            mouseDown = false;
        });

        document.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });

        document.addEventListener('mousemove', (event) => {
            if (!mouseDown) return;

            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;

            if (mouseButton === 0) { // Left mouse - rotate
                cameraTheta -= deltaX * 0.01;
                cameraPhi += deltaY * 0.01;

                // Clamp phi
                cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraPhi));

                // Update camera position
                camera.position.x = lookAtTarget.x + cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
                camera.position.y = lookAtTarget.y + cameraDistance * Math.cos(cameraPhi);
                camera.position.z = lookAtTarget.z + cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);

                camera.lookAt(lookAtTarget);
            } else if (mouseButton === 2) { // Right mouse - pan
                // Get camera's right and up vectors
                const cameraRight = new THREE.Vector3();
                const cameraUp = new THREE.Vector3();
                camera.getWorldDirection(new THREE.Vector3()); // Update camera matrix
                cameraRight.setFromMatrixColumn(camera.matrixWorld, 0);
                cameraUp.setFromMatrixColumn(camera.matrixWorld, 1);

                // Pan speed proportional to distance
                const panSpeed = cameraDistance * 0.001;
                
                // Move lookAt target and camera position
                const panOffset = new THREE.Vector3();
                panOffset.addScaledVector(cameraRight, -deltaX * panSpeed);
                panOffset.addScaledVector(cameraUp, deltaY * panSpeed);

                lookAtTarget.add(panOffset);
                camera.position.add(panOffset);
                camera.lookAt(lookAtTarget);
            }

            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        document.addEventListener('wheel', (event) => {
            cameraDistance *= (1 + event.deltaY * 0.001);
            cameraDistance = Math.max(1, Math.min(100, cameraDistance));

            // Update camera position
            camera.position.x = lookAtTarget.x + cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            camera.position.y = lookAtTarget.y + cameraDistance * Math.cos(cameraPhi);
            camera.position.z = lookAtTarget.z + cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);
        });

        // Distance mode toggle
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

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>