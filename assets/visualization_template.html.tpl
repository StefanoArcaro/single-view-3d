
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
                    0, 0,   // vertex 0
                    1, 0,   // vertex 1
                    1, 1,   // vertex 2
                    0, 1    // vertex 3
                ]);
                geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            }

            const loader = new THREE.TextureLoader();
            loader.load(
                meshData.texture,
                texture => {
                    // Texture loaded successfully
                    material = new THREE.MeshBasicMaterial({
                        map: texture,
                        side: THREE.DoubleSide
                    });
                    const mesh = new THREE.Mesh(geometry, material);
                    scene.add(mesh);
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

        // Position camera
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);

        // Mouse controls (simple orbit)
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let cameraDistance = Math.sqrt(camera.position.x**2 + camera.position.y**2 + camera.position.z**2);
        let cameraTheta = Math.atan2(camera.position.x, camera.position.z);
        let cameraPhi = Math.acos(camera.position.y / cameraDistance);

        document.addEventListener('mousedown', (event) => {
            mouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        document.addEventListener('mouseup', () => {
            mouseDown = false;
        });

        document.addEventListener('mousemove', (event) => {
            if (!mouseDown) return;

            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;

            cameraTheta -= deltaX * 0.01;
            cameraPhi += deltaY * 0.01;

            // Clamp phi
            cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraPhi));

            // Update camera position
            camera.position.x = cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            camera.position.y = cameraDistance * Math.cos(cameraPhi);
            camera.position.z = cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);

            camera.lookAt(0, 0, 0);

            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        document.addEventListener('wheel', (event) => {
            cameraDistance *= (1 + event.deltaY * 0.001);
            cameraDistance = Math.max(1, Math.min(50, cameraDistance));

            // Update camera position
            camera.position.x = cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            camera.position.y = cameraDistance * Math.cos(cameraPhi);
            camera.position.z = cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);
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