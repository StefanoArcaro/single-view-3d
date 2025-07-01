// Mouse controls for camera manipulation
class CameraControls {
    constructor(camera, lookAtTarget = new THREE.Vector3(0, 0, 0)) {
        this.camera = camera;
        this.lookAtTarget = lookAtTarget;
        
        // Initialize camera spherical coordinates
        this.cameraDistance = Math.sqrt(
            camera.position.x**2 + camera.position.y**2 + camera.position.z**2
        );
        this.cameraTheta = Math.atan2(camera.position.x, camera.position.z);
        this.cameraPhi = Math.acos(camera.position.y / this.cameraDistance);
        
        // Mouse state
        this.mouseDown = false;
        this.mouseButton = 0;
        this.mouseX = 0;
        this.mouseY = 0;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        document.addEventListener('mousedown', (event) => {
            this.mouseDown = true;
            this.mouseButton = event.button;
            this.mouseX = event.clientX;
            this.mouseY = event.clientY;
            event.preventDefault();
        });

        document.addEventListener('mouseup', () => {
            this.mouseDown = false;
        });

        document.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });

        document.addEventListener('mousemove', (event) => {
            if (!this.mouseDown) return;
            this.handleMouseMove(event);
        });

        document.addEventListener('wheel', (event) => {
            this.handleWheel(event);
        });
    }
    
    handleMouseMove(event) {
        const deltaX = event.clientX - this.mouseX;
        const deltaY = event.clientY - this.mouseY;

        if (this.mouseButton === 0) { // Left mouse - rotate
            this.cameraTheta -= deltaX * 0.01;
            this.cameraPhi += deltaY * 0.01;

            // Clamp phi
            this.cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraPhi));

            this.updateCameraPosition();
        } else if (this.mouseButton === 2) { // Right mouse - pan
            this.handlePan(deltaX, deltaY);
        }

        this.mouseX = event.clientX;
        this.mouseY = event.clientY;
    }
    
    handlePan(deltaX, deltaY) {
        // Get camera's right and up vectors
        const cameraRight = new THREE.Vector3();
        const cameraUp = new THREE.Vector3();
        this.camera.getWorldDirection(new THREE.Vector3()); // Update camera matrix
        cameraRight.setFromMatrixColumn(this.camera.matrixWorld, 0);
        cameraUp.setFromMatrixColumn(this.camera.matrixWorld, 1);

        // Pan speed proportional to distance
        const panSpeed = this.cameraDistance * 0.001;
        
        // Move lookAt target and camera position
        const panOffset = new THREE.Vector3();
        panOffset.addScaledVector(cameraRight, -deltaX * panSpeed);
        panOffset.addScaledVector(cameraUp, deltaY * panSpeed);

        this.lookAtTarget.add(panOffset);
        this.camera.position.add(panOffset);
        this.camera.lookAt(this.lookAtTarget);
    }
    
    handleWheel(event) {
        this.cameraDistance *= (1 + event.deltaY * 0.001);
        this.cameraDistance = Math.max(1, Math.min(100, this.cameraDistance));
        this.updateCameraPosition();
    }
    
    updateCameraPosition() {
        this.camera.position.x = this.lookAtTarget.x + this.cameraDistance * Math.sin(this.cameraPhi) * Math.sin(this.cameraTheta);
        this.camera.position.y = this.lookAtTarget.y + this.cameraDistance * Math.cos(this.cameraPhi);
        this.camera.position.z = this.lookAtTarget.z + this.cameraDistance * Math.sin(this.cameraPhi) * Math.cos(this.cameraTheta);
        
        this.camera.lookAt(this.lookAtTarget);
    }
}