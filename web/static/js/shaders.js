// Vertex shader for template meshes
const templateVertexShader = `
    attribute float distance;
    varying vec2 vUv;
    varying float vDistance;
    void main() {
        vUv = uv;
        vDistance = distance;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// Fragment shader for template meshes
const templateFragmentShader = `
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
`;