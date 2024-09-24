#version 330 core
out vec4 fragColor;
in vec3 fragPos;

uniform vec3 objectColor;

float particleSize = 0.1;

// particles are rendered as billboarded squares clipped to look like circles
void main() {
    if (length(fragPos) < particleSize) {
        fragColor = vec4(objectColor, 1.0);
    } else {
        discard;
    }
}