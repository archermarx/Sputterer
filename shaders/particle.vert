#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
layout (location=2) in vec3 aTranslate;

uniform vec3 scale;
uniform mat4 camera;
uniform vec3 cameraRight;
uniform vec3 cameraUp;
uniform float pop;

out vec3 fragPos;

// particles are rendered as billboarded squares clipped to look like circles
void main() {
    fragPos = aPos;
    // billboarding
    vec3 pos_worldspace = cameraRight * aPos.x + cameraUp * aPos.y;
    gl_Position = camera * vec4(scale * pos_worldspace + aTranslate, 1.0f);
}