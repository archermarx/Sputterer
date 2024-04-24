#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
layout (location=2) in vec3 aTranslate;

uniform mat4 scale;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * (scale  * vec4(aPos, 1.0) + vec4(aTranslate, 0.0));
}