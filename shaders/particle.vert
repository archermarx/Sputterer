#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
layout (location=2) in vec3 aTranslate;

uniform vec3 scale;
uniform mat4 camera;

void main() {
    gl_Position = camera * vec4(scale * aPos + aTranslate, 1.0f);
}