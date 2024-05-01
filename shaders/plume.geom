#version 330 core

#define MAX_VERTICES 102
#define PI 3.1415926538

layout (points) in;
layout (triangle_strip, max_vertices = MAX_VERTICES) out;

vec3 world_up = vec3(0.0, 1.0, 0.0);

uniform mat4 camera;
uniform vec3 direction;
uniform float length;
uniform float angle;

void build_cone(vec4 position) {
    int resolution = (MAX_VERTICES - 2) / 2;
    float r = length * tan(angle);

    vec3 right = cross(direction, world_up);
    vec3 up = cross(right, direction);

    for (int i = 0; i <= resolution; i++) {

        int ind = i % resolution;
        float angle = 2 * PI * ind / resolution;
        float s = sin(angle);
        float c = cos(angle);
        vec3 v = r * (s * up + c * right) + length * direction;

        gl_Position = camera * position;
        EmitVertex();

        gl_Position = camera * (position + vec4(v, 0.0));
        EmitVertex();
    }
}


void main() {
    build_cone(gl_in[0].gl_Position);
}
