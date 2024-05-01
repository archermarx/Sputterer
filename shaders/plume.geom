#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 5) out;

#define PI 3.1415926538

uniform mat4 camera;

float r1 = 0.1;
float r2 = 1.0;

void build_cone(vec4 origin, int resolution, float offset) {
    for (float i = 0.0; i < resolution; i++) {
        float angle = 2 * PI * i / resolution;
        float s = sin(angle);
        float c = cos(angle);
        gl_Position = camera * (origin + vec4(r1 * c, r1 * s, 0.0, 0.0));
        EmitVertex();
        gl_Position = camera * (origin + vec4(r2 * c, r2 * s, offset, 0.0));
        EmitVertex();
    }
    EndPrimitive();
}

void main() {
    build_cone(gl_in[0].gl_Position, 4, 1.0);
}
