#version 330 core

layout(points) in;
layout(line_strip, max_vertices = 24) out;

uniform vec3 extent;
uniform mat4 camera;

void build_cube(vec4 center) {
    // four x-aligned lines
    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    // four y-aligned lines
    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    // four z-aligned lines
    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(-extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(-extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, -extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();

    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, -extent.z/2, 0));
    EmitVertex();
    gl_Position = camera * (center + vec4(extent.x/2, extent.y/2, extent.z/2, 0));
    EmitVertex();

    EndPrimitive();
}

void main() {
    build_cube(gl_in[0].gl_Position);
}
