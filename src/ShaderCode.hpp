#ifndef SPUTTERER_SHADERCODE_HPP
#define SPUTTERER_SHADERCODE_HPP

namespace shaders {

// GLSL shader code used in other places
struct ShaderCode {
    const char *vert;
    const char *frag;
    const char *geom;
};

constexpr ShaderCode plume{
    .vert = R"""(
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos, 1.0);
}
    )""",
    .frag = R"""(
#version 330 core

float plume_alpha = 0.1;
uniform bool main_beam = true;

vec3 main_beam_color = vec3(150.0, 229.0, 255.0)/255.0;
vec3 scattered_beam_color = vec3(0.8 * main_beam_color.rg, main_beam_color.b);

void main() {
    vec3 color;
    if (main_beam) {
        color = main_beam_color;
    } else {
        color = scattered_beam_color;
    }
    gl_FragColor = vec4(color, plume_alpha);
}
    )""",
    .geom = R"""(
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

    EndPrimitive();
}


void main() {
    build_cone(gl_in[0].gl_Position);
}
    )""",
};

constexpr ShaderCode particle = {
    .vert = R"""(
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
    )""",
    .frag = R"""(
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
    )""",
    .geom = nullptr,
};

constexpr ShaderCode bvh {
    .vert = R"""(
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos, 1.0);
}
    )""",
    .frag = R"""(
#version 330 core

void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
    )""",
    .geom = R"""(
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
    )""",
};

constexpr ShaderCode mesh {
    .vert = R"""(
#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 normal;
out vec3 fragPos;

void main() {
    gl_Position = projection * view * model * vec4(aPos,1.0);
    fragPos = aPos;
    normal = aNormal;
}
    )""",
    .frag = R"""(
#version 330 core
out vec4 fragColor;

in vec3 normal;
in vec3 fragPos;

const float ambientStrength = 0.1;
const float specularStrength = 0.1;
const int shininess = 8;
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const vec3 lightPos = vec3(10.0, 40.0, 8.0);
const float lightPower = length(lightPos) * length(lightPos);
const float screenGamma = 2.2;

uniform vec3 viewPos;
uniform vec3 objectColor;

void main() {
    vec3 norm = normalize(normal);
    vec3 lightDir = lightPos - fragPos;
    float distance = length(lightDir) * length(lightDir);
    lightDir = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - fragPos);

    vec3 ambient = ambientStrength * lightColor;

    float lambertian = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = lambertian * lightColor * lightPower / distance;

    float spec = 0.0;

    if(lambertian > 0){
        vec3 reflectDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(reflectDir, viewDir), 0.0);
        spec = pow(specAngle, shininess);
    }

    vec3 specular = specularStrength * spec * lightColor * lightPower / distance;

    vec3 resultColor = (ambient + diffuse + specular) * objectColor;
    vec3 resultColorCorrected = pow(resultColor, vec3(1.0 / screenGamma));
    fragColor = vec4(resultColorCorrected, 1.0);
}
    )""",
    .geom = nullptr,
};


}



#endif // SPUTTERER_SHADERCODE_HPP
