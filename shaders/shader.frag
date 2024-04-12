#version 330 core
out vec4 fragColor;

in vec3 Normal;

void main() {
    float len = length(Normal);
    float red   = abs(Normal.x) / len;
    float green = abs(Normal.y) / len;
    float blue  = abs(Normal.z) / len;
    fragColor = vec4(red, green, blue, 1.0);
}