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
