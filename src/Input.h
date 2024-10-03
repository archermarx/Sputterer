#ifndef SPUTTERER_INPUT_H
#define SPUTTERER_INPUT_H

#include <string>
#include <array>

#include "Surface.h"
#include "Window.h"
#include "ThrusterPlume.h"
#include "vec3.h"

struct Input {
    // simulation variables
    double timestep_s = 0.0;
    double max_time_s = 0.0;
    int output_interval = -1;
    int verbosity = 0;
    bool display = false;

    // User-specified geometry
    SceneGeometry geometry;
    
    // Chamber geometry
    double chamber_radius_m = -1.0;
    double chamber_length_m = -1.0;

    // Plume model inputs
    ThrusterPlume plume{};

    // particle weight
    double particle_weight{1.0f};
};

Input read_input(std::string filename);

#endif
