#pragma once
#ifndef SPUTTERER_INPUT_HPP
#define SPUTTERER_INPUT_HPP

#include <string>
#include <vector>
#include <array>

#include "Surface.hpp"
#include "Window.hpp"
#include "ThrusterPlume.hpp"
#include "vec3.hpp"

struct Input {
    // simulation variables
    double timestep_s = 0.0;
    double max_time_s = 0.0;
    double output_interval_s = 0.0;
    int verbosity = 0;
    bool display = false;

    // User-specified geometry
    vector<Surface> surfaces;
    
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
