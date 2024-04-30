#pragma once
#ifndef SPUTTERER_INPUT_HPP
#define SPUTTERER_INPUT_HPP

#include <string>
#include <vector>
#include <array>

#include "Surface.hpp"
#include "Window.hpp"
#include "vec3.hpp"

using std::string, std::vector;


class Input {
public:
  Input () = default;

  explicit Input (string filename)
    : filename{std::move(filename)} {}

  string filename{"input.toml"};
  string current_path{"."};

  // user-specified geometry
  vector<Surface> surfaces;

  // simulation variables
  float timestep{0.0};
  float max_time{0.0};
  float output_interval{0.0};

  // chamber geometry
  float chamber_radius{-1.0};
  float chamber_length{-1.0};

  // plume model inputs
  vec3 plume_origin{};
  vec3 plume_direction{};
  double background_pressure_Torr{};
  double divergence_angle_deg{};
  double ion_current_A{};
  std::array<double, 7> plume_model_params{};

  // initial particles (if any)
  std::vector<float> particle_w;
  std::vector<float> particle_x;
  std::vector<float> particle_y;
  std::vector<float> particle_z;
  std::vector<float> particle_vx;
  std::vector<float> particle_vy;
  std::vector<float> particle_vz;

  void read ();
};

#endif