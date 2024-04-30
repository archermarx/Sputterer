#pragma once
#ifndef SPUTTERER_INPUT_HPP
#define SPUTTERER_INPUT_HPP

#include <string>
#include <vector>

#include "surface.hpp"
#include "window.hpp"

using std::string, std::vector;

class input {
public:
  input () = default;

  explicit input (string filename)
          : filename{std::move(filename)} {}

  string filename{"input.toml"};
  string current_path{"."};

  vector<surface> surfaces;

  float timestep{0.0};
  float max_time{0.0};
  float output_interval{0.0};
  float chamber_radius{-1.0};
  float chamber_length{-1.0};

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