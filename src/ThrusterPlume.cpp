//
// Created by marksta on 4/30/24.
//

#include <numbers>
#include <complex>

// header for erfi
#include "../include/Faddeeva.hpp"

#include "ThrusterPlume.hpp"

using std::numbers::pi;

// Convert a
vec2 ThrusterPlume::convert_to_thruster_coords (const vec3 position) const {

  // vector from plume origin to position
  auto offset = position - this->location;
  auto radius = glm::length(offset);

  // angle between this vector and plume direction
  auto dot_prod = glm::dot(offset, this->direction);
  auto cos_angle = dot_prod/(radius*glm::length(this->direction));
  auto angle = acos(cos_angle);

  return {radius, angle};
}

double current_density_scale (const double angle, const double div_angle, const double frac) {
  using namespace std::complex_literals;
  using namespace Faddeeva;

  const auto pi_threehalves = sqrt(pi*pi*pi);
  auto erfi_arg1 = 0.5*div_angle;
  auto alpha_squared = div_angle*div_angle;
  auto erfi_arg2 = 0.5*(1i - alpha_squared)/div_angle;
  auto erfi_arg3 = 0.5*(1i + alpha_squared)/div_angle;
  auto erfi_1 = erfi(erfi_arg1);
  auto erfi_2 = erfi(erfi_arg2);
  auto erfi_3 = erfi(erfi_arg3);
  auto denom = pi_threehalves*div_angle*exp(-(erfi_arg1*erfi_arg1))*(2*erfi_1 + erfi_2 - erfi_3);

  auto angle_frac = angle/div_angle;
  auto numer = 2*frac*exp(-(angle_frac*angle_frac));

  return numer/denom.real();
}

double ThrusterPlume::current_density (const vec3 position) const {


  const auto coords = convert_to_thruster_coords(position);
  auto radius = coords.x;
  auto angle = coords.y;

  if (angle < 0) {
    // point behind thruster - no ion current
    return 0.0;
  }

  auto radius_squared = radius*radius;

  // divergence angles
  const auto [t0, t1, t2, t3, t4, t5, sigma_cex] = this->model_params;
  auto div_angle_scattered = t2*this->background_pressure + t3;
  auto div_angle_main = t1*div_angle_scattered;

  // neutral density
  auto neutral_density = t4*this->background_pressure + t5;

  // get local currents
  auto exp_factor = exp(-radius*neutral_density*sigma_cex*1e-20);
  auto local_cex_current = this->beam_current*(1.0 - exp_factor);
  auto local_beam_current = this->beam_current*exp_factor;

  // compute cex current density
  auto cex_current_density = local_cex_current/(2*pi*radius_squared);

  // man and scattered beam current densities
  auto beam_current_factor = local_beam_current/radius_squared;
  auto main_current_density = beam_current_factor*current_density_scale(angle, div_angle_main, 1.0 - t0);
  auto scattered_current_density = beam_current_factor*current_density_scale(angle, div_angle_scattered, t0);

  // total beam current density is sum of main, scattered, and cex beams
  return main_current_density + scattered_current_density + cex_current_density;
}