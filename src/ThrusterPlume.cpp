//
// Created by marksta on 4/30/24.
//

#include <complex>

#include "gl_helpers.hpp"

// header for erfi
#include "../include/Faddeeva.hpp"

#include "Constants.hpp"
#include "ThrusterPlume.hpp"


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
  using namespace constants;

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

double ThrusterPlume::scattered_divergence_angle () const {
  return (this->main_divergence_angle())/(this->model_params[1]);
}

double ThrusterPlume::main_divergence_angle () const {
  return this->model_params[2]*this->background_pressure + this->model_params[3];
}

double ThrusterPlume::current_density (const vec3 position) const {

  using namespace constants;

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
  auto div_angle_scattered = this->scattered_divergence_angle();
  auto div_angle_main = this->main_divergence_angle();

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

double sputtering_yield (double energy, double angle, double incident_mass, double target_mass, double incident_z
                         , double target_z) {

  using namespace constants;

  // Model fitting parameters
  constexpr auto threshold_energy = 10.92;
  constexpr auto q = 2.18;
  constexpr auto lambda = 4.05;
  constexpr auto mu = 1.97;
  constexpr auto f = 2.29;
  constexpr auto a = 0.44;
  constexpr auto b = 0.71;

  // no sputtering if energy below the threshold energy
  if (energy < threshold_energy) {
    return 0.0;
  }

  // model computation
  constexpr auto twothirds = 2.0/3.0;
  constexpr auto arg1 = 9*pi*pi/128;
  constexpr auto arg2 = a_0*4*pi*eps_0/q_e;
  auto a1 = cbrt(arg1)*arg2;
  auto a2 = sqrt(pow(incident_z, twothirds) + pow(incident_z, twothirds));
  auto eps_l = a1*target_mass*(target_z*incident_z*(target_mass + incident_mass))*a2*energy;

  // Nuclear stopping power for KrC potential
  auto w = eps_l + 0.1728*sqrt(eps_l) + 0.008*pow(eps_l, 0.1504);
  auto inv_w = 1.0/w;
  auto s_n = 0.5*log(1.0 + 1.2288*eps_l)*inv_w;

  // sputtering yield
  auto term_1 = pow(energy/threshold_energy - 1, mu);
  auto term_2 = 1.0/cos(pow(angle, a));
  auto numerator = q*s_n*term_1*pow(term_2, f)*exp(b*(1 - term_2));
  auto denominator = lambda*inv_w + term_1;
  return numerator/denominator;
}

void ThrusterPlume::set_buffers () {
  glGenBuffers(1, &vbo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  float points[] = {this->location.x, this->location.y, this->location.z};
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(points), 0);
  glBindVertexArray(0);
}

void ThrusterPlume::draw () {
  glBindVertexArray(vao);
  glDrawArrays(GL_POINTS, 0, 1);
}


