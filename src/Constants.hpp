//
// Created by marksta on 5/2/24.
//
#pragma once
#ifndef SPUTTERER_CONSTANTS_HPP
#define SPUTTERER_CONSTANTS_HPP

struct Species {
  double mass;
  double atomic_number;
};

namespace constants {
  constexpr double eps_0 = 8.8541878128e-12;    // Vacuum permittivity [C^2 / eV / Å]
  constexpr double q_e = 1.602176634e-19;       // Fundamental charge [C]
  constexpr double m_e = 9.1093837015e-31;      // Electron mass [kg]
  constexpr double m_u = 1.66053906660e-27;     // One Dalton / atomic mass unit [kg]
  constexpr double k_b = 1.380649e-23;          // Boltzmann constant [J/K]
  constexpr double a_0 = 0.529177210903e-10;    // Bohr radius [m]
  constexpr double pi = 3.14159265359;          // pi

  constexpr Species xenon{.mass = 131.293, .atomic_number = 54};
  constexpr Species carbon{.mass = 12.011, .atomic_number = 6};
}

#endif //SPUTTERER_CONSTANTS_HPP
