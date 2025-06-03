#ifndef SPUTTERER_PLUME_INPUTS_H
#define SPUTTERER_PLUME_INPUTS_H

#include <glm/gtc/type_ptr.hpp>

struct PlumeInputs {
    // origin of thruster in space
    glm::vec3 origin{0.0};
    // direction along which plume is pointing
    glm::vec3 direction{0.0, 0.0, 1.0};

    // Model parameters
    std::array<double, 7> model_params{
        1.0,  // ratio of the main beam to the total beam
        0.25, // ratio for divergence angle of the main to scattered beam
        0.0,  // "slope" for linear divergence angle function
        0.0,  // "intercept" for linear divergence angle function
        0.0,  // "slope" for linear neutral density function
        0.0,  // "intercept" for linear neutral density function
        55.0, // charge exchange collision cross section (square Angstroms)
    };

    // Design parameters
    double background_pressure_Torr{0.0}; // normalized background pressure
    double beam_current_A{5.0};

    // Beam energy
    double beam_energy_eV{300.0};
    double scattered_energy_eV{250.0};
    double cex_energy_eV{50.0};

    // diagnostics
    double probe_distance_m{1.0};     // distance between probe and plume origin
    float resample_interval_s{10e-3}; // Frequency at which the plume is resampled
    bool probe{false};                // whether to run a simulated faraday probe sweep
};

#endif // SPUTTERER_PLUME_INPUTS_H
