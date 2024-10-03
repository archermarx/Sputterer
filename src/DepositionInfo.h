#pragma once
#ifndef SPUTTERER_DEPOSITION_RATE_HPP
#define SPUTTERER_DEPOSITION_RATE_HPP

#include <string>
#include <vector>

#include "Constants.h"
#include "Input.h"
#include "Timer.h"

struct DepositionInfo {
    std::string filename;
    std::ofstream file;
    size_t num_tris = 0;
    double particle_weight;
    std::vector<std::string> surface_names;
    std::vector<size_t> local_indices;
    std::vector<size_t> global_indices;
    std::vector<double> triangle_areas;
    std::vector<int> particles_collected;
    std::vector<double> deposition_rates;
    std::vector<double> carbon_masses;
    std::vector<double> mass_fluxes;

    DepositionInfo (std::string filename) : filename(filename) {
        file.open(filename);
        // Write header
        file << "Step,"
            << "Time (s),"
            << "Surface name,"
            << "Local triangle ID,"
            << "Global triangle ID,"
            << "Macroparticles collected,"
            << "Mass collected (kg),"
            << "Deposition rate (um/khr),"
            << "Surface mass flux (kg/m^2/s)"
            << "\n";
        file.close();
    }

    void init_diagnostics () {
        particles_collected.resize(num_tris, 0);
        deposition_rates.resize(num_tris, 0.0);
        carbon_masses.resize(num_tris, 0.0);
        mass_fluxes.resize(num_tris, 0.0);
    }

    void update_diagnostics (Input &input, double time) {
        using namespace constants;
        for (size_t id = 0; id < num_tris; id++) {
            // Compute deposition rate and carbon flux
            carbon_masses[id] = particles_collected[id] * input.particle_weight * carbon.mass * m_u;
            double carbon_volume = carbon_masses[id]/ graphite_density;
            double layer_thickness_um = carbon_volume / triangle_areas[id] * 1e6;
            double physical_time_kh = time / 3600 / 1000;
            deposition_rates[id] = layer_thickness_um / physical_time_kh;
            mass_fluxes[id] = carbon_masses[id] / triangle_areas[id] / time;
        }
    }

    void write_to_file (size_t step, double time) {
       file.open(filename, std::ios_base::app);
       for (int id = 0; id < num_tris; id++) {
           file << step << ",";
           file << time << ",";
           file << surface_names[id] << ",";
           file << local_indices[id] << ",";
           file << global_indices[id] << ",";
           file << particles_collected[id] << ",";
           file << carbon_masses[id] << ",";
           file << deposition_rates[id] << ",";
           file << mass_fluxes[id] << "\n";
       }
       file.close();
    }
};
#endif // SPUTTERER_DEPOSITION_RATE_HPP
