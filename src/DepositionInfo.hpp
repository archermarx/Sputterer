#pragma once
#ifndef SPUTTERER_DEPOSITION_RATE_HPP
#define SPUTTERER_DEPOSITION_RATE_HPP

#include <string>
#include <vector>

struct DepositionInfo {
    std::string filename;
    std::ofstream file;
    size_t num_tris = 0;
    std::vector<std::string> surf_names;
    std::vector<size_t> local_indices;
    std::vector<size_t> global_indices;
    std::vector<double> areas;
    std::vector<int> particles_collected;
    std::vector<double> deposition_rates;
    std::vector<double> mass_fluxes;

    DepositionInfo(std::string filename) : filename(filename) {
        file.open(filename);
        // Write header
        file << "Time (s),"
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
};
     //void append() {
     //   output_file.open(output_filename, std::ios_base::app);

     //   for (int i = 0; i < collect_inds_global.size(); i++) {
     //       auto triangle_id_global = collect_inds_global[i];

     //       double mass_carbon = collected[i]*input.particle_weight*carbon.mass*m_u;
     //       double volume_carbon = mass_carbon/graphite_density;
     //       double triangle_area = h_triangles[triangle_id_global].area;
     //       double layer_thickness_um = volume_carbon/triangle_area*1e6;
     //       double physical_time_kh = physical_time/3600/1000;
     //       double deposition_rate = layer_thickness_um/physical_time_kh;
     //       double flux = mass_carbon / triangle_area / physical_time;
     //       deposition_rates[i] = deposition_rate;
     //       surface_fluxes[i] = flux;

     //       output_file << physical_time << ",";
     //       output_file << surface_names.at(h_material_ids[triangle_id_global]) << ",";
     //       output_file << collect_inds_local.at(i) << ",";
     //       output_file << triangle_id_global << ",";
     //       output_file << collected[i] << ",";
     //       output_file << mass_carbon << ",";
     //       output_file << deposition_rates[i] << "\n";
     //       output_file << surface_fluxes[i] << "\n";
     //   }
     //   output_file.close();
     //}
#endif // SPUTTERER_DEPOSITION_RATE_HPP
