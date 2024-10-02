// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1

#include <toml++/toml.hpp>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "Input.hpp"

namespace fs = std::filesystem;

template<typename T>
void set_value (toml::table &table, const std::string &input_name, T &value) {
    auto node = table[input_name];
    bool valid = true;

    if constexpr (std::is_same_v<T, string>) {
        if (node.is_string()) {
            value = node.as_string()->get();
        } else {
            valid = false;
        }
    } else if constexpr (std::is_same_v<T, glm::vec3>) {
        if (node.is_table()) {
            auto tab = node.as_table();
            auto x = get_value<float>(*tab, "x");
            auto y = get_value<float>(*tab, "y");
            auto z = get_value<float>(*tab, "z");
            value = glm::vec3(x, y, z);
        } else {
            valid = false;
        }
    } else {
        if (node.is_integer()) {
            value = static_cast<T>(node.as_integer()->get());
        } else if (node.is_boolean()) {
            value = static_cast<T>(node.as_boolean()->get());
        } else if (node.is_floating_point()) {
            value = static_cast<T>(node.as_floating_point()->get());
        } else if (node.is_string()) {
            string str = node.as_string()->get();
            std::istringstream ss(str);
            ss >> value;
        } else {
            valid = false;
        }
    }
    if (!valid) {
        std::cerr << "Invalid input for option " << input_name
                  << ".\n Expected value of type " << typeid(T).name() << "\n.";
    }
}

template<typename T>
T get_value (toml::table &table, const std::string &key) {
    T value;
    set_value(table, key, value);
    return value;
}

template<typename T>
void query_value (toml::table &table, const std::string &key, T &value) {
    if (table.contains(key)) {
        value = get_value<T>(table, key);
    }
}

toml::table get_table(toml::table &parent, const std::string &key){
    if (parent.contains(key)) {
        return *parent.get_as<toml::table>(key);
    } else {
        std::ostringstream msg;
        msg << "TOML parsing error:\n"
            << "Key " << key << " not found in table" << std::endl;
        throw std::runtime_error(msg.str());
    }
}

Input read_input (std::string filename) {
    // Open file
    auto input_table = toml::parse_file(filename);

    // Read simulation parameters
    Input input{};
    auto sim = get_table(input_table, "simulation");
    query_value(sim, "verbosity", input.verbosity);
    query_value(sim, "display", input.display);
    query_value(sim, "output_interval", input.output_interval);
    if (input.output_interval <= 0) {
        input.output_interval = -1;
    }
    set_value(sim, "timestep_s", input.timestep_s);
    set_value(sim, "max_time_s", input.max_time_s);
    set_value(sim, "particle_weight", input.particle_weight);

    // Read chamber features
    auto chamber = get_table(input_table, "chamber");
    set_value(chamber, "radius_m", input.chamber_radius_m);
    set_value(chamber, "length_m", input.chamber_length_m);

    // Read plume parameters
    auto plume = get_table(input_table, "plume");
    set_value(plume, "origin", input.plume.origin);
    set_value(plume, "direction", input.plume.direction);
    input.plume.direction = glm::normalize(input.plume.direction);
    set_value(plume, "background_pressure_Torr", input.plume.background_pressure_Torr);
    set_value(plume, "ion_current_A", input.plume.beam_current_A);
    set_value(plume, "beam_energy_eV", input.plume.beam_energy_eV);
    set_value(plume, "scattered_energy_eV", input.plume.scattered_energy_eV);
    set_value(plume, "cex_energy_eV", input.plume.cex_energy_eV);
    auto plume_params_arr = plume.get_as<toml::array>("model_parameters");
    auto ind = 0;
    for (auto &&plume_param: *plume_params_arr) {
        input.plume.model_params[ind] = static_cast<double>(plume_param.as_floating_point()->get());;
        ind++;
    }

    // Read materials
    std::unordered_map<string, Material> materials;
    std::unordered_map<string, vec3> material_colors;
    auto input_materials = *input_table.get_as<toml::array>("material");

    for (auto &&material_node: input_materials) {
        auto mat = *material_node.as_table();

        // Populate material
        Material material;
        auto material_name = get_value<string>(mat, "name");
        auto material_color = get_value<vec3>(mat, "color");
        set_value(mat, "sticking_coeff", material.sticking_coeff);
        set_value(mat, "diffuse_coeff", material.diffuse_coeff);
        set_value(mat, "temperature_K", material.temperature_K);

        // Add material to list
        materials.insert(std::make_pair(material_name, material));
        material_colors.insert(std::make_pair(material_name, material_color));
    }

    // Read surfaces
    auto geometry = *input_table.get_as<toml::array>("geometry");
    auto num_surfaces = geometry.size();
    input.geometry.surfaces.resize(num_surfaces);

    int id = 0;
    for (auto &&elem: geometry) {
        auto tab = *elem.as_table();
        auto &surf = input.geometry.surfaces.at(id);

        // get material
        auto mat_name = get_value<string>(tab, "material");
        if (materials.find(mat_name) != materials.end()) {
            surf.material = materials.at(mat_name);
            surf.color = material_colors.at(mat_name);
        } else {
            std::cerr << "Material \"" << mat_name << "\" not found in input file!" << std::endl;
        }

        auto &material = surf.material;
        query_value(tab, "name", surf.name);
        query_value(tab, "collect", material.collect);
        query_value(tab, "sputter", material.sputter);

        // need to append the current working directory to make sure mesh files
        // are relative to where the input file is located
        auto mesh_path = get_value<string>(tab, "model");
        //auto mesh_path = fs::absolute({filename}).parent_path()/mesh_file;

        // object positions (optional)
        query_value(tab, "translate", surf.transform.translate);
        query_value(tab, "scale", surf.transform.scale);

        if (tab.contains("rotate")) {
            auto rot_tab = get_table(tab, "rotate");
            query_value(rot_tab, "angle", surf.transform.rotation_angle);
            query_value(rot_tab, "axis", surf.transform.rotation_axis);
        }

        query_value(tab, "color", surf.color);
        query_value(tab, "temperature_K", surf.material.temperature_K);

        auto &mesh = surf.mesh;
        mesh.load({mesh_path});
        id++;
    }

    if (input.verbosity > 0) {
        std::cout << "Input read." << std::endl;
    }
    return input;
}
