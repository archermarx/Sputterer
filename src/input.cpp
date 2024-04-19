// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include <toml++/toml.hpp>

#include "input.hpp"
#include <iostream>

template <typename T>
T readTableEntryAs(toml::table &table, std::string inputName) {
    auto node = table[inputName];
    bool valid = true;
    T value{};

    if constexpr (std::is_same_v<T, string>) {
        if (node.is_string()) {
            value = node.as_string() -> get();
        } else {
            valid = false;
        }
    } else {
        if (node.is_integer()) {
            value = static_cast<T>(node.as_integer() -> get());
        } else if (node.is_boolean()) {
            value = static_cast<T>(node.as_boolean() -> get());
        } else if (node.is_floating_point()) {
            value = static_cast<T>(node.as_floating_point() -> get());
        } else if (node.is_string()) {
            string str = node.as_string() -> get();
            std::istringstream ss(str);
            ss >> value;
        } else {
            valid = false;
        }
    }
    if (!valid) {
        std::cout << "Invalid input for option " << inputName << ".\n Expected value of type " << typeid(T).name() << "\n.";
    }

    return value;
}

void Input::read() {

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }

    // Read chamber features
    auto chamber = *input.get_as<toml::table>("chamber");
    chamberRadius = readTableEntryAs<float>(chamber, "radius");
    chamberLength = readTableEntryAs<float>(chamber, "length");

    // Read surface geometry
    auto geometry = *input.get_as<toml::array>("geometry");

    for (auto&& elem : geometry) {
        auto tab = elem.as_table();
        std::string name  = readTableEntryAs<std::string>(*tab, "name");
        std::string file  = readTableEntryAs<std::string>(*tab, "file");
        bool emit    = readTableEntryAs<bool>(*tab, "emit");
        bool collect = readTableEntryAs<bool>(*tab, "collect");
        surfaces.emplace_back(name, file, emit, collect);
    }

    // Read particles
    auto particles = *input.get_as<toml::array>("particle");

    for (auto&& particle: particles) {
        auto particle_tab = particle.as_table();

        auto pos = particle_tab->get_as<toml::table>("position");
        particle_x.push_back(readTableEntryAs<float>(*pos, "x"));
        particle_y.push_back(readTableEntryAs<float>(*pos, "y"));
        particle_z.push_back(readTableEntryAs<float>(*pos, "z"));

        auto vel = particle_tab->get_as<toml::table>("velocity");
        particle_x.push_back(readTableEntryAs<float>(*vel, "x"));
        particle_y.push_back(readTableEntryAs<float>(*vel, "y"));
        particle_z.push_back(readTableEntryAs<float>(*vel, "z"));

        auto weight = readTableEntryAs<float>(*particle_tab, "weight");
        particle_w.push_back(weight);
    }

    for (auto &surface: surfaces) {
        // enable meshes
        surface.enable();
    }
}
