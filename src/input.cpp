// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include <toml++/toml.hpp>
#include <iostream>
#include <filesystem>

#include "input.hpp"

namespace fs = std::filesystem;

template <typename T>
T readTableEntryAs (toml::table &table, const std::string &inputName) {
    auto node  = table[inputName];
    bool valid = true;
    T    value{};

    if constexpr (std::is_same_v<T, string>) {
        if (node.is_string()) {
            value = node.as_string()->get();
        } else {
            valid = false;
        }
    } else if constexpr (std::is_same_v<T, glm::vec3>) {
        if (node.is_table()) {
            auto tab = node.as_table();
            auto x   = readTableEntryAs<float>(*tab, "x");
            auto y   = readTableEntryAs<float>(*tab, "y");
            auto z   = readTableEntryAs<float>(*tab, "z");
            value    = glm::vec3(x, y, z);
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
            string             str = node.as_string()->get();
            std::istringstream ss(str);
            ss >> value;
        } else {
            valid = false;
        }
    }
    if (!valid) {
        std::cout << "Invalid input for option " << inputName << ".\n Expected value of type " << typeid(T).name()
                  << "\n.";
    }

    return value;
}

toml::table getTable (toml::table input, const std::string &name) {
    if (input.contains(name)) {
        return *input.get_as<toml::table>(name);
    } else {
        std::ostringstream msg;
        msg << "TOML parsing error:\n"
            << "Key " << name << " not found in table" << std::endl;
        throw std::runtime_error(msg.str());
    }
}

void Input::read() {

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error &err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }

    auto sim = getTable(input, "simulation");
    timestep = readTableEntryAs<float>(sim, "timestep_s");

    // Read chamber features
    auto chamber  = getTable(input, "chamber");
    chamberRadius = readTableEntryAs<float>(chamber, "radius_m");
    chamberLength = readTableEntryAs<float>(chamber, "length_m");

    // Read surface geometry
    auto geometry = *input.get_as<toml::array>("geometry");

    auto numSurfaces = geometry.size();
    surfaces.resize(numSurfaces);

    int id = 0;
    for (auto &&elem : geometry) {
        auto  tab  = elem.as_table();
        auto &surf = surfaces.at(id);

        auto &emitter  = surf.emitter;
        auto &material = surf.material;

        surf.name        = readTableEntryAs<string>(*tab, "name");
        emitter.emit     = readTableEntryAs<bool>(*tab, "emit");
        material.collect = readTableEntryAs<bool>(*tab, "collect");

        // need to append the current working directory to make sure mesh files are relative to where
        // the input file was run
        auto meshFile = readTableEntryAs<string>(*tab, "file");
        auto path     = fs::absolute({this->filename});
        std::cout << path << ", " << path.parent_path() << std::endl;

        auto meshPath = path.parent_path();
        meshPath /= meshFile;
        std::cout << meshPath << std::endl;

        // Read emitter options
        if (emitter.emit && tab->contains("emitter")) {
            auto emit_tab    = tab->get_as<toml::table>("emitter");
            emitter.flux     = readTableEntryAs<float>(*emit_tab, "flux");
            emitter.velocity = readTableEntryAs<float>(*emit_tab, "velocity");
            if (emit_tab->contains("reverse_direction")) {
                emitter.reverse = readTableEntryAs<bool>(*emit_tab, "reverse_direction");
            }
            if (emit_tab->contains("spread")) {
                emitter.spread = readTableEntryAs<float>(*emit_tab, "spread");
            }
        }

        // Read material options
        if (tab->contains("material")) {
            auto mat_tab            = tab->get_as<toml::table>("material");
            material.sticking_coeff = readTableEntryAs<float>(*mat_tab, "sticking_coeff");
        }

        // object transformations (optional)
        auto &transform = surf.transform;

        if (tab->contains("translate")) {
            transform.translate = readTableEntryAs<glm::vec3>(*tab, "translate");
        }

        if (tab->contains("scale")) {
            transform.scale = readTableEntryAs<glm::vec3>(*tab, "scale");
        }

        if (tab->contains("rotate")) {
            auto rot_tab = tab->get_as<toml::table>("rotate");
            if (rot_tab->contains("angle")) {
                transform.rotationAngle = readTableEntryAs<float>(*rot_tab, "angle");
            }

            if (rot_tab->contains("axis")) {
                transform.rotationAxis = readTableEntryAs<glm::vec3>(*rot_tab, "axis");
            }
        }

        // Color
        if (tab->contains("color")) {
            surf.color = readTableEntryAs<glm::vec3>(*tab, "color");
        }

        // Read mesh data
        auto &mesh = surf.mesh;
        mesh.readFromObj({meshPath});
        mesh.setBuffers();

        id++;
    }

    // Read particles (optional)

    if (input.contains("particle")) {
        auto particles = *input.get_as<toml::array>("particle");

        for (auto &&particle : particles) {
            auto particle_tab = particle.as_table();

            auto pos = particle_tab->get_as<toml::table>("position");
            particle_x.push_back(readTableEntryAs<float>(*pos, "x"));
            particle_y.push_back(readTableEntryAs<float>(*pos, "y"));
            particle_z.push_back(readTableEntryAs<float>(*pos, "z"));

            auto vel = particle_tab->get_as<toml::table>("velocity");
            particle_vx.push_back(readTableEntryAs<float>(*vel, "x"));
            particle_vy.push_back(readTableEntryAs<float>(*vel, "y"));
            particle_vz.push_back(readTableEntryAs<float>(*vel, "z"));

            auto weight = readTableEntryAs<float>(*particle_tab, "weight");
            particle_w.push_back(weight);
        }
    }
}
