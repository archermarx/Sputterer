#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include <toml++/toml.hpp>

using std::vector, std::string;

#include "Vec3.cuh"
#include "Mesh.cuh"
#include "ParticleContainer.cuh"
#include "Surface.cuh"

int main() {
    string filename("../input.toml");

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return 1;
    }

    auto geometry = *input.get_as<toml::array>("geometry");

    std::vector<Surface> surfaces;

    for (auto&& elem : geometry) {

        auto tab = elem.as_table();
        string name = tab->get_as<string>("name")->get();
        string file = tab->get_as<string>("file")->get();
        bool emit = tab->get_as<bool>("emit")->get();
        bool collect = tab->get_as<bool>("collect")->get();

        Surface s(name, file, emit, collect);
        surfaces.push_back(s);

        std::cout << s.name << "\n" << s.mesh << "\n";
    }

    return 0;
}
