#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include <toml++/toml.hpp>

using std::vector, std::string;

#include "Vec3.hpp"
#include "Mesh.hpp"
#include "Surface.hpp"
#include "ParticleContainer.cuh"
#include "Window.hpp"

vector<Surface> readInput(string filename) {

    std::vector<Surface> surfaces;

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return surfaces;
    }

    auto geometry = *input.get_as<toml::array>("geometry");

    for (auto&& elem : geometry) {

        auto tab = elem.as_table();
        string name = tab->get_as<string>("name")->get();
        string file = tab->get_as<string>("file")->get();
        bool emit = tab->get_as<bool>("emit")->get();
        bool collect = tab->get_as<bool>("collect")->get();

        Surface s(name, file, emit, collect);
        surfaces.push_back(s);
    }

    return surfaces;
}

int main(int argc, char * argv[]) {

    string filename(argv[1]);
    bool display;
    if (argc > 2) {
        string _display(argv[2]);
        display = static_cast<bool>(stoi(_display));
    } else {
        display = false;
    }

    if (display) {

        Window window("Sputterer", 800, 800);

        while (window.open) {

            window.checkForUpdates();
        }
    }

    return 0;
}
