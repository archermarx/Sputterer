#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

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
#include "Shader.hpp"

vector<Surface> readInput(string filename) {

    std::cout << "In readinput" << std::endl;

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
        surfaces.emplace_back(name, file, emit, collect);
    }

    for (auto &surface: surfaces) {
        // enable meshes
        surface.mesh.enable();
    }

    return surfaces;
}


int main(int argc, char * argv[]) {

    // Handle command line arguments
    string filename("input.toml");
    if (argc > 1) {
        filename = argv[1];
    }

    bool display(false);
    if (argc > 2) {
        string _display(argv[2]);
        display = static_cast<bool>(stoi(_display));
    }

    Window window("Sputterer", 800, 800);

    Shader shader("shaders/shader.vert", "shaders/shader.frag");

    auto surfaces = readInput(filename);

    for (const auto& surface: surfaces) {
        std::cout << surface.name << "\n";
        std::cout << surface.mesh << "\n";
    }

    while (window.open && display) {

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        for (int i = 0; i < surfaces.size(); i++) {
            surfaces[i].mesh.draw(shader);
        }

        window.checkForUpdates();
    }

    return 0;
}
