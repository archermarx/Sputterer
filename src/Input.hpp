#ifndef _INPUT_HPP
#define _INPUT_HPP

#include <Surface.hpp>
#include <string>
#include <vector>

class Input {
public:
    Input();
    Input(std::string filename)
        : filename(filename) {}

    std::string          filename{"input.toml"};
    std::vector<Surface> surfaces;

    double chamberRadius;
    double chamberLength;

    std::vector<float> particle_w;
    std::vector<float> particle_x;
    std::vector<float> particle_y;
    std::vector<float> particle_z;
    std::vector<float> particle_vx;
    std::vector<float> particle_vy;
    std::vector<float> particle_vz;

    void read ();

private:
};

#endif