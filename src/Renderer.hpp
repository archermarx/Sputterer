#ifndef SPUTTERER_RENDERER_HPP
#define SPUTTERER_RENDERER_HPP

#include <vector>

#include "Camera.hpp"
#include "Shader.hpp"
#include "Triangle.cuh"
#include "ThrusterPlume.hpp"
#include "ParticleContainer.cuh"
#include "Surface.hpp"
#include "vec3.hpp"
#include "Input.hpp"

constexpr vec3 carbon_particle_color = {0.05f, 0.05f, 0.05f};
constexpr float carbon_particle_scale = 0.05;

struct Grid {
    float scale = 6.0;
    float spacing = 1.0;
    glm::vec3 color = {0.9, 0.9, 0.9};
    float linewidth = 0.005;
};

class GridRenderer {
    public:
        bool enabled = true;
        float opacity = 0.25;

        std::vector<Grid> grids = {
            {6.0, 0.1, {0.8, 0.8, 0.8}, 0.0025},
            {6.0, 1.0, {0.9, 0.9, 0.9}, 0.005},
        };

        GridRenderer();
        void draw_grid (Grid grid, int level, glm::vec3 center = {0.0, 0.0, 0.0});
        void draw (Camera &camera, float aspect_ratio);
    private:
        Shader shader;
        unsigned int vao, vbo, ebo;
};

class BVHRenderer {
    public:
        bool enabled = false;
        int draw_depth = 1;

        BVHRenderer(Scene *scene);
        void draw (Camera &camera, float aspect_ratio);
        void draw_bvh (int depth, int node_idx);
        static void draw_box (Shader &shader, BBox &box, unsigned int &vao, unsigned int &vbo);
    private:
        Scene *scene;
        Shader shader;
        unsigned int vao, vbo;
};

class Renderer {
    public:
        BVHRenderer bvh;
        ThrusterPlume &plume;
        ParticleContainer &particles;
        SceneGeometry &geometry; 
        GridRenderer grid;

        Renderer (Input &input, Scene *scene, ThrusterPlume &plume,
                  ParticleContainer &particles, SceneGeometry &geometry);

        void draw (Input &input, Camera &camera, float aspect_ratio);
};

#endif // SPUTTERER_RENDERER_HPP
