#ifndef SPUTTERER_RENDERER_H
#define SPUTTERER_RENDERER_H

#include <vector>

#include "Camera.h"
#include "Shader.h"
#include "Triangle.h"
#include "ThrusterPlume.h"
#include "ParticleContainer.h"
#include "Surface.h"
#include "Input.h"

#include <glm/glm.hpp>

const glm::vec3 carbon_particle_color = {0.05f, 0.05f, 0.05f};
const float carbon_particle_scale = 0.05;

struct Grid {
    float scale = 6.0;
    float spacing = 1.0;
    glm::vec3 color = {0.9, 0.9, 0.9};
    float linewidth = 0.005;
};

class GeometryRenderer {
  public:
    vector<Surface> &surfaces;
    GeometryRenderer(std::vector<Surface> &surfaces);
    void draw (Camera &camera, float aspect_ratio);

  private:
    ShaderProgram shader;
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
        ShaderProgram shader;
        unsigned int vao, vbo, ebo;
};

class BVHRenderer {
    public:
        bool enabled = false;
        int draw_depth = 1;

        BVHRenderer(Scene *scene);
        void draw (Camera &camera, float aspect_ratio);
        void draw_bvh (int depth, int node_idx);
        static void draw_box (ShaderProgram &shader, BBox &box, unsigned int &vao, unsigned int &vbo);
    private:
        Scene *scene;
        ShaderProgram shader;
        unsigned int vao, vbo;
};

class Renderer {
    public:
        BVHRenderer bvh;
        ThrusterPlume &plume;
        ParticleContainer &particles;
        GeometryRenderer geometry; 
        GridRenderer grid;

        Renderer (Input &input, Scene *scene, ThrusterPlume &plume,
                  ParticleContainer &particles, std::vector<Surface> surfaces);

        void draw (Input &input, Camera &camera, float aspect_ratio);
};

#endif // SPUTTERER_RENDERER_H
