#ifndef SPUTTERER_RENDERER_HPP
#define SPUTTERER_RENDERER_HPP

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

class GridRenderer {
    public:
        bool enabled = false;

        GridRenderer();
        void draw (Camera camera, float aspect_ratio);
    private:
        Shader shader;
        unsigned int vao, vbo();
};

class BVHRenderer {
    public:
        bool enabled = false;
        int draw_depth = 1;

        BVHRenderer(Scene *scene);
        void draw (Camera camera, float aspect_ratio);
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

        Renderer (Input &input, Scene *scene, ThrusterPlume &plume,
                  ParticleContainer &particles, SceneGeometry &geometry);

        void draw (Input &input, Camera &camera, float aspect_ratio);
};

#endif // SPUTTERER_RENDERER_HPP
