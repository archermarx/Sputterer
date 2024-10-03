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

class BVHRenderer {
    public:
        BVHRenderer(Scene *scene): scene(scene) {}

        Scene *scene;
        Shader shader;
        int draw_depth;
        bool render = false;

        static void draw_box (Shader &shader, BBox &box, unsigned int &vao, unsigned int &vbo);
        void draw_bvh (int depth, int node_idx);
        void draw (Camera camera, float aspect_ratio);
        void setup_shaders (); 

    private:
        unsigned int vao, vbo;
};

class Renderer {
    public:
        BVHRenderer bvh;
        ThrusterPlume &plume;
        ParticleContainer &particles;
        SceneGeometry &geometry; 

        Renderer (Input &input, Scene *scene, ThrusterPlume &plume,
                ParticleContainer &particles, SceneGeometry &geometry)
            : bvh(scene), plume(plume), particles(particles), geometry(geometry){
                setup(input);
            }

        void setup (Input &input) {
            if (input.display) {
                geometry.setup_shaders();
                particles.setup_shaders(carbon_particle_color, carbon_particle_scale);
                plume.setup_shaders(input.chamber_length_m / 2);
                bvh.setup_shaders();
            }
        }
        void draw (Input &input, Camera &camera, float aspect_ratio) {
            if (input.display) {
                geometry.draw(camera, aspect_ratio);
                particles.draw(camera, aspect_ratio);
                bvh.draw(camera, aspect_ratio);
                plume.draw(camera, aspect_ratio);
            }
        };
};

#endif // SPUTTERER_RENDERER_HPP
