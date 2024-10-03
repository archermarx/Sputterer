#ifndef SPUTTERER_RENDERER_HPP
#define SPUTTERER_RENDERER_HPP

constexpr vec3 carbon_particle_color = {0.05f, 0.05f, 0.05f};
constexpr float carbon_particle_scale = 0.05;

class Renderer {
    public:
        BVHRenderer bvh;
        ThrusterPlume &plume;
        ParticleContainer &particles;
        SceneGeometry &geometry;

        Renderer(Input &input, Scene *scene, ThrusterPlume &plume,
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
