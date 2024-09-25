//
// Created by marksta on 4/30/24.
//

#include <complex>

#include "gl_helpers.hpp"

// header for erfi
#include "../include/Faddeeva.hpp"

#include "Constants.hpp"
#include "ThrusterPlume.hpp"
#include "Input.hpp"
#include "ParticleContainer.cuh"
#include "ShaderCode.hpp"

void ThrusterPlume::find_hits (Input &input,
                               Scene &h_scene,
                               host_vector<Material> &h_materials,
                               host_vector<size_t> &h_material_ids,
                               host_vector<HitInfo> &hits,
                               host_vector<float3> &hit_positions,
                               host_vector<float> &num_emit) {
    using namespace constants;
    host_vector <float3> vel;
    host_vector<float> ws;

    // plume coordinate system
    auto up = vec3{0.0, 1.0, 0.0};
    auto right = cross(direction, up);
    up = cross(right, direction);

    auto incident = constants::xenon;
    auto target = constants::carbon;

    auto [main_fraction, scattered_fraction, _] = current_fractions();
    auto beam_fraction = main_fraction + scattered_fraction;
    main_fraction = main_fraction/beam_fraction;

    // tune the number of rays to get a max emission probability close to 1
    float target_prob = 0.95;
    float tol = 0.1;
    float max_iters = 5;
    float max_emit_prob = 0.0;
    int num_rays = 10'000;

    for (int iter = 0; iter < max_iters; iter++) {
        // TODO: do this on GPU
        // TODO: use low-discrepancy sampler for this
        float max_emit = 0.0;
        for (int i = 0; i < num_rays; i++) {
            // select whether ray comes from main beam or scattered beam based on
            // fraction of beam that is scattered vs main
            auto u = rand_uniform();
            double div_angle, beam_energy;
            if (u < main_fraction) {
                div_angle = main_divergence_angle();
                beam_energy = beam_energy_eV;
            } else {
                div_angle = scattered_divergence_angle();
                beam_energy = scattered_energy_eV;
            }

            auto azimuth = rand_uniform(0, 2*constants::pi);
            auto elevation = abs(rand_normal(0, div_angle/sqrt(2.0)));

            auto dir = cos(elevation)*direction + sin(elevation)*(cos(azimuth)*right + sin(azimuth)*up);
            Ray r{make_float3(origin + dir*1e-3f), normalize(make_float3(dir))};

            auto hit = r.cast(h_scene);

            if (hit.hits) {
                // get material that we hit
                auto &mat = h_materials[h_material_ids[hit.id]];

                // Only record hit if material is allowed to sputter
                if (mat.sputter) {
                    hits.push_back(hit);
                    auto hit_pos = r.at(hit.t);
                    hit_positions.push_back(hit_pos);
                    vel.push_back({0.0f, 0.0f, 0.0f});
                    ws.push_back(0.0f);

                    auto cos_hit_angle = static_cast<double>(dot(r.direction, -hit.norm));
                    auto hit_angle = acos(cos_hit_angle);

                    auto yield = sputtering_yield(beam_energy, hit_angle, incident, target);
                    auto n_emit = yield*beam_current_A*beam_fraction/q_e/num_rays/input.particle_weight;
                    if (n_emit > max_emit)
                        max_emit = n_emit;

                    num_emit.push_back(n_emit);
                }
            }
        }
        // basic proportional controller
        max_emit_prob = max_emit*input.timestep_s;
        num_rays = static_cast<int>(num_rays*max_emit_prob/target_prob);
        if (abs(max_emit_prob - target_prob) < tol) {
            break;
        }
    }

    if (input.verbosity > 0) {
        std::cout << "Max emission probability: " << max_emit_prob << std::endl;
        std::cout << "Number of plume rays: " << num_rays << std::endl;
    }

    //ParticleContainer pc_plume{"plume", hit_positions.size()};
    ///pc_plume.add_particles(hit_positions, vel, ws);
    particles.initialize(hit_positions.size());
    particles.add_particles(hit_positions, vel, ws);
}

[[maybe_unused]]  vec2 ThrusterPlume::convert_to_thruster_coords (const vec3 position) const {

    // vector from plume origin to position
    auto offset = position - this->origin;
    auto radius = glm::length(offset);

    // angle between this vector and plume direction
    auto dot_prod = glm::dot(offset, this->direction);
    auto cos_angle = dot_prod/(radius*glm::length(this->direction));
    auto angle = acos(cos_angle);

    return {radius, angle};
}

[[maybe_unused]] double current_density_scale (const double angle, const double div_angle, const double frac) {
    using namespace std::complex_literals;
    using namespace Faddeeva;
    using namespace constants;

    const auto pi_threehalves = sqrt(pi*pi*pi);
    auto erfi_arg1 = 0.5*div_angle;
    auto alpha_squared = div_angle*div_angle;
    auto erfi_arg2 = 0.5*(1i - alpha_squared)/div_angle;
    auto erfi_arg3 = 0.5*(1i + alpha_squared)/div_angle;
    auto erfi_1 = erfi(erfi_arg1);
    auto erfi_2 = erfi(erfi_arg2);
    auto erfi_3 = erfi(erfi_arg3);
    auto denom = pi_threehalves*div_angle*exp(-(erfi_arg1*erfi_arg1))*(2*erfi_1 + erfi_2 - erfi_3);

    auto angle_frac = angle/div_angle;
    auto numer = 2*frac*exp(-(angle_frac*angle_frac));

    return numer/denom.real();
}

double ThrusterPlume::scattered_divergence_angle () const {
    return (this->main_divergence_angle())/(this->model_params[1]);
}

constexpr double torr_to_pa = 133.322;

double ThrusterPlume::main_divergence_angle () const {
    return this->model_params[2]*torr_to_pa*this->background_pressure_Torr + this->model_params[3];
}

CurrentFraction ThrusterPlume::current_fractions () const {

    using namespace constants;

    // divergence angles
    const auto [t0, t1, t2, t3, t4, t5, sigma_cex] = this->model_params;

    // neutral density
    auto neutral_density = t4*this->background_pressure_Torr + t5;

    // get fraction of current in beam vs in main
    auto exp_factor = exp(-1.0*neutral_density*sigma_cex*1e-20);
    auto cex_current_factor = (1.0 - exp_factor);
    auto beam_current_factor = exp_factor;

    return {.main = (1 - t0)*beam_current_factor, .scattered = t0*beam_current_factor, .cex = cex_current_factor,};
}

double sputtering_yield (double energy, double angle, Species incident, Species target) {

    using namespace constants;
    auto [incident_mass, incident_z] = incident;
    auto [target_mass, target_z] = target;

    // Model fitting parameters
    constexpr auto threshold_energy = 10.92;
    constexpr auto q = 2.18;
    constexpr auto lambda = 4.05;
    constexpr auto mu = 1.97;
    constexpr auto f = 2.29;
    constexpr auto a = 0.44;
    constexpr auto b = 0.71;

    // no sputtering if energy below the threshold energy
    if (energy < threshold_energy) {
        return 0.0;
    }

    // model computation
    constexpr auto twothirds = 2.0/3.0;
    constexpr auto arg1 = 9*pi*pi/128;
    constexpr auto arg2 = a_0*4*pi*eps_0/q_e;
    auto a1 = cbrt(arg1)*arg2;

    // lindhard screening length
    auto a_l = a1/sqrt(pow(incident_z, twothirds) + pow(incident_z, twothirds));
    auto eps_l = a_l*energy*target_mass/(target_z*incident_z*(target_mass + incident_mass));

    // Nuclear stopping power for KrC potential
    auto w = eps_l + 0.1728*sqrt(eps_l) + 0.008*pow(eps_l, 0.1504);
    auto inv_w = 1.0/w;
    auto s_n = 0.5*log(1.0 + 1.2288*eps_l)*inv_w;

    // sputtering yield
    auto term_1 = pow(energy/threshold_energy - 1, mu);
    auto term_2 = 1.0/cos(pow(angle, a));
    auto numerator = q*s_n*term_1*pow(term_2, f)*exp(b*(1 - term_2));
    auto denominator = lambda*inv_w + term_1;
    return numerator/denominator;
}

void ThrusterPlume::setup_shaders (float length) {
    particles.setup_shaders({0.2f, 0.75f, 0.94f}, 0.15);
    cone_shader.load(shaders::plume.vert, shaders::plume.frag, shaders::plume.geom);
    cone_shader.use();
    cone_shader.set_float("length", length);
    cone_shader.set_vec3("direction", direction);

    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    float points[] = {this->origin.x, this->origin.y, this->origin.z};
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(points), 0);
    glBindVertexArray(0);
}

void ThrusterPlume::draw (Camera camera, float aspect_ratio) {
    particles.draw(camera, aspect_ratio);

    if (render) {
        // get camera matrix
        auto cam_mat = camera.get_matrix(aspect_ratio);

        // enable plume cone shader
        cone_shader.use(); 
        cone_shader.set_mat4("camera", cam_mat);

        // draw main beam
        auto div_angle = main_divergence_angle();
        cone_shader.set_bool("main_beam", true);
        cone_shader.set_float("angle", div_angle);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 1);
        
        // draw scattered beam
        div_angle = scattered_divergence_angle();
        cone_shader.set_bool("main_beam", false);
        cone_shader.set_float("angle", div_angle);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, 1);
    }
}


