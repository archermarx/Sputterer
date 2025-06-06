#include <random>
#include <sstream>

#include <thrust/distance.h>
#include <thrust/partition.h>

#include "Camera.h"
#include "ParticleContainer.h"
#include "cuda_helpers.h"
#include "gl_helpers.h"
#include "Constants.h"
#include "Input.h"
#include "ShaderCode.h"

// Setup RNG
__global__ void k_setup_rng (curandState *rng, size_t N, uint64_t seed) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        curand_init((seed << 20) + tid, 0, 0, &rng[tid]);
    }
}

void ParticleContainer::initialize (size_t capacity) {
    CUDA_CHECK_STATUS();

    // Clear vectors
    this->num_particles = 0;
    d_position.clear();
    d_velocity.clear();
    d_weight.clear();
    d_tmp.clear();
    d_rng.clear();

    // Allocate memory on GPU
    d_position.resize(capacity);
    d_velocity.resize(capacity);
    d_weight.resize(capacity);
    d_tmp.resize(capacity);

    // Allocate RNG
    d_rng.resize(capacity);

    // Reinit RNG
    CUDA_CHECK_STATUS_WITH_MESSAGE("Before RNG initialization");

    auto [grid, block] = get_kernel_launch_params(capacity, k_setup_rng);
    curandState *d_rng_ptr = thrust::raw_pointer_cast(d_rng.data());
    k_setup_rng<<<grid, block>>>(d_rng_ptr, d_rng.size(), time(nullptr));

    CUDA_CHECK_STATUS_WITH_MESSAGE("After RNG initialization");
    CUDA_CHECK(cudaDeviceSynchronize());
}

ParticleContainer::ParticleContainer (string name, size_t num, double mass, int charge)
    : name(std::move(name))
    , mass(mass)
    , charge(charge) {
    initialize(num);
}

void ParticleContainer::add_particles (const host_vector<float3> &pos, const host_vector<float3> &vel,
                                       const host_vector<float> &w) {
    CUDA_CHECK_STATUS();
    auto n = static_cast<int>(std::min(std::min(pos.size(), vel.size()), w.size()));
    if (n == 0)
        return;

    if (this->num_particles + n > MAX_PARTICLES) {
        throw std::runtime_error("Maximum number of particles exceeded! Terminating.");
    }

    position.resize(num_particles + n);
    velocity.resize(num_particles + n);
    weight.resize(num_particles + n);

    // Copy particles to CPU arrays
    for (size_t i = 0; i < n; i++) {
        auto id = num_particles + i;
        position[id] = pos[i];
        velocity[id] = vel[i];
        weight[id] = w[i];
    }

    // Copy particles to GPU
    thrust::copy(position.begin() + num_particles, position.end(), d_position.begin() + num_particles);
    thrust::copy(velocity.begin() + num_particles, velocity.end(), d_velocity.begin() + num_particles);
    thrust::copy(weight.begin() + num_particles, weight.end(), d_weight.begin() + num_particles);

    num_particles += n;
    CUDA_CHECK_STATUS();
}

void ParticleContainer::copy_to_cpu () {
    CUDA_CHECK_STATUS();
    position = host_vector<float3>(d_position.begin(), d_position.begin() + num_particles);
    velocity = host_vector<float3>(d_velocity.begin(), d_velocity.begin() + num_particles);
    weight = host_vector<float>(d_weight.begin(), d_weight.begin() + num_particles);
    CUDA_CHECK_STATUS();
}

void ParticleContainer::setup_shaders (glm::vec3 color, float scale) {
    // Load particle shader
    shader.link(shaders::particle);
    shader.use();
    // TODO: have scale controlled by a slider
    this->scale = scale;
    this->color = color;
    shader.set_uniform("objectColor", color);

    // TODO: have geometric primitives stored as strings in a c++ source file
    // Set up particle meshes
    mesh.load("square");
    mesh.set_buffers();
    glGenBuffers(1, &buffer);
}

void ParticleContainer::draw (Camera &camera, float aspect_ratio) {
    if (!render || num_particles <= 0) {
        return;
    }

    // get camera matrix for use in particle and plume shaders
    auto cam_mat = camera.get_matrix(aspect_ratio);

    // enable shader and set uniforms
    shader.use();
    // Set particle scale
    glm::vec3 scale_vec{this->scale};
    shader.set_uniform("scale", scale_vec);
    shader.set_uniform("cameraRight", camera.right);
    shader.set_uniform("cameraUp", camera.up);
    shader.set_uniform("camera", cam_mat);

    // Bind vertex array
    auto vao = this->mesh.vao;
    GL_CHECK(glBindVertexArray(vao));

    // Send over model matrix data
    auto mat_vector_size = static_cast<GLsizei>(this->num_particles * sizeof(glm::vec3));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, this->buffer));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, mat_vector_size, &position[0], GL_DYNAMIC_DRAW));

    // Set attribute pointers for translation
    GL_CHECK(glEnableVertexAttribArray(2));
    GL_CHECK(glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr));
    GL_CHECK(glVertexAttribDivisor(2, 1));

    // Bind element array buffer
    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->mesh.ebo));

    // Draw meshes
    GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(3 * this->mesh.num_triangles),
                                     GL_UNSIGNED_INT, nullptr, num_particles));

    // unbind buffers
    GL_CHECK(glBindVertexArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

__host__ __device__ float carbon_diffuse_prob (float cos_incident_angle, float incident_energy_ev) {
    // fit parameters to molecular dynamics data
    constexpr auto angle_offset = 1.6823f;
    constexpr auto energy_offset = 65.6925f;
    constexpr auto energy_scale = 34.5302f;

    auto fac = (cos_incident_angle - angle_offset) * logf((incident_energy_ev + energy_offset) / energy_scale);
    auto diffuse_coeff = 0.003f + fac * fac;
    return diffuse_coeff;
}

/*
 * Sample particles that reflect diffusely from a surface with temperature defined by `thermal_speed`
 */
__device__ float3 sample_diffuse_reflection (const Triangle &tri, const float3 norm, float thermal_speed,
                                             curandState *rng) {
    using namespace constants;

    // get tangent vectors
    auto tan1 = normalize(tri.v1 - tri.v0);
    auto tan2 = cross(tan1, norm);

    // Boyd and Schwartzentruber "Non-equilibrium gas dynamics and molecular simulation", pp. 318-319.
    auto r1 = curand_uniform(rng);
    auto r2 = curand_uniform(rng);
    auto r3 = curand_uniform(rng);
    auto r4 = curand_uniform(rng);
    auto r5 = curand_uniform(rng);

    auto c_norm = sqrt(-0.5 * log(r1));
    auto c_tan1 = sinpi(2 * r2) * sqrt(-0.5 * log(r3));
    auto c_tan2 = sinpi(2 * r4) * sqrt(-0.5 * log(r5));

    // Compute new velocity vector
    auto vel_refl = thermal_speed * (c_norm * norm + c_tan1 * tan1 + c_tan2 * tan2);

    return vel_refl;
}

__device__ float sigmund_thompson_edf (float E, float E_B, float m) {
    return E / pow(E + E_B, 3 - 2 * m);
}

/*
 * Sample particles that sputter off a surface in response to an incident atom of a given energy.
 * Currently, we assume a cosine distribution and so there is no dependence on the angle of the incident particle.
 * Ref: https://eplab.ae.illinois.edu/Publications/IEPC-2022-379.pdf
 */
__device__ float3 sample_sputtered_particle (const Triangle &tri, const float3 norm, float incident_energy_ev,
                                             float incident_angle_rad, double target_mass, curandState *rng) {

    using constants::pi;

    constexpr auto E_B = 7.4;
    constexpr auto m = static_cast<float>(1.0 / 3.0);

    // Determine vector by sampling from cosine distribution
    // get tangent vectors
    auto tan1 = normalize(tri.v1 - tri.v0);
    auto tan2 = cross(tan1, norm);

    auto sin2_theta = curand_uniform(rng);
    auto cos_theta = sqrt(1 - sin2_theta);
    auto sin_theta = sqrt(sin2_theta);

    // Uniform distribution in azimuth
    auto plane_angle = curand_uniform(rng) * 2;
    float sin_phi, cos_phi;
    sincospif(plane_angle, &sin_phi, &cos_phi);

    auto c_norm = cos_theta;
    auto c_tan1 = sin_theta * sin_phi;
    auto c_tan2 = sin_theta * cos_phi;

    auto vector = c_norm * norm + c_tan1 * tan1 + c_tan2 * tan2;

    // Determine energy by sampling from Sigmund-Thompson energy distribution
    auto max_sigmund_thompson_edf = sigmund_thompson_edf(E_B / (2 * (1 - m)), E_B, m);
    float energy;

    while (true) {
        auto u = curand_uniform(rng);
        energy = curand_uniform(rng) * incident_energy_ev;
        auto p_accept = sigmund_thompson_edf(energy, E_B, m) / max_sigmund_thompson_edf;
        if (u <= p_accept)
            break;
    }

    // Convert energy to velocity
    auto velocity = sqrt(2 * constants::q_e * energy / target_mass);

    return velocity * vector;
}

DeviceParticleContainer ParticleContainer::data () {
    CUDA_CHECK_STATUS();
    DeviceParticleContainer pc;
    pc.position = thrust::raw_pointer_cast(this->d_position.data());
    pc.velocity = thrust::raw_pointer_cast(this->d_velocity.data());
    pc.weight = thrust::raw_pointer_cast(this->d_weight.data());
    pc.rng = thrust::raw_pointer_cast(this->d_rng.data());
    pc.num_particles = this->num_particles;
    CUDA_CHECK_STATUS();
    return pc;
}

__global__ void k_evolve (DeviceParticleContainer pc, Scene scene, const Material *materials,
                          const size_t *material_ids, int *collected, const HitInfo *hits, const float *emit_prob,
                          size_t num_hits, float input_weight, float dt) {

    // Thread ID, i.e. what particle we're currently moving
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    using namespace constants;

    // Target particle mass
    // FIXME: currently hard-coded to carbon, easy to fix by passing in mass as a param
    const double target_mass = 12.011 * m_u;

    // Particle energy
    const double energy_factor = 0.5 * target_mass / q_e;

    // k_B / m_u (for thermal speed calculations)
    const auto thermal_speed_factor = static_cast<float>(sqrt(2 * k_b / target_mass));

    // Push particles
    if (tid < pc.num_particles) {

        auto pos = pc.position[tid];
        auto vel = pc.velocity[tid];

        // Check for intersections with boundaries
        Ray ray(pos, dt * vel);
        auto closest_hit = ray.cast(scene);

        if (closest_hit.t <= 1) {
            auto t = closest_hit.t;
            auto hit_triangle_id = closest_hit.id;
            auto hit_pos = closest_hit.pos;
            auto norm = closest_hit.norm;

            // Get material info where we hit
            auto &mat = materials[material_ids[hit_triangle_id]];

            // Generate a random number
            auto local_rng = &pc.rng[tid];
            auto uniform = curand_uniform(local_rng);

            // get incident angle and energy
            auto velnorm_2 = dot(vel, vel);
            auto cos_incident_angle = abs(dot(vel, -norm) / sqrt(velnorm_2));
            auto incident_energy_ev = static_cast<float>(energy_factor * velnorm_2);

            // Get sticking and diffuse coeff from model
            // Material coefficients not presently used
            // auto diffuse_coeff = mat.diffuse_coeff;
            auto diffuse_coeff = carbon_diffuse_prob(cos_incident_angle, incident_energy_ev);
            auto sticking_coeff = 1.0f - diffuse_coeff;

            if (uniform < sticking_coeff) {
                // Particle sticks to surface
                pc.position[tid] = hit_pos;
                pc.velocity[tid] = float3{0.0f, 0.0f, 0.0f};

                // Record that we hit this triangle
                atomicAdd(&collected[hit_triangle_id], 1);

                // set weight negative to flag for removal
                // magnitude indicates which triangle we hit
                pc.weight[tid] = static_cast<float>(-hit_triangle_id);

            } else if (uniform < diffuse_coeff + sticking_coeff) {
                // Particle reflects diffusely based on surface temperature
                // TODO: pass thermal speed (or sqrt of temperature) instead of temperature to avoid this
                auto sqrt_temp = sqrtf(mat.temperature_K);
                auto thermal_speed = thermal_speed_factor * sqrt_temp;
                auto vel_refl =
                    sample_diffuse_reflection(scene.triangles[hit_triangle_id], norm, thermal_speed, local_rng);

                // Get particle position
                // (assuming particle reflects ~instantaneously then travels according to new velocity vector)
                // TODO: most of this code is shared with below--worth unifying?
                auto final_pos = hit_pos + (1 - t) * dt * vel_refl;
                pc.position[tid] = final_pos;
                pc.velocity[tid] = vel_refl;
            } else {
                // Particle reflects specularly
                float3 vel_norm = dot(vel, norm) * norm;
                float3 vel_refl = vel - 2 * vel_norm;

                auto final_pos = hit_pos + (1 - t) * dt * vel_refl;
                pc.position[tid] = final_pos;
                pc.velocity[tid] = vel_refl;
            }
        } else {
            pc.position[tid] = pos + dt * vel;
        }
    } else if (tid < num_hits + pc.num_particles) {
        // Emit new particles
        auto &hit = hits[tid - pc.num_particles];
        auto p_emit = emit_prob[tid - pc.num_particles];
        auto local_rng = &pc.rng[tid];

        // generate rng
        auto u = curand_uniform(local_rng);

        // add new particles (negative weight if not real)
        if (u < p_emit * dt) {
            // Sample velocity
            auto &tri = scene.triangles[hit.id];
            auto incident_ion_energy_ev = hit.energy;
            auto incident_ion_angle_rad = hit.angle;
            auto vel = sample_sputtered_particle(tri, hit.norm, incident_ion_energy_ev, incident_ion_angle_rad,
                                                 target_mass, local_rng);
            auto pos_offset = curand_uniform(local_rng) * dt * vel;
            pc.velocity[tid] = vel;

            // offset particles from surface to avoid re-intersecting emission surface and to produce more
            pc.position[tid] = hit.pos + pos_offset;
            pc.weight[tid] = input_weight;
        } else {
            // flag particle for removal
            pc.position[tid] = -1000.000 * hit.pos;
            pc.velocity[tid] = {0.0, 0.0, 0.0};
            pc.weight[tid] = -1.0;
        }
    }
}

std::string print_dim3 (dim3 x) {
    std::ostringstream s;
    s << "[" << x.x << "," << x.y << "," << x.z << "]";
    return s.str();
}

void ParticleContainer::evolve (Scene scene, const device_vector<Material> &mats, const device_vector<size_t> &ids,
                                device_vector<int> &collected, const device_vector<HitInfo> &hits,
                                const device_vector<float> &num_emit, const Input &input) {

    CUDA_CHECK_STATUS();
    // TODO: could move all of the device geometric info into a struct
    auto d_id_ptr = thrust::raw_pointer_cast(ids.data());
    auto d_mat_ptr = thrust::raw_pointer_cast(mats.data());

    auto d_col_ptr = thrust::raw_pointer_cast(collected.data());
    auto d_hit_ptr = thrust::raw_pointer_cast(hits.data());
    auto d_emit_ptr = thrust::raw_pointer_cast(num_emit.data());

    auto [grid, block] = get_kernel_launch_params(num_particles + hits.size(), k_evolve);
    k_evolve<<<grid, block>>>(this->data(), scene, d_mat_ptr, d_id_ptr, d_col_ptr, d_hit_ptr, d_emit_ptr, hits.size(),
                              input.particle_weight, input.timestep_s);

    this->num_particles += hits.size();

    CUDA_CHECK(cudaDeviceSynchronize());

    remove_out_of_bounds(input);

    CUDA_CHECK_STATUS();
}

float rand_uniform (float min, float max) {
    static std::default_random_engine rng;

    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

float rand_normal (float mean, float std) {
    static std::default_random_engine rng;

    std::normal_distribution<float> dist(mean, std);
    return dist(rng);
}

__global__ void k_flag_oob (float3 *pos, float3 *vel, float *weight, float radius2, float halflength, size_t n) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n && weight[id] > 0) {
        auto r = pos[id];
        auto v = vel[id];
        auto vel2 = v.x * v.x + v.y * v.y * v.z * v.z;
        if (vel2 < 1e-3) {
            weight[id] = -1;
        } else {
            auto dist2 = r.x * r.x + r.y * r.y; // distance in x-y plane from center
            auto dist_backcap = dist2 + (r.z + halflength) * (r.z + halflength);
            auto dist_frontcap = dist2 + (r.z - halflength) * (r.z - halflength);
            if (dist2 > radius2 || (r.z < -halflength && dist_backcap > radius2) ||
                (r.z > halflength && dist_frontcap > radius2)) {
                // Particles that are oob get negative weight
                weight[id] = -1;
            }
        }
    }
}

// functor for determining whether a float is positive, used in flagging particles with positive weights
struct IsPositive {
    __host__ __device__ bool operator() (const float &w) {
        return w > 0;
    }
};

void ParticleContainer::remove_out_of_bounds (const Input &input) {
    CUDA_CHECK_STATUS();
    // Get raw pointers to position and weight data
    auto d_pos_ptr = thrust::raw_pointer_cast(d_position.data());
    auto d_vel_ptr = thrust::raw_pointer_cast(d_velocity.data());
    auto d_wgt_ptr = thrust::raw_pointer_cast(d_weight.data());

    // Mark particles that are OOB or have zero velocity with negative weight
    const auto r = input.chamber_radius_m;
    const auto l = input.chamber_length_m;
    const auto [grid, block] = get_kernel_launch_params(num_particles, k_flag_oob);

    k_flag_oob<<<grid, block>>>(d_pos_ptr, d_vel_ptr, d_wgt_ptr, r * r, l / 2 - r, num_particles);
    CUDA_CHECK(cudaDeviceSynchronize());

    // reorder positions and velocities so that particles with negative or zero weight follow those with positive weight
    // This could be done with a single partition if I was smarter
    thrust::partition(d_position.begin(), d_position.begin() + num_particles, d_weight.begin(), IsPositive());
    thrust::partition(d_velocity.begin(), d_velocity.begin() + num_particles, d_weight.begin(), IsPositive());

    // reorder weights according to the same scheme as above
    // copy weights to temporary vector first
    // thrust partition likely is allocating some temporary memory
    // to avoid this, we would probably want to set up a custom allocator
    // c.f. https://github.com/NVIDIA/thrust/blob/1.6.0/examples/cuda/custom_temporary_allocation.cu
    // Alternatively, could use CUB device partition, which gives us more control to allocate temporary data
    // c.f. https://nvidia.github.io/cccl/cub/api/structcub_1_1DevicePartition.html#_CPPv4N3cub15DevicePartitionE
    thrust::copy(d_weight.begin(), d_weight.begin() + num_particles, d_tmp.begin());
    auto ret = thrust::partition(d_weight.begin(), d_weight.begin() + num_particles, d_tmp.begin(), IsPositive());

    // Reset number of particles to the middle of the partition
    num_particles = static_cast<int>(thrust::distance(d_weight.begin(), ret));
    CUDA_CHECK_STATUS();
}

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc) {
    os << "==========================================================\n";
    os << "Particle container \"" << pc.name << "\"\n";
    os << "==========================================================\n";
    os << "Mass: " << pc.mass << "\n";
    os << "Charge: " << pc.charge << "\n";
    os << "Number of particles: " << pc.num_particles << "\n";
    os << "----------------------------------------------------------\n";
    os << "\tx\ty\tz\tvx\tvy\tvz\tw\t\n";
    os << "----------------------------------------------------------\n";
    for (int i = 0; i < pc.num_particles; i++) {
        os << "\t" << pc.position[i].x << " ";
        os << pc.position[i].y << "  ";
        os << pc.position[i].z << "  ";
        os << pc.velocity[i].x << "  ";
        os << pc.velocity[i].y << "  ";
        os << pc.velocity[i].z << "  ";
        os << pc.weight[i] << "\n";
    }
    os << "==========================================================\n";

    return os;
}
