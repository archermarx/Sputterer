#include <random>

#include <thrust/distance.h>
#include <thrust/partition.h>

#include "ParticleContainer.cuh"
#include "cuda_helpers.cuh"

ParticleContainer::ParticleContainer(string name, double mass, int charge)
    : name(name)
    , mass(mass)
    , charge(charge) {

    // Allocate memory on GPU
    d_position.resize(MAX_PARTICLES);
    d_velocity.resize(MAX_PARTICLES);
    d_weight.resize(MAX_PARTICLES);
    d_tmp.resize(MAX_PARTICLES);
}

void ParticleContainer::addParticles(vector<float> x, vector<float> y, vector<float> z, vector<float> ux,
                                     vector<float> uy, vector<float> uz, vector<float> w) {

    auto N = std::min({x.size(), y.size(), z.size(), ux.size(), uy.size(), uz.size(), w.size()});

    if (N == 0) {
        return;
    }

    position.resize(numParticles + N);
    velocity.resize(numParticles + N);
    weight.resize(numParticles + N);

    // Add particles to CPU arrays
    for (size_t i = 0; i < N; i++) {
        auto id = numParticles + i;

        position[id] = {x.at(i), y.at(i), z.at(i)};
        velocity[id] = {ux.at(i), uy.at(i), uz.at(i)};
        weight[id]   = {w.at(i)};
    }

    // Copy particles to GPU
    thrust::copy(position.begin() + numParticles, position.end(), d_position.begin() + numParticles);
    thrust::copy(velocity.begin() + numParticles, velocity.end(), d_velocity.begin() + numParticles);
    thrust::copy(weight.begin() + numParticles, weight.end(), d_weight.begin() + numParticles);

    numParticles += N;
}

void ParticleContainer::copyToCPU() {
    position = host_vector<float3>(d_position.begin(), d_position.begin() + numParticles);
    velocity = host_vector<float3>(d_velocity.begin(), d_velocity.begin() + numParticles);
    weight   = host_vector<float>(d_weight.begin(), d_weight.begin() + numParticles);
}

#define MIN_T 100'000
#define TOL 1e-6

__host__ __device__ HitInfo hits_triangle (Ray ray, Triangle tri) {
    HitInfo info;

    // Find vectors for two edges sharing v1
    auto edge1 = tri.v1 - tri.v0;
    auto edge2 = tri.v2 - tri.v0;

    // Begin calculating determinant
    auto pvec = cross(ray.direction, edge2);
    auto det  = dot(edge1, pvec);

    // If determinant is near zero, ray lies in plane of triangle
    if (abs(det) < TOL) {
        return info;
    }

    // Calculate distance from v0 to ray origin
    auto tvec = ray.origin - tri.v0;

    // Calculate u parameter and test bounds
    auto u = dot(tvec, pvec) / det;
    if (u < 0.0 || u > 1.0) {
        return info;
    }

    auto qvec = cross(tvec, edge1);

    // Calculate v parameter and test bounds
    auto v = dot(ray.direction, qvec) / det;
    if (v < 0.0 || u + v > 1.0) {
        return info;
    }
    // Calculate t, ray intersects triangle
    auto t = dot(edge2, qvec) / det;

    info.hits = true;
    info.t    = t;

    if (dot(ray.direction, tri.norm) > 0) {
        info.norm = -tri.norm;
    } else {
        info.norm = tri.norm;
    }

    return info;
}

__global__ void k_push (float3 *position, float3 *velocity, const int N, const Triangle *tris,
                        const size_t numTriangles, const float dt) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {

        auto pos = position[id];
        auto vel = velocity[id];

        // Check for intersections with boundaries
        Ray ray{.origin = pos, .direction = dt * vel};

        HitInfo closest_hit{.hits = false, .t = static_cast<float>(MIN_T), .norm = {0.0, 0.0, 0.0}};
        HitInfo current_hit;
        for (size_t i = 0; i < numTriangles; i++) {
            current_hit = hits_triangle(ray, tris[i]);
            if (current_hit.hits && current_hit.t < closest_hit.t && current_hit.t >= 0) {
                closest_hit = current_hit;
            }
        }

        if (closest_hit.t <= 1) {
            auto &[_, t, norm] = closest_hit;

            float3 vel_norm = dot(vel, norm) * norm;
            float3 vel_refl = vel - 2 * vel_norm;

            auto hit_pos   = pos + t * dt * vel;
            auto final_pos = hit_pos + (1 - t) * dt * vel_refl;

            position[id] = final_pos;
            velocity[id] = vel_refl;

        } else {
            position[id] = pos + dt * vel;
        }
    }
}

std::pair<dim3, dim3> ParticleContainer::getKernelLaunchParams(size_t block_size) const {
    auto grid_size = static_cast<int>(ceil(static_cast<float>(numParticles) / block_size));
    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);
    return std::make_pair(grid, block);
}

void ParticleContainer::push(const float dt, const thrust::device_vector<Triangle> &tris) {
    auto d_pos_ptr = thrust::raw_pointer_cast(d_position.data());
    auto d_vel_ptr = thrust::raw_pointer_cast(d_velocity.data());
    auto d_tri_ptr = thrust::raw_pointer_cast(tris.data());

    auto [grid, block] = getKernelLaunchParams();
    k_push<<<grid, block>>>(d_pos_ptr, d_vel_ptr, numParticles, d_tri_ptr, tris.size(), dt);

    cudaDeviceSynchronize();
}

float randUniform (float min = 0.0f, float max = 1.0f) {
    static std::default_random_engine     rng;
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

void ParticleContainer::emit(Triangle &triangle, float flux, float dt) {
    auto numEmit    = flux * triangle.area * dt;
    int  intNumEmit = static_cast<int>(numEmit);
    auto remainder  = numEmit - intNumEmit;

    auto u = randUniform();
    if (u < remainder) {
        intNumEmit += 1;
    }

    if (intNumEmit < 1) {
        return;
    }

    float speed = -1.0;

    std::vector<float> x(intNumEmit, 0.0), y(intNumEmit, 0.0), z(intNumEmit, 0.0);
    std::vector<float> ux(intNumEmit, 0.0), uy(intNumEmit, 0.0), uz(intNumEmit, 0.0);
    std::vector<float> w(intNumEmit, 1.0);

    auto velocityJitter = 0.15f;
    for (int i = 0; i < intNumEmit; i++) {
        auto pt  = triangle.sample(randUniform(), randUniform());
        x.at(i)  = pt.x;
        y.at(i)  = pt.y;
        z.at(i)  = pt.z;
        ux.at(i) = speed * (triangle.norm.x + velocityJitter * randUniform(-1, 1));
        uy.at(i) = speed * (triangle.norm.y + velocityJitter * randUniform(-1, 1));
        uz.at(i) = speed * (triangle.norm.z + velocityJitter * randUniform(-1, 1));
    }

    addParticles(x, y, z, ux, uy, uz, w);
}

__global__ void k_flag_oob (float3 *pos, float *weight, float radius2, float halflength, size_t N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N && weight[id] > 0) {
        auto r     = pos[id];
        auto dist2 = r.x * r.x + r.y * r.y;
        if (dist2 > radius2 || r.z < -halflength || r.z > halflength) {
            // Particles that are oob get negative weight
            weight[id] = -1;
        }
    }
}

void ParticleContainer::flagOutOfBounds(float radius, float length) {
    auto [grid, block] = getKernelLaunchParams();

    auto d_pos_ptr = thrust::raw_pointer_cast(d_position.data());
    auto d_wgt_ptr = thrust::raw_pointer_cast(d_weight.data());
    k_flag_oob<<<grid, block>>>(d_pos_ptr, d_wgt_ptr, radius * radius, length / 2, numParticles);
    cudaDeviceSynchronize();
}

struct is_positive {
    __host__ __device__ bool operator() (const float &w) {
        return w > 0;
    }
};

void ParticleContainer::removeFlaggedParticles() {
    // reorder positions and velocities so that particles with negative weight follow those with positive weight
    thrust::partition(d_position.begin(), d_position.begin() + numParticles, d_weight.begin(), is_positive());
    thrust::partition(d_velocity.begin(), d_velocity.begin() + numParticles, d_weight.begin(), is_positive());

    // reorder weights according to the same scheme as above
    // copy weights to temporary vector first
    // thrust partition likely is allocating some temporary memory
    // to avoid this, we would probably want to set up a custom allocator
    // c.f. https://github.com/NVIDIA/thrust/blob/1.6.0/examples/cuda/custom_temporary_allocation.cu
    // Alternatively, could use CUB device partition, which gives us more control to allocate temporary data
    // c.f. https://nvidia.github.io/cccl/cub/api/structcub_1_1DevicePartition.html#_CPPv4N3cub15DevicePartitionE
    thrust::copy(d_weight.begin(), d_weight.begin() + numParticles, d_tmp.begin());
    auto ret = thrust::partition(d_weight.begin(), d_weight.begin() + numParticles, d_tmp.begin(), is_positive());

    // Reset number of particles to the middle of the partition
    numParticles = thrust::distance(d_weight.begin(), ret);
}

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc) {
    os << "==========================================================\n";
    os << "Particle container \"" << pc.name << "\"\n";
    os << "==========================================================\n";
    os << "Mass: " << pc.mass << "\n";
    os << "Charge: " << pc.charge << "\n";
    os << "Number of particles: " << pc.numParticles << "\n";
    os << "----------------------------------------------------------\n";
    os << "\tx\ty\tz\tvx\tvy\tvz\tw\t\n";
    os << "----------------------------------------------------------\n";
    for (int i = 0; i < pc.numParticles; i++) {
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