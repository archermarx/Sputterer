#include "ParticleContainer.cuh"
#include "cuda_helpers.cuh"

ParticleContainer::ParticleContainer(string name, double mass, int charge)
    : name(name)
    , mass(mass)
    , charge(charge)
    , numParticles(0) {

    // Allocate sufficient GPU memory to hold MAX_PARTICLES particles
    CUDA_CHECK(cudaMalloc((void **)&d_pos_x, MAX_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_pos_y, MAX_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_pos_z, MAX_PARTICLES * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void **)&d_vel_x, MAX_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_vel_y, MAX_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_vel_z, MAX_PARTICLES * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void **)&d_weight, MAX_PARTICLES * sizeof(float)));
};

ParticleContainer::~ParticleContainer() {
    std::cout << "GPU memory freed." << std::endl;
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));
    CUDA_CHECK(cudaFree(d_weight));
}

void ParticleContainer::addParticles(vector<float> x, vector<float> y, vector<float> z, vector<float> ux,
                                     vector<float> uy, vector<float> uz, vector<float> w) {

    auto N = std::min({x.size(), y.size(), z.size(), ux.size(), uy.size(), uz.size(), w.size()});

    // Add particles to CPU arrays
    position_x.insert(position_x.end(), x.begin(), x.begin() + N);
    position_y.insert(position_y.end(), y.begin(), y.begin() + N);
    position_z.insert(position_z.end(), z.begin(), z.begin() + N);

    velocity_x.insert(velocity_x.end(), ux.begin(), ux.begin() + N);
    velocity_y.insert(velocity_y.end(), uy.begin(), uy.begin() + N);
    velocity_z.insert(velocity_z.end(), uz.begin(), uz.begin() + N);

    weight.insert(weight.end(), w.begin(), w.begin() + N);

    // Copy particles to GPU
    // The starting memory address is numParticles * sizeof(float)
    int start = numParticles * sizeof(float);
    int size  = N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_pos_x + start, x.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y + start, y.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z + start, z.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x + start, ux.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y + start, uy.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z + start, uz.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight + start, w.data(), size, cudaMemcpyHostToDevice));

    numParticles += N;
}

void ParticleContainer::copyToCPU() {
    int size = numParticles * sizeof(float);
    CUDA_CHECK(cudaMemcpy(position_x.data(), d_pos_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(position_y.data(), d_pos_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(position_z.data(), d_pos_z, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity_x.data(), d_vel_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity_y.data(), d_vel_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity_z.data(), d_vel_z, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weight.data(), d_weight, size, cudaMemcpyDeviceToHost));
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

__global__ void k_push (float *d_pos_x, float *d_pos_y, float *d_pos_z, float *d_vel_x, float *d_vel_y, float *d_vel_z,
                        int N, Triangle *tris, size_t numTriangles, float dt) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        // Check for intersections with boundaries

        float3 pos{d_pos_x[id], d_pos_y[id], d_pos_z[id]};
        float3 vel{d_vel_x[id], d_vel_y[id], d_vel_z[id]};

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

            d_pos_x[id] = final_pos.x;
            d_pos_y[id] = final_pos.y;
            d_pos_z[id] = final_pos.z;
            d_vel_x[id] = vel_refl.x;
            d_vel_y[id] = vel_refl.y;
            d_vel_z[id] = vel_refl.z;

        } else {
            d_pos_x[id] += d_vel_x[id] * dt;
            d_pos_y[id] += d_vel_y[id] * dt;
            d_pos_z[id] += d_vel_z[id] * dt;
        }
    }
}

void ParticleContainer::push(const float dt, const cuda::vector<Triangle> &tris) {
    const int BLOCK_SIZE = 32;
    const int GRID_SIZE  = static_cast<int>(ceil(static_cast<float>(numParticles) / BLOCK_SIZE));
    dim3      grid(GRID_SIZE, 1, 1);
    dim3      block(BLOCK_SIZE, 1, 1);

    k_push<<<grid, block>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numParticles, tris.data(),
                            tris.size(), dt);

    cudaDeviceSynchronize();
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
        os << "\t" << pc.position_x[i] << "\t";
        os << pc.position_y[i] << "\t";
        os << pc.position_z[i] << "\t";
        os << pc.velocity_x[i] << "\t";
        os << pc.velocity_y[i] << "\t";
        os << pc.velocity_z[i] << "\t";
        os << pc.weight[i] << "\t";
        os << "\n";
    }
    os << "==========================================================\n";

    return os;
}