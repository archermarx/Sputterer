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

__global__ void k_push (float *d_pos_x, float *d_pos_y, float *d_pos_z, float *d_vel_x, float *d_vel_y, float *d_vel_z,
                        int N, float dt) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        d_pos_x[id] += d_vel_x[id] * dt;
        d_pos_y[id] += d_vel_y[id] * dt;
        d_pos_z[id] += d_vel_z[id] * dt;
    }
}

void ParticleContainer::push(float dt) {
    const int BLOCK_SIZE = 32;
    const int GRID_SIZE  = static_cast<int>(ceil(static_cast<float>(numParticles) / BLOCK_SIZE));
    dim3      grid(GRID_SIZE, 1, 1);
    dim3      block(BLOCK_SIZE, 1, 1);

    k_push<<<grid, block>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numParticles, dt);

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