#include "ParticleContainer.cuh"
#include "cuda_helpers.cuh"

ParticleContainer::ParticleContainer(string name, double mass, int charge):
    name(name), mass(mass), charge(charge), numParticles(0) {

    // Allocate sufficient GPU memory to hold MAX_PARTICLES particles
    CUDA_CHECK( cudaMalloc((void**) &d_pos_x, MAX_PARTICLES * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**) &d_pos_y, MAX_PARTICLES * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**) &d_pos_z, MAX_PARTICLES * sizeof(float)) );

    CUDA_CHECK( cudaMalloc((void**) &d_vel_x, MAX_PARTICLES * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**) &d_vel_y, MAX_PARTICLES * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**) &d_vel_z, MAX_PARTICLES * sizeof(float)) );

    CUDA_CHECK( cudaMalloc((void**) &d_weight, MAX_PARTICLES * sizeof(float)) );
};

ParticleContainer::~ParticleContainer() {
    // Free GPU memory
    CUDA_CHECK( cudaFree(d_pos_x) );
    CUDA_CHECK( cudaFree(d_pos_y) );
    CUDA_CHECK( cudaFree(d_pos_z) );
    CUDA_CHECK( cudaFree(d_vel_x) );
    CUDA_CHECK( cudaFree(d_vel_y) );
    CUDA_CHECK( cudaFree(d_vel_z) );
    CUDA_CHECK( cudaFree(d_weight) );
}

void ParticleContainer::addParticles(
        vector<float> x, vector<float> y, vector<float> z,
        vector<float> ux, vector<float> uy, vector<float> uz, vector<float> w) {

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
    int size = N * sizeof(float);
    CUDA_CHECK( cudaMemcpy(d_pos_x  + start, x.data(),  size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_pos_y  + start, y.data(),  size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_pos_z  + start, z.data(),  size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_vel_x  + start, ux.data(), size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_vel_y  + start, uy.data(), size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_vel_z  + start, uz.data(), size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_weight + start, w.data(),  size, cudaMemcpyHostToDevice) );

    numParticles += N;
}

void ParticleContainer::copyToCPU() {
    int size = numParticles * sizeof(float);
    CUDA_CHECK( cudaMemcpy(position_x.data(), d_pos_x,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(position_y.data(), d_pos_y,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(position_z.data(), d_pos_z,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(velocity_x.data(), d_vel_x,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(velocity_y.data(), d_vel_y,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(velocity_z.data(), d_vel_z,  size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(weight.data(),     d_weight, size, cudaMemcpyDeviceToHost) );
}

std::ostream &operator<<(std::ostream &os, ParticleContainer const &pc) {
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