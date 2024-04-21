#ifndef _CUDA_CUH
#define _CUDA_CUH

#include <vector>

#include "../include/cuda_helpers.cuh"

namespace cuda {

template <typename T>
class vector {
public:
    // Default constructor - vector of size zero with no data
    // (data pointer is null)
    vector()
        : m_data_ptr(nullptr)
        , m_size(0) {}

    // Initialize a vector with a pre-determined size
    vector(size_t numElements)
        : m_size(numElements) {
        CUDA_CHECK(cudaMalloc((void **)&m_data_ptr, m_size * sizeof(T)));
    }

    // Construct a vector from a std::vector, copying memory to GPU
    vector(const std::vector<T> &host_vec)
        : m_size(host_vec.size()) {
        CUDA_CHECK(cudaMalloc((void **)&m_data_ptr, m_size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(m_data_ptr, host_vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Free up memory when this vector is destroyed
    ~vector() {
        if (m_data_ptr != nullptr) {
            CUDA_CHECK(cudaFree(m_data_ptr));
        }
    }

    // Return number of elements in the vector
    size_t size () const {
        return m_size;
    }

    // Returns the pointer to the data on the device
    T *data () const {
        return m_data_ptr;
    }

    // Copy the contents of the vector to a provided std::vector living on the host
    void copyTo (std::vector<T> &dest) {
        dest.resize(m_size);
        CUDA_CHECK(cudaMemcpy(dest.data(), m_data_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Copy the contents of this vector to host, creating a new std::vector
    std::vector<T> get () {
        std::vector<T> output;
        copyTo(output);
        return output;
    }

private:
    // Number of elements in vector
    size_t m_size;

    // Pointer to data on device
    T *m_data_ptr;
};

class event {
public:
    event() {
        cudaEventCreate(&m_event);
    }
    ~event() {
        cudaEventDestroy(m_event);
    };

    void record () {
        cudaEventRecord(m_event);
        cudaEventSynchronize(m_event);
    }

    cudaEvent_t m_event;
};

float eventElapsedTime (const event &e1, const event &e2);

} // namespace cuda

#endif