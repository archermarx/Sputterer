#pragma once
#ifndef CUDA_CUH
#define CUDA_CUH

#include "../include/cuda_helpers.cuh"

namespace cuda {

class event {
public:
    event() {
        CUDA_CHECK(cudaEventCreate(&m_event));
    }
    ~event() {
        CUDA_CHECK(cudaEventDestroy(m_event));
    };

    void record () const {
        CUDA_CHECK(cudaEventRecord(m_event));
        CUDA_CHECK(cudaEventSynchronize(m_event));
    }

    cudaEvent_t m_event{};
};

float eventElapsedTime (const event &e1, const event &e2);

} // namespace cuda

#endif