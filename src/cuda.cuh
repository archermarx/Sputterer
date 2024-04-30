#pragma once
#ifndef SPUTTERER_CUDA_CUH
#define SPUTTERER_CUDA_CUH

#include "../include/cuda_helpers.cuh"

namespace cuda {

  class Event {
  public:
    Event () {
      CUDA_CHECK(cudaEventCreate(&m_event));
    }

    ~Event () {
      CUDA_CHECK(cudaEventDestroy(m_event));
    };

    void record () const {
      CUDA_CHECK(cudaEventRecord(m_event));
      CUDA_CHECK(cudaEventSynchronize(m_event));
    }

    cudaEvent_t m_event{};
  };

  float event_elapsed_time (const Event &e1, const Event &e2);

} // namespace cuda

#endif