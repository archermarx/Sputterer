#include "cuda.cuh"

namespace cuda {
  float event_elapsed_time (const event &e1, const event &e2) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, e1.m_event, e2.m_event);
    return elapsed;
  }
} // namespace cuda
