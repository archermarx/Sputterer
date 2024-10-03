#include "cuda_helpers.h"

namespace cuda {
  float event_elapsed_time (const Event &e1, const Event &e2) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, e1.m_event, e2.m_event);
    return elapsed;
  }
} // namespace cuda
