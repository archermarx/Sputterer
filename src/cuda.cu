#include "cuda.cuh"

namespace cuda {
float eventElapsedTime (const event &e1, const event &e2) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, e1.m_event, e2.m_event);
    return elapsed;
}
} // namespace cuda
