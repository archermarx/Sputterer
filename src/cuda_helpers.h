#ifndef SPUTTERER_CUDA_HELPERS_H
#define SPUTTERER_CUDA_HELPERS_H

#include <stdio.h>

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

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
