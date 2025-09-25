#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

float scasum(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx
) {
    (void)n;
    (void)x;
    (void)incx;
    return 0.0f;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" float scasum_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    return reidblas::ilp64::scasum(n, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
