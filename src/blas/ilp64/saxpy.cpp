#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void saxpy(
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy
) {
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void saxpy_(
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::saxpy(n, alpha, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
