#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

double dsdot(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy
) {
    (void)n;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    return 0.0;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" double dsdot_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
) {
    return reidblas::ilp64::dsdot(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
