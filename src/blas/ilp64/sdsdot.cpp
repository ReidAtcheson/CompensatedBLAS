#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

float sdsdot(
    const std::int64_t *n,
    const float *sb,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy
) {
    (void)n;
    (void)sb;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    return 0.0f;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" float sdsdot_(
    const reidblas_blas_int *n,
    const float *sb,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
) {
    return reidblas::ilp64::sdsdot(n, sb, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
