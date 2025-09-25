#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

double ddot(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx,
    const double *y,
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
extern "C" double ddot_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy
) {
    return reidblas::ilp64::ddot(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
