#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void daxpy(
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    double *y,
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
extern "C" void daxpy_(
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::daxpy(n, alpha, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
