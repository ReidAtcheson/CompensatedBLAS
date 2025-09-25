#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void drotm(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy,
    const double *param
) {
    (void)n;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    (void)param;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void drotm_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *param
) {
    reidblas::ilp64::drotm(n, x, incx, y, incy, param);
}
#endif  // REIDBLAS_USE_ILP64
