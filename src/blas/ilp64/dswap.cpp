#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void dswap(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy
) {
    (void)n;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void dswap_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::dswap(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
