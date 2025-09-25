#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void sswap(
    const std::int64_t *n,
    float *x,
    const std::int64_t *incx,
    float *y,
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
extern "C" void sswap_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::sswap(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
