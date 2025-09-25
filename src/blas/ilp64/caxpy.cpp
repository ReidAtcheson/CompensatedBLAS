#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void caxpy(
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *y,
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
extern "C" void caxpy_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::caxpy(n, alpha, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
