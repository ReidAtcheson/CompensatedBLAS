#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

reidblas_complex_float cdotu(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy
) {
    (void)n;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    return {};
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" reidblas_complex_float cdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    return reidblas::ilp64::cdotu(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
