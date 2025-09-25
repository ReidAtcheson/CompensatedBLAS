#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

reidblas_complex_double zdotu(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
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
extern "C" reidblas_complex_double zdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    return reidblas::ilp64::zdotu(n, x, incx, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
