#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void drot(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy,
    const double *c,
    const double *s
) {
    (void)n;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    (void)c;
    (void)s;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void drot_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *c,
    const double *s
) {
    reidblas::ilp64::drot(n, x, incx, y, incy, c, s);
}
#endif  // REIDBLAS_USE_ILP64
