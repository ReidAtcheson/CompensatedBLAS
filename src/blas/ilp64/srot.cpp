#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void srot(
    const std::int64_t *n,
    float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy,
    const float *c,
    const float *s
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
extern "C" void srot_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy,
    const float *c,
    const float *s
) {
    reidblas::ilp64::srot(n, x, incx, y, incy, c, s);
}
#endif  // REIDBLAS_USE_ILP64
