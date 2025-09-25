#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void sspr2(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy,
    float *ap
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    (void)ap;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void sspr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *ap
) {
    reidblas::ilp64::sspr2(uplo, n, alpha, x, incx, y, incy, ap);
}
#endif  // REIDBLAS_USE_ILP64
