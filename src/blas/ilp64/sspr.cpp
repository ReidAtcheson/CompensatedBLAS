#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void sspr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *ap
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)ap;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void sspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *ap
) {
    reidblas::ilp64::sspr(uplo, n, alpha, x, incx, ap);
}
#endif  // REIDBLAS_USE_ILP64
