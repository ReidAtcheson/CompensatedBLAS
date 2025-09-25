#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void chpr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *ap
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
extern "C" void chpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *ap
) {
    reidblas::ilp64::chpr(uplo, n, alpha, x, incx, ap);
}
#endif  // REIDBLAS_USE_ILP64
