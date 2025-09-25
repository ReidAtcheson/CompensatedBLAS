#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ssyr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *a,
    const std::int64_t *lda
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)a;
    (void)lda;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void ssyr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *a,
    const reidblas_blas_int *lda
) {
    reidblas::ilp64::ssyr(uplo, n, alpha, x, incx, a, lda);
}
#endif  // REIDBLAS_USE_ILP64
